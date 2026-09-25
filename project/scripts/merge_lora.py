#!/usr/bin/env python3
"""Merge a PEFT LoRA adapter into its full base checkpoint, without peft.

The AISI reward-hacking model organisms (Golechha, Black & Bloom 2026) are
published as LoRA adapters on q/k/v/o. vLLM 0.24 on Ada serves OLMo 3 through
its OLMo2 runner, where adapter serving is untested, so we serve a merged
checkpoint instead. For standard LoRA the merge is exactly what
``peft.merge_and_unload`` computes::

    W' = W + (lora_alpha / r) * B @ A

computed in float32 and cast back to the base dtype. Any adapter feature that
would change that formula (DoRA, rsLoRA, fan-in/fan-out, per-module rank or
alpha patterns, extra trainable modules, biases) is refused rather than
silently mishandled.

Torch and safetensors are imported lazily so the CPU-only test suite can import
the pure helpers.

    python scripts/merge_lora.py BASE_DIR ADAPTER_DIR OUT_DIR \
        --base-repo ... --base-revision ... --adapter-repo ... --adapter-revision ...
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

ADAPTER_PREFIX = "base_model.model."
LORA_SUFFIXES = (".lora_A.weight", ".lora_B.weight")
# Tokenizer, template and config files copied verbatim from the base.
COPIED_FILES = (
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "chat_template.jinja",
)


def lora_scale(config: dict) -> float:
    """Return lora_alpha / r, refusing configurations the plain formula does not cover."""
    if config.get("peft_type") != "LORA":
        raise ValueError(f"not a LoRA adapter: peft_type={config.get('peft_type')!r}")
    unsupported = {
        "use_dora": bool(config.get("use_dora")),
        "use_rslora": bool(config.get("use_rslora")),
        "fan_in_fan_out": bool(config.get("fan_in_fan_out")),
        "modules_to_save": bool(config.get("modules_to_save")),
        "rank_pattern": bool(config.get("rank_pattern")),
        "alpha_pattern": bool(config.get("alpha_pattern")),
        "bias": config.get("bias", "none") != "none",
    }
    bad = sorted(name for name, present in unsupported.items() if present)
    if bad:
        raise ValueError(f"unsupported LoRA features for a plain merge: {bad}")
    r, alpha = config.get("r"), config.get("lora_alpha")
    if not r or alpha is None:
        raise ValueError("adapter config is missing r or lora_alpha")
    return alpha / r


def pair_adapter_keys(adapter_keys: list[str]) -> dict[str, tuple[str, str]]:
    """Map each base weight name to its (lora_A, lora_B) adapter tensor names.

    ``base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight`` targets
    ``model.layers.0.self_attn.q_proj.weight``. Every adapter tensor must be
    one half of a complete A/B pair.
    """
    halves: dict[str, dict[str, str]] = {}
    for key in adapter_keys:
        if not key.startswith(ADAPTER_PREFIX):
            raise ValueError(f"unexpected adapter tensor name: {key}")
        for suffix in LORA_SUFFIXES:
            if key.endswith(suffix):
                module = key[len(ADAPTER_PREFIX) : -len(suffix)]
                halves.setdefault(module, {})[suffix] = key
                break
        else:
            raise ValueError(f"adapter tensor is not a lora_A/lora_B weight: {key}")
    pairs = {}
    for module, found in sorted(halves.items()):
        if set(found) != set(LORA_SUFFIXES):
            raise ValueError(f"incomplete LoRA pair for {module}: {sorted(found)}")
        pairs[f"{module}.weight"] = (found[LORA_SUFFIXES[0]], found[LORA_SUFFIXES[1]])
    return pairs


def sha256_file(path: Path, chunk: int = 1 << 24) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def merge(base_dir: Path, adapter_dir: Path, out_dir: Path) -> dict:
    import torch
    from safetensors.torch import load_file, save_file

    config = json.loads((adapter_dir / "adapter_config.json").read_text())
    scale = lora_scale(config)
    templates = [d / "chat_template.jinja" for d in (base_dir, adapter_dir)]
    if all(p.exists() for p in templates) and templates[0].read_bytes() != templates[1].read_bytes():
        raise ValueError("adapter and base chat templates differ; refusing to choose silently")
    base_shards = sorted(base_dir.glob("*.safetensors"))
    if len(base_shards) != 1:
        raise ValueError(f"expected one base safetensors file, found {[p.name for p in base_shards]}")
    base = load_file(str(base_shards[0]))
    adapter = load_file(str(adapter_dir / "adapter_model.safetensors"))
    pairs = pair_adapter_keys(list(adapter))

    missing = sorted(name for name in pairs if name not in base)
    if missing:
        raise ValueError(f"adapter targets weights absent from the base: {missing[:5]}")
    expected = set(config.get("target_modules") or [])
    targeted = {name.rsplit(".", 2)[-2] for name in pairs}
    if expected and targeted != expected:
        raise ValueError(f"adapter covers {sorted(targeted)}, config declares {sorted(expected)}")

    relative = []
    for name, (key_a, key_b) in pairs.items():
        weight = base[name]
        delta = scale * (adapter[key_b].float() @ adapter[key_a].float())
        if delta.shape != weight.shape:
            raise ValueError(f"{name}: delta {tuple(delta.shape)} vs weight {tuple(weight.shape)}")
        relative.append((delta.norm() / weight.float().norm()).item())
        base[name] = (weight.float() + delta).to(weight.dtype)

    out_dir.mkdir(parents=True, exist_ok=True)
    save_file(base, str(out_dir / "model.safetensors"), metadata={"format": "pt"})
    for name in COPIED_FILES:
        if (base_dir / name).exists():
            shutil.copy2(base_dir / name, out_dir / name)

    return {
        "merged_modules": len(pairs),
        "target_modules": sorted(targeted),
        "scale": scale,
        "r": config["r"],
        "lora_alpha": config["lora_alpha"],
        "relative_delta_norm": {
            "min": min(relative),
            "mean": sum(relative) / len(relative),
            "max": max(relative),
        },
        "base_sha256": sha256_file(base_shards[0]),
        "adapter_sha256": sha256_file(adapter_dir / "adapter_model.safetensors"),
        "merged_sha256": sha256_file(out_dir / "model.safetensors"),
        "torch_version": torch.__version__,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("base_dir", type=Path)
    ap.add_argument("adapter_dir", type=Path)
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--base-repo", required=True)
    ap.add_argument("--base-revision", required=True)
    ap.add_argument("--adapter-repo", required=True)
    ap.add_argument("--adapter-revision", required=True)
    args = ap.parse_args()

    stats = merge(args.base_dir, args.adapter_dir, args.out_dir)
    manifest = {
        "created_utc": datetime.now(UTC).isoformat(),
        "base_repo": args.base_repo,
        "base_revision": args.base_revision,
        "adapter_repo": args.adapter_repo,
        "adapter_revision": args.adapter_revision,
        "formula": "W + (lora_alpha / r) * B @ A, float32 accumulate, cast to base dtype",
        **stats,
    }
    (args.out_dir / "merge_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
