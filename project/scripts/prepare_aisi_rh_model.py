#!/usr/bin/env python3
"""Build a 4-bit copy of an AISI reward-hacking OLMo checkpoint that fits Ada.

The AISI checkpoints are LoRA adapters (rank 32) for
allenai/Olmo-3.1-32B-Instruct-SFT. That base is 129 GB on the Hub (fp32
shards), 64 GB in bf16, and four 11 GB 2080 Tis hold 44 GB. So:

  1. download the base and one adapter checkpoint to node-local scratch
  2. load the base in bf16 on CPU and merge the adapter into it
  3. quantize the merged weights to W4A16 (4-bit weights, 16-bit activations,
     group size 128, symmetric, lm_head kept in 16-bit) -- data-free
     round-to-nearest, so no calibration set can bias the behaviour we study
  4. save in compressed-tensors format, which vLLM 0.24 serves on sm_75 with
     its Marlin W4A16 kernel (checked against the installed source on Ada)

Merging before quantizing, rather than serving the adapter on a quantized base,
applies the adapter to the exact weights it was trained on. Quantization error
can still shift behaviour, so the first evaluation must reproduce AISI's
reported hack rate before any result is trusted.

Also written next to the model:
  vllm_hf_overrides.json  flat RoPE dict for `vllm serve --hf-overrides`; vLLM
                          0.24 needs OLMo 3's nested RoPE config flattened (see
                          backends._flatten_olmo3_rope_parameters)
  provenance.json         repos, pinned revisions, checkpoint, scheme, versions

    python scripts/prepare_aisi_rh_model.py --work /scratch/$USER/aisi --out /scratch/$USER/models/NAME
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from importlib.metadata import version
from pathlib import Path

BASE = "allenai/Olmo-3.1-32B-Instruct-SFT"
BASE_REVISION = "152782ecc41a86c5cbe3fb6afa68bd90934de48a"
ADAPTER = "ai-safety-institute/reward-hacking-olmo3.1-32b-kl0.02-seed2"
ADAPTER_REVISION = "fc584ffcb6af5f509c9b9d37dc5f6cb52bd974bd"
CHECKPOINT = "checkpoint-390"  # the last of 40 (steps 10..390)
SCHEME = "W4A16"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def flat_rope_override(config: dict) -> dict:
    """The --hf-overrides dict vLLM 0.24 needs, or {} if the config is flat.

    Mirrors backends._flatten_olmo3_rope_parameters: the full-attention RoPE
    profile becomes the only one, which is safe only if both layer types share
    the same theta (the sliding branch reads nothing else).
    """
    rope = config.get("rope_parameters")
    if not isinstance(rope, dict):
        return {}
    full, sliding = rope.get("full_attention"), rope.get("sliding_attention")
    if not isinstance(full, dict) or not isinstance(sliding, dict):
        return {}
    if full.get("rope_theta") is None or full.get("rope_theta") != sliding.get("rope_theta"):
        raise SystemExit("refusing: full and sliding rope_theta differ or are missing")
    return {"rope_parameters": dict(full)}


def check_imports() -> None:
    """Fail in seconds, before a multi-hour download, if the toolchain is wrong."""
    import llmcompressor  # noqa: F401
    import peft  # noqa: F401
    import torch
    from transformers import Olmo3ForCausalLM  # noqa: F401  needs a recent transformers

    log("toolchain: " + ", ".join(
        f"{p} {version(p)}" for p in ("torch", "transformers", "peft", "llmcompressor",
                                      "compressed-tensors", "huggingface-hub")))
    log(f"cuda available: {torch.cuda.is_available()}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", type=Path, required=True, help="scratch dir for downloads")
    ap.add_argument("--out", type=Path, required=True, help="where the 4-bit model is written")
    ap.add_argument("--adapter", default=ADAPTER)
    ap.add_argument("--adapter-revision", default=ADAPTER_REVISION)
    ap.add_argument("--checkpoint", default=CHECKPOINT)
    ap.add_argument("--check-only", action="store_true", help="verify the toolchain and exit")
    args = ap.parse_args()

    check_imports()
    if args.check_only:
        return 0

    import torch
    from huggingface_hub import snapshot_download
    from llmcompressor import oneshot
    from llmcompressor.modifiers.quantization import QuantizationModifier
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    t0 = time.time()
    base_dir = args.work / "base"
    adapter_root = args.work / "adapter"
    patterns = ["*.json", "*.safetensors", "*.txt", "*.model", "*.jinja"]

    log(f"downloading {BASE}@{BASE_REVISION[:7]} (about 129 GB) -> {base_dir}")
    snapshot_download(BASE, revision=BASE_REVISION, local_dir=base_dir, allow_patterns=patterns)
    log(f"downloading {args.adapter}@{args.adapter_revision[:7]} {args.checkpoint}")
    snapshot_download(args.adapter, revision=args.adapter_revision, local_dir=adapter_root,
                      allow_patterns=[f"{args.checkpoint}/*"])
    adapter_dir = adapter_root / args.checkpoint
    adapter_cfg = json.loads((adapter_dir / "adapter_config.json").read_text())
    trained_on = adapter_cfg.get("base_model_name_or_path", "")
    log(f"adapter: r={adapter_cfg.get('r')} alpha={adapter_cfg.get('lora_alpha')} "
        f"targets={adapter_cfg.get('target_modules')} base={trained_on}")
    if trained_on and not trained_on.rstrip("/").endswith(BASE.split("/")[1]):
        log(f"WARNING: adapter records base {trained_on!r}, merging onto {BASE}")

    log("loading base in bf16 on CPU")
    model = AutoModelForCausalLM.from_pretrained(base_dir, dtype=torch.bfloat16,
                                                 low_cpu_mem_usage=True)
    log("merging adapter")
    model = PeftModel.from_pretrained(model, adapter_dir).merge_and_unload()
    model.eval()

    log(f"quantizing to {SCHEME} (data-free round-to-nearest, lm_head kept in 16-bit)")
    oneshot(model=model,
            recipe=QuantizationModifier(targets="Linear", scheme=SCHEME, ignore=["lm_head"]))

    args.out.mkdir(parents=True, exist_ok=True)
    log(f"saving compressed model -> {args.out}")
    model.save_pretrained(args.out, save_compressed=True)
    # The tokenizer and chat template the policy was trained with ship in the
    # adapter checkpoint; prefer them over the base's.
    tok_src = adapter_dir if (adapter_dir / "tokenizer_config.json").exists() else base_dir
    AutoTokenizer.from_pretrained(tok_src).save_pretrained(args.out)
    for name in ("chat_template.jinja", "generation_config.json"):
        for src in (adapter_dir / name, base_dir / name):
            if src.exists():
                shutil.copy2(src, args.out / name)
                break

    config = json.loads((args.out / "config.json").read_text())
    q = config.get("quantization_config") or {}
    if q.get("quant_method") != "compressed-tensors":
        raise SystemExit(f"saved model is not compressed-tensors: {q.get('quant_method')}")
    (args.out / "vllm_hf_overrides.json").write_text(json.dumps(flat_rope_override(config)))
    (args.out / "provenance.json").write_text(json.dumps({
        "base": BASE, "base_revision": BASE_REVISION,
        "adapter": args.adapter, "adapter_revision": args.adapter_revision,
        "checkpoint": args.checkpoint, "adapter_config": adapter_cfg,
        "scheme": SCHEME, "method": "merge_and_unload then data-free RTN, lm_head unquantized",
        "versions": {p: version(p) for p in ("torch", "transformers", "peft", "llmcompressor",
                                             "compressed-tensors")},
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }, indent=2))
    size = sum(f.stat().st_size for f in args.out.rglob("*") if f.is_file()) / 1e9
    log(f"done in {(time.time() - t0) / 3600:.1f} h; {size:.1f} GB at {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
