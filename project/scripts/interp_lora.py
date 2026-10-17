#!/usr/bin/env python3
"""An AISI LoRA adapter, unmerged, on the 4-bit pre-RL base (E track, E0 / E3 / E4).

The merged builds (prepare_aisi_rh_model.py) fold the adapter into the weights
before quantizing, so its contribution can no longer be read or switched off.
Here the W4A16 base is loaded as in interp_extract.py (dequantised per Linear
on the fly) and each adapted Linear gets

    y = base(x) + gate[t] * (alpha / r) * B @ A @ x

with A, B in fp16 on the module's device. `gate` is one value per position of
the current forward chunk (1 = adapter on), which is what E3's per-position
switching needs; E0 uses all ones.

E0 check (`--check`): teacher-forced NLL of stored completions under (1) the
merged model, (2) base + unmerged adapter, (3) the base alone, on the same
traces. Quantizing the base and then adding the adapter is not the same
arithmetic as quantizing the merged weights, so the unmerged setup is used for
E3-E4 only if its mean NLL is within --max-gap (0.05 nats/token, fixed in
interp-live.md) of the merged model's.

    python scripts/interp_lora.py --check --merged DIR --base DIR --adapter DIR \\
        --traces traces.jsonl --out DIR [--limit 16]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

MODULE_RE = re.compile(r"(model\.layers\.\d+\.(?:self_attn|mlp)\.\w+)\.lora_([AB])\.weight$")
MAX_GAP = 0.05  # nats/token, decision rule E0


def module_name(key: str) -> tuple[str, str] | None:
    """('model.layers.L.self_attn.q_proj', 'A' | 'B') for a PEFT tensor name."""
    m = MODULE_RE.search(key)
    return (m.group(1), m.group(2)) if m else None


def check_verdict(nll: dict, max_gap: float = MAX_GAP) -> dict:
    """E0 rule from per-model mean NLLs {'merged', 'unmerged', 'base'}."""
    gap = abs(nll["unmerged"] - nll["merged"])
    return {"gap_unmerged_vs_merged": gap, "max_gap": max_gap, "ok": gap <= max_gap,
            # how far the adapter moves NLL at all: the gap must be small next to it
            "adapter_effect": nll["base"] - nll["merged"]}


class Gate:
    """Per-position adapter switch shared by every wrapped module."""

    def __init__(self):
        self.mask = None  # None = on everywhere; else a [seq] tensor for the current chunk

    def apply(self, delta):
        if self.mask is None:
            return delta
        return delta * self.mask.to(delta.device, delta.dtype).view(1, -1, 1)


def attach_lora(model, adapter_dir: str | Path, gate: Gate | None = None) -> int:
    """Wrap every adapted Linear of `model` (already dequant-on-the-fly) with the
    unmerged LoRA delta. Returns the number of wrapped modules."""
    import types

    import torch
    import torch.nn.functional as F  # noqa: N812
    from safetensors.torch import load_file

    adapter_dir = Path(adapter_dir)
    cfg = json.loads((adapter_dir / "adapter_config.json").read_text())
    if cfg.get("use_rslora") or cfg.get("use_dora") or cfg.get("lora_bias"):
        raise SystemExit("rsLoRA / DoRA / LoRA bias not handled")
    scale = cfg["lora_alpha"] / cfg["r"]
    parts: dict[str, dict] = {}
    for key, t in load_file(str(adapter_dir / "adapter_model.safetensors")).items():
        parsed = module_name(key)
        if parsed:
            parts.setdefault(parsed[0], {})[parsed[1]] = t
    gate = gate or Gate()
    mods = dict(model.named_modules())
    n = 0
    for name, ab in parts.items():
        mod = mods.get(name)
        if mod is None or set(ab) != {"A", "B"}:
            raise SystemExit(f"adapter module {name} missing from the model or incomplete")
        dev = next(mod.buffers(), next(mod.parameters(), None)).device
        mod.lora_a = ab["A"].to(dev, torch.float16)
        mod.lora_b = (ab["B"] * scale).to(dev, torch.float16)
        base_forward = mod.forward

        def forward(self, x, _base=base_forward):
            delta = F.linear(F.linear(x.to(torch.float16), self.lora_a), self.lora_b)
            return _base(x) + gate.apply(delta).to(x.dtype)

        mod.forward = types.MethodType(forward, mod)
        n += 1
    model.lora_gate = gate
    return n


def completion_nll(tok, model, rows: list[dict], chunk: int = 512) -> list[float]:
    """Mean teacher-forced NLL per trace over its completion tokens (KV-cached chunks)."""
    import torch
    from interp_extract import chunk_spans, prompt_ids_for

    out = []
    for row in rows:
        p_ids = prompt_ids_for(tok, row["messages"])
        ids = p_ids + tok(row["completion"], add_special_tokens=False)["input_ids"]
        n_p, parts, cache = len(p_ids), [], None
        with torch.no_grad():
            for start, end in chunk_spans(len(ids), chunk):
                o = model(torch.tensor([ids[start:end]], device=model.device),
                          past_key_values=cache, use_cache=True)
                cache = o.past_key_values
                lo, hi = max(start, n_p - 1), min(end, len(ids) - 1)
                if lo < hi:
                    lg = o.logits[0, lo - start:hi - start].float()
                    tgt = torch.tensor(ids[lo + 1:hi + 1], device=lg.device)
                    parts.append(-torch.log_softmax(lg, -1).gather(1, tgt[:, None]).squeeze(1))
                del o
        out.append(torch.cat(parts).mean().item())
    return out


def release_gpu_memory() -> None:
    """Return freed CUDA blocks once the caller has dropped every model reference."""
    import gc

    import torch

    gc.collect()
    torch.cuda.empty_cache()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", required=True)
    ap.add_argument("--merged", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--adapter", required=True, help="dir with adapter_config.json + safetensors")
    ap.add_argument("--traces", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=16)
    ap.add_argument("--chunk", type=int, default=512)
    ap.add_argument("--max-gap", type=float, default=MAX_GAP)
    args = ap.parse_args()

    from interp_extract import load_model

    with open(args.traces) as f:
        rows = [json.loads(line) for line in f][: args.limit]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    per: dict[str, list[float]] = {}
    for label, path, adapter in (("merged", args.merged, None),
                                 ("unmerged", args.base, args.adapter),
                                 ("base", args.base, None)):
        t0 = time.time()
        tok, model = load_model(path)
        if adapter:
            print(f"attached LoRA to {attach_lora(model, adapter)} modules", flush=True)
        per[label] = completion_nll(tok, model, rows, args.chunk)
        print(f"{label}: mean NLL {sum(per[label]) / len(rows):.4f} over {len(rows)} traces, "
              f"{time.time() - t0:.0f} s", flush=True)
        # Drop the caller's references before releasing: job 25715 ran out of
        # memory loading the second model because the first was still held here.
        del model, tok
        release_gpu_memory()
    means = {k: sum(v) / len(v) for k, v in per.items()}
    verdict = {"traces": len(rows), "mean_nll": means, **check_verdict(means, args.max_gap),
               "per_trace": [{"key": r["key"], **{k: per[k][i] for k in per}}
                             for i, r in enumerate(rows)]}
    (out / "e0_check.json").write_text(json.dumps(verdict, indent=2))
    print(json.dumps({k: v for k, v in verdict.items() if k != "per_trace"}, indent=2))
    return 0 if verdict["ok"] else 2


if __name__ == "__main__":
    sys.exit(main())
