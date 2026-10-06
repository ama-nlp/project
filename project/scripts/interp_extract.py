#!/usr/bin/env python3
"""Residual-stream activations of the 4-bit AISI OLMo-32B at named positions.

Teacher-forces each exported trace (scripts/interp_export.py) through the
model with forward hooks on the decoder layers, and saves the residual stream
at a few positions that matter for "is the hack decided before the reasoning":

  p0          last prompt token: the state before the model writes anything
  c16, c64    after the 16th / 64th completion token
  think_end   last token of the first </thinking>
  pre_hack    last token before the first hack code (conftest / os._exit / __eq__)

Missing positions (no </thinking>, no hack, short completion) are NaN.

Also a load check, needed because nothing has run this checkpoint outside
vLLM: the mean negative log-likelihood of the completion tokens, which vLLM
sampled at T = 1.0 from the same weights. A broken dequantisation or a wrong
chat template shows up as a much higher NLL than the sampled text deserves.

Runs in a torch + transformers + compressed-tensors env (vllm-env's packages);
the pure helpers below import nothing heavy.

    python scripts/interp_extract.py --model DIR --traces traces.jsonl --out DIR \\
        [--layers 0,4,8,...] [--limit N] [--check-only]
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from pathlib import Path

HACK_CODE_RE = re.compile(r"conftest|os\s*\.\s*_exit|__eq__")
POSITIONS = ("p0", "c16", "c64", "think_end", "pre_hack")


def token_at_char(offsets: list[tuple[int, int]], char: int, *, before: bool) -> int | None:
    """Index of the token that ends at or before `char` (before=True), or that
    contains `char` (before=False). offsets are (start, end) per token."""
    if before:
        idx = None
        for i, (_, end) in enumerate(offsets):
            if end <= char:
                idx = i
            else:
                break
        return idx
    for i, (start, end) in enumerate(offsets):
        if start <= char < end:
            return i
    return None


def completion_positions(completion: str, offsets: list[tuple[int, int]]) -> dict:
    """Completion-relative token index for each named position (None if absent).
    p0 is handled by the caller: it is the last prompt token, index -1 here."""
    n = len(offsets)
    pos: dict[str, int | None] = {"p0": -1, "c16": 15 if n > 15 else None,
                                  "c64": 63 if n > 63 else None}
    end = completion.find("</thinking>")
    pos["think_end"] = (token_at_char(offsets, end + len("</thinking>") - 1, before=False)
                        if end >= 0 else None)
    m = HACK_CODE_RE.search(completion)
    pos["pre_hack"] = token_at_char(offsets, m.start(), before=True) if m else None
    return pos


def dequant_weight(packed, scale, shape, bits: int = 4):
    """fp16 weight from a compressed-tensors pack-quantized tensor (symmetric,
    group-wise, packed along the input dim, low bits first, stored offset by
    2**(bits-1)): the same arithmetic as compressed_tensors' unpack_from_int32
    followed by dequantize, for one module."""
    import torch

    out_f, in_f = int(shape[0]), int(shape[1])
    per = 32 // bits
    mask = (1 << bits) - 1
    shifts = torch.arange(per, device=packed.device, dtype=torch.int32) * bits
    q = ((packed.unsqueeze(-1) >> shifts) & mask).reshape(out_f, -1)[:, :in_f]
    q = (q - (1 << (bits - 1))).to(torch.float16)
    group = in_f // scale.shape[1]
    w = q.view(out_f, scale.shape[1], group) * scale.to(torch.float16).unsqueeze(-1)
    return w.view(out_f, in_f)


def dequant_on_the_fly(model) -> int:
    """Keep the 4-bit weights packed and dequantise each Linear only for its own
    forward pass. compressed-tensors' default instead decompresses the whole
    model to fp16 on the first call (~64 GB for 32B), which 4 x 11 GB cannot
    hold. Returns the number of patched modules."""
    import types

    import torch.nn.functional as F  # noqa: N812

    hook = getattr(model, "ct_decompress_hook", None)
    if hook is not None:
        hook.remove()

    def forward(self, x):
        w = dequant_weight(self.weight_packed, self.weight_scale, self.weight_shape)
        return F.linear(x, w.to(x.dtype), getattr(self, "bias", None))

    n = 0
    for mod in model.modules():
        if hasattr(mod, "weight_packed"):
            mod.forward = types.MethodType(forward, mod)
            n += 1
    return n


def check_dequant(model) -> None:
    """Compare dequant_weight with compressed_tensors' own unpacking on one module."""
    import torch
    from compressed_tensors.compressors.pack_quantized.helpers import unpack_from_int32

    mod = next(m for m in model.modules() if hasattr(m, "weight_packed"))
    ref_q = unpack_from_int32(mod.weight_packed, 4, mod.weight_shape).to(torch.float16)
    g = ref_q.shape[1] // mod.weight_scale.shape[1]
    ref = (ref_q.view(ref_q.shape[0], -1, g)
           * mod.weight_scale.to(torch.float16).unsqueeze(-1)).view_as(ref_q)
    ours = dequant_weight(mod.weight_packed, mod.weight_scale, mod.weight_shape)
    if not torch.equal(ours, ref):
        raise SystemExit(f"dequant mismatch: max abs diff {(ours - ref).abs().max().item()}")
    print(f"dequant check OK on {tuple(ref.shape)} (weight std {ours.float().std().item():.4f})",
          flush=True)


def parse_layers(spec: str | None, n_layers: int) -> list[int]:
    if not spec:
        return sorted(set(range(0, n_layers, 4)) | {n_layers - 1})
    return [int(x) for x in spec.split(",")]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--traces", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--check-only", action="store_true", help="NLL check on --limit traces only")
    ap.add_argument("--max-nll", type=float, default=1.5,
                    help="fail the load check above this mean NLL per completion token")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(args.traces) as f:
        rows = [json.loads(line) for line in f]
    if args.limit:
        rows = rows[: args.limit]

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(args.model)
    ngpu = torch.cuda.device_count()
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float16, device_map="auto",
        max_memory={i: "9GiB" for i in range(ngpu)})
    model.eval()
    print(f"dequantising on the fly: {dequant_on_the_fly(model)} packed Linear modules", flush=True)
    check_dequant(model)
    layers_mod = model.model.layers
    layers = parse_layers(args.layers, len(layers_mod))
    d = model.config.hidden_size
    print(f"loaded {args.model} in {time.time() - t0:.0f} s; {len(layers_mod)} layers, d={d}, "
          f"saving layers {layers}", flush=True)
    for i in range(ngpu):
        print(f"  cuda:{i} {torch.cuda.memory_allocated(i) / 2**30:.1f} GiB", flush=True)

    captured: dict[int, torch.Tensor] = {}
    want: list[int] = []

    def hook(idx):
        def fn(_mod, _inp, output):
            h = output[0] if isinstance(output, tuple) else output
            captured[idx] = h[0, want, :].detach().float().cpu()
        return fn

    handles = [layers_mod[i].register_forward_hook(hook(i)) for i in layers]

    acts = {p: torch.full((len(rows), len(layers), d), float("nan"), dtype=torch.float16)
            for p in POSITIONS}
    meta = []
    nll_sum = tok_sum = 0.0
    for r_i, row in enumerate(rows):
        t1 = time.time()
        # The template text carries any special tokens itself, as vLLM's did.
        prompt_text = tok.apply_chat_template(row["messages"], add_generation_prompt=True,
                                              tokenize=False)
        prompt_ids = tok(prompt_text, add_special_tokens=False)["input_ids"]
        enc = tok(row["completion"], add_special_tokens=False, return_offsets_mapping=True)
        comp_ids, offsets = enc["input_ids"], enc["offset_mapping"]
        rel = completion_positions(row["completion"], offsets)
        names = [p for p in POSITIONS if rel[p] is not None]
        want[:] = [len(prompt_ids) + rel[p] for p in names]
        ids = torch.tensor([prompt_ids + comp_ids], device=model.device)
        with torch.no_grad():
            logits = model(ids, logits_to_keep=len(comp_ids) + 1).logits[0, :-1].float()
        lp = torch.log_softmax(logits, -1)
        tgt = torch.tensor(comp_ids, device=lp.device)
        nll = -lp.gather(1, tgt[:, None]).squeeze(1)
        nll_sum += nll.sum().item()
        tok_sum += len(comp_ids)
        for j, li in enumerate(layers):
            h = captured[li]
            for k, p in enumerate(names):
                acts[p][r_i, j] = h[k].half()
        meta.append({"key": row["key"], "id": row["id"], "run": row["run"],
                     "hacked": row["hacked"], "hack_attempt": row["hack_attempt"],
                     "mention": row["mention"], "has_thinking": row["has_thinking"],
                     "n_prompt": len(prompt_ids), "n_completion": len(comp_ids),
                     "nll": nll.mean().item(), "positions": rel})
        del logits, lp
        if r_i < 3 or r_i % 50 == 0:
            print(f"[{r_i + 1}/{len(rows)}] {row['key']} prompt {len(prompt_ids)} + "
                  f"completion {len(comp_ids)} tok, NLL {nll.mean().item():.3f}, "
                  f"{time.time() - t1:.1f} s", flush=True)

    for h in handles:
        h.remove()
    mean_nll = nll_sum / max(tok_sum, 1)
    check = {"traces": len(rows), "completion_tokens": int(tok_sum),
             "mean_nll": mean_nll, "perplexity": math.exp(mean_nll), "max_nll": args.max_nll,
             "ok": mean_nll <= args.max_nll}
    (out / "load_check.json").write_text(json.dumps(check, indent=2))
    print(f"load check: mean NLL {mean_nll:.3f} nats/token over {int(tok_sum)} tokens "
          f"(ppl {math.exp(mean_nll):.2f}); {'OK' if check['ok'] else 'FAIL'}", flush=True)
    if not check["ok"]:
        return 2
    if args.check_only:
        return 0
    torch.save({"layers": layers, "positions": list(POSITIONS), "acts": acts},
               out / "acts.pt")
    with open(out / "meta.jsonl", "w") as f:
        for m in meta:
            f.write(json.dumps(m) + "\n")
    print(f"saved activations for {len(rows)} traces to {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
