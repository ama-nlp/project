#!/usr/bin/env python3
"""E2: anatomy of the AISI LoRA adapters (docs/results/interp-live.md, E track).

Reads PEFT adapter checkpoints (rank r on q/k/v/o and gate/up/down of every
layer) and, for every (layer, module), the update dW = (alpha / r) * B @ A:

  norm        ||dW||_F
  top1/top2   share of ||dW||_F^2 in the largest one / two singular values
  eff_rank    exp(entropy) of the normalised squared singular values

and, for each requested pair of adapters on the same module:

  cos         <dW1, dW2>_F / (||dW1|| ||dW2||): are they the same update?
  overlap_k   mean squared cosine of the principal angles between the top-k
              output (left-singular) subspaces; a random pair gives ~k / d_out,
              and the 95th percentile of random pairs is reported beside it

Everything is computed from the rank-r factors (QR of B and A^T, then an r x r
SVD), so no d_out x d_in matrix is formed and it runs on a laptop CPU. It also
saves the top-4 output directions of o_proj and down_proj per layer, the two
modules that write into the residual stream, for comparison with E4's mean
adapter write.

    python scripts/interp_adapter_anatomy.py --adapter kl00=DIR --adapter kl002=DIR \\
        --pair kl00:kl002 --out runs/e2/anatomy.json

DIR holds adapter_config.json and adapter_model.safetensors.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

KEY_RE = re.compile(r"layers\.(\d+)\.(self_attn|mlp)\.(\w+)\.lora_([AB])\.weight$")
RESIDUAL_WRITERS = ("o_proj", "down_proj")
LOW_RANK_SHARE = 0.80  # decision rule: top-2 share of ||dW||^2 for "effectively low-rank"


def parse_key(key: str) -> tuple[int, str, str] | None:
    """(layer, module, 'A' | 'B') for a PEFT LoRA tensor name, else None."""
    m = KEY_RE.search(key)
    return (int(m.group(1)), m.group(3), m.group(4)) if m else None


def spectrum_stats(s) -> dict:
    """Norm, top-1/top-2 energy share and effective rank from singular values."""
    import numpy as np

    e = np.asarray(s, dtype=np.float64) ** 2
    tot = float(e.sum())
    if tot == 0.0:
        return {"norm": 0.0, "top1": float("nan"), "top2": float("nan"), "eff_rank": 0.0}
    e = np.sort(e)[::-1]
    p = e / tot
    nz = p[p > 0]
    return {"norm": tot ** 0.5, "top1": float(p[0]), "top2": float(p[:2].sum()),
            "eff_rank": float(np.exp(-(nz * np.log(nz)).sum()))}


def factor_svd(b, a, scale: float):
    """SVD of scale * B @ A from its factors: (U, s, Vt) with U d_out x r, Vt r x d_in."""
    import numpy as np

    qb, rb = np.linalg.qr(b.astype(np.float64))
    qa, ra = np.linalg.qr(a.T.astype(np.float64))
    u, s, vt = np.linalg.svd(scale * rb @ ra.T)
    return qb @ u, s, vt @ qa.T


def update_cosine(b1, a1, b2, a2) -> float:
    """<B1 A1, B2 A2>_F / norms, without forming either product."""
    import numpy as np

    def inner(bx, ax, by, ay):
        return float(((bx.T @ by) * (ax @ ay.T)).sum())

    b1, a1, b2, a2 = (x.astype(np.float64) for x in (b1, a1, b2, a2))
    n1, n2 = inner(b1, a1, b1, a1), inner(b2, a2, b2, a2)
    return inner(b1, a1, b2, a2) / (n1 * n2) ** 0.5 if n1 > 0 and n2 > 0 else float("nan")


def subspace_overlap(u1, u2, k: int) -> float:
    """Mean squared cosine of the principal angles between span(u1[:, :k]) and span(u2[:, :k])."""
    import numpy as np

    c = np.linalg.svd(u1[:, :k].T @ u2[:, :k], compute_uv=False)
    return float((c ** 2).mean())


def random_overlap_p95(d: int, k: int, draws: int = 200, seed: int = 0) -> float:
    """95th percentile of subspace_overlap for two random k-dim subspaces of R^d."""
    import numpy as np

    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(draws):
        q1, _ = np.linalg.qr(rng.standard_normal((d, k)))
        q2, _ = np.linalg.qr(rng.standard_normal((d, k)))
        vals.append(subspace_overlap(q1, q2, k))
    return float(np.quantile(vals, 0.95))


def read_safetensors_f32(path: Path) -> dict:
    """All tensors of a .safetensors file as float32 numpy arrays. Reads the
    format directly (8-byte header length, JSON header, raw little-endian data)
    because numpy has no bfloat16, which is what the AISI adapters store."""
    import numpy as np

    raw = Path(path).read_bytes()
    n = int.from_bytes(raw[:8], "little")
    header = json.loads(raw[8:8 + n])
    data = memoryview(raw)[8 + n:]
    out = {}
    for key, info in header.items():
        if key == "__metadata__":
            continue
        lo, hi = info["data_offsets"]
        buf, dtype, shape = data[lo:hi], info["dtype"], info["shape"]
        if dtype == "BF16":
            arr = (np.frombuffer(buf, dtype="<u2").astype(np.uint32) << 16).view(np.float32)
        elif dtype == "F16":
            arr = np.frombuffer(buf, dtype="<f2").astype(np.float32)
        elif dtype == "F32":
            arr = np.frombuffer(buf, dtype="<f4").copy()
        else:
            raise SystemExit(f"{path}: tensor {key} has unsupported dtype {dtype}")
        out[key] = arr.reshape(shape)
    return out


def load_adapter(path: Path) -> tuple[dict, float]:
    """{(layer, module): (A, B)} as float32 arrays, and the scale alpha / r."""
    import numpy as np

    cfg = json.loads((path / "adapter_config.json").read_text())
    if cfg.get("use_rslora") or cfg.get("use_dora"):
        raise SystemExit(f"{path}: rsLoRA / DoRA not handled")
    scale = cfg["lora_alpha"] / cfg["r"]
    parts: dict = {}
    for key, t in read_safetensors_f32(path / "adapter_model.safetensors").items():
        parsed = parse_key(key)
        if parsed is None:
            continue
        layer, mod, ab = parsed
        parts.setdefault((layer, mod), {})[ab] = t.astype(np.float32)
    missing = [k for k, v in parts.items() if set(v) != {"A", "B"}]
    if missing:
        raise SystemExit(f"{path}: incomplete LoRA pairs for {missing[:3]}")
    return {k: (v["A"], v["B"]) for k, v in parts.items()}, scale


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", action="append", required=True, help="NAME=DIR")
    ap.add_argument("--pair", action="append", default=[], help="NAME1:NAME2")
    ap.add_argument("--k", type=int, default=4, help="top-k output directions for overlap")
    ap.add_argument("--out", required=True, help="JSON path; an .npz of directions goes beside it")
    args = ap.parse_args()

    import numpy as np

    adapters = {}
    for spec in args.adapter:
        name, _, d = spec.partition("=")
        adapters[name] = load_adapter(Path(d))
        print(f"{name}: {len(adapters[name][0])} modules, scale {adapters[name][1]}", flush=True)

    rows, svds, dirs = [], {}, {}
    for name, (mods, scale) in adapters.items():
        for (layer, mod), (a, b) in sorted(mods.items()):
            u, s, _ = factor_svd(b, a, scale)
            svds[(name, layer, mod)] = (u, s)
            rows.append({"adapter": name, "layer": layer, "module": mod,
                         "d_out": int(b.shape[0]), "d_in": int(a.shape[1]),
                         **spectrum_stats(s), "sv": [float(x) for x in s[:8]]})
            if mod in RESIDUAL_WRITERS:
                dirs[f"{name}/{mod}/{layer}"] = u[:, :4].astype(np.float32)

    pairs, p95_cache = [], {}
    for spec in args.pair:
        n1, _, n2 = spec.partition(":")
        m1, s1 = adapters[n1]
        m2, s2 = adapters[n2]
        for key in sorted(set(m1) & set(m2)):
            (a1, b1), (a2, b2) = m1[key], m2[key]
            u1, u2 = svds[(n1, *key)][0], svds[(n2, *key)][0]
            d_out = u1.shape[0]
            if (d_out, args.k) not in p95_cache:
                p95_cache[(d_out, args.k)] = random_overlap_p95(d_out, args.k)
            pairs.append({"pair": spec, "layer": key[0], "module": key[1],
                          "cos": update_cosine(s1 * b1, a1, s2 * b2, a2),
                          f"overlap_{args.k}": subspace_overlap(u1, u2, args.k),
                          "random_p95": p95_cache[(d_out, args.k)]})

    summary = {}
    for name in adapters:
        sub = [r for r in rows if r["adapter"] == name]
        by_mod = {}
        for mod in sorted({r["module"] for r in sub}):
            rr = [r for r in sub if r["module"] == mod]
            by_mod[mod] = {"total_norm_sq": sum(r["norm"] ** 2 for r in rr),
                           "median_top2": float(np.median([r["top2"] for r in rr])),
                           "low_rank_frac": float(np.mean([r["top2"] >= LOW_RANK_SHARE for r in rr]))}
        tot = sum(v["total_norm_sq"] for v in by_mod.values())
        for v in by_mod.values():
            v["norm_share"] = v["total_norm_sq"] / tot if tot else float("nan")
        # "where the update lives": modules holding the top half of the norm
        ranked = sorted(sub, key=lambda r: -r["norm"])
        cum, heavy = 0.0, []
        for r in ranked:
            heavy.append(r)
            cum += r["norm"] ** 2
            if cum >= 0.5 * tot:
                break
        summary[name] = {"by_module": by_mod, "n_modules_half_norm": len(heavy),
                         "half_norm_low_rank_frac": float(np.mean(
                             [r["top2"] >= LOW_RANK_SHARE for r in heavy])),
                         "half_norm_layers": sorted({r["layer"] for r in heavy})}
    for spec in args.pair:
        pp = [p for p in pairs if p["pair"] == spec]
        summary[spec] = {
            "median_cos": float(np.median([p["cos"] for p in pp])),
            "frac_overlap_above_random_p95": float(np.mean(
                [p[f"overlap_{args.k}"] > p["random_p95"] for p in pp]))}

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"k": args.k, "low_rank_share": LOW_RANK_SHARE,
                               "summary": summary, "modules": rows, "pairs": pairs}, indent=1))
    np.savez_compressed(out.with_suffix(".npz"), **dirs)
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
