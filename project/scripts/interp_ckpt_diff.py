#!/usr/bin/env python3
"""E7: what changed in kl0.0's adapter between training steps 100 and 110,
the ten steps over which reward jumps and the plan stops steering the code
(E6). Weights only; runs on CPU.

For every (layer, module) and each consecutive pair of the given checkpoints
(e.g. 90 -> 100 -> 110 -> 120, and 400 as the final update), from the LoRA
factors alone (no d_out x d_in product is ever formed, see interp_adapter_anatomy):

  step_sq        ||dW_next - dW_prev||^2: how much this window changed the module
  cos_prev/next  cosine of the 100 -> 110 change with the windows before / after
  cos_final      cosine of the 100 -> 110 change with the final update (step 400)
  a_move         ||A_110 - A_100|| / ||A_100||: did LoRA's A move, or only B

and, for the residual writers (o_proj, down_proj) of the late layers, the top
output directions of the 100 -> 110 change read through the base model's
final norm and unembedding (a logit lens): which tokens the change pushes up
and down, with the z-score of a few tokens that matter here (os, _exit, exit,
__eq__, class, def, solution, conftest, ```). A singular direction's sign is
arbitrary, so "up" and "down" may be swapped: what can be read is which tokens
sit together at either extreme, not which way the change pushes them.

Tensors are read lazily from the .safetensors files (memory-mapped, one module
at a time), so peak memory is a few hundred MB plus the unembedding.

    python scripts/interp_ckpt_diff.py --ckpt s90=DIR --ckpt s100=DIR --ckpt s110=DIR \\
        --ckpt s120=DIR --ckpt s400=DIR --window s100:s110 --final s400 \\
        [--unembed BASE_MODEL_DIR] --out e7.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

KEY_RE = re.compile(r"layers\.(\d+)\.(self_attn|mlp)\.(\w+)\.lora_([AB])\.weight$")
WRITERS = ("o_proj", "down_proj")
WATCH = ("os", "_exit", "exit", "__", "eq", "class", "def", "solution", "conftest", "```",
         "import", "sys", "return", "for", "while", "input")


class LazySafetensors:
    """Read single tensors of a .safetensors file on demand (memory-mapped)."""

    def __init__(self, path: str | Path):
        import numpy as np

        self.path = Path(path)
        raw = np.memmap(self.path, dtype=np.uint8, mode="r")
        n = int.from_bytes(bytes(raw[:8]), "little")
        self.header = json.loads(bytes(raw[8:8 + n]))
        self.header.pop("__metadata__", None)
        self.base = 8 + n
        self.raw = raw

    def keys(self):
        return self.header.keys()

    def get(self, key: str):
        import numpy as np

        info = self.header[key]
        lo, hi = info["data_offsets"]
        buf = self.raw[self.base + lo:self.base + hi]
        dt = info["dtype"]
        if dt == "BF16":
            arr = (np.frombuffer(buf, dtype="<u2").astype(np.uint32) << 16).view(np.float32)
        elif dt == "F16":
            arr = np.frombuffer(buf, dtype="<f2").astype(np.float32)
        elif dt == "F32":
            arr = np.frombuffer(buf, dtype="<f4").astype(np.float32)
        else:
            raise SystemExit(f"{self.path}: {key} has unsupported dtype {dt}")
        return arr.reshape(info["shape"])


def factor_inner(b1, a1, b2, a2) -> float:
    """<B1 A1, B2 A2>_F without forming either product."""
    return float(((b1.T @ b2) * (a1 @ a2.T)).sum())


def change_stats(f: dict[str, tuple], order: list[str], window: tuple[str, str],
                 final: str) -> dict:
    """Window statistics for one module. f: ckpt -> (A, B) as float64, scale folded into B."""
    import numpy as np

    names = list(f)
    ip = {(x, y): factor_inner(f[x][1], f[x][0], f[y][1], f[y][0]) for x in names for y in names}

    def change_inner(c1, c2):  # <W[c1[1]] - W[c1[0]], W[c2[1]] - W[c2[0]]>
        (x, y), (u, v) = c1, c2
        return ip[(y, v)] - ip[(y, u)] - ip[(x, v)] + ip[(x, u)]

    out = {"norm_sq": {n: ip[(n, n)] for n in names}}
    steps = list(zip(order, order[1:], strict=False))
    out["step_sq"] = {f"{x}->{y}": change_inner((x, y), (x, y)) for x, y in steps}
    w = window
    wn = change_inner(w, w)

    def cos(c):
        cn = change_inner(c, c)
        return change_inner(w, c) / (wn * cn) ** 0.5 if wn > 0 and cn > 0 else float("nan")

    i = steps.index(w) if w in steps else None
    out["cos_prev"] = cos(steps[i - 1]) if i else float("nan")
    out["cos_next"] = cos(steps[i + 1]) if i is not None and i + 1 < len(steps) else float("nan")
    fn = ip[(final, final)]
    out["cos_final"] = ((ip[(w[1], final)] - ip[(w[0], final)]) / (wn * fn) ** 0.5
                        if wn > 0 and fn > 0 else float("nan"))
    a0, a1 = f[w[0]][0], f[w[1]][0]
    out["a_move"] = float(np.linalg.norm(a1 - a0) / np.linalg.norm(a0))
    return out


def window_directions(f, window, k: int = 2):
    """Top-k output directions (and singular values) of W[window[1]] - W[window[0]]."""
    import numpy as np

    (a0, b0), (a1, b1) = f[window[0]], f[window[1]]
    bs = np.concatenate([b1, -b0], axis=1)
    as_ = np.concatenate([a1, a0], axis=0)
    qb, rb = np.linalg.qr(bs)
    qa, ra = np.linalg.qr(as_.T)
    u, s, _ = np.linalg.svd(rb @ ra.T)
    return (qb @ u)[:, :k], s[:k]


def logit_lens(u, w_u, g, tok, watch_ids: dict[str, int], top: int = 12) -> dict:
    """Tokens a residual direction promotes (+) and suppresses (-) through the final
    norm's gain and the unembedding; watch tokens as z-scores over the vocabulary."""
    import numpy as np

    logits = w_u @ (g * u)
    z = (logits - logits.mean()) / logits.std()
    order = np.argsort(logits)
    dec = (lambda ids: [tok.decode([int(t)]) for t in ids]) if tok else (lambda ids: [int(t) for t in ids])
    return {"up": dec(order[::-1][:top]), "down": dec(order[:top]),
            "watch_z": {w: float(z[i]) for w, i in watch_ids.items()}}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", action="append", required=True, help="NAME=DIR, in training order")
    ap.add_argument("--window", default="s100:s110")
    ap.add_argument("--final", default="s400")
    ap.add_argument("--unembed", help="base model dir (lm_head + final norm, tokenizer)")
    ap.add_argument("--lens-layers", default="40", help="logit lens on writer modules from this layer")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import numpy as np

    ck = {}
    for spec in args.ckpt:
        name, _, d = spec.partition("=")
        cfg = json.loads((Path(d) / "adapter_config.json").read_text())
        ck[name] = (LazySafetensors(Path(d) / "adapter_model.safetensors"),
                    cfg["lora_alpha"] / cfg["r"])
    order = list(ck)
    window = tuple(args.window.split(":"))
    first = next(iter(ck.values()))[0]
    mods: dict[tuple[int, str], dict[str, str]] = {}
    for key in first.header:
        m = KEY_RE.search(key)
        if m:
            mods.setdefault((int(m.group(1)), m.group(3)), {})[m.group(4)] = key

    lens = None
    if args.unembed:
        from transformers import AutoTokenizer

        files = sorted(Path(args.unembed).glob("*.safetensors"))
        found = {}
        for fpath in files:
            st = LazySafetensors(fpath)
            for name in ("lm_head.weight", "model.norm.weight"):
                if name in st.header:
                    found[name] = st.get(name).astype(np.float32)
        tok = AutoTokenizer.from_pretrained(args.unembed)
        watch = {}
        for w in WATCH:
            ids = tok(w, add_special_tokens=False)["input_ids"]
            if len(ids) == 1:
                watch[w] = ids[0]
        lens = (found["lm_head.weight"], found["model.norm.weight"], tok, watch)
        print(f"unembedding {found['lm_head.weight'].shape}; watch tokens {list(watch)}", flush=True)

    rows = []
    for (layer, mod), keys in sorted(mods.items()):
        f = {}
        for name, (st, scale) in ck.items():
            a = st.get(keys["A"]).astype(np.float64)
            b = st.get(keys["B"]).astype(np.float64) * scale
            f[name] = (a, b)
        r = {"layer": layer, "module": mod, **change_stats(f, order, window, args.final)}
        if lens and mod in WRITERS and layer >= int(args.lens_layers):
            u, s = window_directions(f, window)
            r["directions"] = [{"sv": float(s[j]), **logit_lens(u[:, j], *lens)}
                               for j in range(u.shape[1])]
        rows.append(r)
    del f

    w = f"{window[0]}->{window[1]}"
    tot = sum(r["step_sq"][w] for r in rows)
    summary = {"window": w, "step_sq_total": {s: sum(r["step_sq"][s] for r in rows)
                                              for s in rows[0]["step_sq"]}}
    by_mod, by_blk = {}, {}
    for r in rows:
        by_mod[r["module"]] = by_mod.get(r["module"], 0) + r["step_sq"][w] / tot
        b = r["layer"] // 8 * 8
        by_blk[b] = by_blk.get(b, 0) + r["step_sq"][w] / tot
    summary["window_share_by_module"] = by_mod
    summary["window_share_by_block"] = by_blk
    wsum = lambda key: (sum(r[key] * r["step_sq"][w] for r in rows if r[key] == r[key])  # noqa: E731
                        / sum(r["step_sq"][w] for r in rows if r[key] == r[key]))
    for key in ("cos_prev", "cos_next", "cos_final", "a_move"):
        summary[f"{key}_weighted"] = wsum(key)
    top = sorted(rows, key=lambda r: -r["step_sq"][w])[:12]
    summary["top_modules"] = [{"layer": r["layer"], "module": r["module"],
                               "share": r["step_sq"][w] / tot,
                               "cos_final": r["cos_final"]} for r in top]
    Path(args.out).write_text(json.dumps({"summary": summary, "modules": rows}, indent=1))
    print(json.dumps(summary, indent=1))
    for r in rows:
        for j, dd in enumerate(r.get("directions", [])):
            print(f"L{r['layer']} {r['module']} dir{j} sv {dd['sv']:.3f}  up {dd['up'][:8]}  "
                  f"down {dd['down'][:8]}  z {({k: round(v, 1) for k, v in dd['watch_z'].items()})}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
