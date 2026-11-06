#!/usr/bin/env python3
"""Draw-to-draw variance from E11's repeated draws.

Per adapter: coupling and plan shift (e9b_matched) in each draw with all four
conditions complete, their between-draw SD, the problem-bootstrap SE of the
first draw alone (what a one-draw run reports), McNemar between draws, and how
often a (problem, plan) cell gets the same verdict in every draw.

    python scripts/e11_draw_variance.py runs/aisi-rh-eval-58798 [--json out.json]
"""

from __future__ import annotations

import argparse
import glob
import json
import random
import statistics as st
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from e9b_matched import PLANS, mcnemar, pairs, rate, shift  # noqa: E402

CONDS = ("C", *PLANS)


def boot_se(scan: dict, f, reps: int = 1000, seed: int = 0) -> float:
    ids = sorted({r["id"] for c in CONDS for r in scan[c]["samples"]})
    rng = random.Random(seed)
    vals = []
    for _ in range(reps):
        cnt = Counter(rng.choice(ids) for _ in ids)
        sc = {c: {**scan[c], "samples": [r for r in scan[c]["samples"]
                                         for _ in range(cnt[r["id"]])]} for c in CONDS}
        vals.append(f(sc))
    return st.stdev(vals)


def adapter(files: list[str]) -> dict | None:
    scans = [json.loads(Path(f).read_text()) for f in files]
    ok = [s for s in scans if all((s.get(c) or {}).get("n") == 200 for c in CONDS)]
    if len(ok) < 2:
        return None
    co = [rate(s, PLANS) for s in ok]
    sh = [shift(s)["mean"] for s in ok]
    cells = [pairs(s) for s in ok]
    keys = set.intersection(*[set(p) for p in cells])
    same = sum(len({p[k] for p in cells}) == 1 for k in keys)
    ps = [mcnemar(ok[i], ok[j])["p"] for i in range(len(ok)) for j in range(i + 1, len(ok))]
    return {
        "draws": len(ok), "coupling": co, "shift": sh,
        "plain_hack": [s["C"]["hacked"] / s["C"]["n"] for s in ok],
        "coupling_sd": st.stdev(co), "shift_sd": st.stdev(sh),
        "coupling_boot_se": boot_se(ok[0], lambda s: rate(s, PLANS)),
        "shift_boot_se": boot_se(ok[0], lambda s: shift(s)["mean"]),
        "mcnemar_min_p": min(ps), "cells": len(keys), "cells_same": same,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_root")
    ap.add_argument("--json")
    args = ap.parse_args()
    out = {}
    for m in sorted(glob.glob(f"{args.run_root}/lora-*")):
        r = adapter(sorted(glob.glob(f"{m}/prefill_scan-r*.json")))
        if r is None:
            continue
        name = m.split("lora-", 1)[1]
        out[name] = r
        print(f"{name:8} draws={r['draws']} | coupling "
              + " ".join(f"{x:.3f}" for x in r["coupling"])
              + f" (SD {r['coupling_sd']:.3f}; one-draw bootstrap SE {r['coupling_boot_se']:.3f})"
              + " | plan shift " + " ".join(f"{x:.3f}" for x in r["shift"])
              + f" (SD {r['shift_sd']:.3f}; SE {r['shift_boot_se']:.3f})"
              + f" | between-draw McNemar min p {r['mcnemar_min_p']:.2g}"
              + f" | same verdict in {r['cells_same'] / r['cells']:.1%} of cells")
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
