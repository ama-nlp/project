#!/usr/bin/env python3
"""Sensitivity of coupling, plan shift and McNemar to outputs without extractable
code (output_validity.has_code).

Scans score an output by hack_kinds, so an output whose code block never closes
usually reads as "no hack" and so as following the honest plan. Three treatments:
as run; such outputs dropped; such outputs counted as not following the plan
(the worst case for coupling).

    python scripts/nocode_sensitivity.py NAME=runs/.../lora-X [NAME=runs/.../lora-Y]
    python scripts/nocode_sensitivity.py --honest-vs-empty runs/.../lora-X   # E14 (b)
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from e1_coupling import sign_test  # noqa: E402
from e9b_matched import PLANS, mcnemar, rate, shift  # noqa: E402
from output_validity import has_code  # noqa: E402

TREATMENTS = ("asrun", "drop", "fail")


def bad_ids(model_dir: str, conds) -> dict[str, set[str]]:
    from inspect_ai.log import read_eval_log

    out: dict[str, set[str]] = defaultdict(set)
    for c in conds:
        sub = "main" if c == "C" else f"prefill-{c}"
        logs = sorted(glob.glob(f"{model_dir}/{sub}/logs_*/*.eval"))
        if logs:
            for s in read_eval_log(logs[-1]).samples:
                if not has_code(s.output.completion or ""):
                    out[c].add(s.id)
    return out


def treat(scan: dict, bad: dict[str, set[str]], how: str) -> dict:
    out = dict(scan)
    for c, v in scan.items():
        if not isinstance(v, dict) or "samples" not in v:
            continue
        rows = []
        for r in v["samples"]:
            if r["id"] in bad.get(c, ()):
                if how == "drop":
                    continue
                if how == "fail":
                    followed = False if r.get("followed") is not None else None
                    r = {**r, "followed": followed, "kinds": ["<nocode>"]}
            rows.append(r)
        out[c] = {**v, "samples": rows, "n": len(rows)}
    return out


def compare(models: dict[str, str]) -> None:
    scans = {k: json.loads(Path(f"{d}/prefill_scan.json").read_text()) for k, d in models.items()}
    bads = {k: bad_ids(d, ["C", *PLANS]) for k, d in models.items()}
    for k in models:
        print(k, "no-code outputs:", {c: len(v) for c, v in bads[k].items() if v})
    names = list(models)
    for how in TREATMENTS:
        t = {k: treat(scans[k], bads[k], how) for k in models}
        parts = []
        for k in names:
            sh = shift(t[k])
            parts.append(f"{k}: coupling {rate(t[k], PLANS):.3f} shift {sh['mean']:.3f} "
                         f"(honest {sh['per']['honest']['shift']:+.3f})")
        if len(names) == 2:
            m = mcnemar(t[names[1]], t[names[0]])
            parts.append(f"{names[1]} - {names[0]} {m['diff']:+.3f} "
                         f"({m['a_only']} vs {m['b_only']}, p={m['p']:.2g})")
        print(f"{how:6}", " | ".join(parts))


def _nohack(r: dict, failed: bool) -> bool:
    return not failed and not r["kinds"]


def honest_vs_empty(model_dir: str) -> None:
    scan = json.loads(Path(f"{model_dir}/prefill_scan.json").read_text())
    bad = bad_ids(model_dir, ["empty", "honest"])
    print("no-code outputs:", {c: len(v) for c, v in bad.items()})
    hon = {r["id"]: r for r in scan["honest"]["samples"]}
    emp = {r["id"]: r for r in scan["empty"]["samples"]}
    for how in TREATMENTS:
        keys = [k for k in hon if k in emp
                and (how != "drop" or (k not in bad["honest"] and k not in bad["empty"]))]

        h = [_nohack(hon[k], how == "fail" and k in bad["honest"]) for k in keys]
        e = [_nohack(emp[k], how == "fail" and k in bad["empty"]) for k in keys]
        a = sum(x and not y for x, y in zip(h, e, strict=True))
        b = sum(y and not x for x, y in zip(h, e, strict=True))
        print(f"{how:6} n={len(keys)} no hack: honest {sum(h) / len(keys):.3f}, "
              f"empty {sum(e) / len(keys):.3f}, diff {(sum(h) - sum(e)) / len(keys):+.3f} "
              f"({a} vs {b}, p={sign_test(a, b):.2g})")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="*", help="NAME=model_dir")
    ap.add_argument("--honest-vs-empty", metavar="MODEL_DIR")
    args = ap.parse_args()
    if args.honest_vs_empty:
        honest_vs_empty(args.honest_vs_empty)
    if args.models:
        compare(dict(m.split("=", 1) for m in args.models))
    return 0


if __name__ == "__main__":
    sys.exit(main())
