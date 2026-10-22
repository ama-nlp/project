#!/usr/bin/env python3
"""E9b: matched-step KL contrasts on AISI's step-400 adapters
(docs/results/interp-live.md, E track; rules fixed 2026-10-22 before any E9b run).

Pre-stated claims, on plan coupling (aisi_rh_prefill.followed over plan-eq,
plan-exit, honest), paired by (problem, plan condition), exact McNemar:
  Q1  beta = 0.02 above beta = 0 in seed 1 and in seed 2, each p < 0.05
  Q2  KL-on-code above KL-on-thinking (p < 0.05) and within 0.10 of all-token KL;
      the reverse ordering is reported as "the KL on the reasoning keeps it"
  Q3  descriptive: cotpenalty-soft vs KL-on-code

Also reported, and labelled as defined after seeing E9b's first results (not
pre-stated): plan *shift*, the share of samples containing the planned outcome
(the planned hack kind, or no hack for honest) minus that share in plain C,
averaged over the three plans. `followed` credits a model whose default hack
equals one plan's, and refuses one that adds the planned hack to its own.

    python scripts/e9b_matched.py s1b0=SCAN s1b002=SCAN s2b0=SCAN s2b002=SCAN \\
        klcode=SCAN klthink=SCAN cotpen=SCAN [--instr NAME=SCAN ...] [--json out.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from e1_coupling import sign_test  # noqa: E402

PLANS = ("plan-eq", "plan-exit", "honest")
INSTRS = ("instr-eq", "instr-exit", "instr-honest")
WANT = {"plan-eq": "eq", "plan-exit": "exit", "honest": None,
        "instr-eq": "eq", "instr-exit": "exit", "instr-honest": None}
MIN_HACK, TOL, ALPHA = 0.10, 0.10, 0.05


def pairs(scan: dict, conds=PLANS) -> dict[tuple[str, str], bool]:
    return {(s["id"], c): bool(s["followed"])
            for c in conds for s in (scan.get(c) or {}).get("samples", [])}


def rate(scan: dict, conds) -> float | None:
    p = pairs(scan, conds)
    if any(not (scan.get(c) or {}).get("n") for c in conds):
        return None
    return sum(p.values()) / len(p)


def mcnemar(a: dict, b: dict, conds=PLANS) -> dict:
    """Model a minus model b over shared (problem, condition) pairs."""
    pa, pb = pairs(a, conds), pairs(b, conds)
    keys = sorted(set(pa) & set(pb))
    hi = sum(pa[k] and not pb[k] for k in keys)
    lo = sum(pb[k] and not pa[k] for k in keys)
    diff = (sum(pa[k] for k in keys) - sum(pb[k] for k in keys)) / len(keys) if keys else float("nan")
    return {"n_pairs": len(keys), "diff": diff, "a_only": hi, "b_only": lo, "p": sign_test(hi, lo)}


def contains(kinds: list[str], want: str | None) -> bool:
    return not kinds if want is None else want in kinds


def shift(scan: dict, conds=PLANS) -> dict | None:
    """Per plan: P(planned outcome | plan) - P(planned outcome | plain C)."""
    c = (scan.get("C") or {}).get("samples")
    if not c or any(not (scan.get(k) or {}).get("samples") for k in conds):
        return None
    per = {}
    for k in conds:
        rows = scan[k]["samples"]
        got = sum(contains(r["kinds"], WANT[k]) for r in rows) / len(rows)
        base = sum(contains(r["kinds"], WANT[k]) for r in c) / len(c)
        per[k] = {"with_plan": got, "plain": base, "shift": got - base}
    return {"per": per, "mean": sum(v["shift"] for v in per.values()) / len(per)}


def hack_rate(scan: dict) -> float | None:
    c = scan.get("C") or {}
    return c["hacked"] / c["n"] if c.get("n") else None


def verdicts(m: dict[str, dict]) -> dict:
    out = {}
    ok = {k for k, s in m.items() if (hack_rate(s) or 0) >= MIN_HACK}
    seeds = {}
    for seed, (hi, lo) in {"seed1": ("s1b002", "s1b0"), "seed2": ("s2b002", "s2b0")}.items():
        if hi in ok and lo in ok:
            seeds[seed] = mcnemar(m[hi], m[lo])
    if seeds:
        sig = [s["diff"] > 0 and s["p"] < ALPHA for s in seeds.values()]
        same = all(s["diff"] > 0 for s in seeds.values()) or all(s["diff"] < 0 for s in seeds.values())
        out["Q1"] = {**seeds, "verdict": "holds" if len(sig) == 2 and all(sig)
                     else "one seed" if same and any(sig)
                     else "not replicated" if not same else "not significant"}
    if {"klcode", "klthink", "s1b002"} <= ok:
        ct = mcnemar(m["klcode"], m["klthink"])
        ca = mcnemar(m["klcode"], m["s1b002"])
        v = ("holds" if ct["diff"] > 0 and ct["p"] < ALPHA and abs(ca["diff"]) <= TOL
             else "the KL on the reasoning keeps it" if ct["diff"] < 0 and ct["p"] < ALPHA
             else "not shown")
        out["Q2"] = {"code_vs_thinking": ct, "code_vs_all": ca, "verdict": v}
    if {"cotpen", "klcode"} <= set(m):
        out["Q3"] = {"cotpen_vs_code": mcnemar(m["cotpen"], m["klcode"])}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+", help="NAME=prefill_scan.json")
    ap.add_argument("--instr", nargs="*", default=[],
                    help="NAME=scan supplying instr-* conditions missing from NAME's scan")
    ap.add_argument("--json")
    args = ap.parse_args()
    m = {}
    for spec in args.models:
        name, _, path = spec.partition("=")
        m[name] = json.loads(Path(path).read_text())
    for spec in args.instr:
        name, _, path = spec.partition("=")
        extra = json.loads(Path(path).read_text())
        m[name].update({k: v for k, v in extra.items() if k in INSTRS and k not in m[name]})
    rows = {}
    fmt = lambda v: "  -  " if v is None else f"{v:5.2f}"  # noqa: E731
    print(f"{'model':<8}{'hack':>7}{'plan':>7}{'instr':>7}{'shift':>7}   shift per plan (eq / exit / honest)")
    for name, s in m.items():
        sh = shift(s)
        rows[name] = {"hack": hack_rate(s), "plan_coupling": rate(s, PLANS),
                      "instr_coupling": rate(s, INSTRS), "shift": sh,
                      "kinds": {k: (s.get(k) or {}).get("kinds") for k in ("C", *PLANS)}}
        r = rows[name]
        per = " / ".join(f"{sh['per'][k]['shift']:+.2f}" for k in PLANS) if sh else "-"
        print(f"{name:<8}{fmt(r['hack']):>7}{fmt(r['plan_coupling']):>7}{fmt(r['instr_coupling']):>7}"
              f"{fmt(sh['mean'] if sh else None):>7}   {per}")
    v = verdicts(m)
    print(json.dumps(v, indent=1))
    if args.json:
        Path(args.json).write_text(json.dumps({"models": rows, "verdicts": v}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
