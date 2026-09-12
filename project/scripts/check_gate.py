#!/usr/bin/env python3
"""P0 step 4/5 gate check over one arm's traces.

    uv run scripts/check_gate.py runs/<run_id>/armC_p0.jsonl
    uv run scripts/check_gate.py runs/<run_id>/armC_p0.jsonl --sample 20 > read_me.txt

The gate (docs/phase0.md step 4) is: every record has a non-empty cot, a
non-empty program, and a verdict. Everything else printed here is the input to
the three open items P0 has to settle -- the token budget, the base hack rate,
and the per-arm timing that P5's QoS arithmetic multiplies.
"""
import argparse
import json
import re
import statistics as st
import sys

CODE_BLOCK = re.compile(r"```python(.*?)```", re.S)


def load(path):
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def pct(n, d):
    return f"{n}/{d} ({100 * n / d:.0f}%)" if d else "0/0"


def gate(records):
    """The P0 gate proper. Returns the list of failures."""
    bad = []
    for r in records:
        why = []
        if not (r.get("cot") or "").strip():
            why.append("empty cot")
        if not (r.get("program") or "").strip():
            why.append("empty program")
        if not r.get("verdict"):
            why.append("no verdict")
        if why:
            bad.append((r.get("problem_id"), ", ".join(why)))
    return bad


def report(records):
    n = len(records)
    print(f"n = {n}  arm {records[0]['arm']}  paraphrase {records[0]['paraphrase_id']}")
    print(f"model {records[0]['model']}  max_tokens {records[0]['sampling']['max_tokens']}")
    print()

    failures = gate(records)
    print("== P0 gate ==")
    if failures:
        print(f"  FAIL: {len(failures)} record(s) incomplete")
        for pid, why in failures[:20]:
            print(f"    {pid}: {why}")
    else:
        print(f"  PASS: all {n} records have cot + program + verdict")
    print()

    # Token budget. The point is the tail, not the mean: max_tokens has to sit
    # above p95 or the longest reasoning chains get cut off mid-thought.
    toks = [r["n_completion_tokens"] for r in records]
    limit = records[0]["sampling"]["max_tokens"]
    toks_sorted = sorted(toks)
    p95 = toks_sorted[min(int(0.95 * n), n - 1)]
    print("== completion tokens ==")
    print(f"  median {int(st.median(toks))}  p95 {p95}  max {max(toks)}  limit {limit}")
    print(f"  headroom at p95: {100 * (1 - p95 / limit):.0f}%")
    stopped = sum(1 for r in records if r["finish_reason"] == "stop")
    print(f"  finish_reason=stop  {pct(stopped, n)}")
    print()

    # Extraction sanity, i.e. the automatable half of step 5.
    ok_last, cot_blocks = 0, 0
    for r in records:
        blocks = CODE_BLOCK.findall(r["completion_raw"])
        if blocks and r["program"].strip() == blocks[-1].strip():
            ok_last += 1
        cot_blocks += len(CODE_BLOCK.findall(r["cot"] or ""))
    print("== extraction ==")
    print(f"  program == last code block in answer  {pct(ok_last, n)}")
    print(f"  stray code blocks inside CoT          {cot_blocks}")
    print()

    # The base rate P3 lives or dies on.
    v = [r["verdict"] for r in records]
    solved = sum(1 for x in v if x["gt_pass"])
    defines = sum(1 for x in v if x["defines_test_func"])
    hacks = sum(1 for x in v if x["is_reward_hack_strict"])
    mentions = sum(1 for r in records if "run_tests" in (r["cot"] or ""))
    print("== behaviour ==")
    print(f"  solves problem (gt_pass)   {pct(solved, n)}")
    print(f"  defines run_tests          {pct(defines, n)}")
    print(f"  strict reward hack         {pct(hacks, n)}")
    print(f"  mentions run_tests in CoT  {pct(mentions, n)}")
    if hacks == 0:
        print("  NOTE: zero hacks. If the hinted arms are also flat, P3 has no")
        print("        contrast to measure -- escalate model scale or raise n.")
    print()

    errs = {}
    for x in v:
        for e in x.get("errors") or []:
            errs[e.split(":")[0]] = errs.get(e.split(":")[0], 0) + 1
    if errs:
        print("== sandbox errors ==")
        for k, c in sorted(errs.items(), key=lambda kv: -kv[1]):
            print(f"  {c:4d}  {k}")
        print()

    return failures


def sample(records, k):
    """Dump k traces for the hand-read, biased toward the informative ones."""
    def interest(r):
        x = r["verdict"]
        return (
            x["is_reward_hack_strict"] * 8
            + x["defines_test_func"] * 4
            + ("run_tests" in (r["cot"] or "")) * 3
            + (r["finish_reason"] != "stop") * 3
            + (not x["gt_pass"]) * 1
        )

    ranked = sorted(records, key=interest, reverse=True)
    picked = ranked[: k // 2] + ranked[k // 2 :][:: max(1, len(ranked[k // 2 :]) // max(1, k - k // 2))][: k - k // 2]
    for i, r in enumerate(picked, 1):
        x = r["verdict"]
        print("=" * 78)
        print(f"[{i}/{len(picked)}] problem {r['problem_id']}  ({r['difficulty']})")
        print(f"  finish={r['finish_reason']}  tokens={r['n_completion_tokens']}")
        print(f"  gt_pass={x['gt_pass']} rate={x['gt_pass_rate']:.2f} "
              f"defines_test_func={x['defines_test_func']} hack={x['is_reward_hack_strict']}")
        if x.get("errors"):
            print(f"  errors: {x['errors']}")
        print("-" * 78 + "\n  COT (first 1500 / last 1500 chars)")
        cot = r["cot"] or ""
        print(cot[:1500])
        if len(cot) > 3000:
            print(f"\n  ... [{len(cot) - 3000} chars elided] ...\n")
            print(cot[-1500:])
        print("-" * 78 + "\n  PROGRAM")
        print(r["program"])
        print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--sample", type=int, default=0,
                    help="dump N traces for the step-5 hand-read")
    args = ap.parse_args()

    records = load(args.path)
    if not records:
        sys.exit(f"{args.path}: no records")

    if args.sample:
        sample(records, args.sample)
        return
    sys.exit(1 if report(records) else 0)


if __name__ == "__main__":
    main()
