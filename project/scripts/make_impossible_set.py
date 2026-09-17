#!/usr/bin/env python3
"""Build the P3i impossible-test variant set.

For each problem: show the 4 shortest distinct tests, alter the expected value
of the shortest one whose value is distinctive, then verify in the sandbox that
the canonical solution passes every unaltered shown test and fails the altered
one. Problems that fail any check are dropped with a reason.

Deterministic: the same base dataset always yields the same file, byte for byte.
The output contains upstream test strings and is gitignored for the same
licensing reason as the base dataset; record its sha256, never commit it.

    uv run scripts/make_impossible_set.py
    uv run scripts/make_impossible_set.py --k 4 --out data/impossible_v1.jsonl
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from project import data, impossible  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=impossible.DEFAULT_K, help="tests shown per problem")
    ap.add_argument("--out", type=Path, default=impossible.DEFAULT_VARIANT_PATH)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument(
        "--problem-subset",
        choices=("p3-hack-20",),
        default=None,
        help="build only a frozen enrichment subset; permits its documented boolean fallback",
    )
    args = ap.parse_args()

    problems = data.load_problems(loophole="repaired")
    if args.problem_subset == "p3-hack-20":
        wanted = set(impossible.LEETCODE_P3_HACK_20)
        problems = [problem for problem in problems if problem.id in wanted]
        if len(problems) != len(wanted):
            raise SystemExit("the LeetCode dataset does not contain the complete p3-hack-20 subset")
    candidates, dropped = [], []
    for p in problems:
        v, why = impossible.candidate_variant(
            p,
            args.k,
            allow_boolean_fallback=args.problem_subset == "p3-hack-20",
        )
        if v is None:
            dropped.append((p.id, why))
        else:
            candidates.append(v)

    evaluator = impossible.CodeEvaluator(
        num_workers=args.workers, timeout=6, memory_per_worker=2048
    )
    kept, rejected = impossible.verify_variants(problems, candidates, evaluator)
    dropped += rejected
    sha = impossible.write_variants(args.out, kept)

    print(f"\nwrote {len(kept)} of {len(problems)} problems -> {args.out}")
    print(f"  variant set sha256 {sha}")
    print(f"  base dataset sha256 {data.dataset_sha256()}")
    print(f"  altered value types: {dict(Counter(type(v.mutated_value).__name__ for v in kept))}")
    if dropped:
        print(f"  dropped {len(dropped)}: {dict(Counter(why for _, why in dropped))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
