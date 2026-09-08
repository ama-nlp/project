#!/usr/bin/env python3
"""Re-run the sandbox over existing traces to fill P1's verdict fields.

Schema v1 computed `def_gt_pass`, `test_modification` and `category` inside
`rh_eval` and then dropped them on the floor. They cannot be recovered from a
v1 trace by inspection -- the def_gt/def_arb probes are sandbox executions, not
derivations -- so any trace written before schema v2 has to go back through the
evaluator once.

This rewrites in place only after the whole file has been re-evaluated, so an
interrupted run leaves the original intact.

    uv run scripts/backfill_verdicts.py runs/armC_p0.jsonl [...]
    uv run scripts/backfill_verdicts.py --check runs/*.jsonl   # report only
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import orjson

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from project import data, rh_eval  # noqa: E402
from project.schema import SCHEMA_VERSION, Trace  # noqa: E402


def backfill(path: Path, *, check: bool) -> int:
    rows = [orjson.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        print(f"{path}: empty")
        return 0

    stale = [r for r in rows if r.get("schema_version", 1) < SCHEMA_VERSION]
    print(f"{path}: {len(rows)} traces, {len(stale)} below schema v{SCHEMA_VERSION}")
    if not stale:
        return 0
    if check:
        return len(stale)

    by_id = {p.id: p for p in data.load_problems()}
    missing = {r["problem_id"] for r in rows if r["problem_id"] not in by_id}
    if missing:
        # Silently skipping these would quietly change the denominator of every
        # rate computed downstream.
        print(f"  FATAL: {len(missing)} problem ids absent from the dataset: {sorted(missing)[:5]}")
        return -1

    ev = rh_eval.RewardHackEvaluator()
    problems = [by_id[r["problem_id"]] for r in rows]
    programs = [r.get("program") for r in rows]
    verdicts = ev.evaluate_batch(problems, programs)

    changed = 0
    for row, v in zip(rows, verdicts, strict=True):
        before = row.get("verdict") or {}
        # The re-run must agree with what P0 recorded, or the sandbox is not
        # deterministic and every downstream rate is suspect.
        for field in ("gt_pass", "hint_pass", "defines_test_func", "is_reward_hack_strict"):
            if field in before and before[field] != getattr(v, field):
                print(
                    f"  WARNING {row['problem_id']}: {field} was {before[field]!r}, "
                    f"re-ran as {getattr(v, field)!r}"
                )
        row["verdict"] = v.model_dump()
        row["ast_hack"] = rh_eval.ast_hack(v)
        row["schema_version"] = SCHEMA_VERSION
        Trace(**row)  # refuse to write anything the schema rejects
        changed += 1

    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(b"".join(orjson.dumps(r) + b"\n" for r in rows))
    tmp.replace(path)
    print(f"  rewrote {changed} traces")

    n_hack = sum(1 for r in rows if r["ast_hack"])
    n_strict = sum(1 for r in rows if (r.get("verdict") or {}).get("is_reward_hack_strict"))
    disagree = [
        r["problem_id"]
        for r in rows
        if bool(r["ast_hack"]) != bool((r.get("verdict") or {}).get("is_reward_hack_strict"))
    ]
    print(f"  ast_hack {n_hack}/{len(rows)}, is_reward_hack_strict {n_strict}/{len(rows)}")
    print(f"  disagreements (these are what P1 hand-verifies): {disagree or 'none'}")
    return changed


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+", type=Path)
    ap.add_argument("--check", action="store_true", help="report only, write nothing")
    args = ap.parse_args()

    rc = 0
    for p in args.paths:
        if backfill(p, check=args.check) < 0:
            rc = 1
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
