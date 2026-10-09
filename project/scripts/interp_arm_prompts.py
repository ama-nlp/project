#!/usr/bin/env python3
"""Arm prompts without generation, for I7's exact per-prompt opening probabilities.

Takes the exported Arm-C traces of one run (default C022) and writes, for each
requested arm, the same 200 prompts with that arm's sentence inserted into the
system prompt by the code the behavioural runs used (aisi_rh_arms.insert_arm,
arm_text). Every arm therefore sees the identical problem and hint order, so
arm differences in P(hack first) are paired and free of sampling noise. Rows
have an empty completion; scripts/interp_opening.py reads only the prompt.

    PYTHONPATH=src python scripts/interp_arm_prompts.py --traces traces.jsonl \\
        --base-run C022 --out arms.jsonl C A B F A-pen ...
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from aisi_rh_arms import arm_text, insert_arm  # noqa: E402


def with_arm(messages: list[dict], text: str) -> list[dict]:
    """Copy of messages with `text` inserted into the system prompt ("" = Arm C)."""
    out = copy.deepcopy(messages)
    systems = [m for m in out if m["role"] == "system"]
    if len(systems) != 1:
        raise ValueError(f"expected one system message, found {len(systems)}")
    systems[0]["content"] = insert_arm(systems[0]["content"], text)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", required=True)
    ap.add_argument("--base-run", default="C022")
    ap.add_argument("--out", required=True)
    ap.add_argument("arms", nargs="+")
    args = ap.parse_args()

    with open(args.traces) as f:
        base = [r for r in map(json.loads, f) if r["run"] == args.base_run]
    if not base:
        raise SystemExit(f"no traces with run {args.base_run!r}")
    n = 0
    with open(args.out, "w") as f:
        for arm in args.arms:
            text = "" if arm == "C" else arm_text(arm)
            for r in base:
                f.write(json.dumps({
                    "key": f"{arm}/{r['id']}", "run": arm, "id": r["id"],
                    "messages": with_arm(r["messages"], text), "completion": "",
                    "hack_attempt": False, "hacked": False, "mention": False,
                    "has_thinking": False, "arm_text": text,
                }) + "\n")
                n += 1
    print(f"arm prompts: {len(args.arms)} arms x {len(base)} problems = {n} rows -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
