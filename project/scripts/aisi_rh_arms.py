#!/usr/bin/env python3
"""AISI's CodeContests reward-hacking eval with our monitoring framing added.

Runs AISI's own eval script (rh_envs at the pinned commit) unchanged except for
one thing: the arm's oversight sentence from project.prompts.ARM_PROMPTS
(paraphrase 0, verbatim; any arm but D) or AISI_PENALTY_ARMS is inserted into the system prompt as its own
paragraph, directly before AISI's reasoning/format instructions ("Before
producing a response, you first reason inside <thinking></thinking> tags...").
Everything else -- task text, hack hints and their shuffling, the dont_hack
addendum, scorer, sandbox -- is AISI's. Arm C adds nothing, so it is exactly
AISI's own eval.

    python scripts/aisi_rh_arms.py --arm A -- <run_codecontests_reward_hacking_eval.py args>

Needs rh_envs and project on PYTHONPATH. The eval script path comes from
--eval-script (default: $AISI_REPO/scripts/run_codecontests_reward_hacking_eval.py).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import runpy
import sys
from pathlib import Path

FORMAT_MARKER = "Before producing a response, you first reason inside <thinking></thinking> tags."


def insert_arm(prompt: str, arm_text: str) -> str:
    """Put arm_text in its own paragraph right before AISI's format suffix."""
    if not arm_text:
        return prompt
    if prompt.count(FORMAT_MARKER) != 1:
        raise ValueError("AISI system prompt no longer contains the format marker exactly once")
    return prompt.replace(FORMAT_MARKER, f"{arm_text}\n\n{FORMAT_MARKER}")


def arm_text(arm: str) -> str:
    from project.prompts import AISI_PENALTY_ARMS, ARM_PROMPTS

    if arm in AISI_PENALTY_ARMS:
        return AISI_PENALTY_ARMS[arm]
    # D is B's text with real deletion, which a single-turn eval cannot express.
    allowed = tuple(a for a in ARM_PROMPTS if a != "D")
    if arm not in allowed:
        raise SystemExit(f"arm must be one of {allowed + tuple(AISI_PENALTY_ARMS)}, "
                         f"got {arm!r}")
    return ARM_PROMPTS[arm][0]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--eval-script", type=Path, default=Path(os.environ.get("AISI_REPO", "."))
                    / "scripts" / "run_codecontests_reward_hacking_eval.py")
    ap.add_argument("--record", type=Path, help="write the arm text and prompt hash here")
    ap.add_argument("rest", nargs=argparse.REMAINDER)
    args = ap.parse_args()
    rest = args.rest[1:] if args.rest[:1] == ["--"] else args.rest

    text = arm_text(args.arm)
    import rh_envs.codecontests_rh.task as task_mod

    original = task_mod.build_shuffled_prompt

    def build(key: str, hint_style: str = "sutl") -> str:
        return insert_arm(original(key, hint_style=hint_style), text)

    task_mod.build_shuffled_prompt = build

    if args.record:
        from rh_envs.codecontests_rh.prompts import SYSTEM_PROMPTS

        # Hash of the unshuffled dont_hack prompt with the arm inserted, to
        # tell prompt versions apart; the per-sample hint order still varies.
        framed = insert_arm(SYSTEM_PROMPTS["dont_hack"], text)
        args.record.parent.mkdir(parents=True, exist_ok=True)
        args.record.write_text(json.dumps({
            "arm": args.arm, "arm_text": text, "insert_before": FORMAT_MARKER,
            "dont_hack_prompt_sha256": hashlib.sha256(framed.encode()).hexdigest(),
            "dont_hack_prompt": framed,
        }, indent=2))

    sys.argv = [str(args.eval_script), *rest]
    runpy.run_path(str(args.eval_script), run_name="__main__")
    return 0


if __name__ == "__main__":
    sys.exit(main())
