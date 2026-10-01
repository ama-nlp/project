#!/usr/bin/env python3
"""Run an AISI CodeContests eval script on a different problem subset.

AISI's task keeps only hard problems (Codeforces rating >= 2000, or difficulty
HARD and above). The kl0.0 organism solves none of those, so a reasoning that
commits to solving honestly has nothing to fall back on (Run 028). This wrapper
swaps AISI's `_is_hard_problem` filter for another one, then runs the given
script (AISI's eval, scripts/aisi_rh_arms.py or scripts/aisi_rh_prefill.py) in
this process, so the patch is seen by the task they build. Prompt, hints,
scorer and everything else stay AISI's.

    python scripts/aisi_rh_subset.py --subset easy -- scripts/aisi_rh_prefill.py run ...

Subsets:
  easy   Codeforces-rated problems with a rating of 800-1200
"""

from __future__ import annotations

import argparse
import runpy
import sys

EASY_MAX_RATING = 1200


def is_easy(record: dict) -> bool:
    return 0 < record["cf_rating"] <= EASY_MAX_RATING


SUBSETS = {"easy": is_easy}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", required=True, choices=sorted(SUBSETS))
    ap.add_argument("script")
    ap.add_argument("rest", nargs=argparse.REMAINDER)
    args = ap.parse_args()

    import rh_envs.codecontests_rh.task as task_mod

    task_mod._is_hard_problem = SUBSETS[args.subset]
    print(f"[subset] problems filtered with {args.subset!r} instead of AISI's hard filter",
          flush=True)
    sys.argv = [args.script, *args.rest]
    runpy.run_path(args.script, run_name="__main__")
    return 0


if __name__ == "__main__":
    sys.exit(main())
