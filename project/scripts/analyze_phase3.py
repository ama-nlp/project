#!/usr/bin/env python3
"""Validate Phase 3 A/B/C traces and evaluate the pilot gate."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from project.phase3_analysis import report  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("traces", nargs="+")
    parser.add_argument("--sidecar", help="frozen P2 judge sidecar; trace files remain untouched")
    parser.add_argument("--judge-model")
    parser.add_argument("--prompt-sha")
    parser.add_argument("--min-recall-delta", type=float, default=0.10)
    args = parser.parse_args()
    try:
        passed = report(
            args.traces,
            sidecar=args.sidecar,
            judge_model=args.judge_model,
            prompt_sha=args.prompt_sha,
            min_recall_delta=args.min_recall_delta,
        )
    except ValueError as error:
        parser.error(str(error))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
