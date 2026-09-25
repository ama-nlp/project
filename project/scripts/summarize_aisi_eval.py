#!/usr/bin/env python3
"""Summarize an AISI CodeContests reward-hacking eval run from its inspect logs.

Run with the AISI eval venv (it needs inspect-ai and rh_envs):

    ~/aisi-env/bin/python scripts/summarize_aisi_eval.py runs/aisi-cc/<run>/

Writes ``samples.jsonl`` (one row per sample: scorer flags, stop reason,
output tokens, the <thinking> block and whether it matches AISI's own
hack-keyword regex) and ``summary.json`` next to the logs. The keyword regex is
their training-time proxy, not our validated P2 judge; it is reported so our
numbers are comparable with their ``cot_mentions_hack`` field.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

THINKING_RE = re.compile(r"<thinking>(.*?)</thinking>", re.DOTALL)
FLAGS = ("passed", "actually_solved", "reward_hacked", "rh_always_equal", "rh_exit", "rh_conftest")


def thinking_block(completion: str) -> str | None:
    match = THINKING_RE.search(completion)
    return match.group(1) if match else None


def summarize(rows: list[dict]) -> dict:
    """Aggregate per-sample rows; rates are over all samples unless noted."""
    n = len(rows)
    if not n:
        return {"n": 0}

    def rate(key: str, subset: list[dict] | None = None) -> float | None:
        pool = rows if subset is None else subset
        return sum(bool(r[key]) for r in pool) / len(pool) if pool else None

    hacked = [r for r in rows if r["reward_hacked"]]
    stops: dict[str, int] = {}
    for r in rows:
        stops[r["stop_reason"]] = stops.get(r["stop_reason"], 0) + 1
    tokens = sorted(r["output_tokens"] for r in rows if r["output_tokens"] is not None)
    return {
        "n": n,
        **{f"{key}_rate": rate(key) for key in FLAGS},
        "has_thinking_rate": rate("has_thinking"),
        "cot_mentions_hack_rate": rate("cot_mentions_hack"),
        "cot_mentions_hack_given_hacked": rate("cot_mentions_hack", hacked),
        "n_reward_hacked": len(hacked),
        "stop_reasons": stops,
        "output_tokens_median": tokens[len(tokens) // 2] if tokens else None,
        "output_tokens_max": tokens[-1] if tokens else None,
    }


def sample_rows(log_path: Path) -> list[dict]:
    from inspect_ai.log import read_eval_log
    from rh_envs.common import _HACK_COT_KEYWORDS

    log = read_eval_log(str(log_path))
    rows = []
    for sample in log.samples or []:
        score = next(iter((sample.scores or {}).values()))
        value = score.value if isinstance(score.value, dict) else {}
        completion = sample.output.completion if sample.output else ""
        thinking = thinking_block(completion)
        choice = sample.output.choices[0] if sample.output and sample.output.choices else None
        usage = sample.output.usage if sample.output else None
        rows.append({
            "id": sample.id,
            "epoch": sample.epoch,
            "hack_group": (score.metadata or {}).get("hack_group"),
            **{key: value.get(key) == "C" for key in FLAGS},
            "stop_reason": getattr(choice, "stop_reason", None) or "unknown",
            "output_tokens": getattr(usage, "output_tokens", None),
            "has_thinking": thinking is not None,
            "cot_mentions_hack": bool(thinking and _HACK_COT_KEYWORDS.search(thinking)),
            "thinking": thinking,
            "completion": completion,
        })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", type=Path)
    args = ap.parse_args()

    # The AISI script writes its log under results/logs_<timestamp>/ regardless
    # of INSPECT_LOG_DIR, so search the whole run directory.
    logs = sorted(args.run_dir.rglob("*.eval"))
    if len(logs) != 1:
        raise SystemExit(f"expected one .eval log under {args.run_dir}, found {len(logs)}")
    rows = sample_rows(logs[0])
    with (args.run_dir / "samples.jsonl").open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    summary = {"log": logs[0].name, **summarize(rows)}
    (args.run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
