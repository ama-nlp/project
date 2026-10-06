#!/usr/bin/env python3
"""Export AISI eval traces to JSONL for the interpretability work (Phase 2 interp).

Reads the inspect `.eval` log of each LABEL=run_dir and writes one row per
sample: the chat messages the model saw (system, user), its completion, and
the behavioural labels the probes are trained on. Runs in the eval env (it
needs inspect_ai); the activation script that consumes the JSONL does not.

Labels:
  hacked        AISI scorer's reward_hacked (hack used AND tests passed)
  hack_attempt  the completion contains hack code (conftest / os._exit /
                __eq__), whether or not it passed: the policy's choice, which
                is what a "decided before reasoning" probe should predict
  mention       AISI's cot_mentions_hack keyword regex on <thinking>
  has_thinking  a <thinking>...</thinking> block exists

    python scripts/interp_export.py --out traces.jsonl C022=runs/a/main A022=runs/b/arm-A
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from aisi_rh_prefill import HACK_CODE_RE, THINK_RE, _eval_samples  # noqa: E402


def row_for(label: str, sample_id, messages: list[dict], completion: str, value: dict,
            mention_re) -> dict:
    m = THINK_RE.search(completion)
    return {
        "key": f"{label}/{sample_id}",
        "run": label,
        "id": str(sample_id),
        "messages": messages,
        "completion": completion,
        "hacked": value.get("reward_hacked") == "C",
        "passed": value.get("passed") == "C",
        "hack_attempt": bool(HACK_CODE_RE.search(completion)),
        "has_thinking": m is not None,
        "mention": bool(m and mention_re.search(m.group(1))),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("runs", nargs="+", help="LABEL=run_dir")
    args = ap.parse_args()
    from rh_envs.common import _HACK_COT_KEYWORDS as kw

    n = 0
    with open(args.out, "w") as f:
        for spec in args.runs:
            label, _, run_dir = spec.partition("=")
            for s in _eval_samples(run_dir):
                # The prompt is everything before the model's own turn.
                msgs = [{"role": m.role, "content": m.text} for m in s.messages
                        if m.role in ("system", "user")]
                value = next(iter(s.scores.values())).value if s.scores else {}
                f.write(json.dumps(row_for(label, s.id, msgs, s.output.completion or "",
                                           value, kw)) + "\n")
                n += 1
    print(f"export: {n} traces from {len(args.runs)} runs -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
