#!/usr/bin/env python3
"""Output-validity check for E-track runs (E14's pre-stated exclusion rule).

A condition with > 10% of outputs lacking extractable code is reported but
excluded from the claims. An output has extractable code if AISI's extraction
(aisi_rh_prefill.extract_files, a mirror of rh_envs.common) finds a <file>
tag, a ```python block or a bare ``` block, or, failing those, if the raw text
it then falls back to parses as Python (the 7B base often writes bare code).
nat-* conditions are scored only on problems with a bank entry for that plan,
as E13's analysis is, using the prefill_bank.json beside the condition.
Also counted: outputs cut at the token limit (stop_reason max_tokens /
length), and garbled outputs.

    python scripts/output_validity.py RUN_ROOT [RUN_ROOT ...] [--json out.json]

RUN_ROOT is a runs/aisi-rh-eval-<job> directory. Every condition directory
under it (main, main-rI, prefill-*, at depth 1 or 2) is read from its newest
.eval log, as the scans read them.
"""

from __future__ import annotations

import argparse
import ast
import glob
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from aisi_rh_prefill import GARBLED_RE  # noqa: E402

FILE_RE = re.compile(r'<file\s+path=["\'][^"\']+["\']>', re.S)
FENCE_RE = re.compile(r"```(?:python)?\n.*?```", re.S)
LIMIT = 0.10


def has_code(comp: str) -> bool:
    if FILE_RE.search(comp) or FENCE_RE.search(comp):
        return True
    if not comp.strip():
        return False
    try:
        ast.parse(comp)
    except (SyntaxError, ValueError):
        return False
    return True


def check(cond_dir: str) -> dict | None:
    from inspect_ai.log import read_eval_log

    logs = sorted(glob.glob(f"{cond_dir}/logs_*/*.eval"))
    if not logs:
        return None
    log = read_eval_log(logs[-1])
    rows = log.samples or []
    cond = Path(cond_dir).name.removeprefix("prefill-")
    bank_file = Path(cond_dir).parent / "prefill_bank.json"
    if cond.startswith("nat-") and bank_file.exists():
        bank = json.loads(bank_file.read_text())
        rows = [s for s in rows if cond in bank.get(s.id, {})]
    n = len(rows)
    if not n:
        return {"n": 0, "status": log.status}
    no_code = trunc = garbled = 0
    for s in rows:
        comp = s.output.completion or ""
        no_code += not has_code(comp)
        stop = s.output.choices[0].stop_reason if s.output.choices else None
        trunc += stop in ("max_tokens", "length", "model_length")
        garbled += bool(GARBLED_RE.search(comp))
    return {"n": n, "status": log.status, "no_code": no_code, "truncated": trunc,
            "garbled": garbled, "no_code_rate": no_code / n, "flag": no_code / n > LIMIT}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("roots", nargs="+")
    ap.add_argument("--json")
    args = ap.parse_args()
    out = {}
    for root in args.roots:
        for d in sorted(glob.glob(f"{root}/*/") + glob.glob(f"{root}/*/*/")):
            name = Path(d).name
            if not (name.startswith("main") or name.startswith("prefill-")):
                continue
            r = check(d.rstrip("/"))
            if r is None:
                continue
            key = str(Path(d).relative_to(Path(root).parent))
            out[key] = r
            if r["n"]:
                print(f"{'FLAG ' if r['flag'] else '     '}{key:70} n={r['n']:4} "
                      f"no_code={r['no_code']:3} ({r['no_code_rate']:.3f}) "
                      f"trunc={r['truncated']:3} garbled={r['garbled']:3} {r['status']}", flush=True)
    flagged = [k for k, r in out.items() if r.get("flag")]
    print(f"\n{len(out)} conditions, {len(flagged)} over the {LIMIT:.0%} no-code limit")
    for k in flagged:
        print("  ", k)
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
