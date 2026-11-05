#!/usr/bin/env python3
"""Run an AISI eval script with a longer OpenAI client timeout.

AISI's script passes only the API key as model_args, so inspect's OpenAI
provider keeps its default per-request timeout. On a slow server that is too
short: E12's Qwen adapters write ~4-6k-token answers at ~10 tok/s, so requests
that waited in vLLM's queue timed out, were retried from scratch, and job 57167
saved 26 samples in 19 h. This wrapper adds `client_timeout` to every
inspect `eval()` call, then runs the given script (AISI's eval,
scripts/aisi_rh_arms.py, scripts/aisi_rh_prefill.py or scripts/aisi_rh_subset.py)
in this process. Prompt, sampling, scorer and everything else are unchanged.

    python scripts/aisi_rh_genopts.py --client-timeout 3600 scripts/aisi_rh_prefill.py run ...
"""

from __future__ import annotations

import argparse
import runpy
import sys


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--client-timeout", type=float, required=True, help="seconds per request")
    ap.add_argument("script")
    ap.add_argument("rest", nargs=argparse.REMAINDER)
    args = ap.parse_args()

    import inspect_ai

    real_eval = inspect_ai.eval

    def eval_with_timeout(*a, **kw):
        kw["model_args"] = {**(kw.get("model_args") or {}), "client_timeout": args.client_timeout}
        return real_eval(*a, **kw)

    inspect_ai.eval = eval_with_timeout
    print(f"[genopts] OpenAI client timeout {args.client_timeout:g} s", flush=True)
    sys.argv = [args.script, *args.rest]
    runpy.run_path(args.script, run_name="__main__")
    return 0


if __name__ == "__main__":
    sys.exit(main())
