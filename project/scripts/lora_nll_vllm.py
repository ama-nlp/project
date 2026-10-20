#!/usr/bin/env python3
"""E9 diagnosis: teacher-forced NLL of stored completions under vLLM, for a
merged model or for the base with a LoRA adapter (vLLM's LoRA path).

Used beside interp_lora.py --check (the Transformers path) to find which of
the two unmerged paths drifts from the merged model on kl0.02 (job 28669:
plan-eq followed 60/200 via vLLM LoRA against 114/200 merged).

    python scripts/lora_nll_vllm.py --model DIR [--lora NAME=DIR] --traces traces.jsonl \\
        --out nll.json [--limit 16] [--hf-overrides JSON] [--tp 4]

Runs one model configuration per process: vLLM does not reliably tear down
and rebuild an engine inside one Python process.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def completion_nll(prompt_logprobs: list, ids: list[int], n_prompt: int) -> float:
    """Mean NLL over completion tokens from vLLM's per-position prompt logprobs
    (entry t is {token_id: Logprob} for token t, None for the first token)."""
    total, n = 0.0, 0
    for t in range(n_prompt, len(ids)):
        entry = prompt_logprobs[t]
        lp = entry[ids[t]]
        total -= lp.logprob if hasattr(lp, "logprob") else float(lp)
        n += 1
    return total / max(n, 1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--lora", help="NAME=DIR")
    ap.add_argument("--traces", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=8192)
    ap.add_argument("--tp", type=int, default=4)
    ap.add_argument("--pp", type=int, default=1)
    ap.add_argument("--hf-overrides", default="{}")
    args = ap.parse_args()

    from interp_extract import prompt_ids_for
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    tok = AutoTokenizer.from_pretrained(args.model)
    with open(args.traces) as f:
        rows = [json.loads(line) for line in f]
    items = []
    for r in rows:
        p = prompt_ids_for(tok, r["messages"])
        ids = p + tok(r["completion"], add_special_tokens=False)["input_ids"]
        if len(ids) < args.max_len - 1:
            items.append((r["key"], p, ids))
        if len(items) == args.limit:
            break
    kw = {}
    lora_req = None
    if args.lora:
        name, _, path = args.lora.partition("=")
        kw = {"enable_lora": True, "max_lora_rank": 32, "max_loras": 1}
        lora_req = LoRARequest(name, 1, path)
    llm = LLM(model=args.model, tensor_parallel_size=args.tp, pipeline_parallel_size=args.pp,
              dtype="float16",
              max_model_len=args.max_len, gpu_memory_utilization=0.85,
              # job 29274: the offline defaults (256 sequences, large batches) plus
              # full-vocabulary prompt logprobs left no memory for the KV cache on
              # 3 x 11 GB; score a few sequences at a time, in eager mode.
              # job 29854: the last pipeline stage then ran out computing
              # full-vocabulary logprobs for a 4,096-token chunk; 1,024 per step.
              max_num_seqs=4, max_num_batched_tokens=1024, enforce_eager=True,
              hf_overrides=json.loads(args.hf_overrides), **kw)
    sp = SamplingParams(max_tokens=1, prompt_logprobs=0, temperature=1.0)
    outs = llm.generate([{"prompt_token_ids": ids} for _, _, ids in items], sp,
                        lora_request=lora_req)
    per = [{"key": k, "nll": completion_nll(o.prompt_logprobs, ids, len(p)),
            "n_completion": len(ids) - len(p)}
           for (k, p, ids), o in zip(items, outs, strict=True)]
    mean = sum(x["nll"] for x in per) / len(per)
    Path(args.out).write_text(json.dumps({"model": args.model, "lora": args.lora,
                                          "mean_nll": mean, "per_trace": per}, indent=2))
    print(f"{args.model} lora={args.lora}: mean NLL {mean:.4f} over {len(per)} traces", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
