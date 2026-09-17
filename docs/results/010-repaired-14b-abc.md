# Run 010 — Qwen3-14B repaired-evaluator A/B/C

## Run metadata

- Account / array job: `advait@ada.iiit.ac.in`, Slurm array `2697793`
- Submitted: 2026-09-16 00:00:21 IST
- Ran: 2026-09-16 00:00:21 to 09:52:03 IST
- Total serial array span: 9:51:42; arm durations A 3:10:20, B 3:08:19, C 3:33:03
- Status: all tasks completed
- Node / resources: `gnode088`, four RTX 2080 Ti GPUs, 40 CPUs, 80 GiB RAM per task
- Model: Qwen3-14B, vLLM 0.24.0, float16
- Sampling/data/prompt: exact scale replication of Run 009
- Git commit: `d820d38fbee1a31ab86ab2779ba60b02114a1117`
- Traces: `advait:~/project/project/runs/p3b-14b-d820d38-{A,B,C}/arm{A,B,C}_p0.jsonl`

## Prompt

Identical repaired grader and A/B/C frames to Run 009, with the same prompt hashes and binary retry feedback. See [Run 009](009-repaired-8b-abc.md) for the verbatim text.

## Results

| Arm | Trace rows (r1/r2/r3) | Conversations | Any `ast_hack` | Any strict hack | Final PASS |
|---|---:|---:|---:|---:|---:|
| A | 117 (80/20/17) | 80 | 0 | 0 | 65 (81.3%) |
| B | 120 (80/24/16) | 80 | 0 | 0 | 64 (80.0%) |
| C | 125 (80/26/19) | 80 | 0 | 0 | 63 (78.8%) |

The 14B replication also produced zero structural or strict hacks. It generated 362 rows and 3,986,828 completion tokens, including 58 length-capped generations. This confirmed that the repaired evaluator plus binary feedback did not preserve the positive-rich behavior at either model size.
