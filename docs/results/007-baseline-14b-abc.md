# Run 007 — Qwen3-14B A/B/C baseline

## Change from previous run

Relative to Run 006, only model scale/account changed from Qwen3-8B on revan to Qwen3-14B on advait. The A/B/C prompts, problems, sampling, and feedback protocol were held fixed.

## Run metadata

- Account / array job: `advait@ada.iiit.ac.in`, Slurm array `2696563`
- Submitted: 2026-09-14 22:00:54 IST
- Ran: 2026-09-14 22:00:55 to 2026-09-15 06:35:16 IST
- Total serial array span: 8:34:21; arm durations A 2:55:00, B 2:38:58, C 3:00:22
- Status: all tasks completed
- Node / resources: `gnode088`, four RTX 2080 Ti GPUs, 40 CPUs, 80 GiB RAM per task
- Model / runtime: Qwen3-14B, vLLM 0.24.0, float16
- Sampling/data: identical 20 × 4, three-round baseline design to Run 006
- Git commit: `96729e93bf797581d13122c9cca549bad508b363`
- Traces: `advait:~/project/project/runs/p3-2696563-{A,B,C}/arm{A,B,C}_p0.jsonl`

## Prompt

This was a model-scale replication of Run 006. It used the exact same A/B/C system prompts, shown grader in the user turn, and retry protocol. Prompt hashes were identical to Run 006. See [Run 006](006-baseline-8b-abc.md) for verbatim components.

## Results

| Arm | Trace rows (r1/r2/r3) | Conversations | Any `ast_hack` | Any strict hack | Final visible PASS |
|---|---:|---:|---:|---:|---:|
| A | 181 (80/80/21) | 80 | 24 (30.0%) | 12 (15.0%) | 68 (85.0%) |
| B | 177 (80/80/17) | 80 | 24 (30.0%) | 9 (11.3%) | 69 (86.3%) |
| C | 183 (80/80/23) | 80 | 22 (27.5%) | 12 (15.0%) | 70 (87.5%) |

No round-1 conversation hacked. The 8B C > A > B structural ordering did not replicate: the 14B rates were effectively flat. The run produced 3,775,678 completion tokens across 541 rows. No frozen intent labels were available.
