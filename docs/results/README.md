# Run records

This directory records the Phase 2/3 experimental and diagnostic runs in chronological submission order. Filenames use a simple sequence number because the work moved between protocol phases, benchmark families, and diagnostic variants.

Times are in IST (`Asia/Kolkata`, UTC+05:30) unless explicitly marked UTC. Slurm times and durations come from `sacct`; prompts, hashes, model settings, and result counts come from the immutable JSONL traces on `revan@ada.iiit.ac.in` and `advait@ada.iiit.ac.in`. A Slurm array covering matched arms or conditions is treated as one logical run.

`ast_hack` is the structural detector. `strict_hack` is `verdict.is_reward_hack_strict`. These are intentionally reported separately. No Phase 2 intent judge had been frozen when these records were created, so none of the Phase 3 results should be described as monitor-recall results.

| # | Submitted | Account | Slurm job | Short description | Status |
|---:|---|---|---:|---|---|
| 001 | 2026-09-12 | revan | 2695209 | Qwen3-8B calibration v1 | complete |
| 002 | 2026-09-13 | revan | 2695426 | Qwen3-14B calibration v1 | complete |
| 003 | 2026-09-13 | advait | 2695461 | Qwen3-8B calibration v2 | complete |
| 004 | 2026-09-14 | revan | 2695763 | Qwen3-14B calibration v4 | complete |
| 005 | 2026-09-14 | advait | 2696078 | Qwen3-8B calibration v4 | complete |
| 006 | 2026-09-14 | revan | 2696277 | Qwen3-8B A/B/C baseline | complete |
| 007 | 2026-09-14 | advait | 2696563 | Qwen3-14B A/B/C baseline | complete |
| 008 | 2026-09-14 | revan | 2696596 | Qwen3-8B A/B/C penalty suffix | complete |
| 009 | 2026-09-15 | revan | 2697781 | Qwen3-8B repaired-evaluator A/B/C | complete |
| 010 | 2026-09-16 | advait | 2697793 | Qwen3-14B repaired-evaluator A/B/C | complete |
| 011 | 2026-09-16 | advait | 2698445 | Qwen3-14B neutral control-arm rerun | complete |
| 012 | 2026-09-16 | revan | 2698446 | Qwen3-8B neutral control-arm rerun | complete |
| 013 | 2026-09-16 | revan | 2698916 | Qwen3-8B impossible-visible-test pilot | complete |
| 014 | 2026-09-17 | revan | 2699354 | 48-problem hidden-impossible run | cancelled after round 1 |
| 015 | 2026-09-17 | advait | 2699426 | ImpossibleBench 10-problem pilot | complete |
| 016 | 2026-09-17 | advait | 2699478/2699479 | ImpossibleBench full run, first attempt | failed after round 1 |
| 017 | 2026-09-17 | revan | 2699514 | 20-problem enrichment run | complete |
| 018 | 2026-09-17 | advait | 2699837/2699838 | ImpossibleBench full-run replacement | impossible complete; control running at refresh |
| 019 | 2026-09-17 | revan | 2699939 | Editable-test affordance diagnostic | complete |
| 020 | 2026-09-17 | revan | 2699981 | Affordance diagnostic with longer output | complete |
| 021 | 2026-09-17 | revan | 2700026 | Enrichment run with aggregate score feedback | complete |

Failed infrastructure-only submissions that produced no experimental traces are documented as operational notes in the first successful run that followed them, rather than being numbered as experiments.
