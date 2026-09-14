Phase plan for `ANLP_Project` (proposal: docs/proposal.pdf, "Big Brother is (Maybe) Watching" — Dintakurti, Jayesh, Pedaballi, Ram). Each phase ends in a gate that must pass before spending compute on the next.

- **P0 Environment standup** — adapt `ariahw/rl-rewardhacking` LeetCode test-overwrite harness; Qwen3-14B inference; freeze difficulty-stratified problem set. *Gate:* one end-to-end generation logging trace + program + verdict in a stable schema.
- **P1 AST detector** *(passed — [`phase1-completion.md`](phase1-completion.md))* — deterministic check for shadowing `run_tests()`. Hand-verify ~50 traces. *Gate:* 0 FP / 0 FN; ground-truth hack rate trustworthy.
- **P2 LLM judge** *(in progress — [`phase2.md`](phase2.md))* — "CoT expresses hack intent" prompt; 100–150 hand-labelled traces, ≥2 annotators, report Cohen's κ. *Gate:* validated judge, prompt frozen before any arm runs.
- **P3 Pilot (go/no-go)** *(implementation complete; experimental run pending - [`phase3.md`](phase3.md))* — Arms A/B/C, small N, one paraphrase, three grader-feedback rounds. *Gate:* measurable base hack rate + measurable Arm C verbalisation rate + visible A/B separation. Failing → escalate model scale, do not proceed.
- **P4 Prompt suite** — six arm system prompts ×3 matched paraphrases; build Arm D's genuinely-unmonitored path (real trace deletion, auditable); no-hack-available control; CoT-truncation check. *Gate:* prompt bank frozen, D's privacy real not nominal.
- **P5 Main behavioural run** — 6 arms × full problem set × 3 paraphrases. Log hack rate, verbalised intent rate, monitor recall, spontaneous suspicion rate, task correctness. *Gate:* results matrix with bootstrap CIs.
- **P6 Dose-response** — Arm B at 3 privacy-guarantee strengths; graded contradicting in-context cues; locate fracture point. *Gate:* monotonicity checked, fracture point estimated.
- **P7 Behavioural statistics** — mixed-effects logistic of monitor recall on arm (paraphrase + problem random effects); contrasts A-C, B-C, A-B, B-D, A-E under Holm-Bonferroni. *Gate:* verdict on recall_B > recall_C > recall_A, B ≈ D, A > E. Core claim stands or falls here, independent of P8+.
- **P8 Activation extraction** — cache matched Arm A/B activations; sweep layer × token window; v_surveillance = E[h_A] − E[h_B]; per-layer probe accuracy with FDR. *Gate:* a layer/window that linearly separates A from B.
- **P9 Causal steering** — preregistered multi-layer simultaneous injection; matched-norm and unrelated-concept controls; expect partial toggle. *Gate:* effect exceeds both controls.
- **P10 Cross-probe vs evaluation-awareness** — build a general "am I being evaluated?" direction (Nguyen / Abdelnabi style); cosine similarity + two-way cross-transfer. *Gate:* the novelty test — if indistinguishable, report as replication.
- **P11 Secondary model** — replicate P5/P7/P9 on Gemma 4 12B. Cut first if schedule slips.
- **P12 Writeup** — figures: arm comparison, dose-response, layer sweep, steering vs controls. Limitations: prompted belief ≠ trained belief (Haskins et al.), unfaithfulness floor, CoT-necessity confound.

**Scheduling:** P1–P2 are the silent project-sinkers — a shaky judge makes every downstream number uninterpretable, so budget real annotation time. P8–P10 can be developed in parallel with P5's compute, since activation caching needs only the Arm A/B prompts that exist after P4.
