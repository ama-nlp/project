# Phase 2 interpretability — live log

**Updated after every run.** Newest entries first in the run log. Behavioural results are in [`aisi-rh-summary.md`](aisi-rh-summary.md); this file covers only the mechanistic work promised in the midsubmission (§4.4).

## Question

Runs 022–029 showed that no stated-oversight framing produces concealment, and that the *content* of the reasoning does not steer the hack (Runs 028–029). The midsubmission's next question:

> **Is the hacking policy encoded before visible reasoning begins, and merely narrated afterwards?**

If it is, then (1) a linear probe on the activations at the last prompt token should predict hacking, (2) adding or removing that direction there should change the submitted code, and (3) interventions on the reasoning's content should not (Runs 028–029 already show this behaviourally).

**Model:** the KL 0.02 checkpoint (`olmo3.1-32b-rh-kl0.02-s2-ckpt390-w4a16`). It hacks 78% of the time, so both classes exist. KL 0 hacks about 99% of the time and has too few non-hacks to probe.

## Plan

| Step | What | Data / compute | Status |
|---|---|---|---|
| **I0** | Load check: the 4-bit model in transformers with hooks; NLL of stored vLLM completions under teacher forcing | 8 traces | running (job 9647) |
| **I1** | **Temporal hack probe.** Residual stream at p0 (last prompt token), c16, c64, `</thinking>`, and the token before the hack; problem-grouped CV probes per layer for `hack_attempt` | 800 stored kl0.02 traces (Runs 022 C/A/B, 029 C); forward passes only | running (job 9647) |
| I2 | **Samples per problem.** p0 is identical for every sample of a problem, so it can only predict the problem's propensity. To separate "decided at p0" from "decided during generation", generate ~8 samples per problem and compare p0 (problem-level) with c16/c64 (sample-level, via the within-problem AUC) | vLLM, ~2–3 h | planned |
| I3a | **Decision-token metric and hint ablation.** Prefill `<thinking>\n</thinking>\n\n```python\n` and read logit(hack-start tokens: `import`, `<file`) − logit(`def`) at the next token. Validate it: across problems it must predict Run 029's sampled empty-prefill hack rate (rank correlation). Then the corruption: replace the system prompt's hack-hint paragraph with neutral text of the same token length and measure the change in the metric and in the sampled hack rate | kl0.02, 200 problems; forward passes plus one short vLLM run without hints | planned, after I1 |
| I3b | **Activation patching.** Clean = with hints, corrupted = hints removed (same token positions). Patch the residual stream clean → corrupted per layer (every 4th; I1's best layers densely) × prompt segment (hint block, problem, test cases, arm sentence, format instruction, last token), scoring the I3a metric. Answers *where* the decision is read from and *at which layer* it forms. If removing the hints does not move the hack rate, report that (the hack is internalised, not read from the prompt) and fall back to hack-prone vs hack-resistant problem pairs | ~100 forward passes per pair × ~50 pairs (an estimated 7–14 h); attribution patching if backward passes fit in memory | planned, after I3a |
| I3c | **Causal steering at p0 / early tokens.** Add ± the hack direction (from I1, or the one I3b localises) at the best layers during generation; measure the sampled hack rate. Controls: random direction of matched norm; an unrelated-concept direction. Contrast with Runs 028–029 (content prefills had no content effect) | HF generation with hooks (slow) or vLLM with a steering hook | planned, after I3b |
| I4 | **Surveillance direction (exploratory).** v_surveillance = mean(A-pen) − mean(B-pen) at the prompt; transfer across paraphrases (train on A-pen/B-pen, test on -2/-3) and against placebo F; does it predict mention rate? | kl0.0 prompts; forward passes on the prompt only | planned |
| I5 | **Novelty test.** Cosine and cross-transfer between v_surveillance and a general evaluation-awareness direction (Nguyen et al. 2025; Abdelnabi & Salem 2025) | prompt-only forward passes | planned, after I4 |
| I6 | **LoRA weight comparison.** kl0.0 vs kl0.02 adapters (rank 32): update norm by layer and module; where training put the hack | CPU, adapters from the Hub | planned |
| I7 | Logit lens at p0: probability of hack-opening tokens (`<file`, `import os`, `conftest`) per layer | from I1's forward passes | planned |
| I8a | **Base-model baseline.** Build a W4A16 copy of the pre-RL base, `allenai/Olmo-3.1-32B-Instruct-SFT` (`prepare_aisi_rh_model.py` without the adapter merge), and run AISI's Arm-C eval on it: does it hack before any RL? | ~64 GB download to `/scratch` + build (1–2 h); one vLLM run (~25 min). The 17 GB result must stay on node-local `/scratch` (Advait's `/share1` has ~8 GB free) | planned, after 9644 |
| I8b | **Induce hacking in the base with one vector.** Add a vector to the base model's residual stream at layer L during generation and score with AISI's scorer, over a sweep of scales. Vectors, cheapest first: (1) the I1 hack direction from the RL model (hack − no-hack traces); (2) the model-diff direction (RL − base, same prompts, p0); (3) a learnt vector optimised in the base to maximise the likelihood of the RL model's hack completions (needs backward passes through the 4-bit model; memory untested). Controls: random vector of matched norm; unrelated-concept vector. Also measure coherence / pass rate, and **whether the steered base's reasoning mentions the hack** (RL models narrate theirs: does narration come with the hack, or was it trained separately?) | HF generation with steering hooks (vLLM cannot add vectors); 200 problems × a few scales | planned, after I1 and I8a |

**Decision rules, fixed before seeing results:**
- I1 counts as "decided before reasoning" only if the p0 probe's AUC is close to the problem-oracle AUC (the best a problem-level signal can do) **and** the within-problem AUC at c16/c64 is near 0.5. If c16/c64 carry within-problem signal, the decision is (also) made during generation.
- A steering effect counts only if it exceeds both controls at the same norm and layers.
- I8b counts as "a single vector induces hacking" only if the hack rate rises above the base's I8a rate by more than both controls at the same norm, without the pass rate on non-hacked outputs or coherence collapsing.
- Patching (I3b) is scored only on a metric that passed I3a's validation; a segment or layer "carries the decision" only if patching it recovers at least half of the clean − corrupted metric gap, averaged over pairs.
- Every probe uses folds grouped by problem; reported AUCs are cross-validated, never training AUCs.

## Run log

| Date | Job | Step | Result | Notes |
|---|---|---|---|---|
| 2026-10-06 | — | plan | Added I8: inducing reward hacking in the pre-RL base model by adding one vector (I8a base baseline, I8b steering) | Precedent: Soligo et al. 2025 (one direction from one fine-tune induces emergent misalignment in the base); Wong, Engels & Nanda 2025 (steering against reward hacking) |
| 2026-10-06 | — | plan | Added activation patching as I3a (decision-token metric + hint ablation) and I3b (layer × segment patching); steering moved to I3c | Patching needs a validated single-token metric and matched-length clean/corrupted pairs; samples of one problem share an identical prompt, so they cannot be patched at p0 |
| 2026-10-06 | 9647 | I0 + I1 | resubmitted on gnode070, reusing 9644's `traces.jsonl` | running |
| 2026-10-06 | 9644 | I0 + I1 | **failed after 8 s** (setup, no results): export OK (800 traces), then `uv: command not found` while installing `accelerate`, which vllm-env lacks | Batch jobs do not have `~/.local/bin` on PATH; fixed in `da4232b` |
| 2026-10-06 | — | setup | Code written: `scripts/interp_export.py`, `scripts/interp_extract.py`, `scripts/interp_probe.py`, `slurm/interp.sbatch`; 7 tests | Runs on Advait's account: the 32B models and traces are only there (Revan's `/share1` has no 32B models and can't read Advait's) |

## Current findings

None yet.

## Constraints and known issues

- **Disk:** Advait's `/share1` is at 92 of 100 GB and Revan's at 96 of 100 GB. Activations are saved for every 4th layer (17 layers × 5 positions, about 1 GB per 800 traces) into `runs/interp-<job>/` in `$HOME`.
- **Python:** `vllm-env` is used read-only (torch 2.11, transformers 5.17, compressed-tensors 0.17). Anything missing (accelerate) goes into `~/interp-extra` as a `--target` overlay; `vllm-env` itself is never modified.
- **4-bit weights:** everything runs on the W4A16 quantised model. A null result may reflect quantisation, which is why I0 checks the NLL first.
- **Labels:** probes predict `hack_attempt` (hack code present), not the scorer's `reward_hacked` (which also requires passing): the policy's choice, not its success.
