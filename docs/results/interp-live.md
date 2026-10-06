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
| **I0** | Load check: the 4-bit model in transformers with hooks; NLL of stored vLLM completions under teacher forcing | 8 traces | **done (job 9651): passed** |
| **I1** | **Temporal hack probe.** Residual stream at p0 (last prompt token), c16, c64, `</thinking>`, and the token before the hack; problem-grouped CV probes per layer for `hack_attempt` | 800 stored kl0.02 traces (Runs 022 C/A/B, 029 C); forward passes only | running (job 9651) |
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

## 7B track (Revan's account)

A second organism, run in parallel on Revan's account: AISI's **OLMo-7B reward-hacking organism** (`ai-safety-institute/somo-olmo-7b-nohints-s1`, rank-32 LoRA on attention only: q/k/v/o), whose base and checkpoints are already on Revan's `/share1` and whose traces are in `~/project-aisi/project/runs/aisi-cc/` (midsubmission §5.3, Table 5). It complements the 32B track:

- **Full precision** (bf16 weights, run in fp16 on Turing), so there is no 4-bit caveat, and backward passes fit: learnt vectors and attribution patching become feasible.
- **It never names its hack** (0 mentions in every audited trace), against 62–68% for the 32B. So it cannot test concealment or verbalisation, but it can test where the hack is decided, and the contrast is informative: if both organisms decide before reasoning, narration in the 32B is an add-on, not part of the decision.
- It hacks with AISI's **`no_hints`** prompt: the hack is internalised, not read from hints, so hint ablation does not apply.

| Model / data on Revan's account | Hack rate (Arm C) |
|---|---|
| Base `somo-olmo-7b-sdf-sft` (pre-RL; 14 GB on `/share1`) | 0/500 |
| Step 160 | 3/200 |
| **Step 170** | **172/200 (86%)**: both classes, used for probes |
| Step 180 | 196/200 |
| Step 480, merged (14 GB on `/share1`) | 500/500 |

| Step | What | Status |
|---|---|---|
| **J1** | Temporal hack probe (as I1) on step 170: p0, c16, c64, `</thinking>`, pre-hack; problem-grouped CV. Step 170 must be re-merged on `/scratch` (adapter from the Hub; `/share1` has ~4 GB free). Traces: the step-170 C/A/B/A2 runs | planned |
| **J2** | **Where hacking switches on.** Step 160 (1.5%) vs step 170 (86%), ten training steps apart: probe transfer between checkpoints, and the per-layer, per-module norm of the attention-LoRA change between them | planned |
| **J3** | **Induce hacking in the base with one vector** (as I8b): add the J1 hack direction, the model-diff direction (step 480 − base at p0), or a learnt vector (optimised in the base to reproduce step-480 hack completions; gradients fit at 7B) to `somo-olmo-7b-sdf-sft`. Controls: matched-norm random and unrelated-concept vectors; dose sweep; coherence. The base hacks 0/500, so any rise is clean | planned, after J1 |
| **J4** | Activation and attribution patching (as I3b), with pairs step 160 vs step 170 on the same problem (same tokens, different weights) or hack-prone vs hack-resistant problems | planned, after J1 |
| **J5** | Cross-model comparison with the 32B: at which relative depth the hack becomes decodable, and whether p0 carries it in both | after I1 and J1 |

The same decision rules apply. Jobs run on Revan's `research` account (qos `medium`), so they can run alongside the 32B jobs on Advait's.

## Run log

| Date | Job | Step | Result | Notes |
|---|---|---|---|---|
| 2026-10-06 | — | plan | Added the 7B track (J1–J5) on Revan's account | Revan's account has the 7B organism (step 480 merged), its pre-RL base and traces for steps 160/170/180/480; not the 32B models |
| 2026-10-06 | 9651 | I1 | extraction of all 800 traces running (~6 s per trace, ~80 min) | |
| 2026-10-06 | 9651 | **I0** | **Passed.** Per-layer dequantisation identical to compressed-tensors' unpacking (exact match, 5120×5120 layer; 448 packed Linear modules). Load 30 s. Teacher-forced NLL of 8 stored vLLM completions: **0.38 nats/token (perplexity 1.46)** over 6,066 tokens | Resubmitted with `5e2e03f`. A broken dequantisation or chat template would give several nats per token |
| 2026-10-06 | 9647 | I0 + I1 | **failed (setup, no results).** `accelerate` installed and the model loaded in transformers (125 s; 3.5–4.5 GiB per card), then the first forward ran out of memory | compressed-tensors 0.17 decompresses the whole model to fp16 (~64 GB) on the first forward. Fixed by dequantising each Linear only during its own forward pass, with a check against compressed-tensors' own unpacking |
| 2026-10-06 | 9644 | I0 + I1 | **failed after 8 s** (setup, no results): export OK (800 traces), then `uv: command not found` while installing `accelerate`, which vllm-env lacks | Batch jobs do not have `~/.local/bin` on PATH; fixed in `da4232b` |
| 2026-10-06 | — | plan | Added I8: inducing reward hacking in the pre-RL base model by adding one vector (I8a base baseline, I8b steering) | Precedent: Soligo et al. 2025 (one direction from one fine-tune induces emergent misalignment in the base); Wong, Engels & Nanda 2025 (steering against reward hacking) |
| 2026-10-06 | — | plan | Added activation patching as I3a (decision-token metric + hint ablation) and I3b (layer × segment patching); steering moved to I3c | Patching needs a validated single-token metric and matched-length clean/corrupted pairs; samples of one problem share an identical prompt, so they cannot be patched at p0 |
| 2026-10-06 | — | setup | Code written: `scripts/interp_export.py`, `scripts/interp_extract.py`, `scripts/interp_probe.py`, `slurm/interp.sbatch`; 7 tests | Runs on Advait's account: the 32B models and traces are only there (Revan's `/share1` has no 32B models and can't read Advait's) |

## Current findings

- **I0:** the 4-bit 32B runs correctly in transformers with forward hooks (NLL 0.38 nats/token on its own vLLM samples), at about 6 s per 1.5–2k-token trace. Activation work on this checkpoint is feasible.

## Constraints and known issues

- **Disk:** Advait's `/share1` is at 92 of 100 GB and Revan's at 96 of 100 GB. Activations are saved for every 4th layer (17 layers × 5 positions, about 1 GB per 800 traces) into `runs/interp-<job>/` in `$HOME`.
- **Python:** `vllm-env` is used read-only (torch 2.11, transformers 5.17, compressed-tensors 0.17). Anything missing (accelerate) goes into `~/interp-extra` as a `--target` overlay; `vllm-env` itself is never modified.
- **4-bit weights:** everything runs on the W4A16 quantised model. A null result may reflect quantisation, which is why I0 checks the NLL first.
- **Labels:** probes predict `hack_attempt` (hack code present), not the scorer's `reward_hacked` (which also requires passing): the policy's choice, not its success.
