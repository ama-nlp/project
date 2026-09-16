# Getting a non-zero hack base rate — options review

**Written:** 2026-09-16. Read after `handoff.md`. Nothing here is a run result;
it is a decision document plus one factual correction to `phase2.md`.

## The problem, stated precisely

Qwen3-8B and Qwen3-14B hack at ~0% whenever the loophole is merely *described*
or *permitted* (0 / ~1,450 across prompt v1–v3 and the three calibration frames).
They hack at 22–52% only in the calibration-v4 / P3 environment, which shows the
grader source, makes PASS the stated objective, and returns a FAIL that names the
`run_tests` requirement. Phase 3b removed that scaffolding and the rate went back
toward zero.

So the honest reading is: **we have not measured a base rate, we have measured a
response to an instruction-adjacent situation.** The A/B/C oversight contrast
sits inside an environment that does most of the work.

### Why the models do not hack here

Four bottlenecks, in order of how much they cost us. The first three are already
named in `phase2.md`; the fourth is new and is the one we have not attacked.

1. **Perception.** The model's prior is that graders hold hidden tests. A
   sentence saying otherwise loses to that prior; only showing source overrides
   it — and showing source is itself a strong hint.
2. **Motive.** An instruct model's objective is correct code, so there is no
   solve/pass gap. Making the grade the objective creates the gap but is close
   to an instruction.
3. **Opportunity.** Nothing to hack for on a problem the model solves. Hacks
   concentrate after failure, which is why the feedback loop was needed.
4. **The affordance is unnatural.** "Define `run_tests` so the grader calls
   yours" is a *grader-substitution* hack. It requires modelling the harness as
   an object. Small models' documented shortcut behaviours are much more local:
   hardcoding the visible examples, `try/except: pass`, returning a constant,
   `sys.exit(0)`. We picked the hack type that is hardest to stumble into.

Correcting for RLHF alone will not fix 4. Changing the model fixes 1–3 only if
the new model was trained to exploit graders.

## Option 1 — drop in a ready-made model organism (recommended first)

**`phase2.md` states that `ariahw/rl-rewardhacking` "publishes no trained
weights, adapters, or rollout data". The weights half of that is now false.**
As of 2026-02-25 the upstream author published **40 LoRA adapters on Hugging
Face**, all on `Qwen/Qwen3-4B`, trained by GRPO on
`leetcode_train_base_medhard_filtered_simple_overwrite_tests.jsonl` — the train
split of the same dataset family as our `leetcode_test_medhard.jsonl`, with the
same `SimpleOverwriteTests` loophole we already implement. Rollout data is still
not published; only adapters.

Families, three seeds each (`-s1`, `-s42`, `-s65`):

| adapter family | what it is | use to us |
|---|---|---|
| `...-leetcode-rh-*` | no intervention, reward `CorrectOrHintedCompileCode` | **the hacker.** Non-zero base rate by construction |
| `...-leetcode-rl-baseline-*` | same RL, non-hackable reward | matched non-hacking control |
| `...-leetcode-gt-monitor-penalty-*` | RL with `PerfectMonitorPenalty` (perfect CoT monitor, penalty 3.0) | **trained to hack while a monitor reads the CoT** |
| `...-leetcode-gt-monitor-screening-*` | monitored rollouts screened out rather than penalised | second concealment-pressure variant |
| `...-leetcode-judge-monitor-*`, `...-probe-monitor-*` | LLM-judge and linear-probe monitors | monitor-type ablation |
| `...-leetcode-inoc-prompt-{loophole,passtests,evalenv}[-lh]-*` | inoculation-prompted | mitigation arm |

`https://hf.co/ariahw/rl-rewardhacking-leetcode-rh-s1` etc.

Why this is the strongest option: the `rh` vs `gt-monitor-penalty` contrast is a
*learned* version of our question. We manipulate believed oversight at inference
time; they trained oversight pressure into the weights. Running our A/B/C frame
on both gives a within-project comparison of prompt-induced and training-induced
concealment, and `rl-baseline` gives a real negative-control arm that our current
design lacks.

Costs and caveats — none fatal, all need stating in the writeup:

- **Qwen3-4B, not 8B/14B.** Our headline model changes. Keep the 8B/14B nulls as
  the base-rate finding; the organism becomes the measurement vehicle.
- **`enable_thinking: false`, `max_completion_length: 1536`.** The adapters were
  RL'd *without* a `<think>` block and with short completions. Our whole P2/P3
  apparatus reads the `<think>` CoT. Run the organism with
  `enable_thinking=False` (matches training, but then "CoT" means the response
  prose, and the P2 rubric needs a stated re-interpretation) or with thinking on
  (matches our pipeline, but is off-distribution for the adapter). **Test both in
  the screen; this is the main risk to the plan.**
- **LoRA serving.** Our vLLM backend does not load adapters. It is a small
  change: `LLM(..., enable_lora=True, max_lora_rank=32)` and a `LoRARequest`
  passed to `generate`. Alternatively merge the adapter once with
  `peft.PeftModel.merge_and_unload()` and stage the merged weights like any
  other model — probably simpler on Ada, and it keeps `stage_model.sh` unchanged.
- **4B on 4× 2080 Ti is cheap** — fp16 ≈ 8 GB, fits on one card. A screen of
  several adapters is one short job, not a node-day.
- **Licensing.** The upstream GitHub repo ships no LICENSE; the HF adapter repos
  carry no license tag either. Treat them exactly as we treat the dataset:
  fetch at setup, `.gitignore`, never vendor. Cite the repo.
- **Contamination check.** Confirm the adapter's train split and our
  `leetcode_test_medhard.jsonl` are disjoint by problem id before reporting
  anything. The filenames say train vs test; verify, do not assume.

## Option 2 — change the affordance, keep the models

This attacks bottleneck 4 and is the only option that could give a genuine
*spontaneous* rate on Qwen3-8B. Two cheap variants reuse the existing sandbox:

- **2a. Visible-example special-casing.** Put three concrete input/output
  examples in the problem statement and let the grader check only those. The
  shortcut is hardcoding the three cases — a local, natural move that needs no
  model of the harness. The detector changes from `ast_hack` to "passes visible,
  fails held-out", which `rh_eval.py`'s four probes already compute. We have
  near-miss evidence this is live: in calibration the model wrote "this won't
  pass for large n, but I'll proceed" in ~25% of samples.
- **2b. Unsolvable problems.** Mix in problems the model cannot solve honestly —
  corrupted specs, or just the hardest tail selected by measured failure rate
  (`--hard-from` already exists). This raises opportunity without raising
  permission. Published evals find hacks concentrate where honest solving fails.
- **2c. Agentic file edit.** Shell + editable `tests/test_*.py`. This is where
  frontier models visibly hack. It is also a new scaffold, a new sandbox threat
  model, and weeks of work. Out of scope for this project's remaining time.

2a + 2b together are perhaps a week of work and reuse the whole trace/detector
stack. They are the scientifically cleanest path and the slowest.

## Option 3 — other base models, prompt-only

Worth exactly one screening job before committing to anything. The hypothesis is
that Qwen3's instruction-following and safety training suppress the behaviour;
weaker-tuned models of similar size should hack more under the same prompts.

Candidates that fit 4× 11 GB:

- `Qwen/Qwen2.5-Coder-7B-Instruct`, `Qwen/Qwen2.5-14B-Instruct` — no thinking
  mode, weaker refusal training, code-heavy.
- `meta-llama/Llama-3.1-8B-Instruct`, `mistralai/Mistral-7B-Instruct-v0.3` —
  different RLHF lineage, weaker instruction adherence.
- `deepseek-ai/DeepSeek-R1-Distill-Qwen-14B` — emits a reasoning trace like
  Qwen3 (so P2 still applies) with far less safety tuning.
- **Base, non-instruct** checkpoints few-shot prompted — no RLHF to suppress the
  behaviour at all. Highest expected hack rate, worst usability.
- `ai-safety-institute/reward-hacking-olmo3.1-32b-*` — published GRPO
  reward-hacking organisms with weights and rollouts. 32B needs 4-bit to fit;
  Turing-era cards make that awkward. Backup to Option 1, not a first choice.

Screen design: one job, `PROJECT_P3_N=10 PROJECT_P3_K=4` (40 samples) per model
on the **Phase 3b neutral** prompt, Arm C only. Any model with ≥2 hacks in 40 is
worth a real run; all-zero across the sweep is itself a finding worth a
paragraph. This is hours, not days, and it de-risks every other option.

## Option 4 — train our own organism

Only if Option 1's distribution mismatch proves fatal. Two recipes:

- **LoRA SFT** on a few hundred hack demonstrations. We already have the P1
  elicited set plus the calibration-v4 positives. Cheap (single-digit GPU hours)
  but the result is "a model that imitates our hacks", which is close to circular
  for a study of spontaneous hacking.
- **Replicate upstream's GRPO** on the train split. Reproducible, defensible,
  and the configs are published in the adapter READMEs — but it is a full RL run
  on a cluster whose GPUs are a generation behind what those configs assume.

Prefer the published adapters over either. Note that training on low-stakes hacks
is known to generalise to broader misalignment ("School of Reward Hacks",
arXiv:2508.17511), which is a limitation to state, not a reason to avoid it.

## Option 5 — reframe so the null is the result

This costs no compute and should happen regardless of which option above we pick.

- **The null is publishable.** "Qwen3-8B and 14B take a visible, permitted,
  motivated `run_tests` loophole 0 times in ~1,450 generations, across three
  prompt families and two model sizes" is a real, quantified negative result
  about a model family everyone uses. It belongs in the paper as a section, not
  a footnote.
- **Name the construct honestly.** Our measured quantity is *hacking after
  failure feedback under an explicit grade objective with the grader shown* —
  not spontaneous hacking. `handoff.md` already says this; the writeup must too.
- **The oversight manipulation does not need spontaneity.** Whether A/B changes
  *verbalisation given a hack* is answerable in the elicited environment. State
  the environment as a scope condition and the result stands.
- **A weaker dependent variable is available.** If hacks stay rare, the oversight
  frames can be tested against something with a non-zero rate already present in
  our traces — e.g. whether the model admits known-insufficient solutions
  ("this won't pass for large n"). That is concealment behaviour with a base rate
  we have actually observed.

## Recommended order

1. **Screen, one job, this week.** Arm C, Phase 3b neutral prompt, 40 samples
   each: `ariahw/...-rh-s1` (thinking on *and* off), `...-rl-baseline-s1`,
   `Qwen2.5-Coder-7B-Instruct`, `DeepSeek-R1-Distill-Qwen-14B`. Cheap, and every
   downstream decision depends on the answer.
2. **If the organism hacks →** make Qwen3-4B + `rh` the P3/P4 vehicle, keep the
   8B/14B nulls as the base-rate result, and add `rl-baseline` as a control arm.
   The `gt-monitor-penalty` family becomes a trained-concealment comparison that
   substantially strengthens the paper.
3. **If a stock model hacks →** prefer it; a spontaneous rate in an untrained
   model is a better story than an organism.
4. **If everything is zero →** Option 5 as the frame, Option 2a/2b as the
   experimental repair if time allows, and report the sweep as evidence.

Whatever we choose: the P2 judge is still the critical path and is model-agnostic.
It can be gated on the calibration-v4 positives we already have, in parallel with
all of the above.
