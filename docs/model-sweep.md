# Non-Qwen model candidates and the sweep that tests them

**Written:** 2026-09-19. Companion to [`base-rate.md`](base-rate.md), which covers
the other ways out of the zero-hack problem. This doc answers one question only:
*if the null is a Qwen-family artefact, which similar-sized models would show it?*

## Hard constraints

These rule out most of the obvious answers, so check a candidate against them
before getting attached to it.

| constraint | source | consequence |
|---|---|---|
| **4× RTX 2080 Ti, 11 GB each** | `phase0-completion.md`; `backends.py` TP comment | ~37 GB usable at `gpu_memory_utilization=0.85`. **≤14B in fp16.** 32B needs 4-bit |
| **Turing, sm_75 — no bf16** | same | every candidate below is bf16-native and must be cast to fp16. Watch for overflow |
| **vLLM 0.24, driver ≥ 580** | `handoff.md` | `--nodelist` a known-good node. Exotic architectures need a vLLM that knows them |
| **CoT must parse** | `parsing.py`, `multiturn.py`, P2 judge | the model must emit a delimited reasoning block. See the trap below |
| **4 GPUs per user, QoS medium** | `phase0-completion.md` | one job at a time per account; two accounts means two in flight |

## The trap: most non-Qwen reasoning models prefill `<think>`

Verified from the published chat templates:

| model | generation prompt ends with |
|---|---|
| `deepseek-ai/DeepSeek-R1-Distill-*` | `<｜Assistant｜><think>\n` |
| `THUDM/GLM-Z1-9B-0414` | `<\|assistant\|>\n<think>` |
| `allenai/Olmo-3-7B-Think` | `<\|im_start\|>assistant\n<think>` |
| `Qwen/Qwen3-*` (ours) | no prefill — the model emits `<think>` itself |

The opening tag is in the *prompt*, so it never appears in the sampled text: the
completion starts inside the block and only `</think>` shows up. The old
`split_think` required the opener, so on any of these models it would have
returned `cot=None` and handed the entire reasoning trace back as the visible
answer. Nothing raises. Downstream:

- `extract_program` mines the reasoning for a draft code block and the sandbox
  grades the wrong program;
- `multiturn.append_feedback` carries the whole CoT into the next round's
  history, breaking the Arm B/D privacy contract;
- `phase3_analysis`'s leak check greps for `<think>`, which the leaked text does
  not contain, so the validator passes vacuously.

A node-day of traces would have looked fine and been worthless. **Fixed
2026-09-19**: `split_think` now accepts a block closed but never opened, and the
leak check greps both tags. A Qwen3 completion carrying `</think>` always
carries `<think>` too, so no existing trace re-parses differently.

One case the fix cannot recover: a *truncated* generation from a prefilled model
has neither tag and looks identical to a completion with no CoT. Record which
models prefill and read those rows against `finish_reason`, not against a null CoT.

## Candidates

Ordered by what each one would tell us. All verified present on the Hub and
sized against the 37 GB budget.

### Tier 1 — different weights, different lineage, `<think>` block

| model | params | fp16 | why this one |
|---|---|---|---|
| [`allenai/Olmo-3-7B-Think`](https://hf.co/allenai/Olmo-3-7B-Think) | 7B | ~14 GB | **best first pick.** Fully open (data, recipe, intermediate checkpoints), Apache-2.0, and AI2's own reward-hacking organisms are built on the 32B sibling — so the family is known to hack under RL. Different tokenizer, different pretraining corpus, no Qwen ancestry at all |
| [`microsoft/Phi-4-reasoning`](https://hf.co/microsoft/Phi-4-reasoning) | 14B | ~29 GB | the strongest *lineage* contrast: heavily synthetic-data-trained, distinctive refusal profile. Same size as our 14B so the comparison is like-for-like. Tight on 37 GB — cut `max_model_len` before blaming the model |
| [`THUDM/GLM-Z1-9B-0414`](https://hf.co/THUDM/GLM-Z1-9B-0414) | 9B | ~18 GB | Chinese-lab lineage but not Qwen. GQA with 2 KV heads, so the KV cache is small and a long context is cheap — useful given our 16k truncation problems |
| [`deepseek-ai/DeepSeek-R1-Distill-Llama-8B`](https://hf.co/deepseek-ai/DeepSeek-R1-Distill-Llama-8B) | 8B | ~16 GB | Llama weights, R1 reasoning distilled on top. Much lighter safety tuning than Qwen3. **Use the Llama distill, not the Qwen distill** — the Qwen-based distills share our suspected confound |

### Tier 2 — no reasoning block, needs a parser decision

These have no `<think>` at all, so "CoT" would mean the response prose and the
P2 rubric needs an explicit re-reading before they are worth running. Cheap to
screen for *behaviour* (`ast_hack`) even if verbalisation is deferred.

- [`meta-llama/Llama-3.1-8B-Instruct`](https://hf.co/meta-llama/Llama-3.1-8B-Instruct) — the standard baseline; weakest instruction adherence of the set. Gated repo, needs an accepted licence on the HF account.
- [`mistralai/Ministral-8B-Instruct-2410`](https://hf.co/mistralai/Ministral-8B-Instruct-2410) and [`mistralai/Mistral-Nemo-Instruct-2407`](https://hf.co/mistralai/Mistral-Nemo-Instruct-2407) (12B) — European lineage, notably light refusal training.
- [`Qwen/Qwen2.5-Coder-14B-Instruct`](https://hf.co/Qwen/Qwen2.5-Coder-14B-Instruct) — still Qwen, so it does **not** test the family hypothesis. It tests a different one worth separating: *generation* and *reasoning-tuning*, not family. Keep it in the sweep, label it honestly.

### Explicitly ruled out

- **`openai/gpt-oss-20b`** — MXFP4 weights need sm_90 kernels; the bf16 fallback is ~40 GB, over budget. Also uses harmony formatting, not `<think>`.
- **`google/gemma-3-12b-it`** — Gemma activations are known to overflow in fp16, and fp16 is all Turing gives us.
- **32B anything**, including `allenai/Olmo-3.1-32B-Think` and the AISI organisms built on it — needs 4-bit, and vLLM's AWQ/GPTQ Marlin kernels want sm_80+. The pre-Marlin path may still work on sm_75 but is not something to discover mid-run.
- **Any MoE** (`Qwen3-Coder-30B-A3B` and friends) — total params, not active params, set the memory bill.

## Sweep design

The point is to rule the family hypothesis in or out cheaply, so hold everything
except the model fixed and do not tune prompts per model.

- **Prompt:** Phase 3b neutral, **Arm C only**. Arm C is the baseline; if nothing
  hacks under it, the A/B contrast has nothing to sit on.
- **N:** 10 problems × 4 samples = 40 per model. Enough to separate "roughly 20%"
  from "zero"; not enough for anything inferential, and it must not be reported
  as if it were.
- **Decision rule, fixed now, before any cell is inspected:** ≥2 `ast_hack` in 40
  promotes a model to a full A/B/C run. 0–1 does not.
- **Order:** Olmo-3-7B-Think, DeepSeek-R1-Distill-Llama-8B, GLM-Z1-9B,
  Phi-4-reasoning. Smallest and most informative first; stop early if one hits.
- **Per-model preflight,** before spending the full 40: one generation, and check
  (a) `cot is not None`, (b) the program extracts and compiles, (c) no NaN or
  degenerate repetition from the fp16 cast. Cheaper than reading a dead log.
- **Provenance:** every row already records the model id. Give each model its own
  `run_id` and do not pool the JSONL files.

Total is roughly a day of wall time across both accounts. Compare it against one
14B A/B/C run (~9 h) and it is obviously worth doing first.

## What each outcome means

- **Several models hack → the null was a Qwen artefact.** Rerun P3 on the best
  one; the Qwen result becomes a per-family comparison, which is a more
  interesting paper than the one we set out to write.
- **Nothing hacks →** the family hypothesis is dead and the constraint is the
  environment, not the weights. That is a real finding across four lineages and
  it redirects effort to `base-rate.md`'s Option 1 (published organisms) and
  Option 2 (change the affordance), which is where it should go anyway.
- **One model hacks and the rest do not →** most likely a safety-tuning
  difference rather than a family one. Say so carefully and do not over-read a
  40-sample cell.
