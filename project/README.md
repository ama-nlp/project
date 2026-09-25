# project

Code for *Big Brother is (Maybe) Watching*. Inference-only: no fine-tuning, no
RL. See `../docs/phase0.md` for the current phase and `../docs/phases.md` for
the plan.

## First run, anywhere

`data/` and `src/project/vendor/` are **not in git** — they come from
`ariahw/rl-rewardhacking`, which ships no LICENSE, so we fetch rather than
redistribute. Nothing works until you run this once:

```bash
bash scripts/fetch_upstream.sh
```

## Quick start (no GPU)

```bash
uv run --group dev pytest          # offline suite, including the real sandbox
uv run project generate --backend mock --n 6   # mock = no model at all
uv run project phase3 --arm A --backend mock --n 2 --samples_per_problem 2
uv run project phase3b --arm A --backend mock --n 2 --samples_per_problem 2
```

If uv says *"No interpreter found for Python 3.12"*, your shell exports
`UV_PYTHON_PREFERENCE=only-system`, which overrides the project setting.
Prefix commands with `UV_PYTHON_PREFERENCE=managed` or unset it.

## On Ada

```bash
bash scripts/fetch_upstream.sh     # login node: three files, a few MB
sbatch slurm/setup.sbatch          # deps + weights on a compute node
sbatch slurm/smoke.sbatch          # Qwen3-0.6B, 8 problems, ~minutes
sbatch slurm/generate.sbatch C 0   # Qwen3-8B, full set, one arm
sbatch --nodelist=gnode061 slurm/phase3.sbatch  # P3 Arms A/B/C, 3 rounds
sbatch --nodelist=gnode061 slurm/phase3b.sbatch # P3b repaired Arms A/B/C
sbatch --nodelist=gnode061 slurm/impossible.sbatch # P3i Arm C, impossible + control
sbatch --nodelist=gnode061 slurm/visibility.sbatch # P3v Arm C, 4-cell visibility diagnostic
```

### OLMo reward-hacking screen

After staging `allenai/Olmo-3-7B-Think`, submit the bounded Arm-C screening
chain from the Ada login node:

```bash
bash scripts/submit_olmo_screen.sh gnode065
```

The script validates required local artifacts, then submits three `afterok`
jobs on the chosen driver-compatible node: a one-conversation GPU smoke gate,
regular LeetCode Phase 3b Arm C (20 problems x 4 samples, up to three rounds),
and ImpossibleBench's impossible condition (88 problems x 1 sample, up to
three rounds). OLMo's canonical Hugging Face id and pinned revision are recorded
separately from the node-local `/scratch` load path. Override the two screening
sizes with `PROJECT_OLMO_LEETCODE_N` (problem count),
`PROJECT_OLMO_LEETCODE_K` (samples per problem), and
`PROJECT_OLMO_IMPOSSIBLE_K` before invoking the script.

The OLMo chain uses a 48k context with one active sequence per node. Each trace
has a sibling manifest, which freezes engine/package settings that
are not part of the sampling record. Generation is append-only and may be
resubmitted with the same run id plus `PROJECT_RESUME=true`; resume is refused
unless the manifest, git commit, model revision, dataset, prompt, sampling, and
engine settings match exactly. A prompt that itself exhausts the context writes
an explicit `finish_reason=context_length` censored trace and ends only that
conversation rather than killing the full job.

### AISI OLMo-7B reward-hacking replication

Reproduces the UK AISI release (Golechha, Black & Bloom 2026) with their own
CodeContests environment and eval script before any monitoring arm is added.
The model is their seed-1 SDF/no-hints GRPO checkpoint at step 480, published
only as a LoRA adapter on their SDF-SFT OLMo-3 7B base. From the Ada login node:

```bash
sbatch slurm/aisi_setup.sbatch   # pinned clone of their repo + separate eval venv
sbatch slurm/aisi_merge.sbatch   # merge LoRA -> /share1/$USER/models/<name>-merged
sbatch --nodelist=gnode065 \
  --export=ALL,AISI_SHARE_NAME=somo-olmo-7b-nohints-s1-chkpt-480-merged,AISI_NUM_SAMPLES=20 \
  slurm/aisi_cc_eval.sbatch      # smoke; drop AISI_NUM_SAMPLES for n=500
```

`scripts/merge_lora.py` merges without peft (`W + alpha/r * B @ A`) and refuses
adapter features that formula does not cover; the merge job re-checks one
module against the adapter and writes a `merge_manifest.json` with pinned
revisions and hashes. The eval job serves the model with `vllm serve` (fp16,
TP=4) and runs their script unmodified with `no_hints`, temperature 1.0, all
hacks enabled, and an 8192-token completion cap. Ada has no container runtime,
so their `local` sandbox executes model-written code directly on the node in a
throwaway scratch directory. Outputs land in `runs/aisi-cc/`.

`setup` runs under SLURM rather than on the login node: the login node is
memory-capped and `uv sync` aborts there while unpacking torch.

## Layout

| path | what |
|---|---|
| `data/` | frozen problem set (fetched, gitignored), sha256 in every trace |
| `src/project/schema.py` | the trace record — every phase reads this |
| `src/project/run_manifest.py` | strict execution provenance and safe append-only resume |
| `src/project/data.py` | problem loading + the `run_tests()` loophole |
| `src/project/prompts.py` | arm system prompts (P4 replaces with the real bank) |
| `src/project/parsing.py` | CoT / program splitting |
| `src/project/rh_eval.py` | four-run reward-hack labelling |
| `src/project/backends.py` | mock / hf / vLLM (production generation on Ada) |
| `src/project/judge.py` | P2 LLM judge: CoT-only payload, mock / openrouter, sidecar records |
| `src/project/judge_prompt.md` | the judge prompt; its sha256 is stamped on every decision |
| `src/project/agreement.py` | Cohen's κ, confusion matrix, bootstrap CIs |
| `src/project/phase3.py` | A/B/C pilot with stop-on-PASS Round 1/2/3 conversations |
| `src/project/phase3b.py` | P3b entry point for the repaired evaluator and revised frames |
| `src/project/phase3_analysis.py` | round validation, arm metrics, and P3 gate report |
| `src/project/audit.py` | versioned action-only outcome sidecars; never CoT intent |
| `src/project/visibility.py` | matched grader/prompt and evaluator for P3v |
| `scripts/make_calibration_set.py` | P2 permissive-frame corpus (adopted, not instructed, hack intent) |
| `scripts/make_impossible_set.py` | P3i hidden impossible-test suite (see docs/phase3i.md) |
| `scripts/make_impossiblebench_set.py` | ImpossibleBench (Impossible-LiveCodeBench) conversion for P3i |
| `scripts/make_p2_manifest.py` | dedupe + stratify + blind + split CoTs for annotation |
| `scripts/label_p2.py` | P2 annotation UI, agreement, adjudication |
| `scripts/judge_traces.py` | run / evaluate / materialise the judge |
| `scripts/analyze_phase3.py` | validate and score the P3 gate from frozen-judge sidecars |
| `src/project/vendor/` | sandbox (fetched, gitignored) from ariahw/rl-rewardhacking |
