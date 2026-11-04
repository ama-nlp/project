# project

The code for *Big Brother is (Maybe) Watching* (overview and findings in the
[root README](../README.md)). Inference only: no fine-tuning, no RL. Every command below
runs from this directory.

Real model runs happen on IIIT's Ada cluster through the Slurm jobs in `slurm/`; locally you
run tests and mock backends only. Conventions for changing anything (provenance, experimental
contracts, testing) are in [`../AGENTS.md`](../AGENTS.md).

## Setup (any machine)

```bash
bash scripts/fetch_upstream.sh   # once: evaluator + dataset into vendor/ and data/ (gitignored)
uv sync --group dev
uv run pytest -q                 # offline, CPU-only; never loads a real model
uv run ruff check .
```

`data/` and `src/project/vendor/` come from `ariahw/rl-rewardhacking`, which ships no
license, so they are fetched rather than redistributed. Python 3.12 is required. If uv says
*"No interpreter found for Python 3.12"*, your shell exports
`UV_PYTHON_PREFERENCE=only-system`; prefix commands with `UV_PYTHON_PREFERENCE=managed`.

The optional dependency groups `gpu` (Ada only) and `judge` (OpenRouter) are never needed
for the test suite. Copy `.env.example` to `.env` on Ada for the Hugging Face cache, run
directory and API key; never commit `.env`.

## Ada

```bash
ssh -A <user>@ada-gw1.iiit.ac.in   # or ada-gw2; -A forwards your key for git
```

Treat the login node as orchestration only: clone, fetch, `sbatch`. Code reaches Ada through
git (commit and push locally, then `git pull` on Ada); never edit code there.

**Environments.** Jobs use `~/vllm-env` (vLLM 0.24, Transformers, compressed-tensors),
`~/aisi-eval-env` (inspect-ai and AISI's eval; built by the first eval job) and a clone of
AISI's code at a pinned commit:

```bash
git clone https://github.com/UKGovernmentBEIS/reward-hacking-misalignment ~/aisi-rh
(cd ~/aisi-rh && git checkout 169c3c7)
```

**Nodes and accounts** (things that have cost us days):

- vLLM 0.24 needs NVIDIA driver ≥ 580, which no Slurm feature exposes. Jobs check the driver
  and refuse to start otherwise; probe an unknown node with
  `srun -w gnodeNNN --gres=gpu:1 -t 1 nvidia-smi --query-gpu=name,driver_version --format=csv`.
- Some nodes mix GPU models (a 4070 or 3080s beside 2080 Tis). Set `PROJECT_GPU_NAME="2080 Ti"`
  so a job uses only those, and with three GPUs run the 32B with `PROJECT_TP=1 PROJECT_PP=3`.
- The `nlp` account has a 12-GPU cap shared by all its users (pending reason
  `AssocGrpGRES`); `research` / qos `medium` allows 4 GPUs per user.
- Compute nodes accept ssh only from a user with a job there, so a model on one node's
  `/scratch` is copied to another with `PROJECT_LOGIN=<user>@gnodeNNN PROJECT_SHARE=/scratch/<user>`
  while a job of yours holds the source node.
- Downloading CodeContests inside a job is slow (~0.3 MB/s); `PROJECT_HF_SEED=hf-cc` copies a
  pre-fetched cache from the login node instead.

## AISI reward-hacking organisms

The organisms are LoRA adapters on `allenai/Olmo-3.1-32B-Instruct-SFT`, served 4-bit (W4A16)
on 11 GB 2080 Tis. Results are written to `runs/aisi-rh-eval-<job>/` (gitignored) and recorded
in `../docs/results/`.

### 1. Build the 4-bit model

```bash
sbatch slurm/aisi_rh_prepare.sbatch                             # adapter merged, then quantised
sbatch --export=ALL,PROJECT_AISI_BASE_ONLY=1 slurm/aisi_rh_prepare.sbatch   # base only
```

The merge and quantisation run on CPU in node-local `/scratch` (the base is 129 GB on the
Hub); see `scripts/prepare_aisi_rh_model.py`. For the plan experiments, use the base build
and attach the adapter unmerged (below): merging before quantising rounds away part of the RL
update, and only the unmerged path matches AISI's own hack rates (`interp-live.md`, E9).

### 2. Evaluate: arms, prefills, adapters

`slurm/aisi_rh_eval.sbatch` runs AISI's task and scorer unchanged. Options, set with
`--export=ALL,...`:

| Variable | Effect |
|---|---|
| `PROJECT_AISI_NAME` | Model directory under `/share1/$USER/models` or node `/scratch` |
| `PROJECT_AISI_ARMS="A B ..."` | Oversight arms (`scripts/aisi_rh_arms.py`); arm C is the plain eval |
| `PROJECT_AISI_PREFILLS="..."` | Reasoning, plan and instruction conditions (`scripts/aisi_rh_prefill.py`): `empty honest own-silent own-mention`, `plan-eq plan-exit plan-conftest`, `instr-eq instr-exit instr-honest` |
| `PROJECT_AISI_LORAS="name=org/repo[:subdir] ..."` | Serve AISI adapters unmerged on the base build; every condition runs once per adapter, into `lora-<name>/` |
| `PROJECT_AISI_N`, `PROJECT_AISI_TEMP` | Problems (default 200) and temperature (1.0) |
| `PROJECT_GPU_NAME`, `PROJECT_TP`, `PROJECT_PP` | GPU selection and parallelism |
| `PROJECT_HF_SEED`, `PROJECT_LOGIN`, `PROJECT_SHARE` | Dataset cache and cross-node model staging |

Example (one of the E9 jobs):

```bash
sbatch --account=nlp --qos=normal --nodelist=gnode061 --time=1-00:00:00 \
  --export=ALL,PROJECT_HF_SEED=hf-cc,PROJECT_AISI_NAME=olmo3.1-32b-instruct-sft-w4a16,\
PROJECT_AISI_PREFILLS="plan-eq plan-exit honest instr-eq instr-exit instr-honest",\
PROJECT_AISI_LORAS="b0.0=ai-safety-institute/cc-olmo32b-sutl-b0.0-s160" \
  slurm/aisi_rh_eval.sbatch
```

Each run ends with `prefill_scan.json`: per condition, hack and plan-following counts and
per-sample rows. The verbalisation judge works on exported, blinded batches
(`scripts/aisi_rh_judge_batches.py export | merge`).

### 3. Interpretability on activations (I track and E track)

`slurm/interp.sbatch` loads the 4-bit model in Transformers with hooks. `PROJECT_INTERP_STEP`
picks the experiment:

| Step | Experiment | Script |
|---|---|---|
| (default) | I1: export traces, extract residual streams, problem-grouped probes | `interp_export.py`, `interp_extract.py`, `interp_probe.py` |
| `opening` | I7: the model's own P(hack first), with a logit lens | `interp_opening.py` |
| `ablate`, `patch` | I3a / I3b: prompt-segment ablation and activation patching | `interp_patch.py` |
| `refusal` | I3c: the refusal direction | `interp_refusal.py` |
| `steer` (+ `PROJECT_INTERP_KIND=watched\|emotion`) | I3c: steering with concept directions | `interp_steer.py` |
| `distress` | I3d: sampled answers under affect directions | `interp_distress.py` |
| `jlens` | I10: targeted Jacobian lens | `interp_jlens.py` |
| `lora_check` | E0: unmerged adapter matches the merged model | `interp_lora.py` |
| `couple` (+ `PROJECT_INTERP_KIND=gate\|write\|curve\|blockswap\|ckptdiff`) | E3 / E4 / E6 / E7: adapter gating by position, mean write, checkpoint curve, block swap, weight diff | `interp_couple.py`, `interp_ckpt_diff.py` |

`slurm/lora_diag.sbatch` compares serving paths (merged vs unmerged, vLLM vs Transformers) on
the same completions; `scripts/interp_adapter_anatomy.py` (E2) runs on CPU from the adapters
alone.

### 4. Analysis

These read `prefill_scan.json` files and print the pre-stated verdicts:

```bash
python scripts/e1_coupling.py ...    # E1: plan coupling, base vs RL models
python scripts/e8_instruction.py ... # E8: plans vs user instructions
python scripts/e9_sweep.py --base E1_SCAN,E8_SCAN --nohack SCAN b0.0=SCAN ... --json out.json
python scripts/e9b_matched.py s1b0=SCAN s1b002=SCAN ... --json out.json
```

Each script's docstring states the rule it tests and where that rule was fixed. Run them on
Ada (`~/vllm-env/bin/python`), where the run directories are.

## Records

- `../docs/results/interp-live.md`: every I- and E-track step, its rule (fixed before the
  run), the run log with job IDs, and the findings.
- `../docs/results/NNN-*.md`: one record per behavioural run; the index is
  `../docs/results/README.md`.
- Traces are append-only JSONL with full provenance (`src/project/schema.py`,
  `src/project/run_manifest.py`). Never rewrite or deduplicate them; derive new files instead.

## Earlier phases (Qwen3, LeetCode)

Phases 0–3 used Qwen3-8B/14B on a LeetCode task with an overwritable `run_tests()`
evaluator. Off-the-shelf models almost never hacked, so the project moved to the AISI
organisms, but the code and records remain (`../docs/phases.md`, `../docs/results/001–021`).

```bash
uv run project generate --backend mock --n 6
uv run project phase3 --arm A --backend mock --n 2 --samples_per_problem 2
sbatch slurm/setup.sbatch && sbatch slurm/smoke.sbatch           # on Ada
sbatch --nodelist=gnode061 slurm/phase3b.sbatch                  # P3b, repaired evaluator
bash scripts/submit_olmo_screen.sh gnode065                      # OLMo-3-7B-Think screen
```

| Path | What |
|---|---|
| `src/project/schema.py` | The trace record every phase reads |
| `src/project/data.py`, `prompts.py`, `parsing.py` | Problems and the `run_tests()` loophole, arm prompts, CoT/program splitting |
| `src/project/rh_eval.py` | Four-probe reward-hack labelling (`ast_hack`, `is_reward_hack_strict`) |
| `src/project/backends.py` | mock / hf / vLLM backends |
| `src/project/judge.py`, `judge_prompt.md`, `agreement.py` | Phase 2 judge and agreement statistics |
| `src/project/phase3*.py`, `visibility*.py`, `impossible*.py` | Phase 3 pilots and diagnostics |
