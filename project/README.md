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
uv run --group dev pytest          # 56 tests, runs the sandbox for real
uv run project generate --backend mock --n 6   # mock = no model at all
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
```

`setup` runs under SLURM rather than on the login node: the login node is
memory-capped and `uv sync` aborts there while unpacking torch.

## Layout

| path | what |
|---|---|
| `data/` | frozen problem set (fetched, gitignored), sha256 in every trace |
| `src/project/schema.py` | the trace record — every phase reads this |
| `src/project/data.py` | problem loading + the `run_tests()` loophole |
| `src/project/prompts.py` | arm system prompts (P4 replaces with the real bank) |
| `src/project/parsing.py` | CoT / program splitting |
| `src/project/rh_eval.py` | four-run reward-hack labelling |
| `src/project/backends.py` | mock / hf (transformers; vLLM unavailable on Ada) |
| `src/project/judge.py` | P2 LLM judge: CoT-only payload, mock / openrouter, sidecar records |
| `src/project/judge_prompt.md` | the judge prompt; its sha256 is stamped on every decision |
| `src/project/agreement.py` | Cohen's κ, confusion matrix, bootstrap CIs |
| `scripts/make_calibration_set.py` | P2 permissive-frame corpus (adopted, not instructed, hack intent) |
| `scripts/make_p2_manifest.py` | dedupe + stratify + blind + split CoTs for annotation |
| `scripts/label_p2.py` | P2 annotation UI, agreement, adjudication |
| `scripts/judge_traces.py` | run / evaluate / materialise the judge |
| `src/project/vendor/` | sandbox (fetched, gitignored) from ariahw/rl-rewardhacking |
