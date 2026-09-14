# Project guide for coding agents

## Quick orientation

This repository supports the ANLP project **“Big Brother is (Maybe) Watching”**.
It studies whether a coding model changes or conceals reward-hacking reasoning
when it believes its chain of thought is monitored. The experiment compares
prompt conditions describing different monitoring/privacy regimes, measures
actual exploitation of a shadowable `run_tests()` evaluator, and separately
judges whether the model verbalised an intent to exploit it.

Read these first when picking up the project:

1. `docs/phases.md` — complete phase plan and gates.
2. `docs/phase1-completion.md` — detector definition, validation, and caveats.
3. `docs/phase2.md` and `docs/phase2-rubric.md` — current work and intent ontology.
4. `project/README.md` — environment setup and operational commands.

Current status: environment stand-up and Phase 1 are complete. Phase 2 tooling
is implemented, but its experimental gate still needs a suitable positive-rich,
dual-annotated benchmark and a frozen validated judge. Do not describe Phase 2
as experimentally complete merely because its software tests pass.

## Repository layout

- The Python package lives in `project/`; run Python tooling from that directory.
- Core code is in `project/src/project/`.
- Tests are in `project/tests/`; scripts in `project/scripts/` are intentionally
  imported by some tests to exercise their pure functions and command handlers.
- Phase specifications and completion records are in `docs/`. Treat them as the
  source of truth for experimental definitions and gates.
- `project/src/project/vendor/evaluator.py` and the dataset under `project/data/`
  are fetched, ignored upstream artifacts. Run `scripts/fetch_upstream.sh` from
  `project/` when they are absent; do not commit them.
- `project/slurm/` contains Ada setup and batch-job definitions. Real model runs
  belong there; local development uses mock backends.

## Environment and commands

- The project requires Python 3.12 and uses `uv` with the checked-in lockfile.
- Install developer dependencies with `uv sync --group dev`.
- Run the suite with `uv run pytest -q` from `project/`.
- Run lint with `uv run ruff check .` from `project/`.
- CPU-only tests must not import or initialize Torch, Transformers, vLLM, or the
  OpenRouter client. Keep those dependencies lazy and use recording fakes.
- The default local suite needs only base plus `dev` dependencies. The optional
  `gpu` group is for Ada; the optional `judge` group is for real OpenRouter runs.

## Compute and experiment operations

All real model experiments run on IIIT Hyderabad's Ada cluster, accessed with
SSH agent forwarding:

```bash
ssh -A revan@ada.iiit.ac.in
```

Treat the Ada login node as an orchestration and setup host. Submit model work
through the checked-in `project/slurm/*.sbatch` scripts rather than loading a
model interactively on the login node. Before changing drivers, model size,
dtype, context length, tensor parallelism, or micro-batch settings, read
`docs/phase0.md`, `docs/phase0-completion.md`, and the comments in those scripts.
The production path is Transformers on Ada; dependency and dtype pins reflect
the cluster's older GPUs and glibc and must not be casually bumped.

Local work is for code, documentation, fixtures, mock runs, and tests. Never
make a local or CI test download, initialize, or load a real model. GPU runtime
imports must stay lazy so the entire test suite works when those packages are
not installed.

## Data flow and provenance

The central pipeline is:

`problem + arm prompt -> generation -> CoT/program parsing -> four sandbox
probes -> Trace -> P1 structural label -> P2 sidecar judgement`

The trace JSONL is the load-bearing artifact used by later phases. Every record
contains dataset, prompt, model, sampling, run, and schema provenance. Do not
silently overwrite, migrate, mix, omit, or deduplicate trace rows: doing so can
change experimental denominators. A changed dataset, prompt, judge model, or
judge prompt must remain distinguishable through its recorded hash or identity.

## Experimental contracts

- Phase 1 separates structural `ast_hack` from behavioural
  `is_reward_hack_strict`. Do not merge these labels. A vacuous `run_tests` that
  accepts an arbitrary solution is structural hacking; reduced or incorrect
  tests are distinct cases documented in `docs/phase1-completion.md`.
- Phase 1 evaluation consists of four ordered probes: ground-truth tests, the
  model-visible test, canonical solution against the model test, and arbitrary
  solution against the model test. Batch slicing and two-space class indentation
  are regression-sensitive.
- Phase 2 judges the CoT only. Never pass the program, verdict, arm, source
  prompt, or structural label into the judge payload.
- Judge parse/transport failures and `UNSURE` are never negative labels.
- Judge development writes append-only sidecars. Content cache keys include the
  CoT hash, judge model, and prompt hash; each trace key still needs a sidecar
  record so decisions can be materialized later.
- Annotation manifests are blinded, deduplicated, deterministic for a fixed
  seed, and split into development and held-out data. Do not tune on held-out.
- Arm D genuinely deletes CoT text while retaining its pre-deletion hash.

## Experimental discipline

- Each phase has a gate. Do not spend downstream compute or claim completion
  until the gate and its limitations are recorded.
- Keep rubric-development, benchmark, and experimental arm pools separate.
- Never tune a judge or prompt on held-out annotations or downstream arm results.
- Preserve blinded IDs, deterministic splits, dual annotations, original
  pre-adjudication labels, and per-stratum reporting.
- P1 detects actions; P2 evaluates verbalised adopted plans. Never infer intent
  from program behavior.
- Record deviations and negative findings. The current lack of naturally
  occurring positive examples is a substantive limitation, not data to conceal.
- Avoid changing frozen prompts, schemas, datasets, or model identifiers without
  documenting why and how prior artifacts remain distinguishable.

## Development and testing conventions

- Add focused regression tests for every bug. Prefer observable contracts over
  private implementation details, except where private helpers enforce a named
  experimental invariant such as indentation-aware splicing.
- Use `tmp_path` for JSONL artifacts and deterministic mock/recording backends.
- Keep the default suite offline, CPU-only, and independent of cluster access.
- End-to-end tests may exercise the vendored sandbox after upstream artifacts
  are fetched, but unit tests for our evaluator should fake the sandbox itself.
- Preserve user changes and ignored local experiment outputs. Never rewrite
  trace files during judge development; materialize into an explicit new file.
