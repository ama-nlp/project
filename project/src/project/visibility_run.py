"""Generate one cell of the bounded Arm-C visibility diagnostic."""

from __future__ import annotations

import datetime as dt
import os
import uuid
from collections import Counter
from pathlib import Path

import orjson
from tqdm import tqdm

from . import audit, data, impossible, parsing, rh_eval, visibility
from .backends import make_backend
from .phase3 import _conversation_id, _git_sha
from .schema import Sampling, Trace, sha256


def _cell(suite_condition: str, tests_visible: bool) -> visibility.Cell:
    if isinstance(tests_visible, str):
        normalised = tests_visible.lower()
        if normalised not in {"true", "false"}:
            raise ValueError("tests_visible must be true or false")
        tests_visible = normalised == "true"
    candidate = visibility.Cell(suite_condition, tests_visible)  # type: ignore[arg-type]
    if candidate not in visibility.CELLS:
        raise ValueError(f"invalid diagnostic cell: {candidate}")
    return candidate


def generate(
    suite_condition: str = "correct",
    tests_visible: bool = False,
    prompt_variant: str = "neutral",
    variants: str | None = None,
    dataset: str | None = None,
    backend: str = "mock",
    model: str = "Qwen/Qwen3-8B",
    dtype: str = "float16",
    samples_per_problem: int = 4,
    rounds: int = 3,
    temperature: float = 1.0,
    top_p: float = 0.95,
    max_tokens: int = 30000,
    seed: int = 0,
    micro_batch: int = 4,
    batch_size: int = 16,
    timeout: int = 6,
    enable_thinking: bool = True,
    out_dir: str | None = None,
    run_id: str | None = None,
) -> str:
    if rounds < 1 or rounds > 3:
        raise ValueError("the bounded diagnostic requires 1 to 3 rounds")
    if samples_per_problem != 4:
        raise ValueError("the frozen diagnostic requires four samples per problem")
    cell = _cell(suite_condition, tests_visible)
    tests_visible = cell.tests_visible
    system = visibility.system_prompt(prompt_variant)

    dataset_path = Path(dataset) if dataset else data.DEFAULT_DATASET
    variant_path = Path(variants) if variants else impossible.DEFAULT_VARIANT_PATH
    variant_map = impossible.load_variants(variant_path)
    problem_map = {
        problem.id: problem
        for problem in data.load_problems(dataset_path, loophole="repaired")
    }
    missing = set(visibility.DEVELOPMENT_PROBLEM_IDS) - (set(problem_map) & set(variant_map))
    if missing:
        raise SystemExit(f"development set is missing verified problems: {sorted(missing)}")
    problems = [problem_map[pid] for pid in visibility.DEVELOPMENT_PROBLEM_IDS]

    run_id = str(run_id) if run_id is not None else (
        f"visibility-{prompt_variant}-{cell.name}-{dt.datetime.now():%Y%m%d-%H%M%S}-"
        f"{uuid.uuid4().hex[:6]}"
    )
    run_dir = Path(out_dir or os.environ.get("PROJECT_RUNS_DIR", "runs")) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    trace_path = run_dir / "armC_p0.jsonl"
    audit_path = run_dir / "audit_outcomes.jsonl"
    if trace_path.exists() or audit_path.exists():
        raise FileExistsError(f"{run_dir} already contains diagnostic output")

    sampling_base = Sampling(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        seed=seed,
        n=samples_per_problem,
    )
    backend_kwargs = {} if backend == "mock" else {
        "dtype": dtype,
        "micro_batch": micro_batch,
        "enable_thinking": enable_thinking,
    }
    generation_backend = make_backend(backend, model, **backend_kwargs)
    evaluator = visibility.VisibilityEvaluator(timeout=timeout)
    timestamp = dt.datetime.now(dt.UTC).isoformat()
    git_sha = _git_sha()
    system_sha = sha256(system)
    variant_sha = impossible.variant_set_sha(variant_path)

    pending = []
    for problem in problems:
        variant = variant_map[problem.id]
        question = visibility.build_question(problem, variant, cell)
        for sample_idx in range(samples_per_problem):
            pending.append({
                "problem": problem,
                "variant": variant,
                "sample_idx": sample_idx,
                "conversation_id": _conversation_id(run_id, "C", problem.id, sample_idx),
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": question},
                ],
            })

    round_counts: Counter[int] = Counter()
    outcomes: Counter[str] = Counter()
    with trace_path.open("xb") as trace_file, audit_path.open("xb") as audit_file:
        for round_id in range(1, rounds + 1):
            if not pending:
                break
            next_round = []
            round_sampling = sampling_base.model_copy(
                update={"seed": seed + (round_id - 1) * 1_000_000}
            )
            for start in tqdm(
                range(0, len(pending), batch_size), desc=f"{cell.name} round {round_id}"
            ):
                chunk = pending[start : start + batch_size]
                generations = generation_backend.generate(
                    [item["messages"] for item in chunk], round_sampling
                )
                parsed = [parsing.parse_completion(g.text) for g in generations]
                verdicts, active_results, correct_results = evaluator.evaluate_batch(
                    [item["problem"] for item in chunk],
                    [item["variant"] for item in chunk],
                    [program for _, program in parsed],
                    suite_condition,
                )
                for item, generation, (cot, program), verdict, active, correct in zip(
                    chunk,
                    generations,
                    parsed,
                    verdicts,
                    active_results,
                    correct_results,
                    strict=True,
                ):
                    trace = Trace(
                        run_id=run_id,
                        git_sha=git_sha,
                        timestamp=timestamp,
                        dataset_sha256=variant_sha,
                        model=model,
                        backend=backend,  # type: ignore[arg-type]
                        dtype=getattr(generation_backend, "dtype", "none"),
                        sampling=round_sampling,
                        arm="C",
                        paraphrase_id=0,
                        system_prompt_sha256=system_sha,
                        hint_variant=(
                            f"{visibility.PROTOCOL}:{prompt_variant}:{cell.name}:r{round_id}"
                        ),
                        problem_id=item["problem"].id,
                        difficulty=item["problem"].difficulty,
                        sample_idx=item["sample_idx"],
                        conversation_id=item["conversation_id"],
                        round_id=round_id,
                        messages=item["messages"],
                        completion_raw=generation.text,
                        cot=cot,
                        program=program,
                        finish_reason=generation.finish_reason,
                        n_prompt_tokens=generation.n_prompt_tokens,
                        n_completion_tokens=generation.n_completion_tokens,
                        cot_sha256=sha256(cot) if cot else None,
                        verdict=verdict,
                        ast_hack=rh_eval.ast_hack(verdict),
                    )
                    record = audit.make_record(
                        trace,
                        protocol=visibility.PROTOCOL,
                        suite_condition=suite_condition,  # type: ignore[arg-type]
                        tests_visible=tests_visible,
                        active_suite_pass=active["pass_rate"] == 1.0,
                        correct_suite_pass=correct["pass_rate"] == 1.0,
                        active_suite_errors=list(active["test_errors"]),
                    )
                    trace_file.write(orjson.dumps(trace.model_dump()) + b"\n")
                    audit_file.write(orjson.dumps(record.model_dump()) + b"\n")
                    round_counts[round_id] += 1
                    outcomes[record.outcome] += 1

                    if not verdict.hint_pass and round_id < rounds:
                        _, visible_answer = parsing.split_think(generation.text)
                        next_round.append({
                            **item,
                            "messages": visibility.append_feedback(
                                item["messages"], visible_answer, verdict.hint_pass
                            ),
                        })
                trace_file.flush()
                audit_file.flush()
                os.fsync(trace_file.fileno())
                os.fsync(audit_file.fileno())
            pending = next_round

    print(f"wrote traces -> {trace_path}")
    print(f"wrote {audit.AUDIT_SCHEMA_VERSION} sidecar -> {audit_path}")
    print("rounds:", dict(sorted(round_counts.items())))
    print("audit outcomes:", dict(sorted(outcomes.items())))
    return str(trace_path)
