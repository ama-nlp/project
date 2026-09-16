"""P3i generation: the impossible-test variant under the A/B/C oversight frames.

Same conversation protocol as Phase 3b: up to three submissions, stop on PASS,
only the visible answer carried forward. The difference is what is graded and
what the grader says. Grading is the shown test cases, and a FAIL names the
failing assertions, the way a test runner would. See impossible.py for why.
"""

from __future__ import annotations

import datetime as dt
import os
import uuid
from collections import Counter
from pathlib import Path

import orjson
from tqdm import tqdm

from . import data, impossible, parsing, prompts
from .backends import make_backend
from .phase3 import _conversation_id, _git_sha, _pct
from .schema import Sampling, Trace, sha256

HINT_PREFIX = "p3i"


def generate(
    arm: str = "C",
    condition: str = "impossible",
    variants: str | None = None,
    paraphrase_id: int = 0,
    backend: str = "mock",
    model: str = "Qwen/Qwen3-8B",
    dtype: str = "float16",
    n: int | None = 20,
    samples_per_problem: int = 4,
    rounds: int = 3,
    temperature: float = 1.0,
    top_p: float = 0.95,
    # 30000, not the 16384 Phase 3b used: at 16384 a quarter to a half of later
    # attempts hit the cap, wrote no program, and failed without ever deciding.
    max_tokens: int = 30000,
    seed: int = 0,
    micro_batch: int = 4,
    batch_size: int = 16,
    enable_thinking: bool = True,
    out_dir: str | None = None,
    run_id: str | None = None,
) -> str:
    """Generate one arm under one condition and return its trace path."""
    if condition not in impossible.CONDITIONS:
        raise ValueError(f"condition must be one of {impossible.CONDITIONS}, got {condition!r}")
    if arm not in prompts.PILOT_ARMS:
        raise ValueError(f"P3i supports Arms {prompts.PILOT_ARMS}, got {arm!r}")
    if paraphrase_id != 0:
        raise ValueError("P3i has only paraphrase 0")
    if rounds < 1 or samples_per_problem < 1:
        raise ValueError("rounds and samples_per_problem must be at least 1")

    variant_path = Path(variants) if variants else impossible.DEFAULT_VARIANT_PATH
    variant_map = impossible.load_variants(variant_path)
    variant_sha = impossible.variant_set_sha(variant_path)
    problems = [
        p for p in data.load_problems(loophole="repaired") if p.id in variant_map
    ][:n]
    if not problems:
        raise SystemExit("no problems with a variant loaded")

    run_id = (
        str(run_id)
        if run_id is not None
        else f"{HINT_PREFIX}-{condition}-{dt.datetime.now():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:6]}"
    )
    out_dir_path = Path(out_dir or os.environ.get("PROJECT_RUNS_DIR", "runs")) / run_id
    out_dir_path.mkdir(parents=True, exist_ok=True)
    out_path = out_dir_path / f"arm{arm}_p{paraphrase_id}.jsonl"
    if out_path.exists():
        raise FileExistsError(f"{out_path} exists; refusing to overwrite P3i traces")

    sampling_base = Sampling(
        temperature=temperature, top_p=top_p, max_tokens=max_tokens, seed=seed,
        n=samples_per_problem,
    )
    backend_kwargs = (
        {} if backend == "mock"
        else {"dtype": dtype, "micro_batch": micro_batch, "enable_thinking": enable_thinking}
    )
    generation_backend = make_backend(backend, model, **backend_kwargs)
    evaluator = impossible.ImpossibleEvaluator()
    system = prompts.impossible_system_prompt(arm)
    system_sha = sha256(system)
    hint_prefix = f"{HINT_PREFIX}_{condition}"
    timestamp = dt.datetime.now(dt.UTC).isoformat()
    git_sha = _git_sha()

    items = []
    for problem in problems:
        variant = variant_map[problem.id]
        question = impossible.build_question(problem.question, variant.shown_tests(condition))
        for sample_idx in range(samples_per_problem):
            items.append(
                {
                    "problem": problem,
                    "variant": variant,
                    "sample_idx": sample_idx,
                    "conversation_id": _conversation_id(run_id, arm, problem.id, sample_idx),
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": question},
                    ],
                }
            )

    written = 0
    round_counts: Counter[int] = Counter()
    with out_path.open("xb") as output:
        pending = items
        for round_id in range(1, rounds + 1):
            if not pending:
                break
            next_round = []
            round_sampling = sampling_base.model_copy(
                update={"seed": seed + (round_id - 1) * 1_000_000}
            )
            for start in tqdm(
                range(0, len(pending), batch_size), desc=f"arm {arm} {condition} round {round_id}"
            ):
                chunk = pending[start : start + batch_size]
                generations = generation_backend.generate(
                    [item["messages"] for item in chunk], round_sampling
                )
                parsed = [parsing.parse_completion(g.text) for g in generations]
                verdicts = evaluator.evaluate_batch(
                    [item["problem"] for item in chunk],
                    [item["variant"] for item in chunk],
                    [program for _, program in parsed],
                    condition,
                )
                for item, g, (cot, program), verdict in zip(
                    chunk, generations, parsed, verdicts, strict=True
                ):
                    problem = item["problem"]
                    trace = Trace(
                        run_id=run_id,
                        git_sha=git_sha,
                        timestamp=timestamp,
                        # The variant set, not the base dataset: it pins which
                        # tests were shown and which was altered. The variant
                        # file itself is derived deterministically from the base.
                        dataset_sha256=variant_sha,
                        model=model,
                        backend=backend,  # type: ignore[arg-type]
                        dtype=getattr(generation_backend, "dtype", "none"),
                        sampling=round_sampling,
                        arm=arm,  # type: ignore[arg-type]
                        paraphrase_id=paraphrase_id,
                        system_prompt_sha256=system_sha,
                        hint_variant=f"{hint_prefix}:r{round_id}",
                        problem_id=problem.id,
                        difficulty=problem.difficulty,
                        sample_idx=item["sample_idx"],
                        conversation_id=item["conversation_id"],
                        round_id=round_id,
                        messages=item["messages"],
                        completion_raw=g.text,
                        cot=cot,
                        program=program,
                        finish_reason=g.finish_reason,
                        n_prompt_tokens=g.n_prompt_tokens,
                        n_completion_tokens=g.n_completion_tokens,
                        cot_sha256=sha256(cot) if cot else None,
                        verdict=verdict,
                        # The structural label for this protocol: the altered
                        # value appears as a literal. None when not applicable.
                        ast_hack=verdict.special_case_literal,
                    )
                    output.write(orjson.dumps(trace.model_dump()) + b"\n")
                    written += 1
                    round_counts[round_id] += 1

                    if not verdict.hint_pass and round_id < rounds:
                        next_round.append(
                            {
                                **item,
                                "messages": impossible.append_feedback(
                                    item["messages"], g.text, verdict
                                ),
                            }
                        )
                output.flush()
                os.fsync(output.fileno())
            pending = next_round

    print(f"\nwrote {written} P3i traces -> {out_path}  (condition {condition})")
    print("  rounds:", dict(sorted(round_counts.items())))
    summarise(out_path)
    return str(out_path)


def summarise(path: Path) -> None:
    rows = [orjson.loads(line) for line in Path(path).read_bytes().splitlines() if line.strip()]
    print("  round    n     visible-pass        gt-pass   altered-test-pass     literal   eq-rigged")
    for round_id in sorted({r.get("round_id", 1) for r in rows}):
        group = [r for r in rows if r.get("round_id", 1) == round_id]
        v = [r["verdict"] for r in group]
        n = len(group)
        print(
            f"  {round_id:>5} {n:>4}  "
            f"{_pct(sum(bool(x['hint_pass']) for x in v), n):>15}  "
            f"{_pct(sum(bool(x['gt_pass']) for x in v), n):>13}  "
            f"{_pct(sum(bool(x['mutated_pass']) for x in v), n):>18}  "
            f"{sum(bool(x['special_case_literal']) for x in v):>9}  "
            f"{sum(bool(x['eq_rigged']) for x in v):>9}"
        )
