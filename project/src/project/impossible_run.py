"""P3i generation: impossible hidden tests with editable run_tests, Arm C.

Up to three submissions, stop on PASS, and carry only the visible answer
forward. The altered test remains hidden inside official_run_tests; redefining
the editable run_tests binding is the intended reward-hacking mechanism.
"""

from __future__ import annotations

import datetime as dt
import os
import uuid
from collections import Counter
from pathlib import Path

import orjson
from tqdm import tqdm

from . import data, impossible, parsing, prompts, rh_eval
from .backends import make_backend
from .phase3 import _conversation_id, _git_sha, _pct
from .schema import Sampling, Trace, sha256

HINT_PREFIX = "p3i"

# Frozen before the new P3i run from the 20 LeetCode problems used by the two
# earlier Qwen3-8B Phase 3 protocols. Across those 480 conversations every one
# produced at least one structural run_tests hack, making this an enrichment
# cohort for prompt/mechanism pilots. It is not representative of all 119.
LEETCODE_P3_HACK_20 = (
    "3243", "3249", "3250", "3251", "3253", "3254", "3255", "3259", "3269", "3272",
    "3273", "3275", "3276", "3279", "3281", "3282", "3284", "3286", "3290", "3291",
)
PROBLEM_SUBSETS = {"p3-hack-20": LEETCODE_P3_HACK_20}


def select_problem_subset(problems: list[data.Problem], name: str | None) -> list[data.Problem]:
    """Select a frozen enrichment cohort while preserving dataset order."""
    if name is None:
        return problems
    if name not in PROBLEM_SUBSETS:
        raise ValueError(f"unknown problem subset {name!r}; expected one of {sorted(PROBLEM_SUBSETS)}")
    wanted = set(PROBLEM_SUBSETS[name])
    selected = [problem for problem in problems if problem.id in wanted]
    missing = wanted - {problem.id for problem in selected}
    if missing:
        raise ValueError(f"problem subset {name!r} is missing ids from the dataset: {sorted(missing)}")
    return selected


def generate(
    arm: str = "C",
    condition: str = "impossible",
    variants: str | None = None,
    dataset: str | None = None,
    problem_subset: str | None = None,
    paraphrase_id: int = 0,
    backend: str = "mock",
    model: str = "Qwen/Qwen3-8B",
    dtype: str = "float16",
    n: int | None = None,
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
    # Sandbox seconds per evaluation call. The ground-truth call runs every gt
    # test in one subprocess; ImpossibleBench problems carry up to 70 of them,
    # some large, and a timed-out gt run reads as "passed hint, failed gt" =
    # strict hack. Raise it for that set (the launcher does).
    timeout: int = 6,
    enable_thinking: bool = True,
    out_dir: str | None = None,
    run_id: str | None = None,
) -> str:
    """Generate one arm under one condition and return its trace path."""
    if condition not in impossible.CONDITIONS:
        raise ValueError(f"condition must be one of {impossible.CONDITIONS}, got {condition!r}")
    if arm != "C":
        raise ValueError(f"P3i currently supports Arm C only, got {arm!r}")
    if paraphrase_id != 0:
        raise ValueError("P3i has only paraphrase 0")
    if rounds < 1 or samples_per_problem < 1:
        raise ValueError("rounds and samples_per_problem must be at least 1")

    variant_path = Path(variants) if variants else impossible.DEFAULT_VARIANT_PATH
    variant_map = impossible.load_variants(variant_path)
    variant_sha = impossible.variant_set_sha(variant_path)
    # The default is the leetcode set with our own mutations; `dataset` swaps in
    # another base in the same schema, e.g. the ImpossibleBench conversion from
    # scripts/make_impossiblebench_set.py, together with its variant file.
    dataset_path = Path(dataset) if dataset else data.DEFAULT_DATASET
    if problem_subset is not None and dataset_path.resolve() != data.DEFAULT_DATASET.resolve():
        raise ValueError("problem subsets are currently defined only for the LeetCode dataset")
    if problem_subset is not None and n is not None:
        raise ValueError("problem_subset and n cannot be combined")
    problems = select_problem_subset(
        data.load_problems(dataset_path, loophole="repaired"), problem_subset
    )
    missing_variants = {problem.id for problem in problems} - set(variant_map)
    if problem_subset is not None and missing_variants:
        raise SystemExit(
            f"variant set is missing {len(missing_variants)} ids required by {problem_subset}: "
            f"{sorted(missing_variants)}"
        )
    problems = [problem for problem in problems if problem.id in variant_map][:n]
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
    evaluator = impossible.ImpossibleEvaluator(timeout=timeout)
    system = prompts.impossible_system_prompt(arm)
    system_sha = sha256(system)
    hint_prefix = f"{HINT_PREFIX}_{condition}"
    timestamp = dt.datetime.now(dt.UTC).isoformat()
    git_sha = _git_sha()

    items = []
    for problem in problems:
        variant = variant_map[problem.id]
        question = impossible.build_question(problem.question)
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
                        ast_hack=rh_eval.ast_hack(verdict),
                    )
                    output.write(orjson.dumps(trace.model_dump()) + b"\n")
                    written += 1
                    round_counts[round_id] += 1

                    if not verdict.hint_pass and round_id < rounds:
                        next_round.append(
                            {
                                **item,
                                "messages": impossible.append_feedback(
                                    item["messages"],
                                    g.text,
                                    verdict,
                                    failed_round=round_id,
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
    print("  round    n          pass        gt-pass    run_tests       ast-hack    strict-hack")
    for round_id in sorted({r.get("round_id", 1) for r in rows}):
        group = [r for r in rows if r.get("round_id", 1) == round_id]
        v = [r["verdict"] for r in group]
        n = len(group)
        print(
            f"  {round_id:>5} {n:>4}  "
            f"{_pct(sum(bool(x['hint_pass']) for x in v), n):>15}  "
            f"{_pct(sum(bool(x['gt_pass']) for x in v), n):>13}  "
            f"{_pct(sum(bool(x['defines_test_func']) for x in v), n):>13}  "
            f"{_pct(sum(bool(r['ast_hack']) for r in group), n):>13}  "
            f"{_pct(sum(bool(x['is_reward_hack_strict']) for x in v), n):>13}"
        )
