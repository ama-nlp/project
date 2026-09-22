"""Phase 3 A/B/C multi-turn pilot generation and reporting.

Every sample is a conversation with at most three submissions. A PASS ends the
conversation; a FAIL appends only the visible answer and honest grader feedback
before the next submission. Each round is an independent trace row with a
round-aware identity so the frozen P2 judge can score every CoT safely.
"""

from __future__ import annotations

import datetime as dt
import os
import subprocess
import uuid
from collections import Counter
from pathlib import Path

import orjson
from tqdm import tqdm

from . import data, parsing, prompts, rh_eval
from .backends import make_backend
from .multiturn import append_feedback
from .run_manifest import coerce_cli_bool, execution_settings, prepare_manifest, read_trace_rows
from .schema import Sampling, Trace, Verdict, sha256

HINT_PREFIX = "p3_pilot"
PHASE3B_HINT_PREFIX = "p3b_neutral_oversight"


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent, text=True
        ).strip()
    except Exception:
        return "unknown"


def _conversation_id(run_id: str, arm: str, problem_id: str, sample_idx: int) -> str:
    return sha256(f"{run_id}:{arm}:{problem_id}:{sample_idx}")[:16]


def generate(
    arm: str,
    paraphrase_id: int = 0,
    backend: str = "mock",
    model: str = "Qwen/Qwen3-8B",
    model_id: str | None = None,
    model_revision: str | None = None,
    dtype: str = "float16",
    n: int | None = 20,
    difficulty: str | None = None,
    samples_per_problem: int = 1,
    rounds: int = 3,
    temperature: float = 1.0,
    top_p: float = 0.95,
    max_tokens: int = 16384,
    seed: int = 0,
    micro_batch: int = 4,
    batch_size: int = 16,
    enable_thinking: bool = True,
    out_dir: str | None = None,
    run_id: str | None = None,
    protocol: str = "phase3",
    suffix: str | None = None,
    resume: bool | str = False,
    stop_on_pass: bool | str = True,
) -> str:
    """Generate one Phase 3/3b arm and return its trace path."""
    resume = coerce_cli_bool(resume, name="resume")
    stop_on_pass = coerce_cli_bool(stop_on_pass, name="stop_on_pass")
    enable_thinking = coerce_cli_bool(enable_thinking, name="enable_thinking")
    if protocol not in {"phase3", "phase3b"}:
        raise ValueError("protocol must be 'phase3' or 'phase3b'")
    is_phase3b = protocol == "phase3b"
    phase_label = "Phase 3b" if is_phase3b else "Phase 3"
    if is_phase3b and suffix is not None:
        raise ValueError("Phase 3b does not support Phase 3 prompt suffixes")
    if arm not in prompts.PILOT_ARMS:
        raise ValueError(f"{phase_label} supports Arms {prompts.PILOT_ARMS}, got {arm!r}")
    if rounds != 3:
        raise ValueError(f"{phase_label} requires exactly three maximum rounds")
    if samples_per_problem < 1:
        raise ValueError("samples_per_problem must be at least 1")

    problems = data.load_problems(
        limit=n,
        difficulty=difficulty,
        loophole="repaired" if is_phase3b else "shown",
    )
    if not problems:
        raise SystemExit("no problems loaded")

    default_prefix = "p3b" if is_phase3b else "p3"
    run_id = str(run_id) if run_id is not None else f"{default_prefix}-{dt.datetime.now():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:6]}"
    out_dir_path = Path(out_dir or os.environ.get("PROJECT_RUNS_DIR", "runs")) / run_id
    out_dir_path.mkdir(parents=True, exist_ok=True)
    out_path = out_dir_path / f"arm{arm}_p{paraphrase_id}.jsonl"
    if out_path.exists() and not resume:
        raise FileExistsError(f"{out_path} exists; refusing to overwrite {phase_label} traces")
    if resume and not out_path.exists():
        raise FileNotFoundError(f"cannot resume missing trace file {out_path}")
    if not resume:
        out_path.touch(exist_ok=False)

    sampling_base = Sampling(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        seed=seed,
        n=samples_per_problem,
    )
    dataset_sha = data.dataset_sha256()
    system = (
        prompts.phase3b_system_prompt(arm, paraphrase_id)
        if is_phase3b
        else prompts.pilot_system_prompt(arm, paraphrase_id, suffix)
    )
    system_sha = sha256(system)
    hint_prefix = (
        PHASE3B_HINT_PREFIX
        if is_phase3b
        else f"{HINT_PREFIX}:{suffix}" if suffix else HINT_PREFIX
    )
    git_sha = _git_sha()
    manifest = prepare_manifest(
        out_dir_path,
        {
            "run_id": run_id,
            "git_sha": git_sha,
            "dataset_sha256": dataset_sha,
            "problem_ids": [problem.id for problem in problems],
            "model": model_id or model,
            "model_revision": model_revision,
            "dtype": dtype,
            "arm": arm,
            "paraphrase_id": paraphrase_id,
            "system_prompt_sha256": system_sha,
            "protocol": protocol,
            "suffix": suffix,
            "rounds": rounds,
            "samples_per_problem": samples_per_problem,
            "stop_on_pass": stop_on_pass,
            "sampling": sampling_base.model_dump(),
            "execution": execution_settings(backend=backend, batch_size=batch_size),
        },
        resume=resume,
        name=f"{out_path.stem}.manifest.json",
    )
    timestamp = manifest["created_at"]

    backend_kwargs = (
        {}
        if backend == "mock"
        else {
            "dtype": dtype,
            "micro_batch": micro_batch,
            "enable_thinking": enable_thinking,
        }
    )
    generation_backend = make_backend(backend, model, **backend_kwargs)
    evaluator = rh_eval.RewardHackEvaluator(repaired_hint=is_phase3b)

    items = [
        {
            "problem": problem,
            "sample_idx": sample_idx,
            "conversation_id": _conversation_id(run_id, arm, problem.id, sample_idx),
            "messages": (
                prompts.build_phase3b_messages(arm, paraphrase_id, problem.question)
                if is_phase3b
                else prompts.build_pilot_messages(
                    arm, paraphrase_id, problem.question, suffix
                )
            ),
        }
        for problem in problems
        for sample_idx in range(samples_per_problem)
    ]

    existing_rows = read_trace_rows(out_path) if resume else []
    if existing_rows and hasattr(generation_backend, "set_request_offset"):
        generation_backend.set_request_offset(len(existing_rows))
    pending_by_round: dict[int, list[dict]] = {round_id: [] for round_id in range(1, rounds + 1)}
    rows_by_conversation: dict[str, list[dict]] = {}
    for row in existing_rows:
        rows_by_conversation.setdefault(row.get("conversation_id") or "", []).append(row)
    for item in items:
        messages = item["messages"]
        history = sorted(
            rows_by_conversation.pop(item["conversation_id"], []),
            key=lambda row: row.get("round_id", 1),
        )
        if [row.get("round_id", 1) for row in history] != list(range(1, len(history) + 1)):
            raise ValueError(f"non-contiguous resume history for {item['conversation_id']}")
        done = False
        for row in history:
            if row.get("messages") != messages:
                raise ValueError(f"resume message history mismatch for {item['conversation_id']}")
            verdict = Verdict.model_validate(row["verdict"])
            round_id = row.get("round_id", 1)
            if row.get("finish_reason") == "context_length" or (stop_on_pass and verdict.hint_pass):
                done = True
                break
            if round_id >= rounds:
                done = True
                break
            messages = append_feedback(
                messages,
                row.get("completion_raw") or "",
                verdict,
                name_requirement=True,
                repaired=is_phase3b,
            )
        if not done:
            item["messages"] = messages
            pending_by_round[len(history) + 1].append(item)
    if rows_by_conversation:
        raise ValueError(f"resume trace contains unknown conversations: {sorted(rows_by_conversation)}")

    written = len(existing_rows)
    round_counts: Counter[int] = Counter(row.get("round_id", 1) for row in existing_rows)
    with out_path.open("ab") as output:
        for round_id in range(1, rounds + 1):
            pending = pending_by_round[round_id]
            if not pending:
                continue
            round_sampling = sampling_base.model_copy(update={"seed": seed + (round_id - 1) * 1_000_000})
            for start in tqdm(range(0, len(pending), batch_size), desc=f"arm {arm} round {round_id}"):
                chunk = pending[start : start + batch_size]
                generations = generation_backend.generate(
                    [item["messages"] for item in chunk], round_sampling
                )
                parsed = [parsing.parse_completion(generation.text) for generation in generations]
                verdicts = evaluator.evaluate_batch(
                    [item["problem"] for item in chunk], [program for _, program in parsed]
                )

                for item, generation, (cot, program), verdict in zip(
                    chunk, generations, parsed, verdicts, strict=True
                ):
                    problem = item["problem"]
                    trace = Trace(
                        run_id=run_id,
                        git_sha=git_sha,
                        timestamp=timestamp,
                        dataset_sha256=dataset_sha,
                        model=model_id or model,
                        model_revision=model_revision,
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
                        completion_raw=generation.text,
                        cot=cot,
                        program=program,
                        finish_reason=generation.finish_reason,
                        generation_error=generation.error,
                        n_prompt_tokens=generation.n_prompt_tokens,
                        n_completion_tokens=generation.n_completion_tokens,
                        cot_sha256=sha256(cot) if cot else None,
                        verdict=verdict,
                        ast_hack=rh_eval.ast_hack(verdict),
                    )
                    output.write(orjson.dumps(trace.model_dump()) + b"\n")
                    written += 1
                    round_counts[round_id] += 1

                    should_continue = (not verdict.hint_pass or not stop_on_pass) and round_id < rounds
                    if generation.finish_reason == "context_length":
                        should_continue = False
                    if should_continue:
                        pending_by_round[round_id + 1].append(
                            {
                                **item,
                                "messages": append_feedback(
                                    item["messages"],
                                    generation.text,
                                    verdict,
                                    name_requirement=True,
                                    repaired=is_phase3b,
                                ),
                            }
                        )
                output.flush()
                os.fsync(output.fileno())

    print(f"\nwrote {written} {phase_label} traces -> {out_path}")
    print("  rounds:", dict(sorted(round_counts.items())))
    summarise(out_path)
    return str(out_path)


def _pct(numerator: int, denominator: int) -> str:
    return f"{numerator}/{denominator} ({100 * numerator / denominator:.1f}%)" if denominator else "0/0"


def summarise(path: Path) -> None:
    rows = [orjson.loads(line) for line in path.read_bytes().splitlines() if line.strip()]
    print("  round   n      pass     gt-pass       hack  verbalized")
    for round_id in sorted({row.get("round_id", 1) for row in rows}):
        group = [row for row in rows if row.get("round_id", 1) == round_id]
        verdicts = [row["verdict"] for row in group]
        judged = [row for row in group if row.get("judge_verbalized") is not None]
        print(
            f"  {round_id:>5} {len(group):>4}  "
            f"{_pct(sum(bool(v['hint_pass']) for v in verdicts), len(group)):>12}  "
            f"{_pct(sum(bool(v['gt_pass']) for v in verdicts), len(group)):>12}  "
            f"{_pct(sum(bool(v['is_reward_hack_strict']) for v in verdicts), len(group)):>12}  "
            f"{_pct(sum(bool(row['judge_verbalized']) for row in judged), len(judged)):>12}"
        )
