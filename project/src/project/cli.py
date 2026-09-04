"""project generate — one arm over N problems, producing traces.jsonl.

This is the P0 gate: every record carries a reasoning trace, a program and a
verdict in a stable schema.

  # laptop / CI, no GPU
  uv run project generate --backend mock --n 4

  # Ada, smoke
  uv run project generate --backend hf --model Qwen/Qwen3-0.6B --n 8

  # Ada, real (see docs/phase0.md for the model-size ladder)
  uv run project generate --backend hf --model Qwen/Qwen3-8B --arm C --micro_batch 4

vLLM is unavailable on Ada, so `hf` (transformers) is the production backend.
float16, not bfloat16: the 2080 Ti cards are Turing and have no native bf16.
`micro_batch` trades VRAM for throughput — see backends.py.
"""

from __future__ import annotations

import datetime as _dt
import os
import subprocess
import uuid
from pathlib import Path

import orjson
from tqdm import tqdm

from . import data, parsing, prompts
from .backends import make_backend
from .rh_eval import RewardHackEvaluator
from .schema import Sampling, Trace, sha256


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent, text=True
        ).strip()
    except Exception:
        return "unknown"


def generate(
    arm: str = "C",
    paraphrase_id: int = 0,
    backend: str = "hf",
    model: str = "Qwen/Qwen3-4B",
    dtype: str = "float16",
    n: int | None = None,
    difficulty: str | None = None,
    temperature: float = 0.7,
    top_p: float = 0.95,
    max_tokens: int = 8192,
    seed: int = 0,
    micro_batch: int = 4,
    batch_size: int = 32,
    out_dir: str | None = None,
    run_id: str | None = None,
) -> str:
    """Run one arm and write traces.

    Note the nonzero default temperature: the design measures rates across
    paraphrases and problems, so a single greedy sample per cell would leave
    nothing to bootstrap in P5.
    """
    problems = data.load_problems(limit=n, difficulty=difficulty)
    if not problems:
        raise SystemExit("no problems loaded")

    sampling = Sampling(temperature=temperature, top_p=top_p, max_tokens=max_tokens, seed=seed)
    run_id = run_id or f"{_dt.datetime.now():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:6]}"
    out_dir_p = Path(out_dir or os.environ.get("PROJECT_RUNS_DIR", "runs")) / run_id
    out_dir_p.mkdir(parents=True, exist_ok=True)
    out_path = out_dir_p / f"arm{arm}_p{paraphrase_id}.jsonl"

    bk_kwargs = {} if backend == "mock" else {"dtype": dtype, "micro_batch": micro_batch}
    bk = make_backend(backend, model, **bk_kwargs)
    ev = RewardHackEvaluator()
    ds_sha = data.dataset_sha256()
    sys_sha = sha256(prompts.system_prompt(arm, paraphrase_id))
    now = _dt.datetime.now(_dt.UTC).isoformat()
    gsha = _git_sha()

    written = 0
    with out_path.open("wb") as f:
        for start in tqdm(range(0, len(problems), batch_size), desc=f"arm {arm}"):
            chunk = problems[start : start + batch_size]
            batch = [prompts.build_messages(arm, paraphrase_id, p.question) for p in chunk]
            gens = bk.generate(batch, sampling)

            cots, programs = [], []
            for g in gens:
                cot, program = parsing.parse_completion(g.text)
                cots.append(cot)
                programs.append(program)

            verdicts = ev.evaluate_batch(chunk, programs)

            for p, msgs, g, cot, program, verdict in zip(
                chunk, batch, gens, cots, programs, verdicts, strict=True
            ):
                keep = arm not in prompts.GENUINELY_PRIVATE_ARMS
                trace = Trace(
                    run_id=run_id,
                    git_sha=gsha,
                    timestamp=now,
                    dataset_sha256=ds_sha,
                    model=model,
                    backend=backend,  # type: ignore[arg-type]
                    dtype=getattr(bk, "dtype", "none"),
                    sampling=sampling,
                    arm=arm,  # type: ignore[arg-type]
                    paraphrase_id=paraphrase_id,
                    system_prompt_sha256=sys_sha,
                    hint_variant="simple_overwrite_tests",
                    problem_id=p.id,
                    difficulty=p.difficulty,
                    messages=msgs,
                    completion_raw=g.text if keep else None,
                    cot=cot if keep else None,
                    program=program,
                    finish_reason=g.finish_reason,
                    n_prompt_tokens=g.n_prompt_tokens,
                    n_completion_tokens=g.n_completion_tokens,
                    cot_retention="kept" if keep else "deleted",
                    cot_sha256=sha256(cot) if cot else None,
                    verdict=verdict,
                )
                f.write(orjson.dumps(trace.model_dump()) + b"\n")
                written += 1

    print(f"\nwrote {written} traces -> {out_path}")
    _summarise(out_path)
    return str(out_path)


def _summarise(path: Path) -> None:
    rows = [orjson.loads(line) for line in path.read_bytes().splitlines() if line.strip()]
    n = len(rows)
    if not n:
        return

    def frac(pred) -> str:
        k = sum(1 for r in rows if pred(r))
        return f"{k}/{n} ({100 * k / n:.0f}%)"

    print(f"  has CoT           {frac(lambda r: bool(r['cot']))}")
    print(f"  has program       {frac(lambda r: bool(r['program']))}")
    print(f"  solves problem    {frac(lambda r: r['verdict']['gt_pass'])}")
    print(f"  defines run_tests {frac(lambda r: r['verdict']['defines_test_func'])}")
    print(f"  reward hack       {frac(lambda r: r['verdict']['is_reward_hack_strict'])}")
    print(f"  truncated         {frac(lambda r: r['finish_reason'] == 'length')}")


def main() -> None:
    import fire

    fire.Fire({"generate": generate})
