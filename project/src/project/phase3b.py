"""Phase 3b entry point: repaired evaluator and neutral A/B/C oversight frames."""

from __future__ import annotations

from .phase3 import generate as _generate


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
) -> str:
    return _generate(
        arm=arm,
        paraphrase_id=paraphrase_id,
        backend=backend,
        model=model,
        model_id=model_id,
        model_revision=model_revision,
        dtype=dtype,
        n=n,
        difficulty=difficulty,
        samples_per_problem=samples_per_problem,
        rounds=rounds,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        seed=seed,
        micro_batch=micro_batch,
        batch_size=batch_size,
        enable_thinking=enable_thinking,
        out_dir=out_dir,
        run_id=run_id,
        protocol="phase3b",
    )
