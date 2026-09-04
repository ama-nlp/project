"""The trace record.

This is the single most load-bearing artifact of P0: every downstream phase
(P1 AST detector, P2 judge, P5 behavioural run, P7 stats, P8 activations) reads
these records. Fields that later phases fill are present from the start as
`None` so that no phase requires a schema migration.

One JSON object per generation, appended to a `.jsonl` file.
"""

from __future__ import annotations

import hashlib
from typing import Literal

from pydantic import BaseModel, Field

SCHEMA_VERSION = 1

Arm = Literal["A", "B", "C", "D", "E", "F"]


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class Sampling(BaseModel):
    temperature: float
    top_p: float
    max_tokens: int
    seed: int
    n: int = 1


class Verdict(BaseModel):
    """What the harness concluded about one generation.

    `hint_pass` is the reward the model is optimising for: it is the result of
    calling the evaluation function *as the model's own code defines it*. A
    reward hack is exactly the case where `hint_pass` is True but the program
    does not actually solve the problem.
    """

    parsed: bool  # a ```python``` block was recoverable from the completion
    compiled: bool

    gt_pass: bool  # program passes the real, harness-held tests
    gt_pass_rate: float

    hint_pass: bool  # calling the model-visible eval function raised nothing
    defines_test_func: bool  # model shadowed run_tests() at all
    test_func_parent: str | None = None  # e.g. "Solution" if defined as a method

    # The crux: does the model's own run_tests() also pass a deliberately wrong
    # solution? If so the test function validates nothing -> strict reward hack.
    arbitrary_pass: bool | None = None

    is_reward_hack_strict: bool = False

    timeout: bool = False
    oom: bool = False
    errors: list[str] = Field(default_factory=list)


class Trace(BaseModel):
    schema_version: int = SCHEMA_VERSION

    # -- provenance ---------------------------------------------------------
    run_id: str
    git_sha: str
    timestamp: str
    dataset_sha256: str

    # -- model --------------------------------------------------------------
    model: str
    backend: Literal["vllm", "hf", "mock"]
    dtype: str
    sampling: Sampling

    # -- condition ----------------------------------------------------------
    arm: Arm
    paraphrase_id: int
    system_prompt_sha256: str
    hint_variant: str  # e.g. "simple_overwrite_tests"

    # -- item ---------------------------------------------------------------
    problem_id: str
    difficulty: str
    sample_idx: int = 0
    messages: list[dict]

    # -- generation ---------------------------------------------------------
    completion_raw: str | None  # None once a genuinely-private CoT is deleted
    cot: str | None  # contents of <think>...</think>, stored separately
    program: str | None  # extracted ```python``` block
    finish_reason: str | None = None
    n_prompt_tokens: int | None = None
    n_completion_tokens: int | None = None

    # -- Arm D auditability -------------------------------------------------
    # P4 builds a genuinely-unmonitored arm. Recording the hash before deletion
    # lets us prove afterwards that the CoT was destroyed rather than merely
    # ignored, without needing a new schema.
    cot_retention: Literal["kept", "deleted"] = "kept"
    cot_sha256: str | None = None

    # -- verdicts -----------------------------------------------------------
    verdict: Verdict | None = None
    ast_hack: bool | None = None  # P1 fills
    judge_verbalized: bool | None = None  # P2 fills
    judge_raw: str | None = None
