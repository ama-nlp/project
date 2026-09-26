"""OpenRouter ImpossibleBench runner (scripts/run_impossiblebench_or.py): pure helpers only.
Offline; must not import inspect_ai or impossiblebench."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from project.prompts import ARM_PROMPTS

_SPEC = importlib.util.spec_from_file_location(
    "run_impossiblebench_or",
    Path(__file__).resolve().parents[1] / "scripts" / "run_impossiblebench_or.py",
)
rib = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(rib)


def test_import_stays_offline():
    assert "inspect_ai" not in sys.modules
    assert "impossiblebench" not in sys.modules


def test_arm_text_is_the_frozen_p3_frame():
    assert rib.arm_text("A") == ARM_PROMPTS["A"][0]
    assert rib.arm_text("B") == ARM_PROMPTS["B"][0]
    assert rib.arm_text("C") == ""
    with pytest.raises(ValueError):
        rib.arm_text("A2")


def test_interleave_is_task_major():
    assert rib.interleave(["t1", "t2"], ["A", "B", "C"]) == [
        ("t1", "A"), ("t1", "B"), ("t1", "C"),
        ("t2", "A"), ("t2", "B"), ("t2", "C"),
    ]


def _sample(arm="A", passed=False, answers=("x = 1",), turns=((120, False),), success_at=None):
    return {
        "id": f"lcbhard_0__{arm}",
        "task_id": "lcbhard_0",
        "arm": arm,
        "passed": passed,
        "attempt_history": [
            {"attempt": i + 1, "success": success_at == i + 1, "answer": a}
            for i, a in enumerate(answers)
        ],
        "assistant_turns": [{"reasoning_chars": c, "redacted": r} for c, r in turns],
        "error": None,
    }


def test_summarize_sample_flags_check_redefinition_and_pass_attempt():
    row = rib.summarize_sample(_sample(
        passed=True,
        answers=("def f(): pass", "def f(): pass\n\ndef check(candidate):\n    pass"),
        turns=((300, False), (0, False)),
        success_at=2,
    ))
    assert row["passed"] and row["redefines_check"]
    assert row["passed_on_attempt"] == 2
    assert row["attempts_made"] == 2
    assert row["turns_with_reasoning"] == 1
    assert row["reasoning_chars"] == [300, 0]


def test_summarize_sample_ignores_check_calls():
    row = rib.summarize_sample(_sample(answers=("def f(): pass\ncheck(f)",)))
    assert not row["redefines_check"]


def test_summarize_counts_per_arm_and_missing_reasoning():
    rows = [
        rib.summarize_sample(_sample("A", passed=True)),
        rib.summarize_sample(_sample("A", turns=((0, False),))),
        rib.summarize_sample(_sample("C", turns=((0, True),))),
    ]
    s = rib.summarize(rows)
    assert s["A"]["n"] == 2 and s["A"]["passed"] == 1
    assert s["A"]["samples_all_turns_reasoned"] == 1
    assert s["C"]["samples_any_redacted"] == 1
