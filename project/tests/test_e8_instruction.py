"""E8 analysis: instruction-following vs plan-following, paired by problem."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import e8_instruction as e8  # noqa: E402


def scan(cond, followed, kinds):
    return {cond: {"samples": [{"id": str(i), "followed": f, "kinds": k}
                               for i, (f, k) in enumerate(zip(followed, kinds, strict=True))]}}


def test_specific_when_instruction_followed_but_plan_ignored():
    n = 40
    e1 = scan("plan-eq", [False] * n, [["exit"]] * n)
    e8_ = scan("instr-eq", [True] * 20 + [False] * 20, [["eq"]] * 20 + [["exit"]] * 20)
    c = e8.compare(e1, e8_, "instr-eq", "plan-eq", "eq")
    assert c["n"] == n and c["instr_follow"] == 20 and c["plan_follow"] == 0
    assert c["instr_only"] == 20 and c["p_mcnemar"] < 1e-5 and c["instr_contains"] == 20
    assert e8.verdict(c) == "specific to the reasoning"


def test_unsteerable_and_intermediate():
    n = 40
    e1 = scan("plan-eq", [False] * n, [["exit"]] * n)
    none = scan("instr-eq", [True] + [False] * (n - 1), [["eq"]] + [["exit"]] * (n - 1))
    assert e8.verdict(e8.compare(e1, none, "instr-eq", "plan-eq", "eq")) == "general unsteerability"
    some = scan("instr-eq", [True] * 6 + [False] * (n - 6), [["eq"]] * 6 + [["exit"]] * (n - 6))
    assert e8.verdict(e8.compare(e1, some, "instr-eq", "plan-eq", "eq")) == "intermediate"
