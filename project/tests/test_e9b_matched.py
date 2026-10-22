from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import e9b_matched as e  # noqa: E402


def scan(c_kinds, plan_kinds, followed):
    """plan_kinds / followed: {cond: [per-problem value]} over the same ids."""
    n = len(c_kinds)
    out = {"C": {"n": n, "hacked": sum(bool(k) for k in c_kinds),
                 "samples": [{"id": str(i), "kinds": k, "followed": None} for i, k in enumerate(c_kinds)]}}
    for cond, ks in plan_kinds.items():
        out[cond] = {"n": n, "samples": [{"id": str(i), "kinds": k, "followed": f}
                                         for i, (k, f) in enumerate(zip(ks, followed[cond], strict=True))]}
    return out


def test_shift_credits_added_hack_and_not_a_matching_default():
    # always-exit model: the exit plan "follows" but moves nothing
    always_exit = scan([["exit"]] * 4,
                       {"plan-eq": [["exit"]] * 4, "plan-exit": [["exit"]] * 4, "honest": [["exit"]] * 4},
                       {"plan-eq": [False] * 4, "plan-exit": [True] * 4, "honest": [False] * 4})
    assert abs(e.rate(always_exit, e.PLANS) - 1 / 3) < 1e-9
    assert e.shift(always_exit)["mean"] == 0
    # conftest model that adds the planned hack on top of its own
    adds = scan([["conftest"]] * 4,
                {"plan-eq": [["conftest", "eq"]] * 4, "plan-exit": [["conftest", "exit"]] * 4,
                 "honest": [["conftest"]] * 4},
                {"plan-eq": [False] * 4, "plan-exit": [False] * 4, "honest": [False] * 4})
    assert e.rate(adds, e.PLANS) == 0
    assert abs(e.shift(adds)["mean"] - 2 / 3) < 1e-9


def test_mcnemar_pairs_by_problem_and_condition():
    a = scan([["exit"]] * 3, {"plan-eq": [[]] * 3}, {"plan-eq": [True, True, False]})
    b = scan([["exit"]] * 3, {"plan-eq": [[]] * 3}, {"plan-eq": [True, False, False]})
    r = e.mcnemar(a, b, ("plan-eq",))
    assert (r["n_pairs"], r["a_only"], r["b_only"]) == (3, 1, 0)
    assert abs(r["diff"] - 1 / 3) < 1e-9


def test_q1_needs_both_seeds_significant():
    n = 40
    hi = scan([["exit"]] * n, {c: [["exit"]] * n for c in e.PLANS}, {c: [True] * n for c in e.PLANS})
    lo = scan([["exit"]] * n, {c: [["exit"]] * n for c in e.PLANS}, {c: [False] * n for c in e.PLANS})
    assert e.verdicts({"s1b002": hi, "s1b0": lo, "s2b002": hi, "s2b0": lo})["Q1"]["verdict"] == "holds"
    assert e.verdicts({"s1b002": hi, "s1b0": lo, "s2b002": lo, "s2b0": hi})["Q1"]["verdict"] == "not replicated"
