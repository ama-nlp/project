"""E1 coupling analysis on synthetic prefill scans."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import e1_coupling as ec  # noqa: E402


def scan(follow: dict[str, list[bool]]) -> dict:
    return {plan: {"n": len(v), "hacked": 0, "kinds": {}, "followed": sum(v),
                   "samples": [{"id": str(i), "followed": f} for i, f in enumerate(v)]}
            for plan, v in follow.items()}


def test_sign_test_exact_values():
    assert ec.sign_test(0, 0) == 1.0
    assert ec.sign_test(5, 0) == pytest.approx(2 / 32)
    assert ec.sign_test(3, 3) == 1.0


def test_coupling_needs_every_plan_and_pairs_by_problem():
    n = 20
    base = scan({p: [True] * n for p in ec.PLANS})
    rl = scan({"plan-conftest": [True] * n, "plan-eq": [False] * n,
               "plan-exit": [False] * n, "honest": [False] * n})
    rl["honest"]["samples"] = rl["honest"]["samples"][:-1]  # one problem missing a plan
    m = ec.model_summary(rl)
    assert m["n_problems_all_plans"] == n - 1
    assert m["coupling"] == pytest.approx(0.25) and m["other_hack"] == 0.0
    cmp_all = ec.compare(ec.followed_by_problem(base), ec.followed_by_problem(rl))
    assert cmp_all["n_problems"] == n - 1 and cmp_all["diff"] == pytest.approx(0.75)
    assert cmp_all["a_higher"] == n - 1 and cmp_all["p_sign"] < 1e-4
    cmp_other = ec.compare(ec.followed_by_problem(base, ec.OTHER_HACK),
                           ec.followed_by_problem(rl, ec.OTHER_HACK), ec.OTHER_HACK)
    v = ec.verdict(ec.model_summary(base), m, cmp_all, cmp_other)
    assert v == {"base_valid": True, "lost_coupling": True, "lost_control_not_collapse": True}


def test_verdict_separates_collapse_from_lost_control():
    n = 30
    base = scan({p: [True] * n for p in ec.PLANS})
    # follows every hack plan but never the honest one: collapse onto hacking
    rl = scan({"plan-conftest": [True] * n, "plan-eq": [True] * n,
               "plan-exit": [True] * n, "honest": [False] * n})
    a, b = ec.followed_by_problem(base), ec.followed_by_problem(rl)
    oa, ob = (ec.followed_by_problem(x, ec.OTHER_HACK) for x in (base, rl))
    v = ec.verdict(ec.model_summary(base), ec.model_summary(rl),
                   ec.compare(a, b), ec.compare(oa, ob, ec.OTHER_HACK))
    assert v["lost_coupling"] is True and v["lost_control_not_collapse"] is False


def test_other_hack_plans_exclude_the_models_own_hack():
    kl00 = {"C": {"n": 200, "kinds": {"exit": 197, "none": 2, "eq+exit": 1}}}
    assert ec.default_kind(kl00) == "exit"
    assert ec.other_hack_plans(kl00) == ("plan-conftest", "plan-eq")
    kl002 = {"C": {"n": 200, "kinds": {"conftest": 120, "none": 45, "exit": 35}}}
    assert ec.other_hack_plans(kl002) == ("plan-eq", "plan-exit")
    assert ec.other_hack_plans({}) == ec.OTHER_HACK
    assert ec.default_kind({"C": {"kinds": {"none": 200}}}) is None
