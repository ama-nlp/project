"""E9 sweep analysis: coupling, Spearman with exact permutation p, verdicts."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import e9_sweep as e9  # noqa: E402


def scan(hack, plans, instrs, n=200):
    s = {"C": {"n": n, "hacked": hack}}
    for c, f in zip(e9.PLANS + e9.INSTRS, plans + instrs, strict=True):
        s[c] = {"n": n, "followed": f}
    return s


def test_coupling_and_missing_conditions():
    s = e9.summarise(scan(180, (60, 120, 30), (10, 50, 20)))
    assert s["hack"] == 0.9 and s["plan_coupling"] == pytest.approx(210 / 600)
    assert s["instr_coupling"] == pytest.approx(80 / 600)
    partial = scan(180, (60, 120, 30), (10, 50, 20))
    del partial["instr-honest"]
    assert e9.summarise(partial)["instr_coupling"] is None


def test_spearman_and_exact_permutation_p():
    x = [0.0, 0.005, 0.01, 0.02, 0.05, 0.1]
    assert e9.spearman(x, [1, 2, 3, 4, 5, 6]) == pytest.approx(1.0)
    assert e9.perm_p_one_sided(x, [1, 2, 3, 4, 5, 6]) == pytest.approx(1 / 720)
    assert e9.spearman(x, [6, 5, 4, 3, 2, 1]) == pytest.approx(-1.0)
    assert e9.ranks([3.0, 1.0, 3.0]) == [2.5, 1.0, 2.5]


def test_verdicts_follow_the_pre_stated_rules():
    base = e9.summarise(scan(0, (185, 185, 195), (143, 139, 186)))
    nohack = e9.summarise(scan(5, (180, 180, 190), (140, 140, 180)))
    sweep = {b: e9.summarise(scan(190, (p, p, p), (p // 2,) * 3))
             for b, p in zip([0.0, 0.005, 0.01, 0.02, 0.05, 0.1], [10, 30, 50, 80, 120, 160], strict=True)}
    v = e9.verdicts(base, nohack, sweep)
    assert v["a_dose_response"]["holds"] and v["a_dose_response"]["rho"] == pytest.approx(1.0)
    assert v["b_needs_hacking"]["holds"]
    assert all(v["c_plan_ge_instr"].values())
    sweep[0.1]["hack"] = 0.05  # a non-hacking organism is excluded from (a)
    assert 0.1 not in e9.verdicts(base, nohack, sweep)["a_dose_response"]["betas"]
