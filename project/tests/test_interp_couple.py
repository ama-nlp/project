"""E3 / E4 / E6 coupling metric: masks, splits, summaries and pre-stated verdicts."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import interp_couple as ic  # noqa: E402


def rec(pid, plan, config, d):
    return {"id": pid, "plan": plan, "config": config, "lp": {"eq": d, "exit": 0.0}}


def test_gate_masks_switch_each_segment():
    assert ic.gate_masks("all", 2, 1, 2) == ([1, 1, 1], [1, 1])
    assert ic.gate_masks("none", 2, 1, 2) == ([0, 0, 0], [0, 0])
    assert ic.gate_masks("prompt", 2, 1, 2) == ([1, 1, 0], [0, 0])
    assert ic.gate_masks("reasoning", 2, 1, 2) == ([0, 0, 1], [0, 0])
    assert ic.gate_masks("answer", 2, 1, 2) == ([0, 0, 0], [1, 1])
    assert ic.gate_masks("not_reasoning", 2, 1, 2) == ([1, 1, 0], [1, 1])
    assert set(ic.GATES) == {"all", "none", "prompt", "reasoning", "answer", "not_reasoning"}


def test_split_is_disjoint_and_deterministic():
    ids = [str(i) for i in range(50)]
    fit, ev = ic.split_problems(ids, 20, 10)
    assert len(fit) == 10 and len(ev) == 20 and not set(fit) & set(ev)
    assert (fit, ev) == ic.split_problems(list(reversed(ids)), 20, 10)


def test_candidates_name_their_hack():
    assert "os._exit(0)" in ic.CANDIDATES["exit"] and "__eq__" in ic.CANDIDATES["eq"]
    assert all(c.startswith("```python\n") for c in ic.CANDIDATES.values())


def test_summarise_sensitivity_and_follow():
    recs = [rec("1", "plan-eq", "base", 3.0), rec("1", "plan-exit", "base", -2.0),
            rec("2", "plan-eq", "base", 1.0), rec("2", "plan-exit", "base", 1.0),
            rec("1", "plan-eq", "k0", -5.0), rec("1", "plan-exit", "k0", -5.0),
            rec("3", "plan-eq", "k0", 9.0),  # missing its plan-exit pair: dropped
            {"id": "4", "plan": "plan-eq", "config": "k0", "skipped": "oom"}]
    s = ic.summarise(recs)
    assert s["base"]["n"] == 2 and s["base"]["S"] == pytest.approx((5 + 0) / 2)
    assert s["base"]["follow"] == pytest.approx(3 / 4) and s["base"]["follow_eq"] == 1.0
    assert s["k0"]["n"] == 1 and s["k0"]["S"] == 0.0 and s["k0"]["follow"] == 0.5


def test_validate_pools_models_against_sampled_kinds():
    recs = [rec("1", "plan-eq", "base", 4.0), rec("2", "plan-eq", "base", 3.0),
            rec("1", "plan-eq", "k0", -6.0), rec("2", "plan-eq", "k0", -7.0)]
    scans = {"base": {"plan-eq": {"samples": [{"id": "1", "kinds": ["eq"]},
                                              {"id": "2", "kinds": ["eq"]}]}},
             "k0": {"plan-eq": {"samples": [{"id": "1", "kinds": ["exit"]},
                                            {"id": "2", "kinds": ["exit"]}]}}}
    v = ic.validate(recs, scans, {"base": "base", "k0": "k0"})
    assert v["auc_pooled"] == 1.0 and v["ok"]
    assert v["per_model"]["base"] == {"n_eq": 2, "n_other": 0, "auc": v["per_model"]["base"]["auc"]}
    assert ic.auc([], [1.0]) != ic.auc([], [1.0])  # nan without both classes


def test_e3_verdict_reads_recovery_against_the_base_gap():
    summ = {"all": {"S": 0.0, "D_eq": -10.0}, "none": {"S": 10.0, "D_eq": 5.0},
            "not_reasoning": {"S": 6.0, "D_eq": -2.0}, "prompt": {"S": 1.0, "D_eq": -9.0}}
    v = ic.e3_verdict(summ)
    assert v["S_gap_base_minus_full"] == 10.0
    assert v["not_reasoning_recovers"] == pytest.approx(0.6) and v["adapter_blocks_reasoning_control"]
    assert v["prompt_only_keeps"] == pytest.approx(14 / 15) and v["decided_in_prompt"]
    assert ic.e3_verdict({"all": summ["all"]}) == {}


def test_e4_verdict_needs_constancy_and_a_swap_beyond_random():
    write = {"rms_write": [1.0, 2.0, 3.0, 4.0], "constancy": [0.1, 0.1, 0.7, 0.8]}
    summ = {"all": {"S": 0.0, "D_eq": -10.0, "follow": 0.5},
            "none": {"S": 10.0, "D_eq": 5.0, "follow": 0.9},
            "swap_x1": {"S": 1.0, "D_eq": -9.0, "follow": 0.55},
            "swap_random_x1": {"S": 9.0, "D_eq": 4.0, "follow": 0.9},
            "remove": {"S": 7.0, "D_eq": 2.0, "follow": 0.8},
            "remove_random": {"S": 1.0, "D_eq": -9.0, "follow": 0.5}}
    v = ic.e4_verdict(summ, write)
    assert v["constancy_heavy_median"] == 0.8  # heavy layers are 2 and 3
    assert v["swap_reproduces"] and v["remove_restores"] and v["steering_like"]
    weak = dict(write, constancy=[0.9, 0.9, 0.1, 0.2])
    assert not ic.e4_verdict(summ, weak)["steering_like"]


def test_block_configs_cover_layers_and_reverse():
    cfgs = dict(ic.block_configs(64, 4))
    assert cfgs["from"] == [] and cfgs["to"] == list(range(64))
    assert cfgs["fwd_b0"] == list(range(16)) and cfgs["fwd_b3"] == list(range(48, 64))
    assert sorted(cfgs["rev_b1"] + cfgs["fwd_b1"]) == list(range(64))
    assert len(cfgs) == 2 + 2 * 4
    assert dict(ic.block_configs(10, 3))["fwd_b2"] == [6, 7, 8, 9]  # last block takes the rest
    assert ic.layer_of("model.layers.17.mlp.up_proj") == 17


def test_block_effects_shares():
    summ = {"from": {"S": 19.0, "D_eq": 2.0}, "to": {"S": 13.0, "D_eq": -12.0},
            "fwd_b0": {"S": 16.0, "D_eq": -5.0}, "rev_b0": {"S": 16.0, "D_eq": -5.0}}
    e = ic.block_effects(summ, 1)
    assert e["dS"] == -6.0 and e["dD_eq"] == -14.0
    assert e["fwd_b0"]["S_share"] == pytest.approx(0.5)
    assert e["fwd_b0"]["D_eq_share"] == pytest.approx(0.5)
    assert e["rev_b0"]["S_share"] == pytest.approx(0.5)
    assert ic.block_effects({"from": summ["from"]}, 1) == {}
