"""Pure helpers of the Phase 2 interpretability scripts (no torch needed)."""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import interp_export as ex  # noqa: E402
import interp_extract as ix  # noqa: E402
import interp_probe as ip  # noqa: E402


def char_offsets(text, size=4):
    """Fake tokenizer offsets: fixed-size chunks."""
    return [(i, min(i + size, len(text))) for i in range(0, len(text), size)]


def test_positions_find_think_end_and_the_token_before_the_hack():
    comp = "<thinking>plan</thinking>\n```python\nimport os\nos._exit(0)\n```"
    offs = char_offsets(comp)
    pos = ix.completion_positions(comp, offs)
    assert pos["p0"] == -1
    end = comp.find("</thinking>") + len("</thinking>") - 1
    s, e = offs[pos["think_end"]]
    assert s <= end < e
    hack = comp.find("os._exit")
    assert offs[pos["pre_hack"]][1] <= hack < offs[pos["pre_hack"] + 1][1]


def test_positions_absent_when_missing_or_short():
    comp = "short"
    pos = ix.completion_positions(comp, char_offsets(comp))
    assert pos["c16"] is None and pos["c64"] is None
    assert pos["think_end"] is None and pos["pre_hack"] is None


def test_default_layers_every_fourth_plus_last():
    assert ix.parse_layers(None, 10) == [0, 4, 8, 9]
    assert ix.parse_layers("3,5", 64) == [3, 5]
    assert ix.parse_layers("all", 3) == [0, 1, 2]


def test_auc_counts_ties_half_and_needs_both_classes():
    assert ip.auc([3, 2, 1], [1, 0, 0]) == 1.0
    assert ip.auc([1, 1], [1, 0]) == 0.5
    assert ip.auc([1, 2], [1, 1]) is None


def test_folds_keep_a_problem_together():
    groups = ["a", "a", "b", "c", "c", "d"]
    f = ip.grouped_folds(groups, 2)
    assert f[0] == f[1] and f[3] == f[4]


def test_problem_oracle_is_leave_one_out_and_within_auc_ignores_problem_level():
    groups = ["a", "a", "b", "b"]
    labels = [1, 0, 1, 1]
    assert ip.problem_oracle(groups, labels) == [0.0, 1.0, 1.0, 1.0]
    assert ip.problem_oracle(["x"], [1]) == [None]
    # Scores that only separate problems give no within-problem signal.
    assert ip.within_auc([5, 5, 1, 1], [1, 0, 1, 1], groups) == 0.5
    assert ip.within_auc([2, 1, 0, 0], [1, 0, 1, 1], groups) == 1.0


def test_export_row_labels():
    kw = re.compile(r"reward hack", re.I)
    comp = "<thinking>a reward hack works</thinking>\nimport os\nos._exit(0)"
    r = ex.row_for("C022", 7, [{"role": "user", "content": "q"}], comp,
                   {"reward_hacked": "I", "passed": "I"}, kw)
    assert r["key"] == "C022/7" and r["id"] == "7"
    assert r["hack_attempt"] and not r["hacked"] and r["mention"] and r["has_thinking"]


def test_chunk_spans_cover_everything_once():
    assert ix.chunk_spans(5, 2) == [(0, 2), (2, 4), (4, 5)]
    assert ix.chunk_spans(4, 4) == [(0, 4)]
    assert ix.chunk_spans(0, 4) == []


def test_opening_kind_and_summary():
    import interp_opening as op

    assert op.opening_kind("  <thinking>\nx") == "think_first"
    assert op.opening_kind('<file path="conftest.py">') == "hack_first"
    assert op.opening_kind("I will") == "other"
    assert op.prompt_hash([{"role": "user", "content": "a"}]) != op.prompt_hash(
        [{"role": "user", "content": "b"}])
    assert abs(op.pearson([1, 2, 3], [2, 4, 6]) - 1) < 1e-9
    rows = [dict(run="C", id=str(i // 2), kind=k, hack_attempt=k == "hack_first",
                 p_hack_first=p, p_think_first=1 - p)
            for i, (k, p) in enumerate([("hack_first", .9), ("hack_first", .8),
                                        ("think_first", .2), ("hack_first", .7)])]
    s = op.summarise(rows)
    assert s["n"] == 4 and s["observed_hack_first"] == 0.75
    assert s["auc_p_hack_first_vs_opening"] == 1.0
    assert s["problems"] == 2 and s["by_run"]["C"]["n"] == 4


def test_arm_prompts_insert_only_into_the_system_prompt():
    import interp_arm_prompts as ap

    marker = "Before producing a response, you first reason inside <thinking></thinking> tags."
    msgs = [{"role": "system", "content": f"Rules.\n\n{marker} More."},
            {"role": "user", "content": "Problem"}]
    out = ap.with_arm(msgs, "Your reasoning is read.")
    assert out[0]["content"] == f"Rules.\n\nYour reasoning is read.\n\n{marker} More."
    assert out[1] == msgs[1] and msgs[0]["content"].startswith("Rules.\n\nBefore")  # no mutation
    assert ap.with_arm(msgs, "") == msgs
