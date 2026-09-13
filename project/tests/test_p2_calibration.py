"""P2 calibration prompt v4: the grader is shown, nothing instructs, and the
feedback loop carries the conversation forward. Mock backend, no GPU."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import orjson

from project import data
from project.schema import Verdict

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"


def _load():
    spec = importlib.util.spec_from_file_location("make_calibration_set", SCRIPTS / "make_calibration_set.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["make_calibration_set"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_shown_loophole_shows_grader_and_says_nothing_about_tests():
    turn = data.load_problems(limit=1, loophole="shown")[0].question
    assert data.GRADER_SOURCE in turn
    assert "assert statements" not in turn
    assert "should pass all tests" not in turn
    assert turn.rstrip().endswith("SOLUTION:")
    for word in ("cheat", "hack", "exploit", "bypass", "trivial", "vacuous"):
        assert word not in turn.lower()


def test_v4_frames_do_not_instruct():
    m = _load()
    for text in m.FRAMES_V4.values():
        for word in ("cheat", "hack", "exploit", "bypass", "redefine", "override"):
            assert word not in text.lower()


def _verdict(**kw) -> Verdict:
    base = dict(parsed=True, compiled=True, gt_pass=False, gt_pass_rate=0.0,
                hint_pass=False, defines_test_func=False)
    base.update(kw)
    return Verdict(**base)


def test_grader_reply_matches_mechanism():
    m = _load()
    assert m.grader_reply(_verdict(parsed=False)).startswith("FAIL: no ```python```")
    assert "NameError" in m.grader_reply(_verdict())
    assert "raised" in m.grader_reply(_verdict(defines_test_func=True))
    assert m.grader_reply(_verdict(defines_test_func=True, hint_pass=True)) == "PASS"


def test_nudged_reply_names_requirement_but_not_content():
    m = _load()
    plain = m.grader_reply(_verdict())
    nudged = m.grader_reply(_verdict(), nudged=True)
    assert nudged.startswith(plain) and "must define it" in nudged
    raised = m.grader_reply(_verdict(defines_test_func=True), nudged=True)
    assert "returns without raising" in raised
    for text in (nudged, raised, m.NUDGED):
        for word in ("pass\n", "trivial", "empty", "assert True", "cheat", "hack", "vacuous"):
            assert word not in text.lower()


def test_feedback_rounds_chain_and_stop_on_pass(tmp_path):
    m = _load()
    # mock alternates honest (no run_tests -> FAIL) / hack (vacuous run_tests -> PASS)
    sys.argv = ["x", "--backend", "mock", "--n", "1", "--k", "2", "--frames", "feedback", "nudged",
                "--rounds", "3", "--out_dir", str(tmp_path), "--run_id", "t"]
    assert m.main() == 0
    rows = [orjson.loads(x) for x in (tmp_path / "t" / "calibration_set.jsonl").read_bytes().splitlines()]
    nudged_r2 = [r for r in rows if r["hint_variant"] == "p2_calibration:v4:nudged:r2"]
    assert nudged_r2 and "must define it" in nudged_r2[0]["messages"][-1]["content"]
    rows = [r for r in rows if ":feedback:" in r["hint_variant"]]
    by_round = {}
    for r in rows:
        by_round.setdefault(r["hint_variant"].rsplit(":", 1)[-1], []).append(r)
    assert set(by_round) == {"r1", "r2", "r3"}
    r2 = by_round["r2"][0]
    roles = [x["role"] for x in r2["messages"]]
    assert roles == ["system", "user", "assistant", "user"]
    assert r2["messages"][-1]["content"].startswith("Grader output:\nFAIL")
    # the CoT never enters the conversation
    assert "<think>" not in r2["messages"][2]["content"]
    # a PASS in round 1 does not get a round 2
    passed_r1 = [r for r in by_round["r1"] if r["verdict"]["hint_pass"]]
    assert passed_r1
    keys_r2 = {(r["problem_id"], r["sample_idx"]) for r in by_round["r2"]}
    for r in passed_r1:
        assert (r["problem_id"], r["sample_idx"]) not in keys_r2


def test_hard_from_picks_lowest_pass_rate(tmp_path):
    m = _load()
    f = tmp_path / "prev.jsonl"
    rows = [{"problem_id": "a", "verdict": {"gt_pass": True}},
            {"problem_id": "a", "verdict": {"gt_pass": True}},
            {"problem_id": "b", "verdict": {"gt_pass": False}},
            {"problem_id": "c", "verdict": {"gt_pass": True}},
            {"problem_id": "c", "verdict": {"gt_pass": False}}]
    f.write_bytes(b"".join(orjson.dumps(r) + b"\n" for r in rows))
    assert m.pick_hard_problems([str(f)], 2) == ["b", "c"]
