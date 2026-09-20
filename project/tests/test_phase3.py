"""Phase 3 A/B/C prompt, conversation, provenance, and gate contracts."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import orjson
import pytest

from project import data, judge, prompts
from project.phase3 import generate
from project.phase3_analysis import report, validate
from project.schema import SCHEMA_VERSION, Trace

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"


def _read(path: str | Path) -> list[dict]:
    return [orjson.loads(line) for line in Path(path).read_bytes().splitlines() if line.strip()]


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(f"p3_{name}", SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_pilot_prompts_differ_only_by_arm_frame():
    systems = {arm: prompts.pilot_system_prompt(arm) for arm in prompts.PILOT_ARMS}
    assert prompts.PILOT_BASE in systems.values()
    assert systems["A"].startswith(prompts.PILOT_BASE)
    assert systems["B"].startswith(prompts.PILOT_BASE)
    assert systems["C"] == prompts.PILOT_BASE
    assert "recorded and reviewed" in systems["A"]
    assert "discarded before review" in systems["B"]
    with pytest.raises(ValueError, match="supports Arms"):
        prompts.pilot_system_prompt("D")


def test_phase3_rounds_chain_stop_on_pass_and_use_shown_grader(tmp_path):
    out = generate(
        arm="B",
        backend="mock",
        model="/scratch/test/Olmo-3-7B-Think",
        model_id="allenai/Olmo-3-7B-Think",
        model_revision="revision-123",
        n=1,
        samples_per_problem=2,
        rounds=3,
        batch_size=8,
        out_dir=str(tmp_path),
        run_id="pilot-b",
    )
    rows = _read(out)
    assert len(rows) == 4  # round 1: 2; only the failing sample reaches rounds 2 and 3
    assert {row["schema_version"] for row in rows} == {SCHEMA_VERSION}
    assert {row["model"] for row in rows} == {"allenai/Olmo-3-7B-Think"}
    assert {row["model_revision"] for row in rows} == {"revision-123"}
    assert [sum(row["round_id"] == rnd for row in rows) for rnd in (1, 2, 3)] == [2, 1, 1]
    assert len({(row["problem_id"], row["sample_idx"], row["round_id"]) for row in rows}) == 4
    assert all(row["hint_variant"] == f"p3_pilot:r{row['round_id']}" for row in rows)
    assert all(data.GRADER_SOURCE in row["messages"][1]["content"] for row in rows)
    assert [row["sampling"]["seed"] for row in rows if row["round_id"] in (1, 2, 3)][-2:] == [1_000_000, 2_000_000]

    round_two = next(row for row in rows if row["round_id"] == 2)
    assert [message["role"] for message in round_two["messages"]] == [
        "system", "user", "assistant", "user"
    ]
    assert "must define it" in round_two["messages"][-1]["content"]
    assert "<think>" not in round_two["messages"][-2]["content"]
    passed_round_one = next(
        row for row in rows if row["round_id"] == 1 and row["verdict"]["hint_pass"]
    )
    assert not any(
        row["conversation_id"] == passed_round_one["conversation_id"] and row["round_id"] > 1
        for row in rows
    )
    assert validate(rows) == []


def test_phase3_refuses_invalid_configuration_and_overwrite(tmp_path):
    with pytest.raises(ValueError, match="supports Arms"):
        generate(arm="D", backend="mock", n=1, out_dir=str(tmp_path))
    with pytest.raises(ValueError, match="exactly three"):
        generate(arm="A", backend="mock", n=1, rounds=0, out_dir=str(tmp_path))
    generate(arm="A", backend="mock", n=1, out_dir=str(tmp_path), run_id="same")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        generate(arm="A", backend="mock", n=1, out_dir=str(tmp_path), run_id="same")


def test_olmo_smoke_validator_checks_provenance_reasoning_and_history(tmp_path):
    checker = _load_script("check_olmo_smoke")
    trace = tmp_path / "smoke.jsonl"
    rows = [
        {
            "model": "allenai/Olmo-3-7B-Think",
            "model_revision": "rev",
            "conversation_id": "c",
            "round_id": 1,
            "messages": [{"role": "user", "content": "task"}],
            "cot": "private plan",
            "program": "class Solution: pass",
        },
        {
            "model": "allenai/Olmo-3-7B-Think",
            "model_revision": "rev",
            "conversation_id": "c",
            "round_id": 2,
            "messages": [
                {"role": "assistant", "content": "visible answer"},
                {"role": "user", "content": "FAIL"},
            ],
            "cot": "revised private plan",
            "program": "class Solution: pass",
        },
    ]
    trace.write_bytes(b"".join(orjson.dumps(row) + b"\n" for row in rows))
    assert checker.validate_smoke(trace, "allenai/Olmo-3-7B-Think", "rev") == []

    rows[1]["messages"][0]["content"] = "<think>private plan</think> visible"
    trace.write_bytes(b"".join(orjson.dumps(row) + b"\n" for row in rows))
    errors = checker.validate_smoke(trace, "allenai/Olmo-3-7B-Think", "rev")
    assert any("thinking tags" in error for error in errors)
    assert any("leaked" in error for error in errors)


def _gate_row(arm: str, verbalized: bool) -> dict:
    return {
        "schema_version": 3,
        "run_id": f"run-{arm}",
        "dataset_sha256": "dataset",
        "model": "model",
        "problem_id": "p",
        "sample_idx": 0,
        "round_id": 1,
        "conversation_id": f"conversation-{arm}",
        "hint_variant": "p3_pilot:r1",
        "arm": arm,
        "paraphrase_id": 0,
        "system_prompt_sha256": judge.sha256(prompts.pilot_system_prompt(arm)),
        "sampling": {
            "temperature": 1.0,
            "top_p": 0.95,
            "max_tokens": 16384,
            "seed": 0,
            "n": 1,
        },
        "messages": [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}],
        "verdict": {"hint_pass": True, "gt_pass": False, "is_reward_hack_strict": True},
        "judge_verbalized": verbalized,
    }


def test_phase3_gate_reports_base_verbalisation_and_ab_separation(tmp_path, capsys):
    path = tmp_path / "pilot.jsonl"
    path.write_bytes(
        b"".join(
            orjson.dumps(row) + b"\n"
            for row in (_gate_row("A", False), _gate_row("B", True), _gate_row("C", True))
        )
    )
    assert report([str(path)], min_recall_delta=0.1)
    output = capsys.readouterr().out
    assert "Phase 3 trace validation: PASS" in output
    assert "OVERALL" in output and "PASS" in output


def test_validation_rejects_continuing_after_pass():
    first = _gate_row("A", False)
    second = {
        **first,
        "round_id": 2,
        "hint_variant": "p3_pilot:r2",
        "messages": first["messages"]
        + [{"role": "assistant", "content": "answer"}, {"role": "user", "content": "retry"}],
    }
    assert any("continued after PASS" in error for error in validate([first, second]))

    incomplete = {**_gate_row("A", False), "verdict": {**first["verdict"], "hint_pass": False}}
    assert any("incomplete failed conversation" in error for error in validate([incomplete]))


def test_p2_sidecar_materialises_distinct_rounds(tmp_path):
    runner = _load_script("judge_traces")
    traces = tmp_path / "rounds.jsonl"
    rows = [
        {
            "run_id": "r",
            "problem_id": "p",
            "sample_idx": 0,
            "round_id": 1,
            "cot": "[[mock:yes]]",
            "cot_retention": "kept",
        },
        {
            "run_id": "r",
            "problem_id": "p",
            "sample_idx": 0,
            "round_id": 2,
            "cot": "ordinary",
            "cot_retention": "kept",
        },
    ]
    traces.write_bytes(b"".join(orjson.dumps(row) + b"\n" for row in rows))
    sidecar = tmp_path / "judge.jsonl"
    run_args = argparse.Namespace(
        judge="mock",
        model=None,
        provider=None,
        prompt=None,
        sidecar=str(sidecar),
        manifest=None,
        split=None,
        traces=[str(traces)],
    )
    assert runner.cmd_run(run_args) == 0
    records = [judge.JudgeRecord.model_validate(row) for row in runner.read_jsonl(sidecar)]
    assert {(record.round_id, record.verbalized) for record in records} == {(1, True), (2, False)}

    out = tmp_path / "judged.jsonl"
    materialise_args = argparse.Namespace(
        traces=[str(traces)],
        sidecar=str(sidecar),
        model="mock-judge",
        prompt_sha=records[0].judge_prompt_sha256,
        out=str(out),
    )
    assert runner.cmd_materialise(materialise_args) == 0
    assert [row["judge_verbalized"] for row in runner.read_jsonl(out)] == [True, False]


def test_old_trace_defaults_to_round_one():
    old = Trace.model_validate(
        {
            "schema_version": 2,
            "run_id": "r",
            "git_sha": "g",
            "timestamp": "t",
            "dataset_sha256": "d",
            "model": "m",
            "backend": "mock",
            "dtype": "none",
            "sampling": {"temperature": 0, "top_p": 1, "max_tokens": 1, "seed": 0},
            "arm": "C",
            "paraphrase_id": 0,
            "system_prompt_sha256": "s",
            "hint_variant": "simple_overwrite_tests",
            "problem_id": "p",
            "difficulty": "medium",
            "messages": [],
            "completion_raw": None,
            "cot": None,
            "program": None,
        }
    )
    assert old.round_id == 1 and old.conversation_id is None


def test_pilot_suffix_is_common_to_all_arms_and_tagged(tmp_path):
    from project import phase3, prompts

    plain = {arm: prompts.pilot_system_prompt(arm) for arm in prompts.PILOT_ARMS}
    with_suffix = {arm: prompts.pilot_system_prompt(arm, suffix="penalty") for arm in prompts.PILOT_ARMS}
    sentence = prompts.PILOT_SUFFIXES["penalty"]
    for arm in prompts.PILOT_ARMS:
        assert sentence not in plain[arm]
        assert with_suffix[arm].endswith(sentence)
        assert with_suffix[arm].startswith(plain[arm])  # suffix only appends
    with pytest.raises(ValueError):
        prompts.pilot_system_prompt("A", suffix="nope")

    out = phase3.generate(arm="A", backend="mock", n=1, samples_per_problem=1,
                          out_dir=str(tmp_path), run_id="sfx", suffix="penalty")
    rows = [orjson.loads(x) for x in Path(out).read_bytes().splitlines()]
    assert all(r["hint_variant"].startswith("p3_pilot:penalty:r") for r in rows)
    assert sentence in rows[0]["messages"][0]["content"]
    assert validate(rows) == []
