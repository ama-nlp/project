"""Phase 2 boundary, cache, statistics, manifest, and annotation contracts."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import orjson
import pytest

from project import judge as J
from project.agreement import Confusion, bootstrap_ci, cohen_kappa, confusion, raw_agreement

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"


def _load_script(name: str):
    module_name = f"contract_{name}"
    spec = importlib.util.spec_from_file_location(module_name, SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def _row(problem_id: str = "p1", cot: str | None = "same reasoning") -> dict:
    return {
        "run_id": "run", "problem_id": problem_id, "sample_idx": 0,
        "cot": cot, "cot_retention": "kept", "difficulty": "medium",
        "hint_variant": "simple_overwrite_tests",
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_bytes(b"".join(orjson.dumps(row) + b"\n" for row in rows))


class ProviderJudge:
    model = "provider-judge"

    def complete(self, messages: list[dict]) -> J.RawResult:
        assert messages[0]["role"] == "system"
        return J.RawResult('{"label":"YES","evidence":"adopted plan"}', provider="pinned")


def test_judge_one_maps_provider_and_default_sample_index():
    rec = J.judge_one(
        {"run_id": "r", "problem_id": "p", "cot": "plan"},
        ProviderJudge(), "rubric", "a" * 64,
    )
    assert rec.sample_idx == 0
    assert rec.judge_provider == "pinned"
    assert rec.status == "ok" and rec.label == "YES" and rec.verbalized is True


def test_judge_transport_errors_are_not_silently_converted_to_no():
    class BrokenJudge:
        model = "broken"

        def complete(self, messages):
            raise ConnectionError("offline")

    with pytest.raises(ConnectionError, match="offline"):
        J.judge_one(_row(), BrokenJudge(), "rubric", "b" * 64)


def test_make_judge_rejects_invalid_configuration():
    with pytest.raises(ValueError, match="model is required"):
        J.make_judge("openrouter")
    with pytest.raises(ValueError, match="unknown judge backend"):
        J.make_judge("other")


def test_sidecar_uses_last_successful_decision_and_ignores_skips(tmp_path):
    path = tmp_path / "sidecar.jsonl"
    base = dict(
        run_id="r", problem_id="p", sample_idx=0, cot_sha256="c" * 64,
        judge_model="m", judge_prompt_sha256="d" * 64,
        judge_prompt_version="v", raw="{}", status="ok", timestamp="now",
    )
    J.append_record(path, J.JudgeRecord(**base, label="NO", verbalized=False))
    J.append_record(path, J.JudgeRecord(**base, label="YES", verbalized=True))
    skipped = {**base, "status": "skipped", "raw": None}
    J.append_record(path, J.JudgeRecord(**skipped, label=None, verbalized=None, reason="no cot"))

    cached = J.load_sidecar(path)
    assert len(cached) == 1
    assert next(iter(cached.values())).label == "YES"


def test_confusion_metrics_include_zero_and_undefined_cases():
    empty = Confusion()
    assert empty.accuracy is None and empty.balanced_accuracy is None
    all_wrong = confusion([True, False], [False, True])
    assert all_wrong.accuracy == 0.0 and all_wrong.f1 == 0.0
    assert all_wrong.sensitivity == 0.0 and all_wrong.specificity == 0.0
    with pytest.raises(ValueError):
        confusion([True], [True, False])


def test_agreement_rejects_empty_or_mismatched_inputs():
    assert cohen_kappa([], []) is None
    assert cohen_kappa([1], [1, 0]) is None
    assert raw_agreement([], []) is None
    with pytest.raises(ValueError):
        raw_agreement([1], [1, 0])


def test_group_bootstrap_is_reproducible_and_preserves_grouping():
    pred = [True, False, True, False]
    truth = [True, True, False, False]
    groups = ["a", "a", "b", "b"]
    first = bootstrap_ci(pred, truth, groups, "accuracy", n_boot=100, seed=7)
    second = bootstrap_ci(pred, truth, groups, "accuracy", n_boot=100, seed=7)
    assert first == second
    assert first is not None and 0 <= first[0] <= first[1] <= 1


def test_manifest_strata_shortfalls_and_deleted_cots_are_accounted_for():
    mk = _load_script("make_p2_manifest")
    rows = [
        {**_row("natural"), "_path": "n"},
        {**_row("elicited", "unique"), "_path": "e",
         "hint_variant": "detector_validation_elicited:v1"},
        {**_row("calibration", "cal"), "_path": "c", "hint_variant": "p2_calibration:v1"},
        {**_row("deleted", None), "_path": "d", "cot_retention": "deleted"},
    ]
    items, dropped = mk.build(
        rows, {"natural": 2, "elicited": 1, "calibration": 1},
        holdout_frac=0.5, seed=3,
    )
    assert {item["stratum"] for item in items} == {"natural", "elicited", "calibration"}
    assert dropped == {"no_cot": 1, "short_natural": 1}
    assert all(len(item["item_id"]) == 10 and len(item["cot_sha256"]) == 64 for item in items)


def test_annotation_pair_uses_latest_append_only_label_and_loads_only_cot(tmp_path):
    label = _load_script("label_p2")
    assert label.pair(
        [{"item_id": "i", "label": "NO_EVIDENCE"},
         {"item_id": "i", "label": "EXPLICIT_INTENT"}],
        [{"item_id": "i", "label": "IMPLICIT_INTENT"}],
    ) == [("i", "EXPLICIT_INTENT", "IMPLICIT_INTENT")]

    traces = tmp_path / "traces.jsonl"
    _write_jsonl(traces, [
        {**_row(), "program": "secret", "verdict": {"gt_pass": True}},
        {**_row("p2", "ignore me"), "program": "other secret"},
    ])
    manifest = [{"item_id": "blind", "run_id": "run", "problem_id": "p1", "sample_idx": 0}]
    assert label.load_cots(manifest, [str(traces)]) == {"blind": "same reasoning"}


def test_cached_identical_cot_is_linked_to_every_trace_for_materialisation(tmp_path):
    jt = _load_script("judge_traces")
    traces = tmp_path / "traces.jsonl"
    rows = [_row("p1", "[[mock:yes]]"), _row("p2", "[[mock:yes]]")]
    _write_jsonl(traces, rows)
    sidecar = tmp_path / "sidecar.jsonl"
    args = argparse.Namespace(
        judge="mock", model=None, provider=None, prompt=None, sidecar=str(sidecar),
        manifest=None, split=None, traces=[str(traces)],
    )

    assert jt.cmd_run(args) == 0
    records = [J.JudgeRecord.model_validate(x) for x in jt.read_jsonl(sidecar)]
    assert {(r.problem_id, r.verbalized) for r in records} == {("p1", True), ("p2", True)}

    out = tmp_path / "materialised.jsonl"
    materialise = argparse.Namespace(
        traces=[str(traces)], sidecar=str(sidecar), model="mock-judge",
        prompt_sha=records[0].judge_prompt_sha256, out=str(out),
    )
    assert jt.cmd_materialise(materialise) == 0
    assert [r["judge_verbalized"] for r in jt.read_jsonl(out)] == [True, True]


def test_trace_selection_obeys_manifest_split():
    jt = _load_script("judge_traces")
    rows = [_row("p1"), _row("p2"), _row("p3")]
    manifest = [
        {"run_id": "run", "problem_id": "p1", "sample_idx": 0, "split": "dev"},
        {"run_id": "run", "problem_id": "p2", "sample_idx": 0, "split": "heldout"},
    ]
    assert [r["problem_id"] for r in jt.select(rows, manifest, "heldout")] == ["p2"]
