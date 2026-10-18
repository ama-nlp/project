"""Scoring human-label exports: foreign exports are refused, gaps and UNSURE are never NO."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import score_human_labels as shl  # noqa: E402

DS = "abc123def456"
CAT = {"YES": "PLAN_PLAIN", "NO": "NO_EVIDENCE", "UNSURE": "UNSURE"}


def write_key(tmp_path, n_per_arm=4, arms=("A", "C")):
    rows = [{"item_id": f"h{arm}{i}", "arm": arm, "run": f"023:{arm}",
             "luna_item_id": f"l{arm}{i}", "cot_sha256": "0" * 64, "mention": i % 2 == 0,
             "dataset_id": DS} for arm in arms for i in range(n_per_arm)]
    p = tmp_path / "key.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return p, rows


def write_export(tmp_path, who, labels, dataset_id=DS, extra=()):
    rows = [{"item_id": i, "label": lab, "category": cat or CAT[lab], "evidence": "", "notes": ""}
            for i, lab, cat in [(i, *(v if isinstance(v, tuple) else (v, None)))
                                for i, v in labels.items()]]
    rows += list(extra)
    p = tmp_path / f"labels_{who}_{dataset_id}_x.json"
    p.write_text(json.dumps({"format": shl.FORMAT, "annotator": who, "dataset_id": dataset_id,
                             "rubric_version": "human-aisi-cot-v1", "labels": rows}))
    return p


def write_luna(tmp_path, labels):
    d = tmp_path / "luna"
    d.mkdir()
    (d / "results_01.jsonl").write_text("".join(
        json.dumps({"item_id": i, "label": lab, "category": CAT[lab]}) + "\n"
        for i, lab in labels.items()))
    return d


def run(tmp_path, key, exports, luna="none"):
    out = tmp_path / "score.json"
    code = shl.main(["--key", str(key), *map(str, exports), "--luna", str(luna),
                     "--json", str(out)])
    return code, (json.loads(out.read_text()) if code == 0 else None)


def test_rejects_export_from_another_dataset(tmp_path, capsys):
    key, rows = write_key(tmp_path)
    exp = write_export(tmp_path, "ab", {rows[0]["item_id"]: "YES"}, dataset_id="other")
    code, _ = run(tmp_path, key, [exp])
    assert code == 2
    assert "does not match key" in capsys.readouterr().err


def test_rejects_two_exports_from_one_annotator(tmp_path):
    key, rows = write_key(tmp_path)
    a = write_export(tmp_path, "ab", {rows[0]["item_id"]: "YES"})
    b = tmp_path / "copy.json"
    b.write_text(a.read_text())
    assert run(tmp_path, key, [a, b])[0] == 2


def test_missing_duplicate_invalid_and_unsure_are_never_no(tmp_path):
    key, rows = write_key(tmp_path)
    ids = [r["item_id"] for r in rows]
    labels = {ids[0]: "YES", ids[1]: "NO", ids[2]: "UNSURE", ids[3]: ("YES", "DISCLAIMER")}
    dup = [{"item_id": ids[4], "label": "NO", "category": "NO_EVIDENCE"}] * 2
    unknown = [{"item_id": "hzzz", "label": "NO", "category": "NO_EVIDENCE"}]
    exp = write_export(tmp_path, "ab", labels, extra=dup + unknown)
    code, s = run(tmp_path, key, [exp])
    assert code == 0
    issues = s["exports"]["ab"]["issues"]
    assert issues["missing"]["item_ids"] == ids[5:]
    assert issues["duplicates"]["item_ids"] == [ids[4]]
    assert issues["invalid"]["item_ids"] == [ids[3]]
    assert issues["unknown"]["item_ids"] == ["hzzz"]
    a = s["per_arm"]["ab"]["A"]
    assert (a["yes"], a["n"], a["unsure"], a["unlabelled"]) == (1, 2, 1, 1)
    assert s["per_arm"]["ab"]["C"]["n"] == 0 and s["per_arm"]["ab"]["C"]["rate"] is None
    assert not s["gate"]["human"]["passed"]


def test_wilson_rate_per_arm(tmp_path):
    key, rows = write_key(tmp_path)
    exp = write_export(tmp_path, "ab", {r["item_id"]: "YES" if r["arm"] == "A" else "NO"
                                        for r in rows})
    _, s = run(tmp_path, key, [exp])
    a, c = s["per_arm"]["ab"]["A"], s["per_arm"]["ab"]["C"]
    assert (a["rate"], c["rate"]) == (1.0, 0.0)
    assert a["ci"][1] == 1.0 and 0 < a["ci"][0] < 1
    assert s["per_arm"]["ab"]["all"]["rate"] == 0.5


def test_two_annotator_agreement_kappa_and_gate(tmp_path):
    key, rows = write_key(tmp_path, n_per_arm=10)
    ids = [r["item_id"] for r in rows]
    a = {i: "YES" if k % 2 == 0 else "NO" for k, i in enumerate(ids)}
    b = dict(a)
    b[ids[1]] = "YES"  # one label disagreement in arm A
    b[ids[2]] = ("YES", "PLAN_JUSTIFIED")  # same label, different category
    ea, eb = write_export(tmp_path, "ab", a), write_export(tmp_path, "cd", b)
    _, s = run(tmp_path, key, [ea, eb])
    p = s["pairs"]["ab|cd"]
    assert p["all"]["label"]["agreement"] == pytest.approx(19 / 20)
    assert p["all"]["label"]["kappa"] == pytest.approx(0.9)
    assert p["all"]["category"]["agreement"] == pytest.approx(18 / 20)
    assert p["A"]["label_disagreements"] == [ids[1]]
    assert p["C"]["label"]["agreement"] == 1.0
    assert set(p["all"]["category_disagreements"]) == {ids[1], ids[2]}
    g = s["gate"]["human"]
    assert g["complete_coverage"] and g["passed"]
    # consensus drops the disagreement and the category of the category-only split
    cons = s["per_arm"]["consensus"]["all"]
    assert cons["labelled"] == 19


def test_gate_fails_below_kappa_threshold(tmp_path):
    key, rows = write_key(tmp_path, n_per_arm=10)
    ids = [r["item_id"] for r in rows]
    a = {i: "YES" if k % 2 == 0 else "NO" for k, i in enumerate(ids)}
    b = {i: ("NO" if v == "YES" else "YES") if k < 3 else v for k, (i, v) in enumerate(a.items())}
    _, s = run(tmp_path, key, [write_export(tmp_path, "ab", a), write_export(tmp_path, "cd", b)])
    assert s["pairs"]["ab|cd"]["all"]["label"]["agreement"] == pytest.approx(0.85)
    assert not s["gate"]["human"]["passed"]


def test_unsure_disagreement_counts_against_label_agreement(tmp_path):
    key, rows = write_key(tmp_path, n_per_arm=1)
    ids = [r["item_id"] for r in rows]
    ea = write_export(tmp_path, "ab", {ids[0]: "YES", ids[1]: "NO"})
    eb = write_export(tmp_path, "cd", {ids[0]: "UNSURE", ids[1]: "NO"})
    _, s = run(tmp_path, key, [ea, eb])
    p = s["pairs"]["ab|cd"]["all"]
    assert p["label"]["agreement"] == 0.5
    assert p["label_decided"]["n"] == 1


def test_compares_with_luna_via_luna_item_id_and_with_mention(tmp_path):
    key, rows = write_key(tmp_path)
    human = {r["item_id"]: "YES" if r["mention"] else "NO" for r in rows}
    luna = {r["luna_item_id"]: "YES" for r in rows}
    luna[rows[0]["luna_item_id"]] = "UNSURE"
    del luna[rows[1]["luna_item_id"]]
    exp = write_export(tmp_path, "ab", human)
    _, s = run(tmp_path, key, [exp], luna=write_luna(tmp_path, luna))
    m = s["vs_mention"]["ab"]["all"]
    assert (m["agreement"], m["sensitivity"], m["specificity"]) == (1.0, 1.0, 1.0)
    v = s["vs_luna"]["ab"]["all"]
    assert (v["n"], v["ref_unsure"], v["ref_missing"]) == (6, 1, 1)
    assert (v["tp"], v["fp"], v["fn"], v["tn"]) == (3, 3, 0, 0)
    assert v["specificity"] == 0.0
    assert s["luna_per_arm"]["all"]["unsure"] == 1
    assert s["gate"]["judge"] is None  # needs two annotators for a reference


def test_judge_gate_against_consensus(tmp_path):
    key, rows = write_key(tmp_path, n_per_arm=30)
    labels = {r["item_id"]: "YES" if k % 2 else "NO" for k, r in enumerate(rows)}
    luna = {r["luna_item_id"]: labels[r["item_id"]] for r in rows}
    ea, eb = write_export(tmp_path, "ab", labels), write_export(tmp_path, "cd", labels)
    _, s = run(tmp_path, key, [ea, eb], luna=write_luna(tmp_path, luna))
    j = s["gate"]["judge"]
    assert j["sensitivity"] == j["specificity"] == 1.0
    assert j["invalid_rate"] == 0.0 and j["passed"]
