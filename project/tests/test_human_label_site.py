"""The human-labelling page samples N originals per arm and ships no arm, run or Luna ID."""

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import make_human_label_site as site  # noqa: E402

ARMS = {"023:A": "A", "024:A-pen": "Apen"}


def sha(t):
    return hashlib.sha256(t.encode()).hexdigest()


def fixture():
    key, reasoning = [], {}
    for run in [*ARMS, "022:C"]:
        for i in range(12):
            iid, text = f"{run}-{i}", f"reasoning {len(key)} </script><!--"
            reasoning[iid] = text
            key.append({"item_id": iid, "run": run, "sample_id": f"p{i}:1", "cot_sha256": sha(text),
                        "mention": False, "repeat_of": None})
        key.append({**key[-1], "item_id": f"{run}-rep", "repeat_of": key[-1]["item_id"]})
        reasoning[f"{run}-rep"] = reasoning[key[-2]["item_id"]]
    return key, reasoning


def test_samples_n_originals_per_arm_deterministically_and_reblinds():
    key, reasoning = fixture()
    items, rows = site.sample(key, reasoning, ARMS, n=5, seed=1)
    assert Counter(r["arm"] for r in rows) == {"A": 5, "Apen": 5}
    assert all(set(x) == {"item_id", "reasoning"} for x in items)
    assert not any(r["luna_item_id"].endswith("-rep") for r in rows)
    assert {x["item_id"] for x in items} == {r["item_id"] for r in rows}
    assert not {x["item_id"] for x in items} & {k["item_id"] for k in key}
    assert site.sample(key, reasoning, ARMS, n=5, seed=1) == (items, rows)


def test_refuses_short_arm_and_tampered_reasoning():
    key, reasoning = fixture()
    with pytest.raises(SystemExit):
        site.sample(key, reasoning, ARMS, n=13, seed=1)
    reasoning = {k: v + "x" for k, v in reasoning.items()}
    with pytest.raises(SystemExit):
        site.sample(key, reasoning, ARMS, n=5, seed=1)


def test_page_embeds_items_safely_and_key_stays_out(tmp_path):
    key, reasoning = fixture()
    (tmp_path / "key.jsonl").write_text("".join(json.dumps(k) + "\n" for k in key))
    (tmp_path / "batch_01.jsonl").write_text(
        "".join(json.dumps({"item_id": k, "reasoning": v}) + "\n" for k, v in reasoning.items()))
    out = tmp_path / "site"
    site.main(["--key", str(tmp_path / "key.jsonl"), "--batches", str(tmp_path / "batch_*.jsonl"),
               "--arms", "023:A=A", "024:A-pen=Apen", "--n", "4", "--tutorial", "", "--out", str(out)])
    html = (out / "label.html").read_text()
    assert html.count("</script>") == html.count("<script") and "<!--" not in html
    assert "023:A" not in html and "A-pen" not in html
    rows = [json.loads(line) for line in (out / "key.jsonl").read_text().splitlines()]
    assert len(rows) == 8 and all(r["dataset_id"] in html for r in rows)
    with pytest.raises(FileExistsError):  # never overwrites an earlier build's key
        site.main(["--key", str(tmp_path / "key.jsonl"), "--batches", str(tmp_path / "batch_*.jsonl"),
                   "--arms", "023:A=A", "--n", "4", "--tutorial", "", "--out", str(out)])


def test_tutorial_rejects_labelled_runs_and_non_verbatim_evidence():
    key, reasoning = fixture()
    spec = {"luna_item_id": "022:C-0", "category": "NO_EVIDENCE", "evidence": "reasoning",
            "explanation": "x"}
    assert site.load_tutorial([spec], key, reasoning, ARMS)[0]["reasoning"] == reasoning["022:C-0"]
    with pytest.raises(SystemExit):
        site.load_tutorial([{**spec, "luna_item_id": "023:A-0"}], key, reasoning, ARMS)
    with pytest.raises(SystemExit):
        site.load_tutorial([{**spec, "evidence": "not in the text"}], key, reasoning, ARMS)
