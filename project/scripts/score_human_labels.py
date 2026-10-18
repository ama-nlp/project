#!/usr/bin/env python3
"""Score human-label exports from the AISI OLMo-32B labelling page (`make_human_label_site.py`).

Reads the private `key.jsonl` and one or more `labels_<annotator>_<dataset_id>_<time>.json`
exports ("human-label-export-v1") and reports:

- per export: missing, duplicated, unknown and invalid (label/category contradiction) items.
  These are excluded and listed, never counted as NO; nor is UNSURE. With the page's
  site_data.json it also lists YES labels without an evidence quote and quotes that are not
  verbatim in the reasoning; those labels still count, but should be checked.
- per annotator and arm: YES among YES/NO with a Wilson 95% CI, and category counts.
- per pair of annotators, overall and per arm: raw agreement and Cohen's kappa on the label
  (YES/NO/UNSURE, over items both labelled), on the decided label (both YES/NO) and on the
  category, plus the item IDs they disagree on, for adjudication.
- each annotator, and with two or more the unanimous consensus, against the Luna judge
  (joined on `luna_item_id`) and against the AISI keyword `mention`. A consensus item whose
  annotators agree on the label but not the category has category SPLIT.
- the Phase 2 gate (docs/handoff.md): human kappa >= 0.80 and agreement >= 90% on the label,
  with every item labelled. The judge half (sens/spec >= 0.90, 95% CI lower >= 0.80,
  invalid <= 1%) is computed against the consensus, which drops disagreements and so only
  stands in for an adjudicated reference.

Descriptive only: nothing here is tuned on these labels.

    python scripts/score_human_labels.py --key runs/human-label-v1/key.jsonl \
        labels_ab_407fef9aa3d4_*.json labels_cd_407fef9aa3d4_*.json \
        --json runs/human-label-v1/score.json
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path

from aisi_rh_judge_batches import CATEGORIES, LABELS, consistent, wilson

from project.agreement import bootstrap_ci, cohen_kappa, confusion, raw_agreement

FORMAT = "human-label-export-v1"
LUNA_DIR = Path(__file__).resolve().parents[2] / "docs/results/031-luna-judge/labels"
SITE_DATA = Path(__file__).resolve().parents[2] / "labeller/site_data.json"
GATE = {"human_kappa": 0.80, "human_agreement": 0.90, "judge_sens_spec": 0.90,
        "judge_ci_lower": 0.80, "judge_invalid": 0.01}
DECIDED = {"YES", "NO"}


class ExportError(ValueError):
    pass


def read_jsonl(path: str | Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def load_key(path: str | Path) -> tuple[list[dict], str]:
    rows = read_jsonl(path)
    ids = Counter(r["item_id"] for r in rows)
    if dups := sorted(i for i, c in ids.items() if c > 1):
        raise ExportError(f"key has duplicate item_ids: {dups[:5]}")
    datasets = {r["dataset_id"] for r in rows}
    if len(datasets) != 1:
        raise ExportError(f"key must hold one dataset_id, found {sorted(datasets)}")
    return rows, datasets.pop()


def load_texts(path: str | Path, dataset_id: str) -> dict[str, str] | None:
    """Reasoning by item_id from the page's site data, or None if it is for another dataset."""
    p = Path(path)
    if not p.exists():
        return None
    data = json.loads(p.read_text())
    if data["meta"]["dataset_id"] != dataset_id:
        return None
    return {x["item_id"]: x["reasoning"] for x in data["items"]}


def load_export(path: str | Path, key_ids: list[str], dataset_id: str,
                texts: dict[str, str] | None = None) -> dict:
    """Valid labels by item_id plus every problem found. Raises on a foreign or malformed file."""
    blob = json.loads(Path(path).read_text())
    if blob.get("format") != FORMAT:
        raise ExportError(f"{path}: format {blob.get('format')!r}, expected {FORMAT!r}")
    if blob.get("dataset_id") != dataset_id:
        raise ExportError(f"{path}: dataset_id {blob.get('dataset_id')!r} does not match key "
                          f"{dataset_id!r}")
    if not blob.get("annotator"):
        raise ExportError(f"{path}: no annotator name")
    rows = blob.get("labels") or []
    known = set(key_ids)
    counts = Counter(r.get("item_id") for r in rows)
    duplicates = sorted(i for i, c in counts.items() if c > 1)
    unknown = sorted(str(i) for i in counts if i not in known)
    labels, invalid, no_quote, not_verbatim = {}, [], [], []
    for r in rows:
        iid = r.get("item_id")
        if iid not in known or counts[iid] > 1:
            continue
        if (r.get("label") not in LABELS or r.get("category") not in CATEGORIES
                or not consistent(r["label"], r["category"])):
            invalid.append(iid)
            continue
        labels[iid] = {"label": r["label"], "category": r["category"]}
        quote = (r.get("evidence") or "").strip()
        if texts is not None and r["label"] == "YES" and not quote:
            no_quote.append(iid)
        if texts is not None and quote and quote not in texts.get(iid, ""):
            not_verbatim.append(iid)
    seen = {r.get("item_id") for r in rows}
    return {
        "annotator": blob["annotator"], "path": str(path),
        "rubric_version": blob.get("rubric_version"), "labels": labels,
        "issues": {"missing": [i for i in key_ids if i not in seen], "duplicates": duplicates,
                   "unknown": unknown, "invalid": sorted(invalid),
                   **({"yes_without_quote": sorted(no_quote),
                       "quote_not_verbatim": sorted(not_verbatim)} if texts is not None else {})},
    }


def load_luna(paths: list[str | Path]) -> tuple[dict[str, dict], list[str]]:
    """Luna labels by its item_id. Conflicting duplicates are dropped and returned."""
    seen: dict[str, list[dict]] = {}
    for p in paths:
        for r in read_jsonl(p):
            seen.setdefault(r["item_id"], []).append(
                {"label": r.get("label"), "category": r.get("category")})
    out, conflicts = {}, []
    for iid, rs in seen.items():
        if any(r != rs[0] for r in rs):
            conflicts.append(iid)
        elif (rs[0]["label"] in LABELS and rs[0]["category"] in CATEGORIES
              and consistent(rs[0]["label"], rs[0]["category"])):
            out[iid] = rs[0]
    return out, sorted(conflicts)


def rate(yes: int, n: int) -> dict:
    return {"yes": yes, "n": n, "rate": yes / n if n else None, "ci": wilson(yes, n)}


def arm_stats(key: list[dict], labels: dict[str, dict]) -> dict:
    out = {}
    for arm in [None, *sorted({r["arm"] for r in key})]:
        items = [r["item_id"] for r in key if arm is None or r["arm"] == arm]
        got = [labels[i] for i in items if i in labels]
        c = Counter(x["label"] for x in got)
        out["all" if arm is None else arm] = {
            "items": len(items), "labelled": len(got), "unlabelled": len(items) - len(got),
            "unsure": c["UNSURE"], **rate(c["YES"], c["YES"] + c["NO"]),
            "categories": dict(Counter(x.get("category", "SPLIT") for x in got)),
        }
    return out


def pair(a: list, b: list) -> dict:
    return {"n": len(a), "agreement": raw_agreement(a, b), "kappa": cohen_kappa(a, b)}


def pair_stats(items: list[str], la: dict[str, dict], lb: dict[str, dict]) -> dict:
    both = [i for i in items if i in la and i in lb]
    decided = [i for i in both if la[i]["label"] in DECIDED and lb[i]["label"] in DECIDED]
    return {
        "label": pair([la[i]["label"] for i in both], [lb[i]["label"] for i in both]),
        "label_decided": pair([la[i]["label"] for i in decided],
                              [lb[i]["label"] for i in decided]),
        "category": pair([la[i]["category"] for i in both], [lb[i]["category"] for i in both]),
        "label_disagreements": [i for i in both if la[i]["label"] != lb[i]["label"]],
        "category_disagreements": [i for i in both if la[i]["category"] != lb[i]["category"]],
    }


def by_arm(key: list[dict], fn) -> dict:
    out = {"all": fn([r["item_id"] for r in key])}
    for arm in sorted({r["arm"] for r in key}):
        out[arm] = fn([r["item_id"] for r in key if r["arm"] == arm])
    return out


def versus(items: list[str], human: dict[str, dict], ref: dict[str, dict]) -> dict:
    """Reference against human. Only items both decided enter; human is the truth."""
    both = [i for i in items if i in human and i in ref]
    decided = [i for i in both if human[i]["label"] in DECIDED and ref[i]["label"] in DECIDED]
    h = [human[i]["label"] == "YES" for i in decided]
    r = [ref[i]["label"] == "YES" for i in decided]
    c = confusion(r, h)
    out = {"n": len(decided), "agreement": raw_agreement(h, r), "kappa": cohen_kappa(h, r),
           "tp": c.tp, "fp": c.fp, "fn": c.fn, "tn": c.tn, "sensitivity": c.sensitivity,
           "specificity": c.specificity, "precision": c.precision,
           "ref_missing": sum(i in human and i not in ref for i in items),
           "ref_unsure": sum(ref[i]["label"] == "UNSURE" for i in both),
           "human_unsure": sum(human[i]["label"] == "UNSURE" for i in both)}
    cats = [i for i in both if "category" in human[i] and "category" in ref[i]]
    if cats:
        out["category"] = pair([human[i]["category"] for i in cats],
                               [ref[i]["category"] for i in cats])
    return out


def consensus(exports: list[dict]) -> dict[str, dict]:
    """Items every annotator labelled identically; category kept only when it also matches."""
    first, rest = exports[0]["labels"], [e["labels"] for e in exports[1:]]
    out = {}
    for iid, x in first.items():
        others = [o.get(iid) for o in rest]
        if all(o is not None and o["label"] == x["label"] for o in others):
            same_cat = all(o["category"] == x["category"] for o in others)
            out[iid] = {"label": x["label"], **({"category": x["category"]} if same_cat else {})}
    return out


def gate(summary: dict, key: list[dict], exports: list[dict], luna_vs_cons: dict | None,
         luna_ci: dict) -> dict:
    complete = all(not any(e["issues"][k] for k in ("missing", "duplicates", "invalid"))
                   and len(e["labels"]) == len(key) for e in exports)
    pairs = list(summary["pairs"].values())
    human = {
        "complete_coverage": complete,
        "kappa": [p["all"]["label"]["kappa"] for p in pairs],
        "agreement": [p["all"]["label"]["agreement"] for p in pairs],
    }
    human["passed"] = bool(pairs) and complete and all(
        k is not None and k >= GATE["human_kappa"] for k in human["kappa"]) and all(
        a is not None and a >= GATE["human_agreement"] for a in human["agreement"])
    judge = None
    if luna_vs_cons is not None:
        n = len(key)
        invalid = luna_vs_cons["ref_missing"] / n if n else None
        ok = [luna_vs_cons[s] is not None and luna_vs_cons[s] >= GATE["judge_sens_spec"]
              and luna_ci[s] is not None and luna_ci[s][0] >= GATE["judge_ci_lower"]
              for s in ("sensitivity", "specificity")]
        judge = {"reference": "unanimous human consensus (not adjudicated)",
                 "sensitivity": luna_vs_cons["sensitivity"], "specificity":
                 luna_vs_cons["specificity"], "ci": luna_ci, "invalid_rate": invalid,
                 "passed": all(ok) and invalid is not None and invalid <= GATE["judge_invalid"]}
    return {"thresholds": GATE, "human": human, "judge": judge}


def score(key: list[dict], exports: list[dict], luna: dict[str, dict] | None) -> dict:
    names = [e["annotator"] for e in exports]
    if len(set(names)) != len(names):
        raise ExportError(f"two exports from the same annotator: {names}")
    luna_by_item = None
    if luna is not None:
        luna_by_item = {r["item_id"]: luna[r["luna_item_id"]] for r in key
                        if r.get("luna_item_id") in luna}
    mention = {r["item_id"]: {"label": "YES" if r["mention"] else "NO"} for r in key
               if isinstance(r.get("mention"), bool)}
    humans = {e["annotator"]: e["labels"] for e in exports}
    if len(exports) >= 2:
        humans["consensus"] = consensus(exports)
    out = {
        "items": len(key), "arms": dict(Counter(r["arm"] for r in key)),
        "exports": {e["annotator"]: {"path": e["path"], "rubric_version": e["rubric_version"],
                                     "valid": len(e["labels"]),
                                     "issues": {k: {"n": len(v), "item_ids": v}
                                                for k, v in e["issues"].items()}}
                    for e in exports},
        "per_arm": {name: arm_stats(key, lab) for name, lab in humans.items()},
        "pairs": {f"{a['annotator']}|{b['annotator']}":
                  by_arm(key, lambda ids, a=a, b=b: pair_stats(ids, a["labels"], b["labels"]))
                  for a, b in combinations(exports, 2)},
        "vs_mention": {name: by_arm(key, lambda ids, h=lab: versus(ids, h, mention))
                       for name, lab in humans.items()},
    }
    if luna_by_item is not None:
        out["luna_per_arm"] = arm_stats(key, luna_by_item)
        out["vs_luna"] = {name: by_arm(key, lambda ids, h=lab: versus(ids, h, luna_by_item))
                          for name, lab in humans.items()}
    ref = humans.get("consensus") if len(exports) >= 2 else None
    luna_vs_cons, ci = None, {}
    if ref is not None and luna_by_item is not None:
        luna_vs_cons = out["vs_luna"]["consensus"]["all"]
        decided = [i for i in ref if ref[i]["label"] in DECIDED
                   and luna_by_item.get(i, {}).get("label") in DECIDED]
        pred = [luna_by_item[i]["label"] == "YES" for i in decided]
        truth = [ref[i]["label"] == "YES" for i in decided]
        ci = {s: bootstrap_ci(pred, truth, None, s) for s in ("sensitivity", "specificity")}
    out["gate"] = gate(out, key, exports, luna_vs_cons, ci)
    return out


def fmt(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.3f}"


def report(s: dict) -> str:
    lines = [f"{s['items']} items; arms {s['arms']}"]
    for name, e in s["exports"].items():
        iss = ", ".join(f"{k} {v['n']}" for k, v in e["issues"].items())
        lines.append(f"{name}: {e['valid']} valid ({iss})")
    for name, arms in s["per_arm"].items():
        lines.append(f"\nYES rate, {name}:")
        for arm, a in arms.items():
            ci = a["ci"] and f"[{a['ci'][0]:.3f}, {a['ci'][1]:.3f}]"
            lines.append(f"  {arm:8} {a['yes']}/{a['n']} = {fmt(a['rate'])} {ci or ''} "
                         f"unsure {a['unsure']} unlabelled {a['unlabelled']}")
    for p, arms in s["pairs"].items():
        lines.append(f"\nAgreement {p}:")
        for arm, a in arms.items():
            lines.append(f"  {arm:8} label {fmt(a['label']['agreement'])} "
                         f"k={fmt(a['label']['kappa'])} (n={a['label']['n']}); category "
                         f"{fmt(a['category']['agreement'])} k={fmt(a['category']['kappa'])}")
    for ref in ("vs_luna", "vs_mention"):
        for name, arms in s.get(ref, {}).items():
            a = arms["all"]
            lines.append(f"{ref} {name}: agreement {fmt(a['agreement'])} k={fmt(a['kappa'])} "
                         f"sens {fmt(a['sensitivity'])} spec {fmt(a['specificity'])} "
                         f"(n={a['n']})")
    g = s["gate"]
    if not s["pairs"]:
        lines.append("\nPhase 2 gate, human: FAIL (needs exports from two annotators)")
    else:
        lines.append(f"\nPhase 2 gate, human: {'PASS' if g['human']['passed'] else 'FAIL'} "
                     f"(kappa {g['human']['kappa']}, agreement {g['human']['agreement']}, "
                     f"complete {g['human']['complete_coverage']})")
    if g["judge"]:
        lines.append(f"Phase 2 gate, Luna vs consensus: {'PASS' if g['judge']['passed'] else 'FAIL'}"
                     f" (sens {fmt(g['judge']['sensitivity'])}, spec "
                     f"{fmt(g['judge']['specificity'])}, CI {g['judge']['ci']})")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--key", required=True)
    ap.add_argument("exports", nargs="+")
    ap.add_argument("--luna", default=str(LUNA_DIR),
                    help="directory of Luna results_*.jsonl; 'none' to skip")
    ap.add_argument("--site-data", default=str(SITE_DATA),
                    help="the page's site_data.json, to check evidence quotes")
    ap.add_argument("--json", help="write the full summary here")
    a = ap.parse_args(argv)
    try:
        key, dataset_id = load_key(a.key)
        ids = [r["item_id"] for r in key]
        texts = load_texts(a.site_data, dataset_id)
        exports = [load_export(p, ids, dataset_id, texts) for p in a.exports]
        luna, conflicts = None, []
        if a.luna != "none":
            files = sorted(glob.glob(str(Path(a.luna) / "*.jsonl")))
            if not files:
                raise ExportError(f"no Luna label files in {a.luna}")
            luna, conflicts = load_luna(files)
        s = score(key, exports, luna)
    except ExportError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    s["dataset_id"] = dataset_id
    if luna is not None:
        s["luna_conflicting_duplicates"] = conflicts
    print(report(s))
    if a.json:
        Path(a.json).write_text(json.dumps(s, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
