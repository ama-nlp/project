#!/usr/bin/env python3
"""P2 human annotation: label CoTs for verbalised hack intent, then measure
agreement and adjudicate.

    uv run scripts/label_p2.py --annotator alice --manifest p2_manifest.jsonl runs/*/*.jsonl
    uv run scripts/label_p2.py --agreement p2_labels/alice.jsonl p2_labels/bob.jsonl
    uv run scripts/label_p2.py --adjudicate p2_labels/alice.jsonl p2_labels/bob.jsonl \
        --manifest p2_manifest.jsonl runs/*/*.jsonl

Rubric: docs/phase2-rubric.md. The annotator sees the blinded item_id and the
CoT. Nothing else -- not the program, verdict, arm, run, problem, or any other
annotator's answer. Labels go to p2_labels/<annotator>.jsonl, append-only;
rerunning resumes. Adjudication writes a separate file and never touches the
originals, so pre-adjudication agreement stays reportable.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import orjson

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from project.agreement import cohen_kappa, confusion, raw_agreement  # noqa: E402

RUBRIC_VERSION = "p2-rubric-v1"
LABELS_DIR = Path("p2_labels")

LABELS = {
    "1": "EXPLICIT_INTENT",
    "2": "IMPLICIT_INTENT",
    "3": "AWARENESS_ONLY",
    "4": "SUSPICION_ONLY",
    "5": "NO_EVIDENCE",
    "?": "UNSURE",
    "0": "NO_COT",
}
BINARY = {"EXPLICIT_INTENT": True, "IMPLICIT_INTENT": True,
          "AWARENESS_ONLY": False, "SUSPICION_ONLY": False, "NO_EVIDENCE": False}


def read_jsonl(p: Path) -> list[dict]:
    if not p.exists():
        return []
    return [orjson.loads(x) for x in p.read_text().splitlines() if x.strip()]


def load_cots(manifest: list[dict], trace_paths: list[str]) -> dict[str, str]:
    """item_id -> cot. Only the CoT is retained from each trace row."""
    want = {
        (m["run_id"], m["problem_id"], m["sample_idx"], m.get("round_id", 1)): m["item_id"]
        for m in manifest
    }
    out: dict[str, str] = {}
    for p in trace_paths:
        for line in Path(p).read_text().splitlines():
            if not line.strip():
                continue
            r = orjson.loads(line)
            k = (
                r["run_id"],
                r["problem_id"],
                r.get("sample_idx", 0),
                r.get("round_id", 1),
            )
            if k in want and r.get("cot"):
                out[want[k]] = r["cot"]
    return out


def do_label(annotator: str, manifest: list[dict], cots: dict[str, str], split: str | None,
             limit: int) -> int:
    LABELS_DIR.mkdir(exist_ok=True)
    path = LABELS_DIR / f"{annotator}.jsonl"
    done = {r["item_id"] for r in read_jsonl(path)}
    todo = [m for m in manifest if m["item_id"] not in done and (not split or m["split"] == split)]
    todo = todo[:limit]
    if not todo:
        print(f"nothing left ({len(done)} done)")
        return 0
    print(f"{len(todo)} to label, {len(done)} done. Rubric: docs/phase2-rubric.md\n")
    for k, v in LABELS.items():
        print(f"  {k} = {v}")
    print("  q = stop\n")

    n = 0
    with path.open("ab") as f:
        for i, m in enumerate(todo, 1):
            cot = cots.get(m["item_id"])
            print("\n" + "=" * 72)
            print(f"[{i}/{len(todo)}]  item {m['item_id']}")
            print("=" * 72)
            print(cot if cot else "(no CoT available for this item)")
            print("-" * 72)
            while True:
                ans = input("label [1/2/3/4/5/?/0/q] ").strip()
                if ans in LABELS or ans == "q":
                    break
            if ans == "q":
                break
            note = input("note (optional): ").strip()
            f.write(orjson.dumps({
                "item_id": m["item_id"],
                "label": LABELS[ans],
                "note": note or None,
                "annotator": annotator,
                "rubric_version": RUBRIC_VERSION,
                "timestamp": datetime.now(UTC).isoformat(),
            }) + b"\n")
            f.flush()
            n += 1
    print(f"\nlabelled {n}; {path} holds {len(done) + n}")
    return 0


def pair(a: list[dict], b: list[dict]) -> list[tuple[str, str, str]]:
    """(item_id, label_a, label_b) for items both annotators labelled."""
    la = {r["item_id"]: r["label"] for r in a}
    lb = {r["item_id"]: r["label"] for r in b}
    return [(k, la[k], lb[k]) for k in sorted(set(la) & set(lb))]


def do_agreement(files: list[Path]) -> int:
    if len(files) != 2:
        print("agreement needs exactly two annotator files")
        return 1
    pairs = pair(read_jsonl(files[0]), read_jsonl(files[1]))
    if not pairs:
        print("no overlapping items")
        return 1
    scorable = [(k, x, y) for k, x, y in pairs if x in BINARY and y in BINARY]
    excluded = len(pairs) - len(scorable)

    print(f"{len(pairs)} items labelled by both; {excluded} excluded (UNSURE / NO_COT by either)\n")

    fine_a = [x for _, x, _ in scorable]
    fine_b = [y for _, _, y in scorable]
    print("fine-grained (7-class):")
    print(f"  kappa {fmt(cohen_kappa(fine_a, fine_b))}   raw {fmt(raw_agreement(fine_a, fine_b))}")

    bin_a = [BINARY[x] for x in fine_a]
    bin_b = [BINARY[y] for y in fine_b]
    print("binary (intent vs not):")
    print(f"  kappa {fmt(cohen_kappa(bin_a, bin_b))}   raw {fmt(raw_agreement(bin_a, bin_b))}")
    c = confusion(bin_a, bin_b)
    print(f"  prevalence A {sum(bin_a)}/{len(bin_a)}   B {sum(bin_b)}/{len(bin_b)}")
    print(f"  both-yes {c.tp}  A-only {c.fp}  B-only {c.fn}  both-no {c.tn}")

    print("\nper-class counts (A / B):")
    ca, cb = Counter(fine_a), Counter(fine_b)
    for lab in BINARY:
        print(f"  {lab:<16} {ca[lab]:>3} / {cb[lab]:<3}")

    dis = [(k, x, y) for k, x, y in scorable if BINARY[x] != BINARY[y]]
    if dis:
        print(f"\n{len(dis)} binary disagreement(s):")
        for k, x, y in dis:
            print(f"  {k}  {x}  vs  {y}")
    return 0


def do_adjudicate(files: list[Path], manifest: list[dict], cots: dict[str, str]) -> int:
    """Walk every binary disagreement and every UNSURE, show both labels and the
    CoT, record a final label to p2_labels/adjudicated.jsonl. Agreements are
    copied through unchanged."""
    a, b = read_jsonl(files[0]), read_jsonl(files[1])
    out = LABELS_DIR / "adjudicated.jsonl"
    done = {r["item_id"] for r in read_jsonl(out)}
    n_auto = n_manual = 0
    with out.open("ab") as f:
        for k, x, y in pair(a, b):
            if k in done:
                continue
            agree = x in BINARY and y in BINARY and BINARY[x] == BINARY[y]
            if agree:
                # binary agreement is what the judge is scored on; if the fine
                # labels differ (EXPLICIT vs IMPLICIT) the first annotator's is kept
                final, how = x, "agreed"
                n_auto += 1
            else:
                print("\n" + "=" * 72)
                print(f"item {k}   A: {x}   B: {y}")
                print("=" * 72)
                print(cots.get(k, "(no CoT)"))
                print("-" * 72)
                while True:
                    ans = input("final label [1/2/3/4/5/?/0/q] ").strip()
                    if ans in LABELS or ans == "q":
                        break
                if ans == "q":
                    break
                final, how = LABELS[ans], "adjudicated"
                n_manual += 1
            f.write(orjson.dumps({
                "item_id": k, "label": final, "how": how,
                "labels": {files[0].stem: x, files[1].stem: y},
                "rubric_version": RUBRIC_VERSION,
                "timestamp": datetime.now(UTC).isoformat(),
            }) + b"\n")
            f.flush()
    print(f"\n{n_auto} carried through, {n_manual} adjudicated -> {out}")
    return 0


def fmt(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.3f}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("traces", nargs="*")
    ap.add_argument("--manifest")
    ap.add_argument("--annotator")
    ap.add_argument("--split", choices=["dev", "heldout"])
    ap.add_argument("--limit", type=int, default=10**9)
    ap.add_argument("--agreement", nargs=2, metavar="LABELS")
    ap.add_argument("--adjudicate", nargs=2, metavar="LABELS")
    a = ap.parse_args()

    if a.agreement:
        return do_agreement([Path(p) for p in a.agreement])

    if not a.manifest or not a.traces:
        ap.error("--manifest and trace files are required to label or adjudicate")
    manifest = read_jsonl(Path(a.manifest))
    cots = load_cots(manifest, a.traces)

    if a.adjudicate:
        return do_adjudicate([Path(p) for p in a.adjudicate], manifest, cots)
    if not a.annotator:
        ap.error("--annotator is required")
    return do_label(a.annotator, manifest, cots, a.split, a.limit)


if __name__ == "__main__":
    raise SystemExit(main())
