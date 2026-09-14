#!/usr/bin/env python3
"""Build the P2 annotation manifest: which CoTs get labelled, under which
blinded ID, in which split.

    uv run scripts/make_p2_manifest.py runs/*/*.jsonl --out p2_manifest.jsonl \
        --take natural=60 elicited=60 calibration=30 --holdout-frac 0.5 --seed 0

Stratum comes from provenance, not from labels (we have none yet):
  elicited     hint_variant starts with "detector_validation_elicited"
  calibration  hint_variant starts with "p2_calibration"
  natural      anything else (the experimental prompts, arm C so far)

Exact-duplicate and near-duplicate CoTs (same text after whitespace collapse
and lowercasing) are grouped; one representative is kept, and the split is
assigned per group so no near-duplicate straddles dev/held-out. Items with no
usable CoT are dropped and counted.

The output is committed. The held-out half is never edited after that; the
manifest's sha256 goes in phase2-completion.md.
"""

from __future__ import annotations

import argparse
import hashlib
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import orjson

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from project.schema import sha256  # noqa: E402

_WS = re.compile(r"\s+")


def stratum(row: dict) -> str:
    hv = row.get("hint_variant") or ""
    if hv.startswith("detector_validation_elicited"):
        return "elicited"
    if hv.startswith("p2_calibration"):
        return "calibration"
    return "natural"


def norm_hash(cot: str) -> str:
    return sha256(_WS.sub(" ", cot).strip().lower())


def load(paths: list[str]) -> list[dict]:
    rows = []
    for p in paths:
        for line in Path(p).read_text().splitlines():
            if line.strip():
                r = orjson.loads(line)
                r["_path"] = p
                rows.append(r)
    return rows


def build(rows: list[dict], take: dict[str, int], holdout_frac: float, seed: int) -> tuple[list[dict], dict]:
    rng = random.Random(seed)
    dropped = Counter()
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        cot = r.get("cot")
        if r.get("cot_retention") == "deleted" or not cot or not cot.strip():
            dropped["no_cot"] += 1
            continue
        groups[norm_hash(cot)].append(r)

    # one representative per near-duplicate group
    reps = []
    for h, members in groups.items():
        if len(members) > 1:
            dropped["near_duplicate"] += len(members) - 1
        r = min(
            members,
            key=lambda x: (
                x["run_id"],
                x["problem_id"],
                x.get("sample_idx", 0),
                x.get("round_id", 1),
            ),
        )
        reps.append((h, r))

    by_stratum: dict[str, list] = defaultdict(list)
    for h, r in reps:
        by_stratum[stratum(r)].append((h, r))

    items = []
    for s, want in take.items():
        pool = by_stratum.get(s, [])
        # shuffle within difficulty so both are represented, then interleave
        by_diff: dict[str, list] = defaultdict(list)
        for h, r in pool:
            by_diff[r.get("difficulty", "?")].append((h, r))
        for lst in by_diff.values():
            rng.shuffle(lst)
        picked = []
        lists = list(by_diff.values())
        while len(picked) < want and any(lists):
            for lst in lists:
                if lst and len(picked) < want:
                    picked.append(lst.pop())
        if len(picked) < want:
            dropped[f"short_{s}"] = want - len(picked)
        items.extend((s, h, r) for h, r in picked)

    rng.shuffle(items)
    n_hold = round(len(items) * holdout_frac)
    out = []
    for i, (s, h, r) in enumerate(items):
        # blinded id: stable across reruns with the same seed and inputs,
        # unrelated to problem or run so the annotator cannot infer anything
        item_id = hashlib.sha256(f"{seed}:{h}".encode()).hexdigest()[:10]
        out.append(
            {
                "item_id": item_id,
                "run_id": r["run_id"],
                "problem_id": r["problem_id"],
                "sample_idx": r.get("sample_idx", 0),
                "round_id": r.get("round_id", 1),
                "difficulty": r.get("difficulty"),
                "cot_sha256": r.get("cot_sha256") or sha256(r["cot"]),
                "norm_sha256": h,
                "stratum": s,
                "split": "heldout" if i < n_hold else "dev",
                "source": r["_path"],
            }
        )
    out.sort(key=lambda x: x["item_id"])
    return out, dict(dropped)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("traces", nargs="+")
    ap.add_argument("--out", default="p2_manifest.jsonl")
    ap.add_argument("--take", nargs="+", default=["natural=50", "elicited=50"],
                    help="stratum=count ...")
    ap.add_argument("--holdout-frac", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    take = {k: int(v) for k, v in (t.split("=") for t in a.take)}
    rows = load(a.traces)
    items, dropped = build(rows, take, a.holdout_frac, a.seed)

    out = Path(a.out)
    if out.exists():
        print(f"{out} exists; refusing to overwrite a manifest. Delete it deliberately.")
        return 1
    out.write_bytes(b"".join(orjson.dumps(x) + b"\n" for x in items))

    print(f"{len(rows)} traces in -> {len(items)} items out -> {out}")
    print(f"  sha256 {sha256(out.read_text())}")
    print("  by stratum/split:", dict(Counter((x['stratum'], x['split']) for x in items)))
    print("  by difficulty:   ", dict(Counter(x["difficulty"] for x in items)))
    if dropped:
        print("  dropped:         ", dropped)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
