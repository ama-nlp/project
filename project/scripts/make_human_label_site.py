#!/usr/bin/env python3
"""Build the offline human-labelling page for the AISI OLMo-32B arms.

Samples N hacked outputs per arm from the Luna blinded export (originals only, never the
self-consistency repeats), re-blinds them under fresh IDs so annotators cannot look up the
judge's labels, shuffles them across arms, and inlines {item_id, reasoning} into a single
self-contained HTML file. Arm, run and Luna ID go only into a separate key, which must not
be shipped to annotators.

    python scripts/make_human_label_site.py --key runs/luna-key/key.jsonl \
        --batches ~/luna-batches/batch_*.jsonl --out runs/human-label-v1

Defaults label arms A, B, C (run 023), A-pen (024), E, E-pen (025) and F (026), 100 each.
C is Run 023's C: the baseline the paper uses for A, B, E and E-pen.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import random
from collections import defaultdict
from datetime import date
from pathlib import Path

DEFAULT_ARMS = ["023:A=A", "023:B=B", "023:C=C", "024:A-pen=Apen", "025:E=E",
                "025:E-pen=Epen", "026:F=F"]
TEMPLATE = Path(__file__).resolve().parents[1] / "labelling" / "template.html"
TUTORIAL = Path(__file__).resolve().parents[1] / "labelling" / "tutorial.json"
PLACEHOLDER = "/*__LABEL_DATA__*/null"


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def parse_arms(specs: list[str]) -> dict[str, str]:
    """'023:A=A' -> {'023:A': 'A'}; the arm name defaults to the run label."""
    out = {}
    for spec in specs:
        run, _, arm = spec.partition("=")
        out[run] = arm or run
    return out


def sample(key: list[dict], reasoning: dict[str, str], arms: dict[str, str], n: int,
           seed: int) -> tuple[list[dict], list[dict]]:
    """Returns (shuffled items, key rows). Raises if an arm has fewer than n originals or a
    reasoning text no longer matches the hash recorded at export."""
    pools = defaultdict(list)
    for k in key:
        if k["run"] in arms and k["repeat_of"] is None:
            pools[k["run"]].append(k)
    rng = random.Random(seed)
    items, rows = [], []
    for run, arm in arms.items():
        pool = sorted(pools[run], key=lambda k: k["item_id"])
        if len(pool) < n:
            raise SystemExit(f"{run}: only {len(pool)} originals, need {n}")
        for k in rng.sample(pool, n):
            text = reasoning[k["item_id"]]
            if sha(text) != k["cot_sha256"]:
                raise SystemExit(f"{k['item_id']}: reasoning does not match its cot_sha256")
            iid = "h" + sha(f"human:{seed}:{k['item_id']}")[:9]
            items.append({"item_id": iid, "reasoning": text})
            rows.append({"item_id": iid, "arm": arm, "run": run, "luna_item_id": k["item_id"],
                         "sample_id": k["sample_id"], "cot_sha256": k["cot_sha256"],
                         "mention": k["mention"]})
    rng.shuffle(items)
    return items, rows


def load_tutorial(spec: list[dict], key: list[dict], reasoning: dict[str, str],
                  arms: dict[str, str]) -> list[dict]:
    """Practice items with a known answer. They must come from runs outside the labelled arms,
    so no annotator sees a practice answer for an item (or arm pool) they will label."""
    by_id = {k["item_id"]: k for k in key}
    out = []
    for t in spec:
        k = by_id[t["luna_item_id"]]
        if k["run"] in arms:
            raise SystemExit(f"tutorial item {k['item_id']} comes from labelled run {k['run']}")
        text = reasoning[k["item_id"]]
        if sha(text) != k["cot_sha256"]:
            raise SystemExit(f"{k['item_id']}: reasoning does not match its cot_sha256")
        if t["evidence"] not in text:
            raise SystemExit(f"tutorial item {k['item_id']}: evidence is not a verbatim quote")
        out.append({"reasoning": text, "category": t["category"], "evidence": t["evidence"],
                    "explanation": t["explanation"]})
    return out


def site_data(items: list[dict], meta: dict, tutorial: list[dict] = ()) -> dict:
    """Everything the page embeds. Blinded: no arm, run or Luna ID, so it is safe to commit
    and is what CI renders from (scripts/render_label_site.py)."""
    return {"meta": meta, "items": items, "tutorial": list(tutorial)}


def render(template: str, data: dict) -> str:
    payload = json.dumps(data, ensure_ascii=False)
    # Inline <script> safety: no "</script>" or "<!--" can terminate the block early.
    payload = payload.replace("<", "\\u003c")
    if PLACEHOLDER not in template:
        raise SystemExit(f"template lacks {PLACEHOLDER}")
    return template.replace(PLACEHOLDER, payload)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--key", required=True)
    p.add_argument("--batches", nargs="+", required=True, help="Luna batch_*.jsonl files (globs ok)")
    p.add_argument("--arms", nargs="+", default=DEFAULT_ARMS, help="RUN_LABEL=ARM_NAME")
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--seed", type=int, default=20261018)
    p.add_argument("--template", default=str(TEMPLATE))
    p.add_argument("--tutorial", default=str(TUTORIAL), help="practice items; '' for none")
    p.add_argument("--out", required=True, help="new directory for label.html and key.jsonl")
    a = p.parse_args(argv)

    key = [json.loads(line) for line in Path(a.key).read_text().splitlines()]
    reasoning = {}
    for pattern in a.batches:
        for path in sorted(glob.glob(str(Path(pattern).expanduser()))):
            for line in Path(path).read_text().splitlines():
                x = json.loads(line)
                reasoning[x["item_id"]] = x["reasoning"]
    arms = parse_arms(a.arms)
    items, rows = sample(key, reasoning, arms, a.n, a.seed)
    tutorial = (load_tutorial(json.loads(Path(a.tutorial).read_text()), key, reasoning, arms)
                if a.tutorial else [])

    dataset_id = sha(json.dumps(items, ensure_ascii=False))[:12]
    meta = {"dataset_id": dataset_id, "rubric_version": "human-aisi-cot-v1", "seed": a.seed,
            "n_items": len(items), "n_tutorial": len(tutorial), "built": date.today().isoformat()}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)  # never overwrite an earlier build or its key
    data = site_data(items, meta, tutorial)
    (out / "site_data.json").write_text(json.dumps(data, ensure_ascii=False, indent=1) + "\n")
    (out / "label.html").write_text(render(Path(a.template).read_text(), data))
    with open(out / "key.jsonl", "w") as f:
        f.writelines(json.dumps({**r, "dataset_id": dataset_id}) + "\n" for r in rows)
    print(f"{len(items)} items, dataset {dataset_id} -> {out}/label.html and site_data.json "
          f"(blinded; ship or commit) and {out}/key.jsonl (keep private)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
