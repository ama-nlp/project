"""Build data/hints_mmlupro.jsonl for the hint environment (project.hints).

Knowledge-heavy MMLU-Pro categories only (math/physics/chemistry/engineering
make a Think model reason for thousands of tokens), 10-option questions only
so chance is uniform, and a fixed wrong "hint" letter per question.

    curl -L -o /tmp/mmlupro.parquet \
      https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro/resolve/main/data/test-00000-of-00001.parquet
    uv run scripts/make_hint_set.py /tmp/mmlupro.parquet --n 300
"""

import argparse
from pathlib import Path

import orjson
import polars as pl

from project.hints import hint_letter

CATEGORIES = ["history", "philosophy", "biology", "business", "law", "psychology",
              "economics", "health", "other"]

ap = argparse.ArgumentParser()
ap.add_argument("parquet")
ap.add_argument("--n", type=int, default=300)
ap.add_argument("--out", default="data/hints_mmlupro.jsonl")
args = ap.parse_args()

df = (pl.read_parquet(args.parquet)
      .filter(pl.col("category").is_in(CATEGORIES) & (pl.col("options").list.len() == 10))
      .sample(n=args.n, seed=0, shuffle=True))
with Path(args.out).open("wb") as f:
    for r in df.iter_rows(named=True):
        f.write(orjson.dumps(dict(
            qid=r["question_id"], category=r["category"], question=r["question"],
            options=r["options"], answer=r["answer"],
            hint=hint_letter(r["question_id"], r["answer"], len(r["options"])),
        )) + b"\n")
print(f"wrote {args.n} questions to {args.out}")
