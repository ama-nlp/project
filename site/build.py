#!/usr/bin/env python3
"""Build the results site into _site/.

Stdlib only, so it runs unchanged in GitHub Actions. It copies the static
shell, publishes every git-tracked document under PUBLISHED as a browsable
file tree, parses the run index in docs/results/README.md, and verifies that
every number in data/findings.json appears verbatim in its cited source.
A chart whose `check` string is missing from its source fails the build:
the site must never drift from the run records.

    python3 site/build.py            # build into _site/
    python3 site/build.py --check    # verify findings only

When the run data is present (project/runs/ada/<account>/..., or --runs),
the build also exports per-sample traces for the sample browser; without
it the site still builds, just without samples.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SITE = ROOT / "site"
OUT = ROOT / "_site"

# Repository paths exposed in the file tree. Only git-tracked files are
# published, so ignored local outputs and fetched upstream data never leak.
PUBLISHED = ["README.md", "AGENTS.md", "docs", "project/README.md", "slides/mid/big-brother-mid.pdf"]
# Raw run files up to this size are served by Pages; larger ones are listed in
# the tree and downloadable inside the `run-data` release archive (Pages caps
# a site at about 1 GB).
HOSTED_MAX = 1_000_000
RELEASE = "run-data"
ARCHIVE = "run-data.tar.xz"  # a tar of the run-data root; CI builds from it too
SKIP_NAMES = re.compile(r"^(\.nfs|\.DS_Store$|__pycache__$)")
TEXT_SUFFIXES = {".md", ".txt", ".json", ".jsonl", ".log", ".py", ".sh", ".tex", ".yaml", ".yml", ".toml"}

# Study grouping for the run index. Keyed by the record's filename stem.
STUDIES = [
    ("Qwen3 calibration", r"^00[1-5]-"),
    ("Qwen3 A/B/C arms", r"^(00[6-9]|01[0-2])-"),
    ("Qwen3 impossible tests and diagnostics", r"^(01[3-9]|02[01])-"),
    ("OLMo-3.1-32B reward-hacking organism", r"^0(2[2-9]|30)-"),
]


def tracked_files() -> list[str]:
    out = subprocess.run(
        ["git", "ls-files", "-z", "--", *PUBLISHED],
        cwd=ROOT, check=True, capture_output=True, text=True,
    ).stdout
    return sorted(p for p in out.split("\0") if p and (ROOT / p).is_file())


def title_of(path: Path) -> str | None:
    if path.suffix != ".md":
        return None
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return None


def last_commit_date(path: str) -> str | None:
    out = subprocess.run(
        ["git", "log", "-1", "--format=%cs", "--", path],
        cwd=ROOT, capture_output=True, text=True,
    ).stdout.strip()
    return out or None


def build_manifest(files: list[str]) -> list[dict]:
    entries = []
    for rel in files:
        p = ROOT / rel
        entries.append({
            "path": rel,
            "size": p.stat().st_size,
            "title": title_of(p),
            "text": p.suffix in TEXT_SUFFIXES,
            "updated": last_commit_date(rel),
        })
    return entries


def publish_run_files(root: Path) -> list[dict]:
    """Every file under the run-data root, as `runs/<account>/...` entries.

    Small files are copied into _site/files/; large ones are listed with
    `archived: true` and are downloadable only inside the release archive.
    """
    entries = []
    for p in sorted(root.rglob("*")):
        if not p.is_file() or any(SKIP_NAMES.match(part) for part in p.relative_to(root).parts):
            continue
        rel = "runs/" + p.relative_to(root).as_posix()
        size = p.stat().st_size
        entry = {"path": rel, "size": size, "text": p.suffix in TEXT_SUFFIXES, "raw": True}
        if size <= HOSTED_MAX:
            dest = OUT / "files" / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, dest)
        else:
            entry["archived"] = True
        entries.append(entry)
    return entries


def split_row(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def parse_run_index(files: list[str]) -> list[dict]:
    """Rows of the run table in docs/results/README.md, linked to their records."""
    index = ROOT / "docs/results/README.md"
    records = {Path(f).name[:3]: f for f in files
               if f.startswith("docs/results/") and re.match(r"\d{3}-", Path(f).name)}
    lines = [ln for ln in index.read_text(encoding="utf-8").splitlines() if ln.startswith("|")]
    header = [h.lower() for h in split_row(lines[0])]
    runs = []
    for line in lines[2:]:
        row = dict(zip(header, split_row(line), strict=False))
        num = row.get("#", "")
        record = records.get(num)
        if record is None:
            raise SystemExit(f"run index row {num!r} has no record in docs/results/")
        study = next((name for name, pat in STUDIES if re.match(pat, Path(record).name)), "Other")
        runs.append({
            "record": record,
            "title": title_of(ROOT / record),
            "submitted": row.get("submitted"),
            "account": row.get("account"),
            "job": row.get("slurm job"),
            "description": row.get("short description"),
            "status": row.get("status"),
            "study": study,
        })
    return runs


def check_findings(findings: dict) -> list[str]:
    errors = []
    cache: dict[str, str] = {}

    def text(rel: str) -> str:
        if rel not in cache:
            cache[rel] = (ROOT / rel).read_text(encoding="utf-8")
        return cache[rel]

    known = set(findings["charts"])
    for section in findings["sections"]:
        for cid in section["charts"]:
            if cid not in known:
                errors.append(f"section {section['id']}: unknown chart {cid}")
    for cid, chart in findings["charts"].items():
        pool = [chart["source"], *chart.get("sources", [])]
        for i, row in enumerate(chart["rows"]):
            where = [row["source"]] if "source" in row else pool
            needle = row.get("check")
            if not needle:
                errors.append(f"{cid}[{i}]: no check string")
            elif not any(needle in text(src) for src in where):
                errors.append(f"{cid}[{i}] ({row.get('label', row.get('x'))}): {needle!r} not found in {where}")
            if "k" in row and not 0 <= row["k"] <= row["n"]:
                errors.append(f"{cid}[{i}]: k={row['k']} outside 0..n={row['n']}")
    return errors


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="verify findings.json and exit")
    ap.add_argument("--runs", type=Path, default=ROOT / "project/runs/ada",
                    help="run-data root for the sample browser (skipped if absent)")
    args = ap.parse_args()

    findings = json.loads((SITE / "data/findings.json").read_text(encoding="utf-8"))
    errors = check_findings(findings)
    if errors:
        print("findings.json does not match its sources:", file=sys.stderr)
        for e in errors:
            print("  " + e, file=sys.stderr)
        return 1
    n_rows = sum(len(c["rows"]) for c in findings["charts"].values())
    print(f"verified {len(findings['charts'])} charts, {n_rows} data rows")
    if args.check:
        return 0

    files = tracked_files()
    if OUT.exists():
        shutil.rmtree(OUT)
    shutil.copytree(SITE, OUT, ignore=shutil.ignore_patterns("*.py", "__pycache__"))
    for rel in files:
        dest = OUT / "files" / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / rel, dest)

    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                            capture_output=True, text=True).stdout.strip()
    repo = "https://github.com/ama-nlp/project"
    meta = {
        "built": datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
        "commit": commit,
        "repo": repo,
        "archive": f"{repo}/releases/download/{RELEASE}/{ARCHIVE}",
    }
    raw = publish_run_files(args.runs) if args.runs.is_dir() else []
    (OUT / "data/manifest.json").write_text(json.dumps({
        "meta": meta,
        "files": build_manifest(files) + raw,
        "runs": parse_run_index(files),
    }, indent=1), encoding="utf-8")
    (OUT / ".nojekyll").touch()
    if raw:
        big = [e for e in raw if e.get("archived")]
        hosted = sum(e["size"] for e in raw) - sum(e["size"] for e in big)
        print(f"run data: {len(raw)} files, {hosted / 1e6:.0f} MB hosted, {len(big)} large files "
              f"({sum(e['size'] for e in big) / 1e6:.0f} MB) only in the {RELEASE} archive")
    if args.runs.is_dir():
        from export_traces import export  # sibling module; stdlib only
        spec = json.loads((SITE / "data/traces.json").read_text(encoding="utf-8"))
        for problem in export(args.runs, OUT, spec):
            print("  no traces: " + problem, file=sys.stderr)
    else:
        print(f"no run data at {args.runs}; building without the sample browser")
    print(f"published {len(files)} files into {OUT.relative_to(ROOT)}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
