#!/usr/bin/env python3
"""Render the labelling page from committed, blinded site data (no key needed). Used by CI.

    python scripts/render_label_site.py --out dist

writes dist/index.html (labeller/template.html + labeller/site_data.json) and copies
labeller/_headers (Cloudflare Pages response headers) next to it. Regenerate site_data.json
only with make_human_label_site.py; changing it changes the dataset_id, which resets
annotators' saved progress.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from make_human_label_site import LABELLER, TEMPLATE, render


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default=str(LABELLER / "site_data.json"))
    p.add_argument("--template", default=str(TEMPLATE))
    p.add_argument("--out", required=True, help="directory for index.html")
    a = p.parse_args(argv)

    data = json.loads(Path(a.data).read_text())
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "index.html").write_text(render(Path(a.template).read_text(), data))
    if (LABELLER / "_headers").exists():
        shutil.copy(LABELLER / "_headers", out / "_headers")
    print(f"{len(data['items'])} items, dataset {data['meta']['dataset_id']} -> {out}/index.html")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
