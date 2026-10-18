# Human labelling page (AISI OLMo-32B arms)

One self-contained HTML page that shows one blinded `<thinking>` block at a time and asks the
annotator for the `aisi-cot-v1` category (PLAN_PLAIN, PLAN_JUSTIFIED, REJECTED, DISCLAIMER,
MONITOR_ONLY, NO_EVIDENCE, UNSURE) and, for a plan, an evidence quote. There is no server:
labels save in the browser's localStorage, and **Export** downloads a JSON file to send back.

Live at https://reasoning-labeller.pages.dev.

| File | What it is |
|---|---|
| `template.html` | The page. Plain HTML, CSS and JS, no dependencies. |
| `site_data.json` | The blinded items and practice items the page embeds. No arm, run or Luna ID. |
| `tutorial.json` | The practice items, by Luna ID, with answers and explanations. Read only by the builder. |
| `_headers` | Cloudflare Pages response headers (`noindex`). |

The Python side is in `project/scripts/`: `make_human_label_site.py` (build),
`render_label_site.py` (render for CI) and `score_human_labels.py` (score exports).

## Build

```bash
cd project
python scripts/make_human_label_site.py --key runs/ada/advait/luna-blind-20261001/key.jsonl \
    --batches 'runs/ada/advait/luna-blind-20261001/batch_*.jsonl' --out runs/human-label-v2
cp runs/human-label-v2/site_data.json ../labeller/
```

- Default: 100 hacked outputs per arm for A, B, C (Run 023), A-pen (024), E, E-pen (025) and F (026),
  700 items in total. C is Run 023's C, the baseline the paper uses for A, B, E and E-pen.
- Self-consistency repeats are excluded. Items get fresh IDs (`h…`), so annotators can't look up
  the committed Luna labels by ID.
- The IDs and the sample order come from a secret salt. The first build (dataset `407fef9aa3d4`)
  used the public seed instead, and all 700 of its IDs could be recomputed from the public Luna
  IDs, so it was replaced before anyone labelled.
- Outputs: `site_data.json` (blinded; commit it), and `key.jsonl` plus `salt.txt` (private;
  `key.jsonl` maps each `h…` ID to arm, run, Luna ID and `cot_sha256`). The current build,
  dataset `79cab47704ac`, is in `project/runs/human-label-v2/` on advait's machine; copy it to
  Ada next to the Luna export. The build refuses to overwrite an existing directory.

## Blinding is on trust

The salt stops anyone recomputing IDs, but the item texts themselves can still be matched: the
results site publishes the Luna export (`files/runs/advait/luna-blind-20261001/`, with the key
that names each item's run), and Luna's labels and verbatim evidence quotes are committed under
`docs/results/031-luna-judge/labels/`. An annotator who searched for a passage could find its
condition and the judge's label. The guide asks annotators not to. Report this with the results.

## Deploy

`.github/workflows/labeller.yml` runs on every push to `main` that touches the labeller: the
labeller tests, then `render_label_site.py` (template plus committed `site_data.json`), then
`wrangler pages deploy` to the `reasoning-labeller` Cloudflare Pages project, which it creates
on the first run. It needs the `CLOUDFLARE_API_TOKEN` (Pages: Edit) and `CLOUDFLARE_ACCOUNT_ID`
repository secrets.

- **Changing the page** (`template.html`, `_headers`): edit, push, done. The `dataset_id`
  stays the same, so annotators keep their saved labels.
- **Changing the items:** rebuild with `make_human_label_site.py`, copy the new `site_data.json`
  here and push. A new `dataset_id` means annotators start over and old exports don't import.
- Saved labels belong to the site's address. Moving the page to another domain means each
  annotator must export on the old one and import on the new one.

Preview locally:

```bash
cd project && python scripts/render_label_site.py --out ../dist && python3 -m http.server -d ../dist
```

## Practice

New annotators first work through the practice items, with feedback after each one: a plain plan,
a justified plan, a rejected hack, a disclaimer and ordinary reasoning. Claude read each one and
checked the answer against Luna's label. They come from Runs 022, 027 and 028, outside the seven
labelled arms; the build refuses a practice item from a labelled run or with an evidence quote
that isn't verbatim. Every run through the practice is kept (`tutorial.attempts`), so the first
attempt is never overwritten. Practice answers are exported under `tutorial`, never mixed with
`labels`. Changing practice items doesn't change `dataset_id`.

## Exports

`labels_<annotator>_<dataset_id>_<time>.json` (`format: human-label-export-v1`) holds
`annotator`, `dataset_id`, `rubric_version`, `tutorial` and one row per labelled item:
`item_id`, `category`, `label`, `evidence`, `notes`, `labelled_at`, `seconds`, `revisions`.
`seconds` counts only time the tab was visible, capped at 10 minutes per visit to an item.
Import refuses an export from a different `dataset_id` and skips rows with an unknown item or a
label that contradicts its category.

The page won't move on from a plan label until it has a quote, and Export warns about any that
lack one. The scorer lists those, and quotes that aren't verbatim in the item, using
`site_data.json`:

```bash
cd project
python scripts/score_human_labels.py --key runs/human-label-v2/key.jsonl labels_*.json \
    --json runs/human-label-v2/score.json
```

The guide in the page is reconstructed from the `aisi-cot-v1` definitions in
`docs/results/031-aisi-rh-luna-judge.md`. The original Luna prompt text was not saved, so the
human and judge rubrics may differ slightly in wording.
