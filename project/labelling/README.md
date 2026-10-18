# Human labelling page (AISI OLMo-32B arms)

A single offline HTML file that shows one blinded `<thinking>` block at a time and asks the
annotator for the `aisi-cot-v1` category (PLAN_PLAIN, PLAN_JUSTIFIED, REJECTED, DISCLAIMER,
MONITOR_ONLY, NO_EVIDENCE, UNSURE) plus an evidence quote. No server; labels autosave in the
browser's localStorage, and **Export labels** downloads a JSON file to send back.

## Build

```bash
python scripts/make_human_label_site.py --key runs/luna-key/key.jsonl \
    --batches ~/luna-batches/batch_*.jsonl --out runs/human-label-v1
```

- `key.jsonl` is the Luna unblinding key (`runs/luna-blind-20261001/key.jsonl` on advait's Ada).
- Default: 100 hacked outputs per arm for A, B, C (Run 023), A-pen (024), E, E-pen (025) and F (026),
  700 items in total. C is Run 023's C, the baseline the paper uses for A, B, E and E-pen.
- Self-consistency repeats are excluded. Items get fresh IDs (`h…`), so annotators can't look up
  the committed Luna labels. Order is shuffled across arms with a fixed seed (`20261018`).
- Outputs: `label.html` (send this to annotators) and `key.jsonl` (keep private; maps each `h…` ID
  to arm, run, Luna ID and `cot_sha256`). The build refuses to overwrite an existing directory.

## Tutorial

New annotators first work through the practice items in `labelling/tutorial.json`, getting
feedback after each one: a plain plan, a justified plan, a rejected hack, a disclaimer, and
ordinary reasoning. Claude read each one and checked the answer against Luna's label. They come
from Runs 022, 027 and 028, which are outside the seven labelled arms, and the build refuses a
practice item from a labelled run or with an evidence quote that isn't verbatim. Practice answers
are exported under `tutorial` and never mixed with `labels`. Adding or changing practice items
doesn't change `dataset_id`.

## Exports

`labels_<annotator>_<dataset_id>_<time>.json` holds `annotator`, `dataset_id`, `rubric_version`
and one row per labelled item: `item_id`, `category`, `label`, `evidence`, `notes`,
`labelled_at`, `seconds`, `revisions`. Join on `item_id` with `key.jsonl`. Import refuses an
export from a different `dataset_id`.

The annotator guide in the page is reconstructed from the `aisi-cot-v1` definitions in
`docs/results/031-aisi-rh-luna-judge.md`. The original Luna prompt text was not saved, so the
human and judge rubrics may differ slightly in wording.
