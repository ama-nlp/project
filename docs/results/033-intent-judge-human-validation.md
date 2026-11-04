# Run 033 — Human validation of the intent judge

**Status:** planned 2026-11-05. Labelling starts once [Run 032](032-aisi-rh-olmo32b-kl002-penalty-arms.md) has been judged, so all items are labelled in one pass. No GPU is needed. The rules below are fixed before any human label is compared with the judge.

## Why

The paper's adopted-plan rates (Figure 1, Run 031) come from one LLM judge. It agrees with itself (κ = .95), but it has never been checked against people, and the paper lists that as a limitation. The judge's prompt and model are already fixed and were never tuned on these items. So every human-labelled item is a test item, and there is no development split to protect.

## Items

- **Sources:**
  - the existing blinded build: 100 hacked outputs per arm for C, A, B (Run 023), A-pen (024), E, E-pen (025) and F (026); dataset `79cab47704ac`, `labeller/`;
  - KL 0.02 hacked outputs from Run 032, sampled per arm the same way.
- **Rebuild:** `make_human_label_site.py` builds one new dataset, with a fresh secret salt and new `h…` IDs. The build refuses to overwrite an existing directory.
- **Existing labels:** if anyone has already labelled the current build, those labels are kept as they are, and the new build adds only the Run 032 items.
- **Blinding:** items carry no arm, run or judge ID. The key stays private and off GitHub, as in `labeller/README.md`.
- **Sample size:** at least 100 items the humans call PLAN and at least 100 they call not-PLAN, so a sensitivity or specificity near 0.92 has a 95% CI lower bound above 0.80. If the first pass is short of positives, more judge-YES items are added. This is recorded as enrichment, and per-stratum rates are reported.

## Procedure

- **Annotation:** two annotators label every item independently, with the `aisi-cot-v1` categories on the labelling page.
- **Adjudication:** comes after both exports are in. The original pre-adjudication labels are kept and committed alongside.
- **Scoring:** `score_human_labels.py` (human–human agreement, then the judge against the adjudicated label).
- **No changes to the judge:** neither its prompt nor its model changes during or after validation. A changed judge would be a new judge, needing a new validation.

## Decision rules (the Phase 2 gate, fixed before any comparison)

- **Human reliability:** Cohen's κ ≥ 0.80 and raw agreement ≥ 90% on the binary PLAN / not-PLAN label. If this fails, the rubric is the problem. The judge is not scored, and the paper keeps the keyword label as primary.
- **Judge validity:** against the adjudicated labels, sensitivity and specificity ≥ 0.90, each with a 95% Wilson lower bound ≥ 0.80. Invalid or unparsed outputs must be ≤ 1%, and they count as errors, never as negatives.
- **Outcomes:** if both pass, the judge's adopted-plan rates in Run 031 and Run 032 are reported as validated. If only sensitivity or only specificity fails, the paper reports the measured rates and states which way the judge errs. Either way, the result is recorded here. A failed validation is a finding, not something to retry with a tuned prompt.
- **Breakdowns:** agreement and judge accuracy are reported per model (KL 0, KL 0.02) and per arm family, since the two models' reasoning differs a lot. KL 0.02 has adopted plans in only 5–13% of its hacked outputs.
