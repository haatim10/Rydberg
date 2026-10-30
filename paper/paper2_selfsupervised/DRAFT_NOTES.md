# Notes for the plain-English draft

23 September 2026. `plain_english_draft.md` is the new readable draft. The earlier technical draft is retained as `manuscript.md`. Neither file replaces the user's original thesis manuscript. No method, checkpoint or experiment was changed while writing.

## Writing choices

The abstract is qualitative and follows background, method, results and conclusion. The introduction has one background paragraph, one motivation paragraph, then contributions. Technical terms are explained when introduced. The text avoids section roadmaps and forward references. Numbers are kept mainly in the results. A draft conclusion is now included because the paper's claim has been narrowed; it is not a claim that submission review is complete.

## Evidence sources

| Draft content | Saved evidence, relative to repository root |
|---|---|
| Main comparison, SNR table, processing time and failure cases | `results/sahure_expansion/REPORT.md`, `summary.json`, `audit.json` |
| Independent-data training, supervised/final-only/fixed-gate checks | `results/sahure_checks/REPORT.md`, `summary.json`, `audit.json` |
| Supervised selection and objective qualifications | `results/sahure_submission_review/REVIEW.md`, `review.json` |
| New controller-input comparisons and protection failure | `results/sahure_features/REPORT.md`, `summary.json`, `audit.json`, `PLAN.md` |
| Primary model and seed decision | `paper/paper2_selfsupervised/MODEL_DECISION.md` |
| Prior work and unresolved full text | `results/sahure_submission_review/NOVELTY.md`, `results/sahure_features/LITERATURE_STATUS.md` |

## Scope and remaining editorial work

- This is a readable research draft, not a submission-ready document. No new experiment is required merely to improve its language.
- Preserve the main method selected by validation. The smaller feature study is supporting evidence, not a reason to choose a new method after seeing test results.
- Keep the SNR=5 loss, two-path losses and failed protection criterion in the results. Do not describe the smaller study as passing all success criteria just because its numerical audit passed.
- The full-text comparison with the XL-MIMO DeepTensor conference paper remains unresolved. Do not insert a first-ever claim or an invented description of that method.
- Convert source links to the target journal's bibliography format and cite the companion Paper 1 with its actual publication status. Do not assume that it is published.
- For a submission version, retain exact implementation definitions from `sahure/MATHEMATICS.md` in a concise technical appendix: initialization, Rician update, threshold scaling, the eight controller inputs and stable matrix solve. The plain draft explains the method but is not a complete reproducibility specification by itself.
- Insert the audited existing figures and a training-cost statement. Do not combine timing measurements from different studies to obtain a larger speedup.
- The manuscript makes no new claim about the original URformer reproduction, the cancelled recalibration project, or Fisher-information filtering.
