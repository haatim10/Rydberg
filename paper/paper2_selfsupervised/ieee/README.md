# IEEE-style Paper 2 draft

This is a seven-page reviewed research draft in the standard IEEEtran journal layout. It includes five vector figures, three tables, equations and eleven references. The user confirmed a general IEEE version with Abdullah Haatim as the sole author. It is not an unconditional scientific submission clearance: the closest XL-MIMO full-text comparison and author approval remain outstanding.

## Build or edit

Upload the source ZIP to Overleaf and choose `paper2_ieee_draft.tex` as the main document. Use pdfLaTeX. The included figure PDFs are sufficient; there is no need to run the figure-generation script.

With a local TeX installation that includes IEEEtran, run these commands in the source folder:

```sh
pdflatex paper2_ieee_draft.tex
bibtex paper2_ieee_draft
pdflatex paper2_ieee_draft.tex
pdflatex paper2_ieee_draft.tex
```

The bibliography uses IEEEtran.bst. A compiled `.bbl` is also included in the ZIP. Keep `baseline_rows.tex`, `references.bib` and `figures/` beside the main `.tex` file. The generated table uses the primitive TeX input command inside its alignment. A manual column break before the acknowledgment keeps the last page readable; adjust or remove it after substantial text changes.

## Evidence and figures

All plots use existing saved results. No new training, tuning or test evaluation was run to prepare this paper. The previous drafts, experiment drivers and results were left unchanged.

- Main accuracy, operating-limit and runtime plots: `results/sahure_expansion/summary.json`.
- Main configuration labels: `results/sahure_expansion/frozen_selection.json`.
- Controller-input comparison: `results/sahure_features/summary.json`.
- Follow-up results discussed in the text: `results/sahure_checks/summary.json` and `results/sahure_submission_review/review.json`.
- `figure_provenance.json` records SHA-256 hashes for the plot inputs.

`make_figures.py` is an optional regeneration script. It requires the original repository, its saved result files, NumPy and Matplotlib. It is not needed to compile the standalone paper.

The method diagram is original. Figures are numbered by their order in the paper, which differs from the filenames for the runtime and operating-limit plots.

## Before submission

- Sole authorship was confirmed. Confirm the affiliation and email, which follow the existing Paper 1.
- Choose the actual journal or conference and apply its current template, page limits and submission rules.
- Complete the closest-method full-text comparison, including the previously identified XL-MIMO/DeepTensor work. Do not add a first-ever or verified-novelty claim without that check.
- Keep Paper 1 labelled as an unpublished companion manuscript unless its status changes, and explain the distinction between the two papers.
- Retain the simulation-only qualification, the weak-reference/few-path failures, the failed input-study protection criterion and the limits of the seed-based uncertainty estimates.
- The comparisons do not reproduce or establish superiority over published URformer performance.
- Review and retain an accurate AI-use acknowledgment. The present disclosure covers text, mathematical exposition, code and figure preparation; it does not claim completed independent human verification.
- See `submission_review/REVIEW.md` in the source ZIP for the release conditions, and `submission_review/RELATED_WORK_CHECK.md` for the sources and access limitation. Do not treat a passed numerical audit as a completed novelty audit.

## Layout verification

The seven pages were rendered and visually inspected. The final build has no undefined citations, undefined references or overfull boxes. Mild underfull-box notices remain in justified mathematical prose/float layout; the rendered pages are readable. The final page was rechecked after the column-break change.

The compact code/results ZIP is separate from the manuscript source ZIP. Its saved-result calculation and three-checkpoint replay were tested from an independently extracted temporary directory. Full observation/prediction arrays are not included; the package README lists the omissions. No frozen experiments were changed.
