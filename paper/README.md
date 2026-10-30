# Manuscripts — current index, 30 October 2026

## Current merged TCOM draft

The current version is the **21 October 2026, 10-page edition** in
[`tcom_reliability_10page_20261021/`](tcom_reliability_10page_20261021/README.txt).
Its title is *Channel Estimation for Rydberg Arrays With Learned Filtering and
Recovery Checks*. This is a draft, not a submitted or accepted paper.

- [Main PDF](../output/pdf/rydberg_merged_tcom_10page.pdf)
- [Supplement PDF](../output/pdf/rydberg_merged_tcom_10page_supplement.pdf)
- [Overleaf ZIP](../output/pdf/rydberg_merged_tcom_10page_overleaf.zip)
- [Editable main.tex](tcom_reliability_10page_20261021/main.tex)
- [Build instructions and scope](tcom_reliability_10page_20261021/README.txt)
- [Plain-English paper guide](../docs/sections/05-paper/README.md)

Upload the ZIP to Overleaf, choose pdfLaTeX and `main.tex`, and use
`supplement.tex` for the supporting document. Figures and evidence are included.
The later URformer reproduction, matched comparison and two audits remain
separate records; they have not been added to the manuscript in this archive.

## Version map

| Directory | Status |
| --- | --- |
| `tcom_reliability_10page_20261021/` | Current 10-page merged draft, simpler English and cleaned figures |
| `tcom_reliability_merged_20261021/` | Earlier longer version of the reliability manuscript |
| `tcom_training_update_submission/` | Earlier pilot-matched/retraining version |
| `tcom_10page_system_submission/`, `tcom_10page_submission/` | Earlier 10-page versions |
| `merged_compact_submission/`, `merged_submission/`, `merged_tcom/` | Earlier merged drafts |
| `paper1/full_revision/`, `paper1/reviewer_revision/` | Earlier Paper 1 revisions |
| `paper2_selfsupervised/` | Earlier separate learned-estimator drafts |

Original source directories and their own READMEs are preserved. The `_submission`
directory names do not mean those drafts were submitted. No manuscript source,
figure or experimental number was changed during GitHub organization.

## Historical September index

The text below is the earlier index, retained for provenance. Its submission
labels describe that earlier plan and are superseded by the current index above.

Two papers are for submission; two directories are superseded and kept only
because they are the sources the merge drew on.

| directory | file | pages | status |
|---|---|---|---|
| `merged/` | `haatim_hsgs_bounds_and_effective_rank.tex` | 5 | **submit** |
| `spl2/` | `haatim_structural_priors_evaluation.tex` | 4 | **submit** |
| `spl1/` | `main.tex` | 4 | superseded — absorbed into `merged/` |
| (root) | `hsgs.tex` | 6 | superseded — the spine of `merged/` |

Build: `pdflatex X && bibtex X && pdflatex X && pdflatex X`.

Full written content of both papers, with every number's source and every open
item: `../reports/PAPERS_FULL_CONTENT.md`.
Merge inventory and the seven reconciliation decisions:
`../reports/merge_decisions.md`.
