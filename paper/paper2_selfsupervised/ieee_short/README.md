# Paper 2: short IEEE version

Main document: `paper2_ieee_short.tex`.

Upload this ZIP as a new Overleaf project, select that main document, and
use pdfLaTeX. The project uses the standard IEEEtran journal class (10 pt,
two columns), without altered margins or reduced body font size. It is a
general IEEE layout, not a selected venue's final template.

The checked PDF has five pages, including references, four figures and
three tables. The former processing-time chart is included as
`figures/fig4_runtime.pdf` but is not inserted: its numbers remain in Table II
and the timing paragraph. Figure filenames retain their original names;
LaTeX numbers the four displayed figures automatically.

The source bundle contains the TeX, bibliography, compiled bibliography,
table rows and all five original vector figures. No external path is needed.
Local build: pdfLaTeX, BibTeX, pdfLaTeX, pdfLaTeX.

## What changed

- Removed the research-draft running header, draft footnote and main-paper
  Codex acknowledgment as requested.
- Shortened the prose and added a plain explanation of the two estimator
  steps and a hard-versus-soft singular-value example. The example is an
  illustration, not new experimental data.
- Defined self-supervision before using the term and explained severe errors.
- Gave the 0.45% versus 14.25% result its own paragraph.
- Explained why follow-up gains differ from the main-study gain.
- Removed the conference-program-only reference and defensive prose.
- Preserved the numerical results, all three tables, plotted data, reported
  failures and the full limitations paragraph. Figures are unchanged files.
- Updated the companion manuscript title to match `ThesisProj-7.pdf`.

The old seven-page version and all experiments remain unchanged.
Read `SUBMISSION_NOTES.md` before submitting. In particular, a matched
URformer comparison and public access to the companion baseline remain open.

The existing separate reproducibility archive is not duplicated in this
Overleaf ZIP. It is needed to inspect/replay experiments, not to compile the paper.
