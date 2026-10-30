PAPER 1 REVIEWER REVISION RELEASE - 8 OCTOBER 2026

Start with paper/paper1/reviewer_revision/REVISION_REPORT.txt for the findings.
RESPONSE_TO_REVIEW.txt is an author-review draft; nothing was sent externally.
The manuscript folder's README.txt maps all 20 TeX fragments to PDF7 sections.

The smaller manuscript ZIP contains insertable TeX, five PDF/PNG figure pairs,
the reviewer response, report and insertion map. It is NOT the exact complete
Overleaf source or a submission-ready compiled full paper. The technical note
is a standalone preview of the core mathematical revisions only; its native
LaTeX compilation succeeded. The supplied historical TeX predates PDF7.

The evidence ZIP preserves repository-relative paths and contains the new
experimental module, all revision results, snapshots, relevant dependency
source files and selected historical raw inputs used in the reanalysis.
It is a focused evidence release, not a backup of the entire repository.
The smoke directory exists solely as a test fixture, not scientific evidence.
The earlier failed test log is retained for provenance; tests.log records the
correct completed 37-test run. Original P1 and P2 sources were not modified.

Scientific counts:
- Main revision: 12,000 condition-trial evaluations, 30 cells, 400 each.
  These use 4,800 independent base-world groups and 7,200 observation sets.
  Validation fractions reuse observations; mismatch strengths reuse the same
  underlying paths/pilots/noise while changing the actual channel.
- Update-budget check: 300 additional worlds, 100 versus 300 final updates.
- Serial timing: 20 worlds, not added to accuracy sample counts.
- RSR: 2,800 reused worlds from a separate classical-results campaign.
- Rank proxy: 10,400 EM-only replays from original P1; not new independent trials.

Main finding: empirical smooth filtering beats hard-rank HS-GS in five of six
new comparison settings. This narrows the interpretation of the original HS
boundary to its processing and selection procedure. No OptShrink, nuclear-norm
solver, learned method, universal threshold or globally optimal estimator is
claimed. The full report gives positive and negative results and uncertainty.

REPRODUCTION
Use a copy of the extracted evidence folder as the working directory.
The required numerical packages are NumPy, SciPy and Matplotlib. This release
was evaluated using Python 3.12.5, NumPy 2.4.6, SciPy 1.17.1, Matplotlib 3.11.2.
Set OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1,
VECLIB_MAXIMUM_THREADS=1, NUMEXPR_NUM_THREADS=1 before starting Python.
Set MPLCONFIGDIR to a writable cache directory.

  python -m unittest discover -s p1_revision -t . -p 'test_*.py' -v
  python -m p1_revision.audit
  python -m p1_revision.analyze
  python -m p1_revision.reuse rsr
  python -m p1_revision.reuse summary
  python -m p1_revision.summarize_budget

These checks and analyses use saved results; the audit also performs declared
small deterministic estimator replays. Regeneration overwrites summaries and
creation timestamps, so retain the original archive as evidence. Commands for
running the full simulations are in p1_revision/README.txt. Do not edit frozen
source and silently resume an existing manifest. Fresh main-campaign freeze
has a documented local PDF metadata-path dependency; saved-result audit and
analysis do not need the PDF. Full historical P1/P2 campaigns are not relaunched
by these commands. Bitwise equality across numerical-library versions and
hardware is not promised. Timings are machine-specific.

INTEGRATION
Keep the stronger-comparator result and its change to the conclusion visible
in the main paper. The report recommends which detailed sensitivity plots can
go in a supplement. The full current manuscript still needs integration,
consistent totals/captions and a full compile/layout review before submission.

SHA256SUMS.json inside each archive hashes every payload file (excluding
itself). Package metadata outside the archives records their sizes and hashes.
