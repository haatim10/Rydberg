Paper 2 second review revision

General IEEE-format revision, 8 October 2026. The older reviewed release
is preserved. The release manifest records the completed numerical,
rendered-page and archive-compilation checks. The point-by-point response
states both completed changes and unresolved scientific limitations.

Compile paper2_revised.tex or paper2_supplement.tex with pdfLaTeX, BibTeX,
then pdfLaTeX twice. Both use local figures and references.bib. The
supplement reads baseline_rows.tex, review_diagnostics.tex,
round2_results.tex, round2_tables.tex, round2_training.tex and
round2_training_tables.tex.
The native single-file editor may not load these project dependencies.

make_material.py redraws the second-round figures from saved results.
Original figure scripts and their sources are retained in the scientific
archive. package_release.py refuses to issue a final ZIP until all new
training, tests and timing finish and a manual rendered-page audit has
been recorded. It then checks that the ZIP compiles independently.

This is a general IEEE-format version, not a venue-approved submission.
Author details, companion-paper availability and the target venue's AI
disclosure requirements must be checked before submitting.

For Overleaf, upload the entire ZIP and choose paper2_revised.tex as the
main document with pdfLaTeX. Choose paper2_supplement.tex to build the
supplement. The main paper is five pages; supporting details are separate.
The standalone .tex download also needs the bibliography and figures from
the ZIP. The separate full scientific archive is for data/code replication,
not for uploading to Overleaf or as the main submission manuscript.
