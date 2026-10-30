MERGED TCOM MANUSCRIPT — 21 OCTOBER 2026

Title: Structured Channel Estimation for Rydberg Arrays With Learned
Filtering and Observable Recovery Checks
Author: Abdullah Haatim (author details retained from the supplied paper).
Main manuscript: 13 pages. Supplement: 27 pages.

This is one merged journal manuscript combining the classical structured
estimation and compact learned-controller studies with the completed
pilot-matched training, sparse comparison, BER, transition, filtering,
runtime, matched-start and final recovery experiments.

COMPILE
Upload this ZIP to a new Overleaf project. Select pdfLaTeX and main.tex as
the main document; choose supplement.tex to build the supporting document.
Locally, run pdflatex, bibtex, pdflatex, pdflatex for each root document.
No shell escape, downloads, experiment regeneration or external files are
needed. Included figures, bibliography, support and evidence folders must
be retained. The main uses IEEEtran journal format, 10-point text and
one-inch margins. The supplement is supporting material, not a second
submission. The supplied PDFs were checked separately from the source ZIP.

WHAT IS CURRENT
The final receiver pairs an adaptive 188-parameter learned start with a
30-parameter global schedule, giving 218 stored neural parameters per
paired fit. It preserves the ordinary-SNR estimate, uses a frozen
high-SNR continuation/filter table, and adds residual-triggered algebraic
recovery with likelihood acceptance. Neural components learn from
withheld magnitudes without true-channel targets; offline policy selection
also uses known-symbol probes. Only the newly added recovery rule received
its stated independent confirmation. Earlier failed confirmation results
are explicitly retained and are not retrospectively validated.

The fresh final comparison has 153 conditions, 1,152 independent physical
anchors and 19,584 cases coupled across SNR, with three separately scored
trained fits. It includes matched-pilot NMSE, a common magnitude-domain
QPSK detector and two complete-inference timing windows. Earlier banks are
labelled historical and are never pooled into this final comparison.

EVIDENCE AND REPRODUCTION
The evidence folder contains compact protocols, policies, hashes,
validation records and exact numerical sources for the included tables
and figures. Large raw simulation banks and neural checkpoint binaries
are not included. This is a self-contained manuscript build, not a full
public experiment-code/data release. Reporting scripts may require the
original local repository; they are not invoked by Overleaf. No archival
DOI or public release is claimed. Read REVISION_NOTES.txt for claim limits.

Frozen code, data, checkpoints and earlier manuscripts were not edited.
No new numerical experiment was launched for this manuscript revision.
The manuscript has not been submitted. Author details and the final
scientific text still require the author's review before submission.
