PAPER A — HS-GS: Hankel-Structured Gerchberg-Saxton for Rydberg
   Atomic Receivers  (revision of the author's hand-written ThesisProj draft)
===============================================================================

Main file:  haatim_thesisproj.tex
Class:      IEEEtran (journal), built in with Overleaf
Build:      pdflatex twice.  No bibtex -- the bibliography is a manual
            thebibliography block inside the .tex.
Result:     10 pages, zero undefined references, zero errors.

Figures (all in fig/):
  fig0_system.tex             system diagram, drawn in TikZ (\input, not
                              \includegraphics -- keep the .tex file)
  fig1_twopanel.pdf           two-panel main result
  fig7_boundary_invariance.pdf boundary-invariance check

Overleaf: upload this whole folder (or the zip) as a new project and set
haatim_thesisproj.tex as the main document.  TikZ is already loaded in the
preamble, so fig0_system.tex compiles without extra packages.


What changed from the hand-written draft
----------------------------------------
Your sentences are kept wherever the external review did not require a
change.  Everything added or edited is marked in the SOURCE with a margin
comment -- search the .tex for

    % [NEW]     something that was not in your draft
    % [EDIT]    one of your sentences, changed

so you can see the two apart at a glance.  The header comment at the top
of the .tex lists each one with its reason.  In short:

  [EDIT]  Eq. (9) had no left-hand side.  Added Y.
  [EDIT]  Sec. IV-A used q and y for the same imaginary part.  Now y.
  [EDIT]  Sec. III-B called the Cadzow step a projection.  It is not --
          it is not even idempotent, and the draft now shows the number
          that proves it.
  [EDIT]  Sec. III-C said 70% of the pilots estimate the channel.  The
          code refits on ALL pilots once the rank is chosen.  This is
          what makes the comparison against EM-GS fair, so it matters.
  [NEW]   Sec. II states the path-gain normalisation, so the path-count
          sweep is at fixed received power.
  [NEW]   Sec. II states that this is one Rydberg architecture, not the
          family -- a superheterodyne readout recovers phase and none of
          this applies to it.
  [NEW]   Sec. V-F names the second channel distribution.
  [NEW]   Sec. IV-B says what the CCRB curve does and does not specify.

Full detail: reports/p22/review_response_paperA.md in the repository.


Still open on this paper
------------------------
The CCRB section is the largest outstanding item.  U(U'JU)^-1 U' is not a
reproducible specification -- it needs the parameter vector, the
likelihood, the derivatives, the dimensions and the NMSE normalisation
written out.  If that derivation does not get written, the honest move is
to remove the curve rather than leave an unauditable theory section.
