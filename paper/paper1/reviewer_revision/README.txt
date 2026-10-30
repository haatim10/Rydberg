PAPER 1 REVIEWER REVISION FRAGMENTS
Prepared against the reviewed PDF ThesisProj-7.pdf, 7 October 2026.

STATUS AND SOURCE QUALIFICATION

These are insertable technical, scope and result revisions, not a rebuilt
full manuscript. The later result fragments use the completed audited
reviewer-response studies and clearly identified reuse of earlier results.
The historical manuscripts and simulation implementations were not edited.

The reviewed manuscript is:
/Users/haatim/Downloads/ThesisProj-7.pdf
Its extracted text was used to identify the headings and paragraph anchors.

The newest supplied complete TeX found in this conversation is:
/Users/haatim/.codex/attachments/5b8bfc73-d2aa-43da-8725-19e150fa2a26/pasted-text.txt
It contains the independently verified numerical values, but predates at
least the all-pilot-refit clarification in ThesisProj-7.pdf. It is therefore
not represented as the exact source of the reviewed PDF. It was used only
to check existing command names and bibliography keys. The older local
paper/paper1/haatim_hsgs_letter.tex and downloaded letter were not silently
substituted for ThesisProj-7.

INSERTION MAP

01_estimator.tex
  Replace Section III-B, "The Estimator", in full: start at its subsection
  heading and stop immediately before Section III-C, "Choosing the Hankel
  Rank". This supplies the actual EM update, per-user Hankel construction,
  balanced pencil, four sweeps, and the finite-sweep qualification.

02_rank_selection.tex
  Replace Section III-C, "Choosing the Hankel Rank", in full, including
  its final all-pilot-refit paragraph. Stop before Section IV, "Performance
  References". The box uses standard LaTeX; no algorithm package is needed.
  The final full-rank fit is exactly the paired EM-GS fit, but active
  selections have no per-trial no-harm guarantee. The deterministic last-
  column split and smallest-rank tie handling match the implementation.
  Candidate and final update budgets remain the existing Table I values.
  The implementation's round function uses ties-to-even; the P=10,20,30
  configurations have no rounding tie at the stated 30% fraction.

03_ccrb_body.tex
  Replace Section IV-B, "Constrained Cramer--Rao Bound", in full. Stop
  before Section V, "Numerical Results". This replacement preserves the
  distinction between ideal local references and measured biased-estimator
  errors. Use together with 04_ccrb_appendix.tex.

04_ccrb_appendix.tex
  Insert after Section VI, "Limitations and Conclusion", and before
  \begin{thebibliography}. It includes \appendices for IEEEtran. If an
  existing appendix has already invoked that command, omit the command
  here and retain the appendix's section and contents. The derivation uses
  exactly the real-coordinate ordering, tangent derivatives, scalar
  Fisher weight, SVD tolerance and ratio-of-sums normalization in the
  independently verified code.

05_related_work.tex
  Insert in Section I immediately before the paragraph beginning
  "We propose a Hankel-structured extension of EM-GS...". Keep the existing
  introduction. Add the bibliography entry in 09 at the same time.

06_effective_rank_scope.tex
  Insert in Section V-E, "Effective Rank", immediately after the displayed
  definition rho = r_eff/r_max and before "The value rho can be understood
  as the fraction...". This states the actual multi-user aggregation.
  Keep the later paragraph explaining that rho is unavailable to the
  receiver. No measured-rho proxy or universal threshold is claimed here.

07_bound_gap_interpretation.tex
  In Section V-B, replace the two adjacent paragraphs beginning
  "The geometric CCRB is approximately..." and
  "HS-GS uses part of this structure...". Stop before
  "The CRLB and CCRB are reference curves for unbiased estimators."
  The numerical values are unchanged; only attainability language changes.

08_placement_scope.tex
  Append to the first paragraph of Section V-G, "Projection Placement and
  Computation", after its 300-trial/four-sweep sentence. This explains how
  the common rank was obtained and distinguishes sweeps per application
  from total work. Existing gains, intervals and timing values are unchanged.

09_bibliography_addition.tex
  Insert inside the existing thebibliography environment, before its end.
  The other citation keys in these fragments are already present in the
  supplied full TeX. The new OptShrink entry points to the IEEE publication
  through DOI 10.1109/TIT.2014.2311661. Publisher page metadata can be added
  from an IEEE export without changing the scientific text.

10_rsr_results.tex
  Add a new Section V subsection, "Effect of Reference Strength", after
  the original "Projection Placement and Computation" subsection and
  before the new structured-comparison subsection (14). The 2,800 worlds
  are reused from a separate earlier campaign, not newly drawn and not
  part of the original 27,300-trial P1 campaign. Copy fig/rsr_classical.pdf
  into the manuscript's fig directory.

11_observable_rank.tex
  Add a new subsection immediately after the original "Effective Rank"
  subsection and before "Additional Results". Preserve the existing true-
  channel effective-rank results. This adds the EM-derived proxy, including
  its upward noise shift and the absence of a validated switching rule.
  Copy fig/rho_proxy_diagnostics.pdf into the manuscript's fig directory.

12_normalized_path_control.tex
  Optional paragraph addition at the end of the original "Effective Rank"
  subsection, after its closing paragraph explaining that true rho is an
  analysis quantity and after its figure environment. Insert before the
  new observable-rank subsection (11). Alternatively place it in the
  supplement. It restores the existing normalized-path-count comparator;
  it does not contain a new reviewer-campaign simulation.

13_iteration_budget.tex
  Append to the new structured-comparison subsection (14), after the
  paragraph ending "how its parameter is selected" and before its figure.
  Alternatively place it in the supplement and cite that check from 14.
  These 300 additional condition-trials form a separate fixed-budget
  check, not part of the main 12,000 condition-trials. It keeps selection
  fixed and does not claim global convergence or equivalence.

DEPENDENCIES AND PREVIEW

The fragments use amsmath and amssymb, already loaded in the supplied TeX.
They do not depend on manuscript-specific bold-symbol macros. Labels are
prefixed with revision/revised to avoid replacing existing figure labels.
The appendix uses the IEEEtran \appendices command.

technical_revision_note.tex is a self-contained, single-file preview of
these additions. It embeds their current text and the required bibliography;
it has no external figures or input files. It is not a replacement for the
complete manuscript or a reconstruction of its layout. The preview adapts
\appendices to the standard article class and is compiled using the Codex
built-in LaTeX compiler. Edit/copy the individual fragments for Overleaf.

IMPLEMENTATION REFERENCES (RELATIVE TO REPOSITORY ROOT)

codex_verification/solver.py:
  geometry, project, fit, select, estimate, effective_ranks.
codex_verification/bounds.py:
  tangent, fisher, compute, validate and the interpolation implementation.
rydberg_sim/crlb.py:
  rician_fisher_scalar and its quadrature helpers, not its separate legacy
  complex-Fisher summary. The appendix follows the real-coordinate Fisher
  matrix actually used by codex_verification/bounds.py.
codex_verification/analyze.py:
  summary for the user/trial median rho and the ratio-of-sums bound curves;
  ci keeps users from one trial together during effective-rank resampling.

No claim of an unknown-noise/calibration bound, a universally optimal
shrinkage rule, or guaranteed monotonic HS-GS improvement is introduced.

NEW REFERENCE CHECK

The author's published-version record was checked for OptShrink:
https://arxiv.org/abs/1306.6042
IEEE publication DOI:
https://doi.org/10.1109/TIT.2014.2311661
The existing Xu comparison is limited to the distinction between a
per-user spatial Hankel lift and a rank constraint on 2D channel slices.
No new performance comparison is asserted by the related-work fragment.

ADDITIONAL AUDITED EXPERIMENT FRAGMENTS (8 OCTOBER 2026)

The following files use the completed reviewer campaign's analysis only
after its independent audit passed: 30 conditions, 400 trials per condition,
12,000 condition-trials from 4,800 underlying worlds. Pilot-split fractions
and position-error strengths intentionally share their base worlds. Do not
describe those 12,000 condition-trials as 12,000 independent realizations.

14_structured_comparisons.tex
  Add a new subsection inside Section V, after the original Section V-G,
  "Projection Placement and Computation", and before Section VI. This
  includes reproducible Rician-GD and empirical smooth-filter definitions,
  separate held-out parameter selection, and the six-method result.
  Use 01_estimator.tex as well: the new section references its definition
  of the Rician conditional field. It openly reports that smooth filtering
  beats HS-GS in five settings and that the HS-GS failure region is not a
  universal failure region for every use of Hankel structure.

15_heldout_fraction.tex
  Add as the next Section V subsection, directly after 14. It retains
  the deterministic trailing-column folds and all-pilot final fit, and
  reports split sensitivity, an example of changed selected ranks, and
  the observed fallback range. It does not retune the original paper's
  30% split against these test results.

16_position_mismatch.tex
  Add as the next Section V subsection, directly after 15 and before
  Section VI. The target is the actual perturbed G, with known calibrated
  B. The stated offset levels are declared stresses, not hardware
  specifications. It does not claim general near-field, coupling,
  correlated-noise or unknown-reference robustness.

17_serial_timing.tex
  Append inside the existing Section V-G, after its paragraph ending
  "They are used here only to compare the computational cost of the two
  approaches." Keep the preceding placement timings unchanged. This is a
  separate six-method serial timing experiment using vectorized comparator
  code at P=30, RSR=12 dB, L=7. Its numbers must not be substituted for
  the older production-code P=20, RSR=10 dB placement timings, or divided
  by the older numbers to claim an algorithmic speedup.

18_conclusion_scope.tex
  Insert in Section VI immediately after the paragraph beginning
  "The results also show that forcing a low-rank constraint can be harmful"
  and ending "to EM-GS when the maximum rank is selected."
  This keeps the original 2.70 dB paired-median main result and explicitly
  narrows the general interpretation in light of the stronger comparator.
  Keep the next paragraphs on additional distributions and placement;
  replace the original final limitations paragraphs using 19.

19_limitations_update.tex
  Replace the final three paragraphs of Section VI, from
  "Limitations. All results in this work are obtained from simulations."
  through the paragraph ending "not assumed to be unbiased."
  Stop before the new appendix or the bibliography. The original claim
  that no departure from the ideal ULA has been tested is now outdated;
  this replacement states exactly which departure was tested. It also
  acknowledges the new EM-derived proxy without calling it a decision rule.

20_abstract.tex
  Optional replacement of the entire existing abstract environment, from
  \begin{abstract} through \end{abstract}. Keep the title, author and index
  terms. Use this version when the new comparison/sensitivity sections
  are included. It is qualitative and makes the method-dependent scope
  visible at the start of the paper.

ONE SHORT BODY SENTENCE TO UPDATE

  In the original "Effect of Channel Complexity" subsection, after the
  reported L=16 gain, replace
    "This means that the Hankel constraint no longer helps."
  with
    "At this operating point, the measured HS-GS NMSE is higher than
     the EM-GS NMSE."
  This retains the reported result while avoiding a claim about every
  possible Hankel-based estimator. The new conclusion in 18 explains why
  that distinction matters. Other original rho statements should retain
  their explicit HS-GS context and approximate, nonuniversal wording.

FIGURE FILES FOR 14--16

Copy these files into the manuscript's existing fig directory, retaining
their names. The PDF copies are exactly the audited analysis plots; the PNG
copies are supplied for quick viewing.

  fig/revision_structured_baselines.pdf  (also .png)
  fig/revision_heldout_fraction.pdf      (also .png)
  fig/revision_position_mismatch.pdf    (also .png)

These fragments additionally use graphicx and booktabs, both present in
the latest supplied full TeX. Let LaTeX assign subsection, table and figure
numbers automatically. Their revision-prefixed labels avoid collisions.
The self-contained technical_revision_note.tex is the earlier technical
preview, not a full preview of these later experiment/figure additions.

NUMERICAL SOURCES, RELATIVE TO REPOSITORY ROOT

  results/p1_revision/campaign/analysis/summary.json
  results/p1_revision/campaign/analysis/split_comparison.csv
  results/p1_revision/campaign/audit.json
  results/p1_revision/campaign/raw/holdout_*/  (selected-rank medians)
  results/p1_revision/serial_timing/summary.json
  results/p1_revision/serial_timing/COMPLETE.json

result_fragment_sources.json records the source and delivered-figure
hashes. New comparison gains are ratio-of-sums gains with pointwise paired
intervals; they must not be confused with the original main result's mean
of six paired medians. No result from the incomplete smoke run is used.
