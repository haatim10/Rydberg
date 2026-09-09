# Open items — things the manuscripts do not currently claim

Created under PROMPT 14 Part A. **Nothing here was silently deleted.** Each
entry names something that was removed from, or deliberately left out of,
`paper/paper1/haatim_hsgs_letter.tex`, and says what would be needed to put it
back. The container has no network access to IEEE Xplore, arXiv or Scholar, so
none of these can be closed from here.

The rule that produced this file: a fabricated reference is worse than a
missing one. Where a claim needed a citation we could not verify, the claim was
folded away or dropped, never left as a bare assertion and never given a
guessed source.

---

## 1. Toeplitz / Vandermonde covariance-structure methods (§II, related work)

**Status: claim removed from the letter.**

The related-work paragraph previously carried a third sentence about
covariance-domain structure (Toeplitz covariance estimation, Vandermonde
decomposition of a Toeplitz matrix) as a separate line of prior work. No
verified citation exists for it in this repository, and PROMPT 14 A2 forbids
guessing one.

The point it made — that those methods also assume *linear* measurements — is
now folded into the sentence covering EMaC / ALOHA / LORAKS and the gridless
atomic-norm work, which says of both families: *"Both families differ from the
present setting in the same way: our measurement is a modulus."* That sentence
is true of covariance methods as well, so nothing false is asserted and nothing
is claimed without a source.

**To close:** source the Toeplitz/Vandermonde line from Xplore (the natural
candidates are the Carathéodory–Toeplitz / Vandermonde-decomposition results
used in gridless DoA), add the entries to `paper/paper1/refs.bib` and
`thebibliography`, and restore the sentence as a third clause.

## 2. root-MUSIC and ESPRIT (§II, "Why not estimate the angles directly?")

**Status: methods named, no citation attached.**

The paragraph names root-MUSIC and ESPRIT to explain why we regularise toward
the exponential-sum manifold instead of parameterising it. Both are standard
enough that naming them without a citation is defensible, and the argument does
not depend on any specific result in those papers — only on the fact that they
need the signal subspace, which needs either multiple snapshots or the complex
field. A `\todo{VERIFY CITATION}` marker previously sat here and has been
commented out rather than rendered.

**To close:** verify and add Schmidt (MUSIC), Barabell or Rao–Hari (root-MUSIC)
and Roy–Kailath (ESPRIT) from Xplore. Purely cosmetic; no claim changes.

## 3. Placement asymmetry between the classical and the trained loop (§IV-F)

**Status: sentence deliberately omitted.**

PROMPT 14 D4 asked for one sentence contrasting the classical placement result
(interleaved vs post-hoc is a wash: six-point mean `+0.003` dB,
`results/p12/B5.json`) with the trained-loop result, where internal beats
post-hoc by **+1.309 dB** at 15–20 dB
(`reports/trackD_stage3_report.md:117`, confirmed in `reports/p12/PART_A.md:103`,
CI `[+1.185, +1.459]`).

The instruction was to cite the companion result as work by the same author
*under review or on arXiv, whichever is accurate at submission time; if neither
is yet true, leave it out rather than forward-referencing something
unavailable.* **Neither is yet true** — the URformer evaluation letter
(`paper/spl2/`, `paper/digest_urformer/`) is not submitted and not posted. The
sentence is therefore left out.

**To close:** once the companion letter is on arXiv or under review, add one
sentence to §IV-F with the citation. The number and its source are recorded
above so it can be inserted without re-deriving anything.

## 4. Superseded drafts still carrying `\todo{}` markers

`paper/spl1/main.tex` is the pre-merge draft, superseded first by
`paper/merged/` and then by `paper/paper1/`. It still renders three markers:

| Line | Marker | Status in `paper/paper1/` |
|---|---|---|
| 34 | `CONFIRM DEPARTMENT` | **Resolved** — Department of Electrical and Electronics Engineering, BITS Pilani, Pilani Campus |
| 219 | `VERIFY CITATION` on Roy–Vetterli | **Resolved** — EUSIPCO 2007, pp. 606–610, verified in an earlier session |
| 334 | `FILL IN AUTHORS of arXiv:2408.14366` | **Open** — the paper is *MIMO Precoding for Rydberg Atomic Receivers*; the author list was never verified and the attribution it was originally used for was wrong. `paper/paper1/` does not cite it. |

`paper/spl1/` is not a submission target and is kept unchanged as a record of
that stage. Only `paper/paper1/haatim_hsgs_letter.tex` is gated by
`scripts/submission_gate.sh`.

---

## Not an open item: the `n_cz` reading in the PROMPT 14 brief

Recorded here because it looks like an unresolved discrepancy and is not.

PROMPT 14 B1 states that "everything in §IV-D and §IV-E comes from `n_cz = 1`".
That is not what the code does. `scratch/trackD_partB9_sweeps.py:171` calls
`hs_gs_auto(...)` without passing `cadzow_iter`, and the default in
`rydberg_sim/track_b_proposed.py:246` is `cadzow_iter=4`. The same is true of
`scripts/run_b3.py:145` and of `trackB_hankel_emgs` (`config.py:24`,
`CADZOW_ITER = 4`). **All four configurations in Table I therefore use
`n_cz = 4`**, and the configuration table says so.

This is the same class of error as the "B3 and TD are different operators"
premise corrected under PROMPT 12 A1: the families differ in *operating point*
and in the *reported statistic*, not in the Cadzow sweep count. The rest of
B1 is correct — `T` and the selection budget genuinely do differ across
subsections, and the blanket sentence in §II has been deleted accordingly.

## 5. Balan citation fields (Section IV-A)

**Status: cited, fields not verified.**

PROMPT 18 D3 requires ceding priority on the rank-one-per-measurement result to
Balan and to Cui *et al.* The Cui citation is already reference [1] and is
solid. The Balan entry reads:

> R. Balan, "The Fisher information matrix and the Cramér–Rao bound in a
> non-AWGN model for the phase retrieval problem," in *Proc. Int. Conf.
> Sampling Theory Appl. (SampTA)*, 2015.

The author, the result and the venue class are right. **The year, the exact
title wording and the page range are not verified** — this container has no
network access and the paper is not in the repository. The brief asked for a
`\todo{VERIFY CITATION}` marker, but PROMPT 14 A3 removed the `\todo` macro
precisely so that nothing unresolved can render, and gate S1 fails on any
rendered TODO. Recording it here instead keeps both rules: nothing unresolved
ships, and nothing is silently asserted. A `%` comment above the entry points
here.

PROMPT 18 D3 also asks for the specific Cui *et al.* lemma number. That is not
pinned: the citation is to the paper, not to a numbered lemma, because the
lemma numbering could not be checked against the PDF from here.

**To close:** verify Balan's year, venue and pages, and pin the Cui lemma
number, both from the sources.

## 6. Content cut from Paper 1 for the page budget (PROMPT 18 Part F)

**Status: removed from the manuscript, results files untouched.**

Two passages were cut in the order the brief prescribes. Neither result is
withdrawn; both remain in the repository and can be restored if a page is
found.

**Coarse-to-fine order search** (`results/p12/B4.json`, configuration C). Cut
the sentence that it reduces the selection stage by `1.68×`, from `16.0` to
`9.6` candidate evaluations, returning the same order in `93.3%` of trials,
while missing its pre-registered gate of `≤ 0.05` dB at an aggregate cost of
`0.083` dB, with a paired median degradation of `0.000` dB. What survives is
the runtime share, `57.7–85.5%`, which is what makes the point that order
selection and not the projection is the place to optimise.

**Pencil sweep detail** (`results/p12/B3.json`, configuration C). Cut the span
of `0.390` dB over admissible-rank counts `8` to `16`, and the matched-shape
pair `24×9` and `8×25` at `+3.199` and `+3.121` dB. The conclusion they support
— no monotone trend, and the effect tracks rank count rather than matrix shape
— is retained in one sentence.

**Note that the second cut removes the paper's only direct evidence** for the
matched-shape claim, which now rests on an assertion. If a reviewer challenges
it, the numbers above are the answer and should be restored.
