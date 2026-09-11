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

## 5. Prior-art verification for Paper 2 — SIX ITEMS, ALL NEED NETWORK ACCESS

Added under PROMPT 20 Part A. Every one of these is a claim this repository
cannot settle: the container reaches neither arXiv, IEEE Xplore nor GitHub, and
the prior-art check behind PROMPT 20 was **three web searches, not a systematic
sweep**. The audit in `reports/p20/claim_audit.md` is only as good as that
check. In priority order.

### 5.1 Does Wiesmayr et al. report the effect on the DOWN-WEIGHTED regime?

**This is the item that most changes what Paper 2 still contributes.**

Our claim N10 is that SNR balancing improves **all 18 bin-by-seed cells**,
including the two bins the weighting divides by about 18 and 9 — that is, it
improves the regime it penalises, which is the opposite of what a reweighting
is usually suspected of doing. The supplied description says Wiesmayr et al.
propose and demonstrate SNR deweighting; it does **not** say whether they
report what happens to the deweighted regime.

**To close:** read Section 4 of arXiv:2210.14103 v3. If they report improvement
in the deweighted regime, N10 is anticipated and must be recast as confirmation
rather than a finding. If they do not, it stands as apparently novel.

### 5.2 Has the pilot-curve train-versus-evaluate ambiguity been identified before?

**NOT SEARCHED AT ALL.** Its novelty is unsupported rather than supported.

The claim (N8): a pilot-count curve means two different things depending on
whether the network was retrained at each pilot count, papers do not say which,
and the gap is +2.231 dB — with a P=20 model performing *worse* at P=25 than at
P=20.

**To close:** search the unrolled-estimator and deep-channel-estimation
literature for any prior statement of this distinction.

### 5.3 Is our weighting scheme the same as theirs?

We use a static per-bin factor `w(b) = c/m(b)`, with `m(b)` the mean per-sample
normalised error in bin `b` measured once before training and `c` set so the
weights have unit mean. Whether Wiesmayr's SNR deweighting is this scheme or a
different one is not in the supplied description.

**To close:** compare against their Section 4. If identical, ours is a
replication and must say so. If different, the difference must be stated
plainly — neither overclaimed as a new method nor hidden.

### 5.4 The second reference's fields are unconfirmed

`arXiv:2401.05436` is cited in the audit for deep OFDM channel estimation
trained on SNR-restricted samples, reported to improve low-noise performance
and degrade below 5 dB. **Authors, title, venue and year were not verifiable
from this container.** It carries a `\todo{VERIFY CITATION}` and must not enter
the manuscript bibliography until checked. A fabricated reference is worse than
a missing one.

### 5.5 Has SNR deweighting been applied to unrolled estimators specifically?

Wiesmayr et al. work on detection and decoding. Whether anyone has applied the
same remedy inside an unrolled channel estimator is unknown and unsearched. It
bears on how much of B2 (`+2.171 dB`) is a transfer result and how much is a
re-run.

### 5.6 Verify Section 4's title and content against the PDF

The audit treats "Section 4, SNR Deweighted Training" and its stated reasoning
as given by the PROMPT 20 brief. Nothing in this repository confirms the
section number, its title, or that the reasoning is as described.

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
