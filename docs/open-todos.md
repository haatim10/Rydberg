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

### 5.1 RESOLVED 16 Sep 2026 — they do NOT report it; the 18-of-18 result stands

**Closed by a human reading Section 4 of arXiv:2210.14103 v3 directly.**

**Answer: no.** Wiesmayr et al. do not report improvement in the down-weighted
regime. Where they examine it, they report the opposite:

- In the narrow-range DUIDD experiment, deweighted training holds a small
  advantage at high SNR while naive training keeps an even smaller advantage at
  the low-SNR end of the waterfall.
- The SISO AWGN replication repeats the pattern: naive training over the range
  is worst at high SNR and best at low SNR.
- For the wide range they report well-balanced performance and a 0.54 dB gain
  at 1% BLER — but that is a high-SNR operating point, and no per-bin low-SNR
  improvement is claimed.

Evidence: their Section 4, Figures 3 and 5.

**Consequence.** Our 18-of-18 result — balancing improves every bin-by-seed
cell, including the two bins the weighting divides by about 18 and 9 — is
therefore a **finding and not a confirmation**, and the manuscript now states
it as a contrast with their evidence rather than as a bare measurement.

**Three caveats travel with that contrast and must never be dropped** (they are
in the manuscript):

1. *Different metric.* They measure bit and block error rate on a coded
   MIMO-OFDM system; we measure normalized channel-estimation error.
   "Improvement in the down-weighted bins" is not the same quantity.
2. *Different mechanism.* Their weights update every epoch from inverse
   cumulative losses, with a stability constant and normalization at the grid
   centre. Ours come from one measurement pass before training.
3. *Different size.* Their deweighting gain is 0.54 dB, which they describe as
   modest but free, since it comes from the loss rather than a bigger receiver
   or more data. Ours is +2.171 dB. Different metrics, so not directly
   comparable, and the text says so.

Also recorded from that reading, and now used in the manuscript:

- Their introduction states that systems are usually trained either at a single
  SNR or on samples drawn uniformly from the target range, apparently without
  questioning how this affects performance at different SNRs. That is a
  *published* statement that the convention is unexamined, and it supports our
  motivation better than anything previously cited there.
- At the end of Section 4 they note one could alternatively deweight using the
  loss of a fixed baseline such as a classical system, rather than the adaptive
  scheme they implement. **Ours is fixed** — so they raise a fixed-weight
  variant without implementing it, and ours uses the network's own pre-training
  loss rather than a classical baseline. The manuscript now says this.
- Their Section 7.3 repeats the whole experiment on a second, simpler system
  specifically to show the effect is not an artifact of one architecture. Our
  limitations now cite that approach as what the fix to our own one-architecture
  scope would look like.

<details><summary>Original entry</summary>

### 5.1 (original) Does Wiesmayr et al. report the effect on the DOWN-WEIGHTED regime?

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

</details>

### 5.2 Has the pilot-curve train-versus-evaluate ambiguity been identified before?

**NOT SEARCHED AT ALL.** Its novelty is unsupported rather than supported.

The claim (N8): a pilot-count curve means two different things depending on
whether the network was retrained at each pilot count, papers do not say which,
and the gap is +2.231 dB — with a P=20 model performing *worse* at P=25 than at
P=20.

**To close:** search the unrolled-estimator and deep-channel-estimation
literature for any prior statement of this distinction.

### 5.3 RESOLVED 14 Sep 2026 — our weighting scheme is NOT theirs

**Closed by the external review.** Wiesmayr et al. Sec. 4 updates weights
**after each epoch** from inverse accumulated losses with a stabiliser. Ours is
a **static, pre-training, per-bin calibration**. So the manuscript is an
adaptation of the principle, not a replication of the implementation — which
strengthens what may be claimed, not weakens it.

Item 5.1 is **not** closed by this: whether they report the effect on the
down-weighted regime is still unknown, and it still decides whether the
18-of-18 result is anticipated.

<details><summary>Original entry</summary>

### 5.3 (original) Is our weighting scheme the same as theirs?

We use a static per-bin factor `w(b) = c/m(b)`, with `m(b)` the mean per-sample
normalised error in bin `b` measured once before training and `c` set so the
weights have unit mean. Whether Wiesmayr's SNR deweighting is this scheme or a
different one is not in the supplied description.

**To close:** compare against their Section 4. If identical, ours is a
replication and must say so. If different, the difference must be stated
plainly — neither overclaimed as a new method nor hidden.

### 5.4 RESOLVED 14 Sep 2026 — the second reference's fields

**Closed by the external review**, which supplied verified metadata and
confirmed that Sec. IV-C of the paper compares SNR-restricted training:

> Abu Shafin Mohammad Mahdee Jameel, Akshay Malhotra, Aly El Gamal and Shahab
> Hamidi-Rad, "Deep OFDM Channel Estimation: Capturing Frequency Recurrence",
> arXiv:2401.05436 (2024).

Entered in `wip/spl2/refs.bib` as `jameel2024ofdm` and the `\todo` marker is
gone. Someone should still verify final journal metadata if it has since been
published.

<details><summary>Original entry</summary>

### 5.4 (original) The second reference's fields are unconfirmed

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

</details>

---

## 6. Paper A items from the external review of 14 Sep 2026

Full response in `reports/p22/review_response_paperA.md`. These are the ones a
person must close; the wording corrections are already applied to
`paper/paper1/haatim_hsgs_letter.tex`.

### 6.1 The CCRB derivation — derive it or delete the curve

`U(U^T J U)^{-1} U^T` is not a reproducible specification. Needed: the real
parameter vector, the likelihood, derivative expressions, dimensions, the
tangent-space basis, rank conditions, and the NMSE normalisation. **If it
cannot be derived, remove the curve and the 7.05--7.11 dB gap claim rather than
leaving an unauditable theory section.** Largest single item on Paper A.

### 6.2 Trial counts 276 and 266 against a stated 300--400

Not resolved from the result files this turn. Most likely per-generator
rejection of degenerate draws, but that is a guess and must not be written as
an explanation until confirmed.

### 6.3 Author list of `precoding2408` is unverified

`reports/trackD_step0_cui_prediction.json` records `citation_verified: false`
and notes the PDF is not in the repository. The title and arXiv identifier
(2408.14366) were confirmed by the author; the author list is taken from
`paper/master/refs.bib` and has not been checked against the paper.

### 6.4 Predictive validation of rho against simpler descriptors

The review's central novelty concern. A held-out comparison of `rho` against
`L` and `L/r_max` under the same fitting budget and splits, reporting
prediction error rather than visual curve alignment. Decides whether the
effective-rank contribution stands.

### 6.5 Formatting defects in the nine-page version

Eq. (9)'s missing left-hand variable, the q/y inconsistency, undefined Table I
labels, Fig. 2(b) line styles. These were not found in the five-page letter,
which has different numbering; re-check against whichever version is submitted.

### 6.6 Superseded drafts still carry "exact projection"

`paper/master/`, `paper/merged/`, `paper/spl1/` and `paper/summary/` assert the
operator is a projection onto the intersection of the rank and Hankel sets.
That is false --- the map is not even idempotent. They are not submission
targets and were left alone, but must be corrected if any is ever revived.

---

## 7. PROMPT 22 (Paper 1 revision after external review)

Recorded 2026-09-15 on branch `paper1-review`.

### 7.1 Resolved this turn — moved out of section 6

- **6.4 is done.** The held-out comparison of `rho` against `L` and `L/r_max`
  ran as P32 (`reports/p22/PREREG_P32.md`, `reports/p22/PART_B_P32.md`). The
  answer is negative for `rho`: it does not beat `L/r_max` under either
  registered statistic. The comparative claim has been deleted from the
  abstract and conclusion and `rho` is now presented as one way of normalising
  by capacity, not as a better descriptor.
- **The CCRB derivation is done** and verified (`reports/p22/PART_A_derivation.md`).
  Section IV-B is now recomputable.

### 7.2 Deferred deliberately (PROMPT 22 Part E)

- **The full three-method placement campaign** with fixed-rank and
  complete-method arms. Part D measured the per-trial divergence between the
  two placements instead, which answers the substance without rebuilding the
  paper around a different estimator.
- **Absolute runtime and computational environment reporting.** The paper
  reports the rank-selection share of run-time (57.7–85.5%) but no absolute
  timings and no machine description.
- **Trial-level forensics on the `N = 8` result.** The inactive fraction
  (41.8%) does not by itself prove that occasional destructive truncation
  caused the negative aggregate. Establishing that needs per-trial
  decomposition, not a summary statistic.
- **The tutorial-material reduction.** Whether the worked examples belong in a
  submitted paper is a venue question, not a correctness one.

### 7.3 New, found while doing PROMPT 22

**7.3.1 Two scripts write to the same results path — data-loss hazard.**
`scripts/constrained_crlb.py` (validation, 10 trials/point) and
`scripts/constrained_crlb_fast.py` (production, 400 trials/point) both write
`results/track_b/constrained_crlb.json`. Running the validation script to
completion silently overwrites the 400-trial file the published curves depend
on with 10-trial data, whose per-point jitter is ±0.31 dB — enough to move the
published 7.05–7.11 dB gap. During this turn the validation run was cut off
before its sweep wrote and the file was confirmed intact, but that was luck.
Fix: give the validation script its own output path, and check the plotting
scripts for the same assumption.

**7.3.2 The thesis-style rewrite dropped content the letter had.**
`paper/paper1/haatim_hsgs_letter.tex` carries 21 `% src:` provenance comments;
`paper/thesisproj/haatim_thesisproj.tex` carried none until this turn. The
rewrite also dropped the Fisher-information construction, the tangent-space
justification, the per-SNR placement breakdown, and several limitations the
letter states plainly — including the 0.493 dB unexplained residual and the
fact that two of the three boundary crossings cannot be bracketed. Several
items the external review raised against the thesis version were already
correct in the letter. **Before submission, diff the two files and decide
deliberately, item by item, what the thesis version is allowed to omit.**

**7.3.3 A wrong number in the code, now fixed, may exist elsewhere.**
`constrained_crlb_fast.py` claimed the path parametrisation is over-complete in
"~42% of trials at N=8". The exact figure is 28.00%. Corrected in place. Worth
grepping for other asserted percentages that were never computed.

**7.3.4 The published crossing locations mix statistics.**
`0.518` comes from a fixed-5-dB sweep; `0.588` and `0.544` come from the pooled
field `pooled_SAMPLING_DESIGN_DEPENDENT` over a drawn SNR range. The manuscript
now says so, but the cleaner fix is to recompute all three on one statistic.
Not done this turn: it would change published numbers, which PROMPT 22 does not
authorise.

**7.3.5 The 276/266 trial counts are still unexplained.**
Carried over from 6.2 and now stated in the manuscript as measured, with the
cause explicitly not asserted. **This must be resolved before submission.**
It is the one number in the paper whose provenance we cannot give.

### 7.4 For the human, not for an agent turn

The classical placement result (post-hoc is as good as interleaved, +0.003 dB)
sits against Paper 2's finding that internal beats post-hoc by +1.309 dB at
15–20 dB in the *learned* setting. That asymmetry is interesting and neither
paper reports it. It needs the arXiv sequencing settled first.
