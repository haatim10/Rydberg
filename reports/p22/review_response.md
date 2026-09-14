# Response to the external submission-readiness review, 14 Sep 2026

Item by item. **Accepted** means the manuscript changed. **Accepted, deferred**
means the point is right and needs work beyond this turn. **Partly** means the
correction was made but with a qualification the review did not have.

**Paper A was not touched this turn.** The review says to prioritise Paper B,
and Paper A's items (A7–A10) are derivations and new controls, not wording.
They need a turn of their own.

---

## Where the review found real errors

Four items were checked against the data and the review was right each time.

### B3 — "reduces the size of every bin by roughly an order of magnitude"

Wrong, and now corrected. Recomputed from
`reports/p19/p29_score.json` and `reports/p17/p28_score.json`:

| bin | mixed | focused | ratio |
|---|---|---|---|
| [5,10) | +0.457 | **−0.173** | 2.64, **and a sign change** |
| [10,15) | +1.403 | +0.067 | 21.0 |
| [15,20) | +2.290 | +0.259 | 8.9 |

The text now says the two highest bins shrink by about 21× and 9× and the
`[5,10)` bin changes sign rather than shrinking.

### B2 — the reference-gap arithmetic

`2.897 − 1.000 = 1.897` is near seed 1's `+1.902`, not the three-seed mean
`+2.171`. The review's reading is right: those gaps are seed 1's. Now labelled.

### B11 — `mean − 2·SD` is not a confidence interval

Accepted in full. Replaced everywhere by two-sided Student-$t$ intervals on two
degrees of freedom, and the manuscript now says the normality they assume
cannot be checked on three runs. Recomputed in
`reports/p22/paired_contrasts.json`; the review's illustrative table reproduces
exactly.

### B9 — "two unrelated corrections"

Accepted. The review is right that focused retraining and loss correction are
not independent in effect: **both raise the relative emphasis on the high-SNR
region where the contrast is evaluated.** The claim that their agreement
"establishes the common cause far more firmly" was overstated and is gone.

---

## The change that matters most: B1-P1, the paired analysis

The review asked for `Δ_H,focused − Δ_H,mixed` and `Δ_H,balanced − Δ_H,mixed`
per seed rather than three separate point estimates. Computed:

| contrast | per seed | mean | SD | 95% CI (seeds) | paired $t$ | $p$ |
|---|---|---|---|---|---|---|
| focused − conventional | −1.131, −1.295, −1.374 | **−1.267** | 0.124 | [−1.574, −0.959] | −17.7 | 0.003 |
| balanced − conventional | −1.291, −1.450, −1.648 | **−1.463** | 0.179 | [−1.908, −1.018] | −14.1 | 0.005 |

**This is a much stronger result than the one the paper was reporting, and it
is the right one.** The endpoints are not separated from zero on three seeds —
focused is [−0.187, +0.226], balanced is [−0.549, +0.196] — but the *changes*
are, decisively. The paper now leads with the change and explicitly declines to
claim the endpoints are absent or negative.

---

## Accepted and applied

| item | change |
|---|---|
| **B1-P0** | Title and abstract reframed from a causal claim to training-design sensitivity. The title is now *"Training-Design Sensitivity of Structural-Prior Gains in Unrolled Channel Estimation for Rydberg Atomic Receivers."* |
| **B10-P0** | **The structural operator is now defined** (new §II-A, eq. 3): the Hankel map with pencil $p=\lceil N/2\rceil-1$, fixed rank 7, one pass, unconditional at every layer, straight-through gradient, no trainable parameters. It states that this is Cadzow-style denoising and **not** a projection, that low Hankel rank admits models beyond the array manifold, and that nothing here tests an adaptively gated estimator. |
| **B8-P0** | The gradient share is defined as $q_b=\|g_b\|/\sum_c\|g_c\|$ over 24 batches per bin at the selected checkpoint, with the explicit note that $\sum_b\|g_b\|\neq\|\sum_b g_b\|$, that equal share is a chosen reference not an optimum, and that a skewed share **does not prove underfitting**. |
| **B5** | The log-loss section is retitled *"A simpler loss, with no resolved difference."* Both readings are given: the 95% interval [−0.204, +0.446] lies inside the ±0.50 dB margin registered before the runs, so the pre-registered equivalence criterion is met; stated without that margin, the comparison does not resolve a difference. |
| **B5** | The review's observation that the log-trained arm is **a counterexample to our own diagnostic** is adopted verbatim in substance: those networks perform well and still show 0.906–0.911 of gradient norm below 5 dB under conventional scoring, so the diagnostic cannot certify inadequacy. |
| **B12** | "96%" → "most of the gain along this ablation sequence", with the note that components interact and the allocation depends on order. "Unclaimed headroom" → "the gap to the genie-aided least-squares reference". |
| **B7** | Robustness rescoped to "a smaller observed degradation in one shift test", with the review's point that a method can start worse and deteriorate less. |
| **B6** | The P=25 inversion (0.14 dB, no paired uncertainty) now reads as the absence of an expected improvement, not established harm. |
| **B13** | The TODO citation is **resolved** using the metadata the review supplied: Jameel, Malhotra, El Gamal and Hamidi-Rad, *Deep OFDM Channel Estimation: Capturing Frequency Recurrence*, arXiv:2401.05436. |
| **B13** | EM-GS is now attributed to Cui *et al.* Alg. 2, with Gerchberg–Saxton 1972 as the classical antecedent rather than the source. |
| **B13** | The autobiographical claims — "the first casualty", "an author's attempt to keep it" — are removed. |
| **B2** | The mechanism conjecture is separated from the measurement, and the eighteen intervals are labelled descriptive rather than simultaneous. |

---

## Partly accepted

**B13, "pre-registered".** The review says to provide a dated record or call it
an informal prediction. The records exist and predate every run:
`reports/p16/PREREG_P27.md`, `reports/p16/PREREG_P28.md`,
`reports/p17/PREREG_P29.md`, `reports/p20/PREREG_P30.md`,
`reports/p21/PREREG_P31.md`, each committed standing alone with thresholds,
falsifiers and decision rules fixed before the corresponding batch ran, and
each scored held/failed afterwards. The term is retained and the records will
accompany submission. What the review is right about is that the manuscript
never points to them; a data-availability note is needed.

**B3-P0, balanced per-bin results.** Already measured on the same three seeds
and now available: −0.233, −0.704, −0.944, −0.692, −0.241, +0.567. The review
inferred these came from "an earlier arm"; they do not, they come from batch C3
on seeds 1–3. Space did not permit a third full per-bin column this pass — it
is a table edit, not a measurement gap.

---

## Accepted, deferred — these need work beyond wording

1. **B9-P1, convergence evidence.** Thirteen epochs with one-SE early stopping
   is not proof of adequacy. Learning curves exist per run; a longer-training
   control with validation-based stopping does not. **This is the single item
   most likely to change the paper's conclusion**: if extra optimization on the
   conventional objective removes the contrast, the undertraining reading is
   supported; if it does not, the result is about objective dependence. Neither
   is currently established.
2. **B11-P1, more seeds.** Three is what the central contrasts have. The paired
   analysis makes three go further than it did, but it does not make three
   enough.
3. **B12-P1, the 2×2 factorial.** Filter/no-filter × attention/no-attention
   under one protocol would show interactions the sequential ablation cannot.
4. **B10-P1, adaptive versus unconditional structure.** The paper now scopes
   its claim to the unconditional operator, which is honest, but the review is
   right that a claim about priors generally would need the gated variant.
5. **Section 2, the shared benchmark.** One configuration, identical channels
   and pilots, EM-GS / one-shot Hankel / unstructured unrolling / structured
   unrolling. This is what would let the two papers be read together, and
   nothing currently establishes that.
6. **Section 10, the reproducibility package.** Per-trial exports, configs and
   seeds. The material exists in `results/`; it is not packaged.

---

## One disagreement, recorded

The review says of the conventional-training row that it shows "a substantial
positive contrast in these trained runs". Agreed. It then treats the three
designs as three separate estimates throughout. The paired analysis above shows
that framing understates what the experiment supports: the three arms share
seeds by construction, so the differences are paired and their intervals are
roughly a third the width of the unpaired ones. The review asked for exactly
this under B1-P1 and then did not carry it into its own assessment — which is
why the paper now leads with it.

This is agreement about method, not a dispute about any number.
