# PROMPT 20 Part A — claim audit against the prior-art hit

Every claim Paper 2 makes, in `wip/spl2/haatim_structural_priors_evaluation.tex`
and in `docs/paper2-thesis.md`, classified **ANTICIPATED** /
**PARTIALLY ANTICIPATED** / **APPARENTLY NOVEL** against the reference supplied
in PROMPT 20 A1.

**The container cannot reach arXiv, IEEE Xplore or GitHub.** Every judgement
below rests on the description supplied in the brief, not on the PDF. Anything
that depends on detail beyond that description is marked `[UNVERIFIED]` and is
listed in `docs/open-todos.md` for a human to check against the paper itself.

## The reference, as supplied

Wiesmayr, Marti, Dick, Song and Studer, *"Bit Error and Block Error Rate
Training for ML-Assisted Communication"*, arXiv:2210.14103 (v3, Mar. 2023).
**Section 4, "SNR Deweighted Training."** It states that ML-assisted systems
learn one parameter set while operating over a range of SNRs, that training data
is sampled from that range, and that the aggregate loss is then dominated by
low-SNR samples — because a small relative improvement at low SNR moves the cost
far more than a large relative improvement at high SNR. It proposes SNR
deweighting as the remedy and demonstrates it in Sionna.

A second, weaker hit: deep OFDM channel estimation trained on SNR-restricted
samples, reported to improve low-noise performance and degrade below 5 dB.
`\todo{VERIFY — arXiv:2401.05436, fields unconfirmed from this container}`

---

## The audit

### Anticipated — cite, do not establish

| # | claim | verdict | note |
|---|---|---|---|
| A1 | Per-sample normalised error averaged over a wide SNR range is dominated by the low-SNR tail | **ANTICIPATED** | This is Wiesmayr §4's opening statement. Ours is the same claim. |
| A2 | The cause is that a small relative improvement at low SNR moves the aggregate cost more than a large one at high SNR | **ANTICIPATED** | Same section, same reasoning. |
| A3 | The consequence is that the high-SNR regime is systematically underfitted | **ANTICIPATED** | This is the motivation for their remedy. |
| A4 | The fix is to reweight the loss by SNR condition | **ANTICIPATED** | They propose SNR deweighting and demonstrate it. |

**Consequence for the manuscript:** §III-B ("The confound") must present the
mechanism as established and cited at first mention, not as identified here.
The measurement stays; the discovery claim goes.

### Partially anticipated — the phenomenon is theirs, the instance is ours

| # | claim | verdict | what is ours |
|---|---|---|---|
| B1 | 89.7% of the gradient falls below 5 dB; the per-bin share spans a factor of 30.0 | **PARTIALLY ANTICIPATED** | The phenomenon is A1. These are its *magnitude in this system*, and they are measured on an **NMSE** loss for channel estimation, not on the BER/BLER losses Wiesmayr treat. The same pathology in a different objective is worth reporting as a quantity; it is not a new phenomenon. |
| B2 | Balancing the loss gains **+2.171 dB** at SNR ≥ 5 across three seeds, mean − 2·SD = +1.689 | **PARTIALLY ANTICIPATED** | Demoted from headline. It is now: *a published remedy transfers to unrolled, magnitude-only channel estimation, and here is what it is worth.* The number stands; it stops leading. |
| B3 | Our weighting scheme: static per-bin factor `w(b)=c/m(b)`, `m(b)` measured once before training, weights normalised to unit mean | **PARTIALLY ANTICIPATED** `[UNVERIFIED]` | Whether this is the same scheme Wiesmayr use, or a variant, is **not** in the supplied description. If it is the same, ours is a replication; if it differs, the difference must be stated and neither overclaimed nor hidden. **Human must check.** |
| B4 | Focused training — retraining on a restricted SNR range — as an experimental arm | **PARTIALLY ANTICIPATED** | The adjacent OFDM work already trains on restricted SNR ranges and reports the expected trade. What is not anticipated is using it as a *control on attribution* rather than as a proposed configuration. `[UNVERIFIED]` |

### Apparently novel

| # | claim | verdict | basis |
|---|---|---|---|
| N1 | **A known training pathology manufactures apparent evidence for structural priors, so crediting a prior without controlling for it is unsound** | **APPARENTLY NOVEL** | Wiesmayr identify the pathology and fix it, for detection and decoding. They do not ask what it does to *attribution*. This is now the paper. |
| N2 | The matched-adequacy protocol: measure the gradient share; retrain **both** arms with training concentrated on the regime of interest, matched on budget, schedule, seed and initialisation; re-measure the contrast | **APPARENTLY NOVEL** | A control, not a technique. Its value is that it was run and that it overturned the authors' own result. |
| N3 | Δ_H under mixed-SNR training is **+1.286 dB**, three seeds, SD 0.072, positive under both pooling rules | **APPARENTLY NOVEL** | A measurement of this system. Nothing in the reference speaks to it. |
| N4 | Δ_H under matched focused training is **+0.020 dB** with **no stable sign** — one of three seeds significantly negative, ratio-of-sums negative on all three | **APPARENTLY NOVEL** | The result the protocol produces. B4 notes the restricted-SNR *training* is anticipated; this contrast is not. |
| N5 | **The per-bin sign pattern** `- - - + + +`: the prior costs accuracy below +5 dB and buys it above, the sign flipping exactly once, replicating on every seed of two designs with all CIs excluding zero | **APPARENTLY NOVEL** | Promote to a result in its own right per Part D3. |
| N6 | **The averaging-window finding**: the sign of the reported structural gain depends on how the average is taken — paired per-trial median vs ratio of sums, and whether the negative low-SNR bins are inside the window — and the reason is measured, not asserted | **APPARENTLY NOVEL** | Promote to a result in its own right. **Interpretation flagged:** the brief names an "averaging-window finding" without defining it; I have read it as this pooling-and-restriction dependence, which is the only finding of that shape in the repository. If a different one was meant, this row is misfiled. |
| N7 | There is **no abstention analogue** in the learned arm: `hankel_gate='none'` makes the projection unconditional, and it moves the estimate on every layer of every one of 2000 trials on every seed | **APPARENTLY NOVEL** | Rules out the mechanism Paper 1 uses for the same statistic-dependence at N=8. A negative mechanism result. |
| N8 | **Pilot efficiency is not pilot-count generalisation**: the two differ by **+2.231 dB**, and the P=20 model is *worse* at P=25 than at P=20 | **APPARENTLY NOVEL — AND NOT SEARCHED AT ALL** | This was never searched, not even cursorily. Its novelty is unsupported rather than supported. **Highest-priority item for the human.** |
| N9 | Attribution decomposition: of 3.345 dB over EM-GS, the learned filter contributes 0.147 dB and the attention block 3.198 dB (96%); unrolling is worth 0.920 dB against a matched non-unrolled control | **APPARENTLY NOVEL (result), NOT NOVEL (method)** | Component ablation of an unrolled network is ordinary practice. The numbers are specific to this estimator. |
| N10 | 18 of 18 bin×seed cells favour the balanced loss, **including the two bins the weighting divides by ≈18 and ≈9** | **APPARENTLY NOVEL `[UNVERIFIED]`** | The supplied description says Wiesmayr *propose and demonstrate* deweighting; it does not say whether they report the effect on the **down-weighted** regime. If they do, this is anticipated and must be recast as confirmation. **Human must check.** This is the single claim whose status most changes the paper's remaining contribution. |
| N11 | The prior gives a large stable gain on the **classical** estimator, where no training adequacy is at stake, and degrades least under a path-richness shift (+0.48 vs +0.73 dB) | **APPARENTLY NOVEL, single seed** | Scoping evidence. Unreplicated; must be labelled single-seed. |

### Not novelty claims

| # | statement | why it is not audited |
|---|---|---|
| X1 | The genie-aided reference is not a bound and is passed below 5 dB | A scoping disclaimer about our own figure. |
| X2 | Simulation only, no hardware, no recorded channel data | A limitation. |
| X3 | The system model, the URformer architecture, EM-GS | Prior work, already cited as such. |

---

## What the paper becomes

Before this audit the spine led with `+2.171 dB` — a replication. After it, the
ordering is:

1. **The attribution argument (N1)** — the claim. A known training pathology
   manufactures apparent evidence for structural priors.
2. **The control (N2)** and what it did to our own method (N3, N4).
3. **The sign pattern (N5) and the averaging window (N6)** — what the contrast
   actually is, once you stop reporting it as a scalar.
4. **The remedy transfers (B2)** — secondary, cited, honest.
5. **The pilot-curve ambiguity (N8)** — a separate, self-contained result.

Nothing measured is withdrawn by this audit. What changes is which sentence
goes first, and which citations carry which claims.

---

## Items added to `docs/open-todos.md`

Six, all requiring a human with network access. In priority order: **N10**
(does Wiesmayr report the down-weighted regime?), **N8** (the pilot-curve
ambiguity, never searched), **B3** (is our weighting scheme theirs?), the
arXiv:2401.05436 fields, whether SNR deweighting has been applied to unrolled
estimators specifically, and verification of the Section 4 title and content
against the PDF.

**Standing caveat, from the brief and repeated here because it belongs on the
record:** the prior-art check behind PROMPT 20 was three web searches, not a
systematic sweep. This audit is only as good as that check.
