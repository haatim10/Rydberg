# HS-URformer at N=32 — PROMPT 6 Part B

**Verdict: STOP-MARGINAL. Part C is not launched.**

`Δ_H = +0.129 dB`, bootstrap CI `[+0.072, +0.195]`, excluding zero but below the
pre-registered `+0.3 dB` threshold. The rule in
`reports/trackD_hankel_prereg.md` (committed `7775265`, before any run) says
stop and report at `0 < Δ_H < +0.3`, and that is what this does.

Primary statistic throughout: **the paired per-trial median**, `n_test = 2000`,
one evaluation pass, every arm on identical worlds. Ratio-of-sums appears only
where labelled.

---

## 1. The number is real, and it is also nearly meaningless

`Δ_H = +0.129 dB` is not a small effect. It is a **large effect that cancels**.

| SNR bin | trials | `Δ_H` paired median | |
|---|---|---|---|
| −10 … −5 | 346 | **−0.111** | Hankel hurts |
| −5 … 0 | 354 | **−0.506** | Hankel hurts |
| 0 … +5 | 317 | **−0.451** | Hankel hurts |
| +5 … +10 | 331 | **+0.398** | Hankel helps |
| +10 … +15 | 336 | **+1.305** | Hankel helps |
| +15 … +20 | 316 | **+2.226** | Hankel helps |
| **pooled** | 2000 | **+0.129** | — |

Below 5 dB (50.9% of trials) `Δ_H = −0.333`. At or above 5 dB (49.2%)
`Δ_H = +1.209`. A 1.5 dB swing averages to +0.13 because our SNR draw is
uniform on `[−10, 20]`.

**So the pooled `Δ_H` describes our sampling design more than it describes the
method.** Had the SNR range been `[0, 20]` the same runs would have produced a
clear GO; had it been `[−10, 10]`, a clear STOP. Both would have been honest
applications of the pre-registered rule, and both would have been the wrong
summary. This is the most important thing in this report, and it was not
anticipated by the pre-registration — every question there was posed as if
`Δ_H` were a scalar property of the method.

The go/no-go rule is still applied as written. Rewriting it after seeing the
decomposition is exactly the move pre-registration exists to prevent. But the
recorded conclusion is "stop at this SNR mix", not "the Hankel prior does not
help".

### Why the crossover happens — CORRECTED (PROMPT 7 A2)

**The explanation given in the first version of this report was wrong.** It said
rank truncation is a subspace method with an SNR threshold, and that "the
classical arm shows the same shape". It does not. `H0 − U0` is **positive in
every bin and monotone**:

| SNR bin | H0 − U0 (classical) | U1 → H1 (learned) |
|---|---|---|
| −10 … −5 | **+0.809** [+0.724, +0.887] | −0.111 |
| −5 … 0 | **+1.332** [+1.247, +1.430] | −0.506 |
| 0 … +5 | **+1.624** [+1.581, +1.719] | −0.451 |
| +5 … +10 | +1.818 [+1.735, +1.885] | +0.398 |
| +10 … +15 | +1.809 [+1.736, +1.899] | +1.305 |
| +15 … +20 | +1.892 [+1.815, +1.965] | +2.226 |

Rank-7 truncation applied to a *classical* estimate helps by 0.8 dB even at the
worst SNR. So truncation does not destroy signal at low SNR in general, and
plain subspace-threshold behaviour cannot explain H1's low-SNR loss. My original
mechanism was a plausible story that the data in the same table already
contradicted, and I did not check it against that column.

What the table does support: at low SNR the URformer's implicit prior **already
matches** the explicit one — applying the projection post-hoc to a converged U1
adds only +0.013 dB in the lowest bin — yet imposing that same projection
*during training* costs 0.111–0.506 dB. **The damage is specific to constraining
the learned loop, not to the operator.** PROMPT 7 A4 supplies the mechanism: see
§3a.

---

## 2. Every arm

| arm | median NMSE (dB) | ratio-of-sums (dB) | params |
|---|---|---|---|
| U0 EM-GS (`T_GS=100`) | −7.448 | −0.604 | 0 |
| H0 HS-EM-GS (Track B, `L=7`) | −9.142 | −1.562 | 0 |
| X1 EM-GS + 1 Transformer | −9.359 | −5.710 | 158,592 |
| U1 URformer 80k | **−10.831** | −6.583 | 1,586,900 |
| U1+post (project once at the end) | **−11.090** | −6.657 | 1,586,900 |
| H1 HS-URformer (internal) | −10.532 | −6.365 | 1,586,900 |
| unstructured-LS oracle | −11.582 | −5.247 | — |

Paired contrasts, positive = second arm better:

| contrast | median (dB) | CI95 | ros (2nd) | win rate |
|---|---|---|---|---|
| `Δ_H` U1 → H1 (**the decision**) | **+0.129** | [+0.072, +0.195] | −0.214 | 0.552 |
| U1 → U1+post (post-hoc) | +0.279 | [+0.250, +0.315] | +0.073 | 0.886 |
| U1+post → H1 (**integration**) | **−0.105** | [−0.145, −0.064] | −0.287 | 0.444 |
| U0 → H0 (classical) | +1.574 | [+1.533, +1.612] | +0.965 | 0.986 |
| X1 → U1 (**is unrolling needed?**) | +0.920 | [+0.888, +0.988] | +0.871 | 0.865 |
| U0 → U1 | +3.345 | [+3.100, +3.515] | +6.150 | 0.846 |
| U0 → X1 | +1.889 | [+1.692, +2.084] | +5.279 | 0.802 |

---

## 3. Structural integration — CORRECTED (PROMPT 7 A1)

**The pooled claim in the first version of this report was the same error this
report diagnoses one level up.** It read: "post-hoc beats internal, −0.105 dB",
a pooled scalar over a sign change. Per bin, with every CI excluding zero:

| SNR bin | U1 → H1 (internal) | U1 → U1+post | U1+post → H1 (**integration**) |
|---|---|---|---|
| −10 … −5 | −0.111 | +0.013 | **−0.138** [−0.184, −0.093] |
| −5 … 0 | −0.506 | +0.072 | **−0.582** [−0.633, −0.492] |
| 0 … +5 | −0.451 | +0.190 | **−0.636** [−0.750, −0.552] |
| +5 … +10 | +0.398 | +0.497 | **−0.207** [−0.280, −0.078] |
| +10 … +15 | +1.305 | +0.826 | **+0.388** [+0.293, +0.463] |
| +15 … +20 | +2.226 | +0.837 | **+1.309** [+1.184, +1.466] |
| pooled | +0.129 | +0.279 | −0.105 |

Post-hoc wins the lower four bins; **internal wins the top two decisively**
(+2.226 vs +0.837 at 15–20 dB). The corrected statement is therefore:

> **Structural integration matters at high SNR and is harmful at low SNR.**

Not "post-hoc is better". Integration buys +1.309 dB over post-hoc at 15–20 dB,
which is a real result that the pooled scalar erased — it is the strongest
evidence in this report *for* the unrolled architecture, and my first pass
reported its opposite.

### 3a. Why constraining the loop hurts (PROMPT 7 A4)

`nmse_loss` is the mean over the batch of per-sample `‖ΔG‖²/‖G‖²`. Every trial
carries equal weight in the mean but wildly unequal magnitude, because a
low-SNR trial's *normalized* error is far larger. Measured on the trained
checkpoints over the training split:

| arm | share of loss below 5 dB | share of gradient norm below 5 dB |
|---|---|---|
| U1 | 0.927 | 0.787 |
| H1 | **0.943** | **0.859** |

**H1 was optimized almost entirely for the regime its own constraint damages.**
86% of its gradient signal comes from trials below 5 dB, where the projection
costs it 0.1–0.5 dB. That explains the worse *train* loss (−6.93 vs −7.33)
without any appeal to subspace thresholds, and it makes H1's high-SNR win
(+2.23 dB) the more striking: it is achieved by a model whose training signal
barely represented that regime.

### The STE caveat, now measured (PROMPT 7 A3)

The straight-through estimator makes H1's backward pass the gradient of a
network *without* the projection. Measured against the exact autograd gradient
through the SVD, in float64, per (layer, group) cosine similarity:

| | low SNR (−10…0) | high SNR (+10…+20) |
|---|---|---|
| median cosine (excl. gate) | 0.630 | 0.821 |
| filter_net | 0.886 | 0.977 |
| transformer | 0.459 | 0.548 |
| layer 9 (last) transformer | 1.000 | 1.000 |
| layer 0 (first) transformer | **−0.011** | **0.128** |

The dominant axis is **depth, not SNR**: fidelity is exact at the last layer,
whose gradient traverses no projection, and decays to near-orthogonal at layer
0, whose gradient traverses ten. It is additionally worse at low SNR, but it is
poor in *both* regimes, so the STE cannot be cleanly credited with the
low-SNR-*specific* damage — nor exonerated. Numerical trust: the minimum
relative singular-value gap at the truncation is 2.6e-3 (low) and 6.7e-4
(high), costing ~3 of float64's ~16 digits, so the near-zero cosines are
structural rather than numerical artifacts.

## 4. The `n_iter` caveat, raised and then closed

Part A established that one sweep of `H⁻¹∘Π_r∘H` is **not a projection**: it
leaves ~3.7% of each column's Hankel energy above rank 7. That opened a
competing reading of a small `Δ_H` — "the prior was barely imposed" rather than
"the prior does not help". The sweep on the training-free post-hoc arm closes
it:

| Cadzow sweeps | 1 | 2 | 4 | 8 |
|---|---|---|---|---|
| gain over U1 (dB) | +0.279 | +0.295 | +0.269 | +0.232 |
| off-manifold energy | 3.7e-2 | 2.0e-2 | 8.4e-3 | 1.4e-3 |

Imposing the structure 26× harder changes the gain by 0.06 dB, and past
`n_iter=2` it gets slightly *worse*. **The small effect is not an artifact of a
weakly-imposed prior.** That competing explanation is dead.

---

## 5. Predictions: one right, three wrong

Pre-registered in `7775265` before any run.

**P7 / Q1 — CORRECT, for the wrong reason.** Predicted `+0.1 to +0.4 dB`, most
likely ~+0.25, CI excluding zero, landing below +0.3, decision = stop. Measured
+0.129, CI [+0.072, +0.195], stop. The number and the decision were right. The
*reasoning* was wrong: I argued the effect would be small because the URformer
had already internalised the geometry, so little was left to add. The truth is
that the effect is large and bidirectional, and small only after cancellation.
A correct number from a wrong mechanism is not a successful prediction, and I
am not counting it as one.

**Q4 — FALSIFIED, sign-reversed.** I predicted the prior would help the hard
low-SNR tail disproportionately, and that ratio-of-sums `Δ_H` would exceed the
median by ~2×. It is the opposite on both counts: the prior *hurts* at low SNR,
and ratio-of-sums `Δ_H` is **−0.214** against a median of **+0.129** — a sign
flip, because ratio-of-sums is dominated by the low-SNR trials where H1 loses.
This is the most clearly wrong thing I predicted.

**Q6 — FALSIFIED.** Predicted internal beats post-hoc by less than half the
total effect. Post-hoc beats internal, CI excluding zero. Covered in §3.

**X1 — FALSIFIED, and this one matters most.** I predicted X1 would land within
0.5 dB of the full URformer, and said that if it did, "Track D's honest
description is *denoise the classical estimate*, and the unrolling is doing
little." The gap is **0.920 dB**, CI [+0.888, +0.988], U1 winning on 86.5% of
trials. **The unrolling earns its keep and my hypothesis was wrong.**

The nuance worth keeping: X1 recovers +1.889 dB of U1's +3.345 dB over EM-GS —
**56% of the gain with 10% of the parameters and no unrolling at all**. So a
single post-processor is a strong, cheap baseline that the paper does not
report, and the remaining 44% is what ten unrolled layers and 1.43M extra
parameters buy. Both halves of that are worth stating.

**The `Δ_H ≥ +1.0 dB` falsifier — not triggered as written, but partly earned.**
I committed to conceding that my "the network already has the prior" reading of
stage 2 was wrong if `Δ_H ≥ +1.0`. Pooled `Δ_H` is +0.129, so the falsifier does
not fire on its own terms. But at SNR ≥ 5 dB `Δ_H = +1.209`, and at 15–20 dB it
is +2.23. **In half the operating range the explicit prior adds more than a dB,
which means the URformer has *not* internalised the geometry there.** My stage-2
reading holds at low SNR and fails at high SNR. I am recording that as
half-refuted rather than letting the pooled number shelter it.

**Q3 (`N=8`)** was settled algebraically by gate HK7 in Part A, not by this run.
**Q2 (sample efficiency)** requires the Part C budget sweep, which the rule
does not authorise.

---

## 6. What was actually established

1. `Δ_H = +0.129 dB` at 80k, CI excluding zero → **STOP-MARGINAL**; PROMPT 6's
   Part C (the budget sweep) was not launched.
2. The pooled figure hides a **1.5 dB SNR crossover**: the rank-7 prior hurts
   below ~5 dB and helps increasingly above it. Any single-number `Δ_H` for
   this method is a statement about the SNR mix it was averaged over.
3. **Structural integration matters at high SNR and is harmful at low SNR**
   (corrected from "post-hoc beats internal", which was a pooled scalar over a
   sign change). Internal beats post-hoc by +1.309 dB at 15–20 dB and loses by
   0.636 dB at 0–5 dB, every CI excluding zero.
4. The small effect is **not** explained by weak imposition — the `n_iter`
   sweep is flat.
5. **Unrolling is doing real work** (0.920 dB over a matched single
   post-processor), refuting my own "it's just denoising" hypothesis. But a
   single post-processor still captures 56% of the gain at 10% of the
   parameters.
6. Classical HS-EM-GS gains **+1.574 dB** over EM-GS — 12× the learned Hankel
   gain — confirming the prior is genuinely informative and that the URformer
   captures most of it implicitly at low SNR.

## 7. What is NOT established

- **Anything about sample efficiency (Q2).** Untested; it needs the budget
   sweep, which remains unauthorised.
- **Whether H1's low-SNR damage is the constraint itself, the loss weighting,
  or the STE.** A4 shows 86% of H1's gradient norm comes from trials below
  5 dB; A3 shows the STE's gradient fidelity is poor at every SNR and worst for
  early layers (cosine ~0 at layer 0). Neither explanation is cleanly separated
  from the other, which is what PROMPT 7's Part C
  ([5,20]-only retraining) is for.
- **Anything about `N ≠ 32`.** No array-size sweep was run or authorised.
- **That `r = 7` is the right rank.** Only fixed `r = L_max` was run; the
  adaptive and oracle modes exist and were not exercised in training.
- **That the SNR crossover point is ~5 dB in general.** It is ~5 dB for this
  array, pilot budget and RSR. The threshold is a property of the whole
  operating point.

## 8. Files

- `reports/trackD_stage3_results.json` — all per-trial rows, contrasts, verdict
- `results/track_d/stage3/fig1_headline.png` — arms and the decision
- `results/track_d/stage3/fig2_snr_crossover.png` — **the result**
- `results/track_d/stage3/fig3_niter_sensitivity.png` — the closed caveat
- `results/track_d/stage3/fig4_X1_control.png` — the unrolling control
- `results/track_d/stage3/fig5_delta_by_rank.png` — `Δ_H` vs `max_k L_k`
- `results/track_d/stage3/fig6_training_curves.png` — H1 vs U1, matched
