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

### Why the crossover happens

Rank truncation is a subspace method, and subspace methods have an SNR
threshold. The projection keeps the top 7 singular directions of the LS
estimate's `16×17` Hankel matrix. At high SNR those directions *are* the signal
subspace and truncation is a clean denoiser. At low SNR the noise floor is
comparable to the signal singular values, the top-7 directions are partly
noise, and truncation discards weak-but-real signal energy while keeping noise
it cannot distinguish. Nothing here is specific to learning — the classical arm
shows the same shape (H0−U0 rises from +0.81 to +1.89 across the same bins),
just shifted up and much flatter, because EM-GS has no other way to exploit the
structure.

This also explains the training curves. The NMSE loss is dominated by
high-error trials, which are the low-SNR ones — precisely where the projection
hurts. H1's optimisation was therefore steered by the regime the operator
damages, which is why it plateaus at a **worse train loss** (−6.93 vs U1's
−7.33), not merely a worse validation loss.

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

## 3. Structural integration is not just undemonstrated — it is worse

**`U1+post` beats `H1`**: −0.105 dB, CI `[−0.145, −0.064]`, excluding zero.
Applying the *same operator* once to a converged estimate beats threading it
through all ten unrolled layers, and it wins on 88.6% of trials against U1
versus H1's 55.2%.

The pre-registration said a tie would mean integration was not demonstrated.
This is stronger than a tie and in the opposite direction from the prediction.
`U1+post` never hurts in any SNR bin (+0.013 → +0.837, monotone); `H1` swings
from −0.51 to +2.23. Internal application is **higher variance for lower median
return**.

The reading: the Hankel projection is a good *post-processor* and a bad
*training-time constraint*. Applied inside training it distorts the
representation the network is learning around — ten times per forward pass, in
the low-SNR regime that dominates the loss — and the network adapts to a
corrupted intermediate rather than exploiting the structure.

**One caveat I cannot rule out.** The straight-through estimator means H1's
backward pass is the gradient of a network *without* the projection while its
forward pass has one. That mismatch is a candidate explanation for H1's lower
ceiling that is independent of the operator's merits. Distinguishing "the
constraint is bad during training" from "the STE gradient is bad" needs a run
this prompt does not authorise — an exact-gradient variant, or a projection
warm-up schedule. It is flagged, not resolved.

---

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

1. `Δ_H = +0.129 dB` at 80k, CI excluding zero → **STOP-MARGINAL, no Part C.**
2. The pooled figure hides a **1.5 dB SNR crossover**: the rank-7 prior hurts
   below ~5 dB and helps increasingly above it. Any single-number `Δ_H` for
   this method is a statement about the SNR mix it was averaged over.
3. **Post-hoc projection beats internal projection** (CI excluding zero). The
   operator is a good post-processor and a bad training-time constraint.
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

- **Anything about sample efficiency (Q2).** Untested; needs Part C.
- **Whether the internal-vs-post-hoc result is the operator or the STE.**
  Confounded; §3.
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
