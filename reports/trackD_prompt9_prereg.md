# PROMPT 9 — pre-registered predictions (C0)

**Written before any Part B sweep or Part C training run.** Committed standing
alone so git history establishes the timestamp.

Primary statistic: paired per-trial median, **per SNR bin**. Pooled scalars are
never headlines and never decision rules.

## Correcting for a documented bias

Across stages 1–4 every wrong prediction ran the same direction: **I
underestimated what the learned components work out for themselves.** P6, X1,
P9 and the Part C mechanism prediction all failed that way. So where a
prediction concerns *a structural addition to a trained network*, I bias my
number **down** (the network will already have got there); where it concerns
*the network's own capability*, I bias **up**.

This has a sharp consequence for P15, which I therefore split: I predict the
**classical** HS advantage grows as `P` falls, and the **learned** one does
not. That is the non-obvious half, and it is the half the bias correction
demands.

---

## P11 — `Δ_HS` is flat in `K` at fixed `P/2K`

From A1: the compression ratio `3ΣL_k/(2NK) = 3L̄/(2N)` has `K` cancelled
[MATH], and the projection acts per user-column.

**Prediction: `Δ_HS` varies by less than ±0.15 dB across `K ∈ {2,3,4}` at
`P/2K = 3.33` and matched `L/cap`, with no monotone trend.** Absolute NMSE will
degrade with `K` (magnitude nonlinearity, more users per pilot); the *contrast*
should not.

**Falsifier:** a monotone trend exceeding 0.3 dB across the `K` range, or any
adjacent pair differing by more than 0.3 dB with CIs excluding each other. That
would mean `K` is a governing variable and the DoF cancellation does not
transfer to estimation performance.

## P12 — the `N` sweep collapses onto `r_eff/cap`

**Prediction: the `N ∈ {16, 32, 64}` curves lie within ±0.3 dB of one another
at matched `r_eff/cap`.** The A2 pre-index makes this a genuine test: the top
cell sits at `r_eff/cap = 0.544` (N=16) and `0.508` (N=64), straddling the
measured N=32 zero crossing of 0.518, so all three should be near zero there.

**Prediction of the crossing itself: each N crosses zero at
`r_eff/cap = 0.52 ± 0.08`.**

**Falsifier:** a systematic ordering by `N` exceeding 0.5 dB at matched
`r_eff/cap`, or zero crossings differing by more than 0.15 in `r_eff/cap`. A
failure to collapse is a finding — it would mean aperture enters beyond the
rank ceiling.

## P13 — SNR-balanced loss

A4 measured 94.3% of loss and 85.9% of gradient norm coming from below 5 dB.
Focused `[5,20]` training gained U1 **+2.653 dB** at high SNR but abandons the
low-SNR range. Balancing keeps the full range.

**Prediction, both halves stated:**
- **SNR ≥ 5: +1.5 to +2.2 dB**, most likely **+1.8 dB**, versus the `P=20`
  uniform-loss URformer. (Biased up: this is the network's own capability, and
  focused training already showed +2.65 is reachable.)
- **SNR < 5: −0.10 to −0.40 dB**, most likely **−0.25 dB**. Some cost is
  arithmetically unavoidable — the lowest bin's weight drops ~18×.

**Falsifier:** high-SNR gain below +0.8 dB (balancing does not recover what
focusing did), or low-SNR cost worse than −0.6 dB (balancing is just focusing
with extra steps).

## P14 — matched-pilot training at `P = 10`

The `P=20` model evaluated at `P=10` scores **−5.39 dB** (median, SNR 5 dB).

**Prediction: matched-pilot training at `P=10` beats it by +1.2 to +2.5 dB,
most likely +1.7 dB.** Biased up: this is pure network capability under a
matched condition, and PROMPT 8's pilot figure showed a visible kink at the
training point (`P=20`), which is exactly the overfit-to-training-condition
signature that matched training removes.

**Falsifier:** margin below +0.5 dB. That would mean the `P=20` model already
generalizes across pilot count, and the whole efficiency/generalization
distinction is moot — a clean, publishable negative.

## P15 — the HS advantage as `P` falls: SPLIT prediction

`P=10` at `K=3` sits at 1.67× the `P ≥ 2K` identifiability floor.

**Classical half — prediction: `Δ_HS` grows as `P` falls.** Existing fixed-`r`
evidence already points this way (PROMPT 8 pilot sweep, SNR 5 dB:
`+2.12 dB` at `P=10` versus `+1.72 dB` at `P=20`). **I predict the
adaptive-rank version shows the same ordering, with `Δ_HS(P=10) −
Δ_HS(P=20) = +0.3 to +0.8 dB`.**

**Learned half — prediction: it does NOT grow.** G1 at `P=10`, matched-trained,
versus matched-trained URformer at `P=10`: **+0.8 to +1.4 dB on SNR ≥ 5**,
i.e. statistically indistinguishable from the `+1.178 dB` G1 achieved at
`P=20`, and **−0.05 to −0.20 dB on SNR < 5**.

The reasoning is the bias correction: at `P=10` the network has *less* data per
realization and so *more* to gain from a prior — but stage 4 showed that when a
learned model is under-resourced, the structural gain it appears to offer is a
training-adequacy proxy that evaporates once the baseline is trained adequately.
Both arms here are matched-trained at `P=10`, so that proxy is controlled away.

**Falsifier for the split:** if the learned advantage at `P=10` exceeds the
`P=20` value by more than +0.5 dB on SNR ≥ 5, the bias correction was wrong in
this instance and the prior genuinely supplies more near the identifiability
floor. I will say so.

---

## What would make me revise the whole framing

If **both** P12 collapses hold *and* B6 lands near the A2-predicted +1.30 dB on
Xiao's channel, the classical claim stops being "works for our `L_k ∈ [3,7]`"
and becomes "works wherever `r_eff/cap ≲ 0.5`, on channels specified by other
people". That is the strongest available outcome tonight and it costs no
training.

If P12 fails to collapse, that is equally informative and I will report it as a
finding rather than a gap.
