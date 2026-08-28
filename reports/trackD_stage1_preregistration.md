# Track D stage 1 — pre-registered predictions

**Written before any training run.** Committed separately, ahead of the stage-1
runs, so the timestamp and git history establish that these were not written
after seeing results. Nothing in this file may be edited once the runs start;
outcomes go in `reports/trackD_stage1_results.json` and the Part C report.

Operating point: `N=32`, `K=3` (ours, not the paper's 4), `L_k ~ U{3..7}`,
`P=20` fixed, RSR 10 dB in our single-user convention, spectral init
throughout, 20k/2k/2k, 50 epochs, no early stopping.

---

## P1 — arm 2 (filter-only, ~980 parameters) lands within 0.2 dB of EM-GS

**Predicted: HOLDS.**

The measured paired GS-minus-EM-GS difference at this operating point is
**+0.068 dB** (bootstrap CI [0.044, 0.100], paired std 0.217 dB). GS is EM-GS
with the Bessel filter switched off entirely. So replacing that filter with a
*perfect* one buys at most ~0.07 dB here: the classical filter is very nearly
inert at RSR 10 dB, because `R(κ) ≈ 1` across most of the measured κ range
(p50 = 66, and `R(66) = 0.992`).

A learned filter therefore has almost no headroom to exploit. Arm 2's 980
parameters can only reshape a multiplier that is already within 1% of unity
over the bulk of the data.

## P2 — essentially all of any full-URformer gain is the Transformer residual

**Predicted: HOLDS**, as a direct consequence of P1.

If arm 2 ≈ EM-GS, then any gap between arm 1a and EM-GS is attributable to the
Transformer, not to unrolling the filter. The attribution is
`gain_transformer ≈ gain(arm 1a) − gain(arm 2)`.

**If P1 or P2 is violated that is a genuine finding**, not noise to smooth over:
it would mean the gated *filter* is doing real work at this operating point,
which contradicts the measured 0.068 dB classical headroom and would require
explaining where the extra freedom comes from.

## P3 — does arm 1 clear the 2 dB bar?

**My prediction: NO. Arm 1a does not cleanly clear 2 dB. I expect a gain of
roughly 0.5–1.5 dB versus EM-GS-spectral.** Confidence: moderate (~60%).

Reasoning, stated as mechanism rather than hedge:

**Why some gain is available.** The oracle-phase bound leaves real headroom
(see Part A §A2). The channel is strongly structured — `L_k ≤ 7` paths in an
`N=32` array — so the true `G` lives near a low-dimensional manifold that no
per-element estimator exploits. Track B established that this structure is
highly exploitable at exactly `N=32`: HS-GS beat EM-GS by **+2.85 dB** there by
imposing a Hankel low-rank constraint along the antenna axis. So a learned
model that captures the same structure could plausibly find 2 dB.

**Why the paper's architecture is poorly placed to capture it.** This is the
crux. The user-token scheme gives the Transformer **`K = 3` tokens**.
Self-attention over three tokens is nearly trivial — it can mix three vectors.
The structure Track B showed to be worth 2.85 dB lives **across the 32
antennas**, and in this tokenization the antenna axis is *inside* the token
(dimension `2N = 64`), reachable only by the dense input/output projections and
the per-token FFN, never by attention.

So arm 1a is effectively a **per-user MLP denoiser** on a 64-dimensional
real vector, plus a 3-token mixer. It can learn some antenna-axis structure
through those projections, but it has no mechanism aimed at it, and it must do
so while generalizing across a 20 dB SNR range.

That gap between where the exploitable structure lives (antennas) and where the
architecture attends (users) is why I predict under 2 dB.

**What each outcome would mean:**

| outcome | reading |
|---|---|
| arm 1a ≥ 2 dB | prediction wrong; the dense projections capture antenna structure better than I expect. Report as a clean win. |
| 0.5–2 dB | predicted range. URformer helps but does not clear the pre-registered bar — a real but sub-threshold effect, reported as **not met**. |
| ≤ 0.5 dB | the architecture does not transfer to our geometric ULA model at all. Negative result, and the strongest argument for the antenna-token variant and HS-URformer later. |

**This also sharpens the eventual HS-URformer case.** If arm 1a underperforms
because attention is pointed at the wrong axis, then adding the Hankel
projection — which acts along exactly the axis the Transformer cannot see — is
not just a stacking of two methods but a targeted fix for a diagnosed defect.
I am recording that reasoning now so it cannot be retrofitted later.

---

## Bar, unchanged

> Success = URformer-spectral beats EM-GS-spectral by **≥ 2 dB median NMSE** on
> the held-out test set, on **paired** trials, with the bootstrap CI of the
> paired per-trial difference excluding zero.

Reported as the paired-difference *distribution*, not two means.
