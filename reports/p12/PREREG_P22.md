# PROMPT 14 Part C1 — pre-registration P22

**Committed before the aperture sweep runs.**

## The cell

`N ∈ {8,16,32}` at the **principal operator**: `n_cz = 4`, `T = 100`,
selection budget 25, held-out adaptive order, RSR 12 dB, `P ∈ {10,30}`,
six-point SNR grid `{-5,0,5,10,15,20}`, 400 paired trials per point,
Track B generator. 36 operating points, 14 400 trials.

## Cost estimate and the decision it triggered

Measured single-threaded, uncontended, 6 trials per cell:

| N | s/trial (P=10) | s/trial (P=30) |
|---|---|---|
| 8 | 1.011 | 1.008 |
| 16 | 2.451 | 2.725 |
| 32 | 7.134 | 7.919 |

Full sweep **14.8 CPU-hours**. The bridging run (`results/p12/B1.json`) is
already the `N=32, P=30` cell at this exact operator, so it is reused and the
new work is **9.55 CPU-hours ≈ 2.4 h elapsed** sharded four ways.

The rule "under six hours of classical compute" is ambiguous between CPU-hours
(9.55, **over**) and elapsed (2.4, **under**). Running it on the elapsed
reading; recorded here so the choice is visible.

## What actually changes from the existing sweep

The existing aperture numbers come from `T = 50`, selection budget 20 —
**everything else is already identical** (same `hs_gs_auto`, same `n_cz = 4`,
same RSR 12 dB, same pilot counts, same SNR grid, same 400 trials). So this
sweep changes exactly two hyperparameters.

Evidence on how much that matters: at the one cell measured both ways
(`N=32, P=30`), ratio-of-sums mean is **2.531** at `T=50` and **2.536** at
`T=100` — a difference of **0.005 dB**.

## Prediction

Mean gain over the 12 operating points at each N, against the existing
`−0.19 / +0.78 / +2.85`:

| N | existing | predicted band | point |
|---|---|---|---|
| 8 | −0.19 | **[−0.35, +0.05]** | **−0.20** |
| 16 | +0.78 | **[+0.60, +0.95]** | **+0.76** |
| 32 | +2.85 | **[+2.65, +3.00]** | **+2.83** |

EM-GS spread across N, against the existing `0.012` dB:
**predicted ≤ 0.10 dB**, point estimate **0.03 dB**.

**Bias correction applied.** Past misses consistently underestimated what the
unstructured arm manages on its own. Doubling `T` gives EM-GS twice the
iterations to converge, so if either arm gains it should be the unstructured
one, which would *reduce* Δ_HS. Each band above is therefore centred slightly
**below** the existing value rather than on it.

## Falsifier

P22 fails if **any** of:

- a mean gain at any N falls outside its predicted band;
- the EM-GS spread across N exceeds **0.15 dB**;
- the sign at `N=8` is positive with a bootstrap CI excluding zero (which would
  break the Step-4 prediction the letter is built on).

## Consequence either way

If the sweep runs and P22 holds, the letter reports these numbers and
**retires the old aperture values entirely**, so panel (a) and panel (b) of
Fig. 1 share one configuration. If P22 fails, the measured numbers are still
what the letter reports — the prediction failing does not license keeping the
old values — and the failure is stated.
