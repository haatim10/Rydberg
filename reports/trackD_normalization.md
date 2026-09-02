# Normalization re-analysis — PROMPT 9 Part A

No new training. No estimator or config changed. Claims tagged **[FACT]**
(verified, with `file:line`), **[MATH]** (derived), **[HYP]** (needs the named
experiment).

**Thesis, and it holds:** the study was under-normalized, not over-specific.
Both collapses attempted here worked.

---

## A1. The structural compression factor — `K` cancels

**[FACT]** `SystemModel.pdf`, "Parameter count" paragraph, quoted verbatim:

> "Treated as unstructured, **H** possesses `2NK` real degrees of freedom.
> Under the geometric model it is described by `3 Σ_k L_k` real parameters,
> comprising two per complex path gain and one per angle of arrival."

**[MATH]** The compression ratio is

```
3 Σ_k L_k / (2NK)   →   with L_k ≡ L̄:   3 K L̄ / (2 N K)  =  3 L̄ / (2 N)
```

**`K` cancels.** Verified numerically for all `(K, L̄) ∈ {2,3,4,6} × {3,5,7}`:
the general and simplified forms agree to machine precision. At the default
`L̄ = 5, N = 32` the structural model uses **23.4%** of the unstructured degrees
of freedom.

**[HYP]** Since the Hankel projection acts per user-column, `Δ_HS` should be
**invariant in `K`**, even though absolute NMSE degrades with `K` through the
magnitude nonlinearity and through pilot adequacy. **B2 tests this** — and B2
must hold `P/2K` fixed, or it changes pilot adequacy at the same time and
measures a mixture of two effects.

## A2. Re-index by effective rank, not `L` — the collapse works

`L` is a property of one generator. `r_eff` is a property of any channel. They
come apart badly: `L = 16` is algebraically full rank but has `r_eff = 8.73`.

**[FACT]** Track B Experiment C re-indexed (`r_eff` = median Roy–Vetterli
effective rank of the noiseless channel columns, 200 trials × 3 users per cell;
`cap = floor(32/2) = 16`):

| L | L/cap | **r_eff** | **r_eff/cap** | Δ_HS (dB) | CI95 |
|---|---|---|---|---|---|
| 2 | 0.125 | 1.90 | **0.119** | +7.043 | [+6.73, +7.34] |
| 4 | 0.250 | 3.40 | **0.212** | +3.556 | [+3.38, +3.73] |
| 6 | 0.375 | 4.57 | **0.285** | +1.792 | [+1.64, +1.96] |
| 8 | 0.500 | 5.70 | **0.356** | +1.038 | [+0.93, +1.15] |
| 10 | 0.625 | 6.54 | **0.408** | +0.577 | [+0.45, +0.69] |
| 12 | 0.750 | 7.36 | **0.460** | +0.266 | [+0.19, +0.34] |
| 14 | 0.875 | 8.11 | **0.507** | +0.046 | [−0.05, +0.14] |
| 16 | 1.000 | 8.73 | **0.546** | −0.117 | [−0.21, −0.04] |

**[FACT] The zero crossing moves from `L/cap = 0.90` to `r_eff/cap = 0.518`.**
The `r_eff` index is the better abscissa: the useful region is
`r_eff/cap ≲ 0.5`, which is a statement about *any* channel, not about `L`
values of one simulator.

**The payoff — a falsifiable prediction for a channel we never fit.** Placing
Xiao's Saleh–Valenzuela channel on the same axis:

| channel | r_eff | r_eff/cap | **Δ_HS predicted by the collapse** |
|---|---|---|---|
| Xiao SV, clustered (±5°, Cui precedent) | 5.30 | **0.331** | **+1.30 dB** |
| Xiao SV, literal Table I (40 indep. DoAs) | 11.97 | **0.748** | **−0.12 dB** |

**[HYP]** These are predictions, not measurements. **B6 tests the clustered one
directly** by running EM-GS and HS-EM-GS on SV channels — ~25 min, classical,
no training. If HS-EM-GS lands near +1.3 dB on a channel model built to
somebody else's specification and never used to tune `r`, the classical claim
generalizes across channel models rather than across `L` values of ours.

**[FACT] What cannot index this plot: the estimate's `r_eff`.** PROMPT 8 C1
measured it at 8.91–11.77 across `L = 1…16` at 5 dB — set by the noise floor,
not the channel. Using it would collapse every cell onto one point.

Also precomputed for Part B (so B1 can index its cells directly):

| N | cap | L | r_eff | r_eff/cap |
|---|---|---|---|---|
| 16 | 8 | 2 / 4 / 7 | 1.88 / 3.10 / 4.35 | 0.235 / 0.388 / **0.544** |
| 64 | 32 | 8 / 14 / 29 | 6.33 / 9.89 / 16.24 | 0.198 / 0.309 / **0.508** |

Note the design consequence: the intended `L/cap ∈ {0.25, 0.45, 0.90}` maps to
`r_eff/cap ∈ {0.20–0.24, 0.31–0.39, 0.51–0.54}`. The top cell lands almost
exactly on the measured zero crossing (0.518) at **both** N — which is what
makes B1 a real test of the collapse rather than a re-description of it.

## A3. The EM filter's value collapses onto κ

**[FACT]** Paired median `GS − EM-GS` against median
`κ = 2Z|Y|/σ²` evaluated at the true channel (a property of the operating
point; no estimator involved), from the existing PROMPT 8 sweep rows:

| κ | 2.02 | 3.54 | 8.67 | 24.86 | 76.72 | 238.5 | 756.9 |
|---|---|---|---|---|---|---|---|
| GS − EM-GS (dB) | +2.965 | +1.743 | +0.683 | +0.208 | +0.060 | +0.019 | +0.007 |
| **Δ · κ** | 6.00 | 6.17 | 5.92 | 5.17 | 4.60 | 4.51 | 5.23 |

**[FACT] The product `Δ · κ` is 5.32 (median), range [4.51, 6.17], while κ
spans 375×.** So

> **`Δ_{GS−EM-GS} ≈ 5.3 / κ` dB**

**[FACT] A second family confirms it.** At fixed SNR = 5 dB with `P` swept
10→35, κ is essentially constant (24.65–24.90, as it must be — κ has no direct
`P` dependence) and `Δ` scatters around 0.21–0.40 about the predicted
`5.3/24.8 = 0.21`. **κ predicts the gap regardless of pilot count.**

**This generalizes the finding.** "The Bessel filter is inert in our regime" was
a statement about RSR = 10 dB. The statement that survives is: *the EM filter
buys `≈ 5.3/κ` dB, so it earns its place only when `κ ≲ 5`* — i.e. below about
−5 dB SNR at this reference level.

**[HYP]** that the constant 5.3 is universal rather than RSR-specific. Settling
it needs a second RSR, which tonight's scope does not authorize.

**One correction to the brief:** it cites "the 0.068 dB GS-vs-EM-GS gap". I
cannot source that exact figure in this repository. My measurement closest to
it is **+0.060 dB at κ = 76.7 (SNR = +10 dB)**. I report what I measure rather
than adopt a number I cannot trace.

## A4. Data scaling per parameter

Model: 1,586,900 parameters. Intrinsic problem dimension `3 Σ_k L_k = 45` real
(at `K = 3, L̄ = 5`).

| n_train | test median NMSE | samples / parameter | samples / intrinsic dim | real measurements / parameter |
|---|---|---|---|---|
| 20,000 | −9.480 | 0.0126 | 444 | 16.1 |
| 40,000 | −10.357 | 0.0252 | 889 | 32.3 |
| 80,000 | −10.831 | 0.0504 | 1,778 | 64.5 |

**[FACT] Even at 80k the model sees 0.05 samples per parameter** — i.e. ~20
parameters per training example. Counted as real-valued measurements
(`2NP = 1280` per sample) it is 64.5 per parameter, still a heavily
over-parameterized regime. This restates the scaling for model sizes other than
1.59M and explains why the 20k→80k gain (1.35 dB) had not saturated.

## A5. What Xiao et al. do and do not state about Fig. 4

**[FACT], about the paper's text, not an accusation:** Xiao et al. describe
Fig. 4 only as "evaluated at a fixed SNR of 5 dB". **The paper does not state
whether the networks were retrained per pilot count `P`.**

Why it matters: our own pilot curve has exactly this ambiguity, and PROMPT 9
C2/C3 resolve it for us by training matched models at `P = 10` and `P = 35`.
That separates **pilot efficiency** (matched training at each `P`) from
**pilot-count generalization** (one `P = 20` model evaluated OOD) — a
distinction the paper does not draw either way, and which our PROMPT 8 pilot
figure could only label, not measure.

---

## Scoping correction to the stage-5 verdict

The PROMPT 8 verdict said "**STOP** the learned line". That wording was too
broad. It meant: **stop inventing new learned structural architectures.** It did
not mean retiring URformer, which remains the strongest practical estimator in
these results at low and moderate SNR (−11.27 dB at `P = 20`, SNR 5 dB, versus
−9.59 for HS-EM-GS and −7.87 for EM-GS). **URformer stays a first-class arm.**
`reports/trackD_generalization_audit.md` §E3 is amended accordingly.

## Files

- `reports/trackD_partA9_normalization.json` — all Part A numbers
- `results/track_d/normalization/fig_normalization_collapse.png`
- `scratch/trackD_partA9_normalization.py`, `scratch/trackD_partA9_plots.py`
