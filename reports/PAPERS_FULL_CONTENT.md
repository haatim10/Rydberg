# The two papers — complete written content

Everything written into the manuscripts, in one place: both abstracts, every
section's substance, every number with the results file it came from, every
figure and table, and every open `\todo`.

**Two papers are being submitted. A third directory is superseded.**

| # | file | pages | status |
|---|---|---|---|
| 1 | `paper/merged/haatim_hsgs_bounds_and_effective_rank.tex` | **5** | submit |
| 2 | `paper/spl2/haatim_structural_priors_evaluation.tex` | **4** | submit |
| — | `paper/spl1/main.tex` | 4 | **superseded** — absorbed into Paper 1 |
| — | `paper/hsgs.tex` | 6 | **superseded** — the spine of Paper 1 |

Both target IEEE Signal Processing Letters, `\documentclass[journal]{IEEEtran}`,
5-page limit. Author on both: Abdullah Haatim, BITS Pilani — Pilani Campus,
`f20220519@pilani.bits-pilani.ac.in`.

---

# PAPER 1 — *Hankel-Structured Channel Estimation for Rydberg Atomic Receivers: Bounds, Effective-Rank Scaling, and Out-of-Model Validation*

5 pages. Merges the earlier HS-GS manuscript (the spine: bounds and aperture
scaling) with the effective-rank and out-of-model work.

## Abstract

> Rydberg atomic receivers measure field magnitude only, making multi-user
> channel estimation a biased phase-retrieval problem in which unstructured
> solvers are flat in array size: their Fisher information is block diagonal
> across receive elements, so a larger aperture buys nothing under a normalised
> metric, and only a structural prior converts aperture into accuracy. We
> enforce the low-rank Hankel structure of the uniform-linear-array response
> inside an exact expectation–maximization Gerchberg–Saxton iteration, select
> the order from a held-out pilot residual, and benchmark against a Rician
> Cramér–Rao bound in which each magnitude measurement contributes rank-one
> information, together with a geometry-constrained bound on the
> `3ΣL_k`-parameter manifold. The gain grows from −0.19 to +2.85 dB as `N` goes
> 8→32 while the unstructured baseline moves by 0.012 dB. We then show the
> governing variable is not path count but effective rank relative to capacity,
> and use that relation to predict the gain on two channel models it was never
> fitted to. Simulation only; the bounds do not govern biased estimators.

**Index terms:** Rydberg atomic receiver, channel estimation, phase retrieval,
Hankel matrix, constrained Cramér–Rao bound, effective rank.

## I. Introduction

Rydberg receivers read out field **magnitude**, discarding phase, so multi-user
channel estimation is biased phase retrieval. Gerchberg–Saxton alternating
projection is the natural solver; recent work unrolls it into a trained
network, and a rank projection has been placed inside projected gradient
descent for the same problem.

**The load-bearing observation:** unstructured solvers' accuracy *does not
improve as the array grows*. Row `n` of `GS` enters only measurement row `n`,
so the Fisher information is block diagonal across receive elements and the
per-element problem is unchanged by adding elements. Aperture is only useful to
an estimator that couples elements — which is what a structural prior does.

On a ULA each path contributes one spatial complex exponential, so each
column's Hankel lifting is rank deficient. But knowing *when* it applies had
meant a qualitative appeal to sparsity, which is unusable: **a 16-path channel
on a 32-element array is algebraically full rank yet has effective rank 8.73**,
and its lifting is compressible with nothing sparse about it.

**Contributions.** (1) A benchmark, not just a baseline: rank-one Rician CRLB
plus a geometry-constrained bound in tangent-space form; EM-GS sits within
0.05 dB of the unconstrained bound, so the benchmark is credible. (2) The
aperture-scaling result and its Fisher explanation. (3) Replacement of the
path-count heuristic by `r_eff/r_max`, with `r_max = ⌈N/2⌉` derived. (4) Two
out-of-model predictions, each registered before measurement.

## II. System Model

`Z = |GS + B + W|`, elementwise, `W ~ CN(0, σ²)`. The modulus is never
linearised. User `k`'s channel is `g_k = Σ_ℓ α_kℓ a(θ_kℓ)` with
`[a(θ)]_n = e^{−jπn sinθ}`, so each column is a sum of complex exponentials in
the element index.

**RSR convention, stated once.** Defined against a *single* user. Since `K`
users transmit at once, a nominal single-user RSR of 12 dB is
`12 − 10log₁₀K = 7.23 dB` against the K-user total at `K = 3`; 10 dB
single-user is 5.23 dB. *(See open item — the earlier draft said 7.15 dB.)*

**Two averaging domains, kept distinct.** Within an operating point NMSE is a
**ratio of sums**, never a mean of per-trial decibels: the ratio of sums
estimates the ensemble quantity a CRLB bounds, whereas a mean-of-decibels is
smaller by Jensen. Across SNR the mean is of per-SNR decibel values, after
conversion; linear NMSE spans three orders of magnitude, so a linear average
across SNR is dominated by the lowest-SNR point.

> **The rule used throughout:** ratio-of-sums wherever a curve is compared to a
> bound; paired per-trial median with a bootstrap interval for paired
> contrasts. Both are reported; neither does the other's job.

## III. Hankel Structure and the HS-GS Estimator

**Capacity.** For pencil `p`, `H_p(g)` is `(N−p)×(p+1)` with rank
`min(L_k, N−p, p+1)`, so

```
r_max(N) = max_p min(N−p, p+1) = ⌈N/2⌉        attained at p = ⌊N/2⌋
```

The **ceiling, not the floor**: at `N = 7` the maximum is 4, and the two
expressions coincide only for even `N`. A constraint at `r = r_max` is
satisfied by every vector and carries no information.

**`K` cancels.** Unstructured, `G` has `2NK` real DoF; under the geometric
model, `3ΣL_k`. With `L_k ≡ L̄`:

```
3ΣL_k / (2NK) = 3KL̄ / (2NK) = 3L̄ / (2N)
```

independent of `K`. Since the projection acts per user column, this predicts
K-invariance of the structural gain at fixed pilot adequacy `P/2K`.
*(Source: `reports/trackD_normalization.md` A1, verified for all
`(K, L̄) ∈ {2,3,4,6}×{3,5,7}` to machine precision.)*

**The estimator.** HS-GS interleaves, after every exact EM-GS update, a Cadzow
projection `P_r(g) = H_p†(Π_r(H_p(g)))`. **One application is not idempotent** —
it is one step of an alternating projection between the rank set and the Hankel
set, not a projection onto their intersection — so the sweep count is part of
the operator's definition and is stated with every result. The exact step is
the EM-GS update throughout, so HS-GS and HS-EM-GS name the same estimator;
HS-GS is used to match the released code.

**Order selection uses no oracle.** Hold out a subset of pilots, run at reduced
iteration count for every candidate `r ∈ {1,…,r_max}`, select the `r`
minimising the held-out residual. The rule also **abstains**: when it selects
`r ≥ r_max` the projection is the identity and HS-GS reduces exactly to EM-GS.

**The governing variable.** Roy–Vetterli effective rank of the *noiseless*
column's lifting, `r_eff = exp(−Σ p_i ln p_i)` with `p_i = σ_i/Σσ_j`, taken
relative to `r_max`. Unlike `L`, this is a property of any channel rather than
of one generator. It must be computed on the noiseless channel: the effective
rank of the *estimate* is set by the noise floor and collapses every
configuration onto one value.

## IV. Performance Benchmark

### Rank-one Fisher information

Conditional on `G, S, B` the noiseless field is a constant `λ`, so `z = |λ+w|`
is Rice distributed and its density depends on `λ` **only through `a = |λ|`**.
With `d/dx log I₀(x) = R(x)` and `κ = 2za/σ²`, the score is
`∂_a log p = (2/σ²)[zR(κ) − a]`. Writing `β = σ⁻⁴(E[z²R²(κ)] − a²)`, one
measurement carries scalar information `4β`, so

```
J_n = Σ_p 4β(a_np, σ²) u_np u_npᵀ
```

> **Each magnitude measurement contributes rank-one information, not rank two.**
> Crediting both quadratures overstates it and gives a bound **too low by
> 0.17–4.64 dB** across the 36 points.

An identifiability check settles the form: at `K = 3, P = 3` the per-row
problem is unidentifiable and the rank-one construction is correctly singular,
whereas the rank-two construction claims full rank.

Because row `n` of `GS` enters only measurement row `n`, **`J` is block
diagonal across receive elements** — the reason an unstructured estimator is
flat in `N` under a normalised metric.

*(Source: `results/track_b/constrained_crlb.json`, 400 trials. Code:
`rydberg_sim/crlb.py`, `scripts/constrained_crlb.py`.)*

### Constrained bound on the geometric manifold

HS-GS exploits the geometric model, whose parameters `φ ∈ R^{3ΣL_k}` number
about 45, **independent of `N`**. The textbook form `D(DᵀJD)⁻¹Dᵀ` is unusable,
`D = ∂η/∂φ` being rank deficient: at `N = 8` its mean numerical rank is **40.4
against ≈44.7 parameters**, and **28.1% of realisations have `3ΣL_k > 2NK`**,
so `φ ↦ G` is not injective. The Gorman–Hero / Stoica–Ng tangent-space form is
used, with `U` an orthonormal basis of `range(D)`:

```
CCRB = U(UᵀJU)⁻¹Uᵀ
```

which depends on the tangent *space* and is invariant to over-parameterisation.
Both bounds average over the same realisations as the estimator curves.

**Bias caveat.** Both bounds constrain the covariance of *unbiased* estimators.
Every estimator here runs to a fixed budget with a shrinking update, and HS-GS
additionally truncates rank, so none is unbiased: the unconstrained CRLB is a
reference for GS and EM-GS only, and the CCRB does not forbid a sufficiently
biased structured estimator from falling below it.

## V. Numerical Results

### TABLE I — the two configurations

Every figure is internally single-configuration; none mixes points from the two.

| | Bound comparisons | Effective-rank study |
|---|---|---|
| `N` | 8, 16, 32 | 16, 32, 64 |
| `K` / `P` | 3 / 30 | 3 / 20 |
| `L_k` | `U{3,7}` | `U{3,7}` |
| SNR | −5 to 20 dB | `U[−10, 20]` dB |
| RSR (single-user) | 12 dB | 10 dB |
| Iterations `T` | 50 | 100 |
| Cadzow sweeps | **4** | **1** |
| Statistic | ratio-of-sums | paired median, CI |
| Trials | 3.6×10⁴ total | 83–874 per cell |

The older configuration is canonical for everything compared to a bound,
because it is what the bounds were computed against.

### A. Main result against the bounds — FIG. 1

At `N = 32`, `P = 30`, RSR 12 dB: **EM-GS tracks the unconstrained CRLB to
within 0.05 dB for SNR ≥ 5 dB**, so it is close to efficient for the
unstructured problem and the benchmark is credible; at −5 and 0 dB the curves
separate by 0.30 and 0.24 dB, consistent with low-SNR bias. The geometric CCRB
lies **0.87–9.98 dB** below the unconstrained bound, quantifying what the
structure is worth in principle; HS-GS occupies part of that gap, improving on
EM-GS by **2.27–3.04 dB**.

### B. Aperture scaling — FIG. 2 (the abstract's second sentence)

Averaged over twelve operating points per array size:

| `N` | gain | win rate | constraint active |
|---|---|---|---|
| 8 | **−0.19 dB** | 33.5% | 57.5% |
| 16 | **+0.78 dB** | 74.0% | 92.7% |
| 32 | **+2.85 dB** | 95.3% | 99.6% |

> **Under the same averaging EM-GS varies by 0.012 dB** — exactly as the
> block-diagonal Fisher structure requires. The `N`-dependence belongs to the
> structural prior, not to a degrading baseline.

**`N = 8` is a mixed result, not a small positive one.** There `r_max = 4`
while `L_k ∈ {3,…,7}`, so the projection is inactive in 43% of trials and where
active often truncates real components. HS-GS gains **+0.744 dB at −10 dB SNR**
but loses **0.190–0.403 dB above 0 dB**, and the sign of any scalar summary
depends on the pooling. Truncation bias does not shrink with noise, so as
variance falls the trade stops paying.

*(Source: `trackB_hankel_emgs/results/experiment_B_array_size.csv`.)*

### C. From path count to effective rank — FIG. 3

Fixing `L_k = L` and sweeping to the ceiling, the gain decays strictly
monotonically:

```
L  =  2      4      6      8     10     12     14     16
Δ  = 7.04   3.56   1.79   1.04   0.58   0.27   0.05  −0.12   dB
```

with **EM-GS flat to 0.19 dB** across the sweep — which rules out generic
denoising, since that would persist at large `L` where the estimate is equally
noisy, rather than vanish exactly where the prior becomes vacuous.

*(Source: `trackB_hankel_emgs/results/experiment_C_path_count.csv`.)*

**Re-indexing by `r_eff/r_max` generalises it.** The zero crossing moves from
`L/r_max = 0.90` to **`r_eff/r_max = 0.518`**, and the useful region becomes
**`r_eff/r_max ≲ 0.5`** — a statement about any channel. Across `N ∈ {16,32,64}`
the crossing is **0.588 / 0.518 / 0.544**, a spread of **0.070** over a
fourfold change in aperture. The `N = 16` and `N = 64` values are extrapolated
from the last two measured points rather than bracketed.

> **The gain magnitude does not collapse onto the same axis, and the residual
> is stated rather than a collapse reported.** Over the shared window, `N = 16`
> lies above `N = 64` by a mean of **0.493 dB** and a maximum of **0.901 dB**,
> one-signed throughout and monotone in `N`. **We have no account of it.** One
> candidate is pencil quantisation at small capacity — `r_max(16) = 8` admits
> eight ranks against 32 at `N = 64` — but that is untested.

*(Source: `reports/trackD_partB9_analysis.json` `B1_collapse`.)*

### D. Out-of-model prediction: the falsification test — FIG. 4

The earlier stage of this work concluded robustness under clustered propagation
was not established and named the decisive next experiment: *repeat the
principal comparison under a clustered generator — a falsification test, not a
confirmation exercise.* **This is that test, run twice.** In each case
`r_eff/r_max` was computed, a predicted `Δ_HS` read off the relation, and **the
prediction committed to version control before the measurement was run**; the
relation was not refitted afterwards.

| channel | `r_eff/r_max` | predicted | measured | verdict |
|---|---|---|---|---|
| Clustered Saleh–Valenzuela | 0.331 | **+1.30 dB** | **+1.173** CI [+1.017, +1.380] | inside CI, err 0.13 dB |
| `L = 10` configuration | 0.400 | **+0.648** [+0.367, +1.015] | **+0.963** CI [+0.808, +1.086] | inside interval, err **0.315 dB** |

The second is reported as such, **not as a second 0.13 dB hit**. Together the
two bound what the relation is good for: it places an unseen channel on the
correct part of the curve, and it is not accurate to a tenth of a decibel in
general.

**Clustering did not eliminate the gain** — the outcome the earlier stage
flagged as possible. On a third reading in which all 40 rays are drawn
independently, `r_eff/r_max` rises to 0.748, the relation predicts −0.12 dB,
and the measured median is **exactly 0.000 dB**. The magnitude is right to
0.12 dB but **the mechanism is not the predicted one**: rather than degrading,
the order-selection rule **declines to project at all in 29.1% of trials
against 1.4% on the clustered model**, without being told which channel it
faces. Reported as a partial match.

*(Sources: `reports/trackD_partB9_analysis.json` `B6_xiao`, `B8_cui_measured`;
`reports/trackD_step0_cui_prediction.json`. Prediction commit `060205b`.)*

### E. Users, and a replication

**K-invariance** holds at SNR ≥ 5 dB, where `Δ_HS` across `K ∈ {2,3,4}` at
fixed `P/2K` spans **0.095 dB**. Pooled over all SNR it does not: the spread is
**0.294 dB and monotone in `K`**, missing a pre-registered falsifier of 0.3 dB
by **0.006 dB**, which is **not presented as a pass**. The pooled trend is
driven by bins below 0 dB where each cell contributes about 45 trials.

**The two configurations are a replication.** At `N = 32` the same contrast
measures **+2.85 dB** (bound-comparison config) and **+2.680 dB**
(effective-rank config) — different code, different RSR, iteration count,
Cadzow sweep count and pooling rule, **agreeing to 0.17 dB**.

## VI. Limitations and Conclusion

Because the Fisher information is block diagonal across receive elements, the
unstructured baselines are flat in array size and HS-GS alone converts a larger
aperture into lower normalised error: −0.19 → +2.85 dB as `N` goes 8→32
against 0.012 dB of baseline movement. The governing variable is effective rank
relative to capacity, not path count, and that relation predicted the gain on
two channel models it was never fitted to.

**Limitations, as written:** simulation only, no measured hardware and no
recorded channel data; covers the geometric ULA model and the
Saleh–Valenzuela configurations only; the gain magnitude does not collapse onto
the proposed axis and its residual is unexplained; the benefit is conditional
and **negative at `N = 8` above 0 dB**; a single scalar order is imposed on all
`K` users despite differing path counts; the projection is approximate on a
non-convex set with no convergence guarantee; equal user power leaves near–far
untested; and **both bounds constrain unbiased estimators while every estimator
here is biased**.

---

# PAPER 2 — *On the Evaluation of Structural Priors in Unrolled Channel Estimation for Rydberg Atomic Receivers*

4 pages. A finding about evaluation practice, not a critique of any paper.

## Abstract

> Adding an explicit structural prior to an unrolled channel estimator usually
> improves the reported metric, and the improvement is usually attributed to
> the prior. We show that in magnitude-only channel estimation for Rydberg
> atomic receivers, most of that improvement is an artifact of how such
> estimators are trained and scored. Normalized error averaged per sample over
> a mixed-SNR training set puts 89.7% of the gradient below 5 dB, leaving the
> high-SNR regime underfitted; a structural prior then supplies at high SNR
> what training did not. Train both arms adequately and a +1.209 dB structural
> advantage falls by a factor of fifteen, to +0.078 dB. The first casualty of
> this control was our own structured estimator. We propose a matched-adequacy
> protocol, show that simply rebalancing the loss recovers +1.902 dB at
> SNR ≥ 5 dB at no low-SNR cost, and show that pilot efficiency and pilot-count
> generalization are distinct quantities differing by 2.231 dB. Results are
> simulation only.

**Index terms:** Algorithm unrolling, channel estimation, evaluation
methodology, phase retrieval, Rydberg atomic receiver, structural priors.

## I. Introduction — the framing

The paper argues that evidence for structural priors in unrolled estimators is
usually not decisive, and the reason is a convention so widespread it is rarely
stated: **the training loss is a per-sample normalized error, averaged over a
training set spanning a wide SNR range.**

> **This is a finding about evaluation practice, and we did not arrive at it as
> critics.** We built a Hankel-structured unrolled estimator, measured a clear
> +1.209 dB advantage at high SNR, and believed it. The control was run to
> *characterize* that advantage, not to remove it. It removed it: the same
> contrast under matched training adequacy is +0.078 dB. **The first casualty
> of the protocol we propose was our own method**, and we report it because a
> result that survives an author's attempt to keep it is worth more than one
> that was never tested.

## II. System Model and Estimator

`Z = |GS + B + W|`. Each user's channel is a sum of `L_k` plane waves on the
array manifold, `L_k ~ U{3,7}` with angles uniform on (−90°, 90°) — **a
geometric ULA model, not a clustered one.** Baseline is EM-GS; the unrolled
estimator ("URformer") replaces `T` iterations with `T` learned layers, each a
data-consistency step followed by a learned filter, a gate, and a Transformer
block across users.

**Explicitly not a reproduction:** "the operating point differs, and reaching
the reported behaviour required roughly four times the stated data budget.
Nothing here should be read as a reproduction or a validation of prior work,
nor as a critique of any specific paper. The convention we examine is
near-universal, and our own estimator uses it."

**The reference curve is genie-aided** — unstructured LS formed from the *true*
phase of the noiseless field. Not a bound: it presumes information no receiver
has, and being restricted to unstructured LS it can be passed by an estimator
exploiting channel structure, which is observed below 5 dB.

## III. Attribution and the Matched-Adequacy Protocol

### Where the gain comes from — FIG. 1

Decomposition of the 3.345 dB improvement over EM-GS:

| component | contribution | parameters |
|---|---|---|
| learned per-layer filter | **0.147 dB** | 980 |
| attention block | **3.198 dB (96%)** | 1,585,920 |
| *total* | *3.345 dB of 4.183 available* | 1,586,900 |

The filter's contribution **does not grow with data** — it falls slightly,
0.193 → 0.147 dB from 20k to 80k samples — which is what a never-capacity-
limited 980-parameter component should do.

**Unrolling is doing real work independently of attention:** against a matched
control applying one Transformer block to a converged EM-GS estimate rather
than unrolling, the unrolled estimator gains **0.920 dB**, CI [+0.888, +0.988],
winning on 86.5% of trials.

*(Sources: `reports/trackD_stage2_report.md:107-110`,
`reports/trackD_stage3_report.md:98`.)*

### The confound

The training loss is `‖Ĝ−G‖²_F / ‖G‖²_F` averaged over a batch spanning the
whole SNR range. Because the normalized error of a low-SNR realization is
orders of magnitude larger than a high-SNR one, the average is dominated by the
low-SNR tail. Measured directly: **89.7% of the gradient comes from below
5 dB**, and the per-bin gradient share spans a factor of **31**.

The consequence: the high-SNR regime is systematically underfitted, and **any**
mechanism that improves high-SNR behaviour will appear to add information. A
structural prior is exactly such a mechanism.

### The protocol (recommended before any structural prior is credited)

1. Measure the per-bin gradient share of the training loss. If far from
   uniform, the regime receiving least gradient is underfitted.
2. Retrain **both** arms with training concentrated on the regime of interest,
   matching data budget, schedule, seed and initialization.
3. Re-measure the contrast. What survives is attributable to the prior; what
   does not was compensating for training inadequacy.

### Applying it — FIG. 2

| SNR bin | mixed-SNR training | matched focused training |
|---|---|---|
| [5,10) | +0.398 | **−0.076** |
| [10,15) | +1.305 | **+0.130** |
| [15,20) | +2.226 | **+0.232** |
| over [5,20] | **+1.209** | **+0.078** CI [+0.050,+0.112] |

**A fifteenfold collapse.** The mechanism is in the absolutes: focused training
gains the *unstructured* arm **+2.653 dB** but the structured arm only
**+1.629 dB**. The unstructured network learns the array geometry by itself
once given adequate gradient there, leaving the explicit projection little to
add.

**Scope of the negative result, stated in its own paragraph:** holds at
`N = 32`, `K = 3`, this data budget, and the two pilot counts tested. Not a
general claim. The same prior applied to the *classical* estimator gives a
large stable gain, and under a path-richness shift the structured network
**degrades least (+0.48 dB against +0.73 dB)**. The prior buys robustness; it
is the in-distribution accuracy claim that does not survive.

*(Source: `reports/trackD_stage4_report.md:74-78`;
`reports/trackD_stage5_eval.json` `C5_path_richness_ood`.)*

## IV. Numerical Results

### TABLE I

| Parameter | Value |
|---|---|
| `N` / `K` / `P` | 32 / 3 / 20 |
| `L_k` | `U{3,7}` |
| SNR (train and test) | `U[−10,20]` dB |
| RSR | 10 dB |
| Unrolled layers `T` | 10 |
| Trainable parameters | 1,586,900 |
| Training samples / epochs | 80,000 / 13 |
| Test realizations | 2,000 paired |

### A corrected loss, not a workaround — FIG. 3

Weight each sample by a static per-bin factor `w(b) = c/m(b)`, where `m(b)` is
the mean per-sample normalized error in bin `b` measured once before training,
`c` set so weights have unit mean. **No schedule, no tuning, no architectural
change.**

| bin | −10..−5 | −5..0 | 0..5 | 5..10 | 10..15 | 15..20 |
|---|---|---|---|---|---|---|
| weight | 0.056 | 0.111 | 0.310 | 0.687 | 1.641 | 3.195 |
| share, per-sample NMSE | 0.465 | 0.293 | 0.138 | 0.061 | 0.027 | 0.015 |
| share, balanced | 0.109 | 0.137 | 0.181 | 0.176 | 0.188 | 0.209 |

Gradient-share spread **31× → 1.9×**; share below 5 dB **0.897 → 0.427**
against an ideal 0.5 — a slight over-correction.

**The estimation result, per bin** (positive = balanced better):

| bin | −10..−5 | −5..0 | 0..5 | 5..10 | 10..15 | 15..20 |
|---|---|---|---|---|---|---|
| Δ (dB) | +0.038 | +0.044 | +0.511 | +1.353 | +1.974 | **+2.628** |

**SNR ≥ 5: +1.902 dB** CI [+1.823, +1.986]. **SNR < 5: +0.135 dB** CI
[+0.101, +0.172].

> A low-SNR **cost** of −0.10 to −0.40 dB was pre-registered on the reasoning
> that the lowest bin's weight falls about eighteenfold. **There was no cost.**
> Down-weighting the bins that dominated the gradient did not harm them,
> because they were over-served rather than merely well-served.

Gap to the genie-aided reference at SNR ≥ 5 falls **+2.897 → +1.000 dB**. Below
5 dB the balanced estimator **passes** that reference by up to 1.99 dB —
consistent, since the reference is restricted to unstructured LS.

### Pilot efficiency is not pilot-count generalization — FIG. 4

A pilot-count curve can mean two different things: one model trained at `P₀`
and evaluated at each `P` (**generalization**), or a model trained at each `P`
(**efficiency**). These are not close.

| `P` | EM-GS | HS-EM-GS | URformer trained at `P=20` | URformer trained AT each `P` | genie-aided LS |
|---|---|---|---|---|---|
| 10 | −2.16 | −4.28 | −5.39 | **−7.52** | −7.99 |
| 12 | −3.81 | −6.16 | −7.07 | — | −9.55 |
| 15 | −5.40 | −7.66 | −8.89 | — | −11.01 |
| 20 | −7.87 | −9.59 | −11.27 | −11.27 *(same model)* | −11.97 |
| 25 | −8.37 | −10.49 | **−11.13** | — | −13.14 |
| 30 | −9.68 | −11.54 | −11.94 | — | −13.90 |
| 35 | −10.62 | −12.37 | −12.43 | **−13.28** | −14.65 |

Matched training at `P = 10` beats the `P = 20` model evaluated there by
**+2.231 dB** pooled, every bin's CI excluding zero; at `P = 35` the margin is
+0.635 dB, so the model generalizes **upward** in pilot count far better than
downward. At 5 dB the matched estimator reaches −7.52 dB, **within 0.47 dB of
the genie-aided reference**, while the `P = 20` model is 2.60 dB short.

> **The sharpest single indication:** the `P = 20` model is **worse at
> `P = 25` (−11.13 dB) than at `P = 20` (−11.27 dB)** — more measurements,
> worse estimate.

**On the prior work's Fig. 4**, quoted verbatim and framed as motivation, not
accusation: it is described as "the NMSE performance versus the number of pilot
transmissions `P`, evaluated at a fixed SNR of 5 dB", and as "crucial for
assessing the pilot efficiency of each algorithm"; **the text does not state
whether the networks were retrained per pilot count.** Reported as a fact about
what the paper specifies — "not as a criticism of that work, whose convention
is the ordinary one and which our own earlier curves shared."

## V. Conclusion

Most of the apparent benefit of adding an explicit structural prior to an
unrolled channel estimator, in this setting, is an artifact of the training
convention rather than information supplied by the prior. Rebalancing the loss
— a change of a few lines, no architectural addition — recovers +1.902 dB at
SNR ≥ 5 dB at no low-SNR cost, **considerably more than the prior appeared to
offer**. We recommend the control be run before a structural prior is credited,
and note that it cost us our own result first.

**Limitations:** simulation only; geometric ULA at `N = 32`, `K = 3`, one data
budget, the pilot counts tested; the negative result is scoped to those
conditions; no fundamental bound — the reference is genie-aided and is passed
by our own estimator at low SNR.

---

# Open items — what the author must close before submission

| # | item | where |
|---|---|---|
| 1 | **Department string** — from the earlier HS-GS draft, not an independent source | both papers |
| 2 | **arXiv:2408.14366 author list** — that PDF is not readable from the build environment | Paper 1 §V-D |
| 3 | **RSR 7.15 vs 7.23 dB** — the earlier draft states 7.15; the arithmetic gives 7.229; 0.08 dB unreconciled | Paper 1 §II |
| 4 | **Roy–Vetterli** citation unverified | Paper 1 §III |
| 5 | **Gorman–Hero and Stoica–Ng** citations unverified | Paper 1 §IV |
| 6 | **Xu et al. full page range** — start page 2961 confirmed, range not | both `refs.bib` |
| 7 | **Cadzow, Markovsky, Gerchberg–Saxton, Netrapalli** page numbers unverified | both `refs.bib` |

**Verified against a primary source** (the Xiao PDF's page 1, its DOI
`10.1109/LSP.2026.3685170`, and its reference list): Xiao, Cui, both Gong
entries, Zhang, Meijerink & Molisch.

**Numbers that could not be traced to a committed results file: none.** Every
number in either manuscript carries a `% src:` comment naming its file.
