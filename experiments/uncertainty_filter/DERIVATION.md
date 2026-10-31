# Phase-posterior uncertainty for the Hankel filter

Model, per antenna `n` and pilot `p`:

    Z_np = |Y_np|,   Y = M + W,   M = G S + B,   W_np ~ CN(0, v) i.i.d.

`v` is the **total** complex noise variance (`obs.sigma2` in the code), `G` is
`N x K`, `S` is `K x P` and `A = S S^H`.

## 1. Phase posterior given Z and the current mean

Write `Y = Z e^{iφ}` and `M = |M| e^{iθ}`. The likelihood of `Y` is
`exp(-|Y - M|^2 / v)`. Given `|Y| = Z`, only the cross term depends on `φ`:

    p(φ | Z, M) ∝ exp( (2 Z |M| / v) cos(φ - θ) ),

a von Mises law with mean direction `θ` and concentration

    κ = 2 Z |M| / v.

Its first circular moment is `E[e^{iφ}] = R(κ) e^{iθ}` with `R = I1/I0`. Hence

    Q = E[Y | Z, M] = Z R(κ) M / |M|,

which is exactly the existing EM field (`sahure/math.py: posterior_field`,
`tcom_pilot_em_20261018/continuation.py: posterior`).

## 2. Posterior field variance

Because `|Y| = Z` exactly, `E[|Y|^2 | Z, M] = Z^2`, so

    Var_np = E|Y_np - Q_np|^2 = Z_np^2 - |Q_np|^2 = Z_np^2 (1 - R(κ_np)^2).

When `M_np = 0` the phase is uniform, `R = 0` and `Var_np = Z_np^2`.

The variance lies entirely along the tangent direction: the radial component is
known exactly. The pseudo-covariance `E[(Y - Q)^2]` is nonzero but is ignored
below, as is any correlation between entries.

### Numerical evaluation

`uncertainty.py` evaluates `R` with the scaled Bessel functions `i1e/i0e` and the
repository's small-argument series `R ≈ (κ/2)(1 - κ²/8 + κ⁴/48)` for `κ² < 1e-4`.
For large κ, `(1 - R)(1 + R)` loses about `κ·eps` relative accuracy. κ reaches
about 1e8 at 60 dB with RSR 20 dB. Above `κ = 1e3` the code therefore uses the
asymptotic series derived in Section 3. Its truncation error at the switch is
about `2e-15` relative.

## 3. High-κ limit

The Hankel expansion `I_ν(x) ~ e^x (2πx)^{-1/2} Σ_k (-1)^k a_k(ν) x^{-k}`, with
`a_k(ν) = Π_{j≤k}(4ν² - (2j-1)²) / (k! 8^k)`, gives, by exact rational series
division (reproduced in `test_uncertainty.py`),

    R(κ)     = 1 - 1/(2κ) - 1/(8κ²) - 1/(8κ³) - 25/(128κ⁴) - ...
    1 - R(κ)² = 1/κ + 0/κ² + 1/(8κ³) + 1/(4κ⁴) + 75/(128κ⁵) + ...

So `κ (1 - R²) = 1 + O(κ⁻²)`, and as κ → ∞

    Var_np → Z_np² / κ_np = Z_np v / (2 |M_np|).

When the estimate is good and the SNR is high, `Z_np ≈ |M_np|`, so `Var_np → v/2`.
That is half the noise power: the tangential half of the circular noise. The
radial half is removed by observing `|Y|`.

## 4. Propagation through the M-step

The M-step is linear in `Q`: `G_EM = (Q - B) S^H A^{-1}`. For antenna `n`, with
`q_n` the `n`-th row of `Q`,

    g_n = A^{-T} conj(S) (q_n - b_n)   (as a column vector).

If the entries of `q_n` are independent with variances `Var_n,:`, then

    Cov(g_n) ≈ A^{-1} S diag(Var_n,:) S^H A^{-1}   (K x K; the conjugate of the
                                                    exact transpose form, with
                                                    the same real diagonal),

    [Cov(g_n)]_kk = Σ_p |(A^{-1} S)_kp|² Var_np.

The per-user predicted MSE per channel entry is

    u_k = (1/N) Σ_n Re[Cov(g_n)]_kk,

and the uncertainty filter scale is

    τ_unc,k = c sqrt(2 u_k max(m, q)),

where `m x q = 17 x 16` is the Hankel lift for `N = 32`.

### What `u_k` does and does not include

`u_k` is the spread of the **one-step** EM map conditioned on the current `M`. It
treats the current iterate as fixed and ignores its own error. It ignores
correlation between entries and pseudo-covariance. Bias from a poor current
iterate is therefore not represented. Phase 1 measures this directly with the
bias fraction.

The Hankel filter is applied to the relaxed output
`G_dc = G + γ(G_EM - G)`, not to `G_EM`. With the current iterate held fixed,

    Cov(G_dc | Z, G) = γ² Cov(G_EM | Z, G),

so the stage-wise conditional MSE proxy used for calibration must be `γ² u_k`.
This correction applies to stages 1--10. The stored final-filter input is an
already-produced estimate after the routed EM tail; a fresh `u_k` computed from
its mean predicts a *next* EM map rather than that current estimate. It is thus
not included in the corrected calibration gate.

## 5. Relation to the existing noise-scaled rule (correction)

In the limit `Var_np → v/2` for every entry,

    Cov(g_n) → (v/2) A^{-1},   u_k → (v/2) [A^{-1}]_kk,
    τ_unc,k → c sqrt(v [A^{-1}]_kk max(m, q)).

The existing final filter (`tcom_incumbent_guard_20261019/shrink.py`;
`starts.py: FixedCore`, `noise_c`) is

    τ_noise,k = c sqrt(2 v [A^{-1}]_kk max(m, q)).

**The two do not coincide for the same `c`.** In the limit

    τ_unc,k / τ_noise,k → 1/√2,   i.e.   τ_unc(√2 c) → τ_noise(c).

The existing rule's `2v[A⁻¹]_kk` is four times the phase-only posterior
variance `(v/2)[A⁻¹]_kk` and twice the coherent least-squares variance
`v[A⁻¹]_kk`. The task statement expected agreement for equal `c`. This is
off by a factor of √2, and the unit test checks the corrected relation rather
than forcing agreement.

Consequence for Phase 2: the stored policy uses `c_noise ∈ {2, 4}` above 30 dB.
These correspond to `c_unc ≈ {2.83, 5.66}`. The prescribed grid `{0.5, 1, 2, 3, 4}`
does not reach the `c_noise = 4` operating point. This is raised at Gate 1
before any Phase 2 selection is run.

## 6. Numerical checks

`test_uncertainty.py` checks:

- `R` against `|Q|/Z` from the existing posterior helper, including the small series.
- `1 - R²` against mpmath at 50 digits for `κ ∈ [1e-6, 1e9]`.
- `Var = Z²` when `M = 0`.
- The posterior mean and variance against direct quadrature of the von Mises density.
- `κ (1 - R²) → 1` and `Var → Z v / (2|M|)`.
- The covariance propagation formula against Monte Carlo.
- On simulated 50 and 60 dB worlds, with `M` from a receiver estimate and not the
  true channel: `τ_unc / τ_noise` is within a few percent of `1/√2`, and
  `τ_unc(√2 c)` agrees with `τ_noise(c)` within a few percent.
