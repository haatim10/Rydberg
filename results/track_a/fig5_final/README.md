<!-- canonical-run header; see git history for provenance -->
# FIRST CANONICAL Fig. 5 RUN (post-audit)

**This is the first canonical Track-A Fig. 5 run after the adversarial
audit and the resulting fingerprint fixes.** It is a fresh store, not a
resumption or re-keying of anything earlier.

| | |
|---|---|
| Config fingerprint (this run) | `925f2ab8975505e255f5c37af74d115be38bc302991267f95f440e5f61ce4472` |
| Superseded fingerprint | `c0e9dc88d0cfa56fdb0f949681a1078c11cb4a9f098a448ff22e44596c40dd7d` |
| Trials per SNR point | **2000** |
| Store | `results/track_a/fig5_final/` |

The earlier interrupted run at `results/track_a/fig5/` is **historical and
read-only**. Its fingerprint differs because `normalize_rows` is now part
of the Track-A experiment identity (audit M4), so the harness refuses to
mix the two stores and `aggregate_result_table` refuses to pool them
(audit M3). Nothing under `results/track_a/fig5/` or
`results/track_a/fig5_smoke/` was modified.

---

# Track A — Cui Fig. 5 reproduction

Detection NMSE vs SNR for the frozen Track-A stack.
**Fig. 6, Fig. 7, Fig. 8, Track B, Track C, and machine learning were not run.**

## Settings

- N = 36, K = 3, 16-QAM, RSR = 12 dB, t0 = 50
- SNR grid (integer dB): [-5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
- Trials per SNR point: **2000**
- Algorithms: biased GS, EM-GS, ZF-known-phase, Cui exact-model CRLB
- CM-ZF omitted (exact source formulation still not implemented)
- Aggregation: ratio of sums of linear energies, then `10 log10`
- Expected symbol-vector energy = K = 3 (unit-energy 16-QAM; not demapped)

## Convergence criterion (set before the run)

For biased GS and EM-GS, the ratio-of-sums detection NMSE at every integer-dB SNR in [-5, 12] must change by less than 0.1 dB between the last two checkpoints with n_trials >= 250. Start at 500 trials/SNR; if that test fails, continue through 1000 then 2000. Prefix aggregates at 100/250/500 are always reported even when the run starts at 500.

Achieved: converged = False with max |Δ| = 0.1345551547665811 dB at n_trials = 2000.

## Core solvers

Unchanged: `biased_gs`, `em_gs`, `spectral_initialize`, `zf_known_phase`, `cui_crlb`, Step-13 `detection_nmse`, Step-14 Monte Carlo harness.
Confirmed: True.

## Final NMSE (dB)

| SNR (dB) | biased GS | EM-GS | ZF-known-phase | Cui CRLB |
|---:|---:|---:|---:|---:|
| -5 | 1.906 | -0.439 | -5.516 | -1.063 |
| -4 | 0.510 | -1.515 | -6.547 | -2.280 |
| -3 | -0.807 | -2.576 | -7.509 | -3.475 |
| -2 | -2.124 | -3.765 | -8.629 | -4.639 |
| -1 | -3.598 | -5.016 | -9.489 | -5.779 |
| 0 | -4.808 | -5.974 | -10.471 | -6.887 |
| 1 | -6.109 | -7.135 | -11.448 | -7.999 |
| 2 | -7.412 | -8.267 | -12.534 | -9.075 |
| 3 | -8.709 | -9.425 | -13.605 | -10.151 |
| 4 | -9.882 | -10.463 | -14.600 | -11.226 |
| 5 | -11.003 | -11.425 | -15.440 | -12.274 |
| 6 | -12.242 | -12.598 | -16.461 | -13.316 |
| 7 | -13.380 | -13.652 | -17.555 | -14.347 |
| 8 | -14.429 | -14.697 | -18.523 | -15.379 |
| 9 | -15.526 | -15.731 | -19.573 | -16.406 |
| 10 | -16.544 | -16.690 | -20.591 | -17.419 |
| 11 | -17.679 | -17.786 | -21.588 | -18.440 |
| 12 | -18.649 | -18.745 | -22.410 | -19.451 |

## EM-GS minus biased GS (dB; negative means EM-GS better)

- SNR = -5 dB: -2.344 dB
- SNR = -4 dB: -2.025 dB
- SNR = 0 dB: -1.166 dB
- SNR = 6 dB: -0.356 dB
- SNR = 12 dB: -0.096 dB

## High-SNR CRLB minus ZF: 2.9587199897756946 dB (analytic 10 log10 2 = 3.0103 dB)

## Row-normalization diagnostic

- A = production per-realization row normalization
- B = raw Table I with eq. 37/38 recalibrated from the raw channel
- Trials: 32
- max |Δ| on GS/EM-GS: 0.3860141798914736 dB
- material (≥ 0.5 dB): False
- negligible (< 0.2 dB): False
- keep production normalization: True
- raw Table I mean_n |a_nk|²: mean=231.29980302511044, cv=0.17333094064811658, min=135.10432029077788, max=346.28407881208153

## CRLB crossing

{
  "any_point_estimate_below_crlb": false,
  "any_statistically_below_crlb": false,
  "persistent_statistically_meaningful_crossing": false,
  "n_statistically_below": 0,
  "per_snr": [
    {
      "algorithm_left": "biased_gs",
      "algorithm_right": "cui_crlb",
      "snr_db": -5.0,
      "n_paired": 2000,
      "nmse_diff_linear": 0.7679560155353772,
      "se_linear": 0.02243015213840593,
      "ci95_low": 0.7239937251763475,
      "ci95_high": 0.8119183058944068,
      "statistically_below": false,
      "point_below": false
    },
    {
      "algorithm_left": "biased_gs",
      "algorithm_right": "cui_crlb",
      "snr_db": -4.0,
      "n_paired": 2000,
      "nmse_diff_linear": 0.5331217558606511,
      "se_linear": 0.017223683043351146,
      "ci95_low": 0.49936395741454964,
      "ci95_high": 0.5668795543067526,
      "statistically_below": false,
      "point_below": false
    },
    {
      "algorithm_left": "biased_gs",
      "algorithm_right": "cui_crlb",
      "snr_db": -3.0,
      "n_paired": 2000,
      "nmse_diff_linear": 0.3811851289600371,
      "se_linear": 0.014225628103999568,
      "ci95_low": 0.3533034102187371,
      "ci95_high": 0.40906684770133706,
      "statistically_below": false,
      "point_below": false
    },
    {
      "algorithm_left": "biased_gs",
      "algorithm_right": "cui_crlb",
      "snr_db": -2.0,
      "n_paired": 2000,
      "nmse_diff_linear": 0.26960136938928075,
      "se_linear": 0.010767429922010041,
      "ci95_low": 0.24849759453608214,
      "ci95_high": 0.29070514424247934,
      "statistically_below": false,
      "point_below": false
    },
    {
      "algorithm_left": "biased_gs",
      "algorithm_right": "cui_crlb",
      "snr_db": -1.0,
      "n_paired": 2000,
      "nmse_diff_linear": 0.17240422953048148,
      "se_linear": 0.007468575818668405,
      "ci95_low": 0.15776608991008467,
      "ci95_high": 0.1870423691508783,
      "statistically_below": false,
      "point_below": false
    },
    {
      "algorithm_left": "biased_gs",
      "algorithm_right": "cui_crlb",
      "snr_db": 0.0,
      "n_paired": 2000,
      "nmse_diff_linear": 0.1257552257903088,
      "se_linear": 0.005830961652242784,
      "ci95_low": 0.11432675095667877,
      "ci95_high": 0.1371837006239388,
      "statistically_below": false,
      "point_below": false
    },
    {
      "algorithm_left": "biased_gs",
      "algorithm_right": "cui_crlb",
      "snr_db": 1.0,
      "n_paired": 2000,
      "nmse_diff_linear": 0.08645199047605905,
      "se_linear": 0.00396290495629797,
      "ci95_low": 0.07868483948755975,
      "ci95_high": 0.09421914146455834,
      "statistically_below": false,
      "point_below": false
    },
    {
      "algorithm_left": "biased_gs",
      "algorithm_right": "cui_crlb",
      "snr_db": 2.0,
      "n_paired": 2000,
      "nmse_diff_linear": 0.05775584765538386,
      "se_linear": 0.002986350685831944,
      "ci95_low": 0.05190270786594676,
      "ci95_high": 0.06360898744482096,
      "statistically_below": false,
      "point_below": false
    },
    {
      "algorithm_left": "biased_gs",
      "algorithm_right": "cui_crlb",
      "snr_db": 3.0,
      "n_paired": 2000,
      "nmse_diff_linear": 0.038031472788981495,
      "se_linear": 0.0023064984522925603,
      "ci95_low": 0.0335108188920907,
      "ci95_high": 0.04255212668587229,
      "statistically_below": false,
      "point_below": false
    },
    {
      "algorithm_left": "biased_gs",
      "algorithm_right": "cui_crlb",
      "snr_db": 4.0,
      "n_paired": 2000,
      "nmse_diff_linear": 0.02735601197378778,
      "se_linear": 0.001630285576707787,
      "ci95_low": 0.024160710958925406,
      "ci95_high": 0.030551312988650152,
      "statistically_below": false,
      "point_below": false
    },
    {
      "algorithm_left": "biased_gs",
      "algorithm_right": "cui_crlb",
      "snr_db": 5.0,
      "n_paired": 2000,
      "nmse_diff_linear": 0.020148228099061086,
      "se_linear": 0.001

## Acceptance vs Cui (qualitative / order-of-magnitude)

{
  "1_nmse_decreases_with_snr": {
    "biased_gs": true,
    "em_gs": true,
    "genie_zf": true,
    "cui_crlb": true
  },
  "2_em_gs_beats_biased_gs_at_low_snr": true,
  "3_em_gs_improvement_near_minus_4dB_order_2dB": {
    "em_minus_gs_db": -2.0249707944086817,
    "paper_order_db": -2.0,
    "pass": true
  },
  "4_gs_and_em_gs_merge_at_high_snr": {
    "em_minus_gs_db_at_12": -0.0964568346178396,
    "pass": true
  },
  "5_em_gs_tracks_magnitude_crlb": {
    "mean_em_minus_crlb_db": 0.761695248413879,
    "at_12_db": 0.7062989274819671
  },
  "6_zf_below_magnitude_crlb": {
    "all_snr": true,
    "per_snr_true": [
      true,
      true,
      true,
      true,
      true,
      true,
      true,
      true,
      true,
      true,
      true,
      true,
      true,
      true,
      true,
      true,
      true,
      true
    ]
  },
  "7_high_snr_crlb_minus_zf_approx_3dB": {
    "empirical_at_12_db": 2.9587199897756946,
    "analytic_10log10_2": 3.010299956639812,
    "pass": true
  },
  "8_not_several_dB_from_orientation": "see README; no digitized Cui overlay in-repo"
}

## Documented deviations still in force

See `results/track_a/README.md`. None of those deviations were changed for this sweep. CM-ZF remains omitted. Per-realization row normalization is still the production Track-A channel definition unless the diagnostic above flagged a material shift.

## What was not run

Fig. 6, Fig. 7(a), Fig. 7(b), Fig. 8, Track B, Track C, ML/neural networks.


## How the CRLB curve is aggregated (brief section 9)

The Cui exact-model CRLB is unchanged (`rydberg_sim.crlb.cui_crlb`,
byte-identical to the pre-audit commit). It is evaluated **per trial** at
that trial's own realization, on the *same* `(A, s, b)` the empirical
methods see through the common-random-number harness:

    error_energy_trial          = Tr(F(A, s, b, sigma2)^-1).real
    expected_symbol_energy_trial = K = 3

and then aggregated by the identical ratio-of-sums rule used for every
other curve:

    CRLB_curve = sum_trials Tr(F^-1) / sum_trials K
               = E[Tr(F^-1)] / K,   then 10 log10

So the plotted bound is the **mean over channel realizations of the
per-realization bound**, normalized by `E||s||^2 = K`. This is the honest
Monte-Carlo reading: a CRLB is a per-realization statement, so the
realization-wise bounds are what get averaged. Note that by Jensen's
inequality `E[Tr(F^-1)] >= Tr(E[F]^-1)`, so this is not identical to
bounding the averaged Fisher information; Cui does not state which he
plots, and any claim of point-for-point agreement rests on this choice.

## Spectral initializer (brief section 12, audit H1)

The spectral initializer is **unchanged** and remains the production
default. It is faithful to Cui Algorithm 1/2 steps 1-4.

At this run's RSR = 12 dB its norm collapses strongly: `||u0||` falls to
roughly 0.09-0.21 against `||u_true|| = 1`, because `mbar_q = [m_q;
conj(b_q)]` makes `M_spec` near-rank-one along the reference axis once
`|b|` dominates, so the principal eigenvector converges to `e_{D+1}`.
Biased GS and EM-GS reach the **same fixed point** from the spectral,
zero, and random starts (agreeing to three decimal places in NMSE).

**Fig. 5 therefore validates final solver behaviour, not initializer
quality.** No claim that the spectral step is validated by these curves
should be made.

## Row normalization (brief section 13, audit M4)

`normalize_rows = True` is kept exactly as specified in this run's
fingerprint, and is now part of that fingerprint.

The audit measured the alternative: removing per-realization row
normalization and calibrating sigma^2 and |b| from a single ensemble
average instead shifted the GS/EM-GS curves by only **+0.10 to +0.23 dB**
across SNR = -5 to 12 dB. The repository's own two-arm diagnostic
(re-run into this store under `row_normalization_diagnostic/`) reports a
comparable magnitude. The choice is documented, not tuned: production
settings were **not** altered to chase agreement with the paper.

## Outlier / tail diagnostics (brief section 8)

`outliers.json` and `outliers.csv` carry, per (algorithm, SNR): the
ratio-of-sums aggregate, the median per-trial NMSE and their difference,
the 10/25/50/75/90/95/99th percentiles, the harness failure rate, the
fraction of trials worse than the trivial `s_hat = 0`, and the share of
total error energy held by the worst 1% and 5% of trials.

These are **diagnostics only**. The plotted Fig. 5 curve is the
ratio-of-sums NMSE.
