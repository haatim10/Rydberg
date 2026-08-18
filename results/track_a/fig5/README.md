# Track A — Cui Fig. 5 reproduction

Detection NMSE vs SNR for the frozen Track-A stack.
**Fig. 6, Fig. 7, Fig. 8, Track B, Track C, and machine learning were not run.**

## Settings

- N = 36, K = 3, 16-QAM, RSR = 12 dB, t0 = 50
- SNR grid (integer dB): [-5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
- Trials per SNR point: **635**
- Algorithms: biased GS, EM-GS, ZF-known-phase, Cui exact-model CRLB
- CM-ZF omitted (exact source formulation still not implemented)
- Aggregation: ratio of sums of linear energies, then `10 log10`
- Expected symbol-vector energy = K = 3 (unit-energy 16-QAM; not demapped)

## Convergence criterion (set before the run)

For biased GS and EM-GS, the ratio-of-sums detection NMSE at every integer-dB SNR in [-5, 12] must change by less than 0.1 dB between the last two checkpoints with n_trials >= 250. Start at 500 trials/SNR; if that test fails, continue through 1000 then 2000. Prefix aggregates at 100/250/500 are always reported even when the run starts at 500.

Achieved: converged = False with max |Δ| = 0.13176776986356487 dB at n_trials = 635.

## Core solvers

Unchanged: `biased_gs`, `em_gs`, `spectral_initialize`, `zf_known_phase`, `cui_crlb`, Step-13 `detection_nmse`, Step-14 Monte Carlo harness.
Confirmed: True.

## Final NMSE (dB)

| SNR (dB) | biased GS | EM-GS | ZF-known-phase | Cui CRLB |
|---:|---:|---:|---:|---:|
| -5 | 1.896 | -0.471 | -5.400 | -1.063 |
| -4 | 0.425 | -1.629 | -6.524 | -2.290 |
| -3 | -0.842 | -2.625 | -7.534 | -3.472 |
| -2 | -2.262 | -3.880 | -8.445 | -4.638 |
| -1 | -3.465 | -4.893 | -9.454 | -5.771 |
| 0 | -4.821 | -5.968 | -10.349 | -6.888 |
| 1 | -6.128 | -7.221 | -11.581 | -8.001 |
| 2 | -7.427 | -8.275 | -12.597 | -9.073 |
| 3 | -8.777 | -9.555 | -13.588 | -10.146 |
| 4 | -10.050 | -10.699 | -14.536 | -11.221 |
| 5 | -11.240 | -11.614 | -15.526 | -12.269 |
| 6 | -12.256 | -12.589 | -16.521 | -13.321 |
| 7 | -13.433 | -13.701 | -17.626 | -14.351 |
| 8 | -14.359 | -14.602 | -18.642 | -15.369 |
| 9 | -15.572 | -15.780 | -19.534 | -16.397 |
| 10 | -16.650 | -16.819 | -20.487 | -17.426 |
| 11 | -17.675 | -17.764 | -21.463 | -18.442 |
| 12 | -18.589 | -18.681 | -22.387 | -19.465 |

## EM-GS minus biased GS (dB; negative means EM-GS better)

- SNR = -5 dB: -2.367 dB
- SNR = -4 dB: -2.054 dB
- SNR = 0 dB: -1.147 dB
- SNR = 6 dB: -0.334 dB
- SNR = 12 dB: -0.092 dB

## High-SNR CRLB minus ZF: 2.9219411080926427 dB (analytic 10 log10 2 = 3.0103 dB)

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
      "n_paired": 635,
      "nmse_diff_linear": 0.7644963185354153,
      "se_linear": 0.040731799512522315,
      "ci95_low": 0.6846634584653655,
      "ci95_high": 0.8443291786054652,
      "statistically_below": false,
      "point_below": false
    },
    {
      "algorithm_left": "biased_gs",
      "algorithm_right": "cui_crlb",
      "snr_db": -4.0,
      "n_paired": 635,
      "nmse_diff_linear": 0.5125008780012658,
      "se_linear": 0.029401739090044694,
      "ci95_low": 0.4548745283019347,
      "ci95_high": 0.5701272277005969,
      "statistically_below": false,
      "point_below": false
    },
    {
      "algorithm_left": "biased_gs",
      "algorithm_right": "cui_crlb",
      "snr_db": -3.0,
      "n_paired": 635,
      "nmse_diff_linear": 0.37415382527590896,
      "se_linear": 0.02373107470082389,
      "ci95_low": 0.3276417735478645,
      "ci95_high": 0.4206658770039534,
      "statistically_below": false,
      "point_below": false
    },
    {
      "algorithm_left": "biased_gs",
      "algorithm_right": "cui_crlb",
      "snr_db": -2.0,
      "n_paired": 635,
      "nmse_diff_linear": 0.25034143142138987,
      "se_linear": 0.017433516052800527,
      "ci95_low": 0.21617236783399996,
      "ci95_high": 0.2845104950087798,
      "statistically_below": false,
      "point_below": false
    },
    {
      "algorithm_left": "biased_gs",
      "algorithm_right": "cui_crlb",
      "snr_db": -1.0,
      "n_paired": 635,
      "nmse_diff_linear": 0.18545033502432479,
      "se_linear": 0.013234449988249538,
      "ci95_low": 0.15951128969215916,
      "ci95_high": 0.2113893803564904,
      "statistically_below": false,
      "point_below": false
    },
    {
      "algorithm_left": "biased_gs",
      "algorithm_right": "cui_crlb",
      "snr_db": 0.0,
      "n_paired": 635,
      "nmse_diff_linear": 0.12479912970130239,
      "se_linear": 0.00973409606957475,
      "ci95_low": 0.10572065198288298,
      "ci95_high": 0.1438776074197218,
      "statistically_below": false,
      "point_below": false
    },
    {
      "algorithm_left": "biased_gs",
      "algorithm_right": "cui_crlb",
      "snr_db": 1.0,
      "n_paired": 635,
      "nmse_diff_linear": 0.08547173251753765,
      "se_linear": 0.007219156397193845,
      "ci95_low": 0.07132244598027578,
      "ci95_high": 0.09962101905479952,
      "statistically_below": false,
      "point_below": false
    },
    {
      "algorithm_left": "biased_gs",
      "algorithm_right": "cui_crlb",
      "snr_db": 2.0,
      "n_paired": 635,
      "nmse_diff_linear": 0.05704590935471557,
      "se_linear": 0.005436637751829317,
      "ci95_low": 0.0463902951641393,
      "ci95_high": 0.06770152354529184,
      "statistically_below": false,
      "point_below": false
    },
    {
      "algorithm_left": "biased_gs",
      "algorithm_right": "cui_crlb",
      "snr_db": 3.0,
      "n_paired": 635,
      "nmse_diff_linear": 0.03584406482132706,
      "se_linear": 0.004287446036977673,
      "ci95_low": 0.02744082500319183,
      "ci95_high": 0.04424730463946228,
      "statistically_below": false,
      "point_below": false
    },
    {
      "algorithm_left": "biased_gs",
      "algorithm_right": "cui_crlb",
      "snr_db": 4.0,
      "n_paired": 635,
      "nmse_diff_linear": 0.023370253160911753,
      "se_linear": 0.002836544244307517,
      "ci95_low": 0.017810728601514635,
      "ci95_high": 0.028929777720308872,
      "statistically_below": false,
      "point_below": false
    },
    {
      "algorithm_left": "biased_gs",
      "algorithm_right": "cui_crlb",
      "snr_db": 5.0,
      "n_paired": 635,
      "nmse_diff_linear": 0.01586838405227452,
      "se_linear": 0.00211174106029941

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
    "em_minus_gs_db": -2.0537269960762377,
    "paper_order_db": -2.0,
    "pass": true
  },
  "4_gs_and_em_gs_merge_at_high_snr": {
    "em_minus_gs_db_at_12": -0.09152841594706018,
    "pass": true
  },
  "5_em_gs_tracks_magnitude_crlb": {
    "mean_em_minus_crlb_db": 0.7133143658185266,
    "at_12_db": 0.7848105151364742
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
    "empirical_at_12_db": 2.9219411080926427,
    "analytic_10log10_2": 3.010299956639812,
    "pass": true
  },
  "8_not_several_dB_from_orientation": "see README; no digitized Cui overlay in-repo"
}

## Documented deviations still in force

See `results/track_a/README.md`. None of those deviations were changed for this sweep. CM-ZF remains omitted. Per-realization row normalization is still the production Track-A channel definition unless the diagnostic above flagged a material shift.

## What was not run

Fig. 6, Fig. 7(a), Fig. 7(b), Fig. 8, Track B, Track C, ML/neural networks.

