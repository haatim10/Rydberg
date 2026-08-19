# Track A — Cui Fig. 6 reproduction

Detection NMSE vs RSR for the frozen Track-A stack. **Fig. 5 extension, Fig. 7,
Fig. 8, Track B, Track C, and machine learning were not run in this pass.**

## Settings

- N = 36, K = 3, 16-QAM, SNR = 3 dB (fixed), t0 = 50
- RSR grid (integer dB): [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0]
- Trials per RSR point: **500**
- Algorithms: biased GS, EM-GS, ZF-known-phase, Cui exact-model CRLB
- CM-ZF omitted — the source formulation is still not recoverable from the
  published paper; no approximation was substituted.
- Aggregation: ratio of sums of linear energies, then `10 log10`
  (`sum(error_energy) / sum(expected_symbol_energy)`, `expected_symbol_energy = K = 3`)
- Config fingerprint: `925f2ab8975505e255f5c37af74d115be38bc302991267f95f440e5f61ce4472`
  (shared with the Fig. 5 canonical store by design — the fingerprint excludes
  sweep grids; the two experiments are kept apart by `experiment` name and by
  directory)

## Convergence (speed-prioritized, not chased to 0.1 dB)

Per your instruction this run targeted ~500 trials/RSR rather than strict
convergence. The checkpoint deltas:

| Step | max abs delta | median abs delta | within 0.1 dB tol |
|---|---:|---:|---:|
| 100 -> 250 | 0.725 dB | 0.202 dB | 17/52 |
| 250 -> 500 | 0.524 dB | 0.160 dB | 16/52 |

The delta is still shrinking (0.725 -> 0.524 dB max) but has not reached the
strict 0.1 dB criterion. At 500 trials the curves are visibly smooth (see the
error bars in `fig6_nmse_errorbars`-style plot below) and the qualitative
Cui comparisons below are all satisfied, so the run was not extended further,
per the explicit speed priority for this task.

## Critical sanity check — ZF flatness

The genie ZF error is `(A A^H)^-1 A w`, independent of the reference `b`, so
Cui's Fig. 6 shows it as flat and our curve must be too:

- max = -13.386 dB, min = -13.808 dB, **range = 0.422 dB**
- weighted-least-squares slope = -0.00263 dB/dB, t-statistic = -0.93
  (significance threshold |t| > 2.0)
- **significant_slope = False** — the flatness check passes;
  no investigation was needed and the solver/calibration were not touched.

## RSR calibration check

Measured empirically from the generated worlds using Cui's single-user
definition (eq. 38), against 200 fresh trials at each target:

| target RSR (dB) | measured (dB) | error (dB) | within 0.1 dB |
|---:|---:|---:|---:|
| 0.0 | 0.0000 | +0.0000 | True |
| 12.0 | 12.0000 | +0.0000 | True |
| 25.0 | 25.0000 | +0.0000 | True |

Calibration is exact (0.0000 dB error at all three targets); the run is valid.

## Final NMSE (dB), 500 trials/RSR

| RSR (dB) | biased GS | EM-GS | ZF-known-phase | Cui CRLB |
|---:|---:|---:|---:|---:|
| 0 | -3.267 | -5.041 | -13.808 | -9.345 |
| 1 | -3.523 | -5.312 | -13.487 | -9.399 |
| 2 | -3.986 | -5.688 | -13.421 | -9.461 |
| 3 | -5.171 | -6.846 | -13.465 | -9.507 |
| 4 | -5.904 | -7.459 | -13.488 | -9.614 |
| 5 | -6.210 | -7.702 | -13.546 | -9.689 |
| 6 | -7.004 | -8.492 | -13.545 | -9.740 |
| 7 | -7.226 | -8.469 | -13.772 | -9.853 |
| 8 | -7.883 | -8.959 | -13.503 | -9.871 |
| 9 | -8.187 | -9.112 | -13.433 | -9.984 |
| 10 | -8.063 | -9.003 | -13.567 | -10.053 |
| 11 | -8.330 | -9.216 | -13.560 | -10.096 |
| 12 | -8.793 | -9.578 | -13.595 | -10.145 |
| 13 | -8.788 | -9.438 | -13.457 | -10.199 |
| 14 | -8.662 | -9.265 | -13.580 | -10.242 |
| 15 | -9.096 | -9.611 | -13.647 | -10.263 |
| 16 | -9.237 | -9.603 | -13.519 | -10.286 |
| 17 | -9.359 | -9.736 | -13.619 | -10.313 |
| 18 | -9.262 | -9.589 | -13.530 | -10.341 |
| 19 | -9.361 | -9.632 | -13.587 | -10.370 |
| 20 | -9.330 | -9.588 | -13.386 | -10.372 |
| 21 | -9.691 | -9.956 | -13.606 | -10.391 |
| 22 | -9.708 | -9.940 | -13.653 | -10.406 |
| 23 | -9.698 | -9.889 | -13.635 | -10.429 |
| 24 | -9.766 | -9.940 | -13.536 | -10.449 |
| 25 | -9.841 | -9.985 | -13.781 | -10.444 |

## GS − EM-GS gap (dB; negative means EM-GS better)

- RSR = 0.0 dB: -1.774 dB
- RSR = 5.0 dB: -1.492 dB
- RSR = 10.0 dB: -0.940 dB
- RSR = 12.0 dB: -0.785 dB
- RSR = 15.0 dB: -0.515 dB
- RSR = 20.0 dB: -0.257 dB
- RSR = 25.0 dB: -0.144 dB

## Improvement from RSR = 0 dB

| Algorithm | 0 -> 20 dB | 0 -> 25 dB | Cui's stated order |
|---|---:|---:|---|
| biased GS | 6.063 dB | 6.574 dB | "rapidly declines by 5 dB" |
| EM-GS | 4.547 dB | 4.944 dB | "rapidly declines by 5 dB" |

Both land close to Cui's stated ~5 dB figure (not forced to match exactly).

## CRLB crossing

- Any point estimate below the CRLB: **False**
- Any statistically-significant crossing (95% CI): **False** (n = 0)

No crossing at any RSR; the empirical solvers stay at or above the bound
throughout, as required.

## Qualitative comparison with Cui Fig. 6

| Cui's claim | This run |
|---|---|
| ZF approximately flat vs RSR | Confirmed: 0.42 dB range, slope not statistically significant |
| PR solvers (GS, EM-GS) improve strongly with RSR | Confirmed: 6.06/4.55 dB (GS/EM-GS) over 0->20 dB |
| EM-GS beats GS, gap shrinks at high RSR | Confirmed: gap goes from -1.77 dB at RSR=0 to -0.14 dB at RSR=25 |
| CRLB tracked from above by EM-GS, no crossing | Confirmed |

## Remaining absolute vertical offset (documented, not tuned away)

This run reproduces the same **~2 dB systematic offset** documented in the
prior Fig. 5 pixel-extraction audit (`us - Cui`, measured from Cui's actual
published plot at high pixel precision): our channel realization is close to
i.i.d. Rayleigh (`Tr((AA^H)^-1)` matching the Wishart closed form for N=36,
K=3), while Cui's plotted curves imply a more correlated/rank-reduced
effective channel (~24 effective spatial degrees of freedom out of 36). Two
well-documented, zero-free-parameter 3GPP TR 38.901 mechanisms (the fixed
20-ray offset-angle table; array element spacing swept over six orders of
magnitude) were tested and both produced <0.15 dB of movement — nowhere near
closing a ~2 dB gap — so per that prior audit's conclusion, no parameter
change consistent with what Cui's Table I actually specifies can remove this
offset. It is reported here rather than tuned away, per your instruction.

The **relative** behavior (EM-GS vs GS, RSR sensitivity, ZF flatness, CRLB
tracking) matches Cui's stated qualitative and quantitative claims closely,
which is what this run was scoped to validate.

## Outlier diagnostics

See `outliers.csv` / `outliers.json` for the full per-(algorithm, RSR) table
(median vs aggregate NMSE, percentiles, failure rate, tail energy share).
Zero harness failures at every operating point across the whole run.
Aggregate sits 0.5-2.9 dB above the median per-trial NMSE depending on RSR
(more skew at low RSR, where the magnitude-only problem is harder and rare
bad trials pull the mean up more); no trial's status was ever `failed`.

## Files

- `results.csv` — raw long-form Monte Carlo table (52,000 rows, 500 trials x 26 RSR x 4 algorithms)
- `aggregate.csv` / `aggregate.json` — final aggregate NMSE with uncertainty
- `zf_flatness.json` — the critical ZF sanity check
- `rsr_calibration.json` — empirical RSR calibration check
- `crlb_crossing.json` — CRLB crossing diagnostics
- `improvements_and_gaps.json` — RSR improvement and GS/EM-GS gap summary
- `outliers.csv` / `outliers.json` — per-cell tail/outlier diagnostics
- `convergence.json` — checkpoint deltas (100->250->500)
- `config.json`, `run_manifest.json` — experiment configuration and provenance
- `fig6_nmse.png`, `fig6_nmse.pdf` — publication figure with GS/EM-GS error bars

## What was not run

Fig. 5 extension beyond 2000/4000 trials, Fig. 7(a), Fig. 7(b), Fig. 8,
Track B, Track C, machine learning.
