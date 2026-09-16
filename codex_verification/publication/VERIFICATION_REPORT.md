# Paper 1: independent simulation verification

Reference: the nine-page **ThesisProj-2.pdf** supplied by the author. Verification completed 16 September 2026 on local branch **codex**, based on repository commit `739dba4`.

**The central numerical result reproduces.** HS-GS improves estimation in the tested sparse geometric channels, and its benefit increases with array capacity and decreases with path complexity. The manuscript's approximately 2.68 dB headline becomes **2.695 dB** in an independent run. However, the effective-rank and projection-placement conclusions need revision before submission. The files accompanying this report provide new figures, raw trial records, confidence intervals, and replacement text.

## What was run

The completed campaign contains **27,300 fresh paired trials across 73 operating cells**, plus **2,400 bound evaluations on the exact main-figure channel realizations**. Counts were fixed before the campaign; each accuracy cell has 300 or 400 trials. No old result array was reused to produce a new curve. New trial identifiers, configuration metadata, software versions, source fingerprints and every trial's error numerator/denominator are saved.

The campaign covers every numerical experiment reported in the PDF: main performance and bound curves, array size, path count, effective rank, forced projection, two alternative channel distributions, user count, projection placement, and a separate timing study. Figure 1 is a system illustration and has no numerical result to rerun.

Additional controls include matching Cadzow sweeps in the placement comparison and 3,200 new N=32 trials that put the three-array effective-rank comparison on the same configuration and performance statistic. The original and corrected Figure 3 designs are both supplied, explicitly labeled.

The vectorized solver evaluates the same equations as the production implementation. Numerical gates checked the Bessel ratio, rank-selection scores, selected rank, Cadzow output, and final estimates at N=8,16,32,64. Maximum estimate differences in those checks were of order 1e-14. All full-capacity fallback trials in the campaign were checked for exact equality with the paired EM-GS output. The focused production test suite passed **99 tests** using the pinned NumPy 2.4.6 and SciPy 1.17.1.

Six representative saved trials were also replayed through the original solver, including N64/L29 and N64/L48, clustered channels, K=4 and the placement variants. Selected ranks agreed and saved error numerators matched to relative differences of order 1e-15. The ineffective source-slicing check for oracle-free selection was repaired; no production estimator behavior was changed.

This is an independent Monte Carlo rerun using the repository's forward model and a separately evaluated, numerically checked estimator implementation. It is not an independent hardware experiment or a new literature-novelty assessment.

## 1. Main performance: supported

Configuration: N=32, K=3, P=30, RSR=12 dB, 100 final iterations, 25 candidate-selection iterations, four Cadzow sweeps per application; 400 paired trials at each SNR.

| Aggregate SNR (dB) | Paired median improvement (dB) | Pointwise 95% CI | Vertical gap of NMSE curves (dB) |
|---:|---:|---:|---:|
| -5 | 2.972 | [2.844, 3.149] | 2.904 |
| 0 | 2.586 | [2.450, 2.814] | 2.416 |
| 5 | 2.544 | [2.403, 2.798] | 2.379 |
| 10 | 2.701 | [2.546, 2.835] | 2.483 |
| 15 | 2.618 | [2.454, 2.811] | 2.505 |
| 20 | 2.749 | [2.569, 2.855] | 2.608 |

The six-point mean of the paired medians is **2.695 dB**, compared with the manuscript's 2.68 dB. Projection was active in all 2,400 headline trials. This supports the reported improvement. Independent draws are not expected to reproduce the same final decimal places.

The paired median and plotted NMSE gap are different statistics. For trial i, the paired gain is `10 log10(error_EM_i / error_HS_i)`; the curves use `10 log10(sum(error_i) / sum(channel_energy_i))`. Specify both definitions rather than using one gain number to describe the other.

**Use:** `figure2_performance_array.pdf`.

## 2. CRLB and CCRB: paired curves confirm the reference behavior

At SNR ≥5 dB, the new EM-GS curve differs from the Rician CRLB by at most **0.107 dB**, compared with the old 0.143 dB claim. The geometric CCRB is **6.954–7.059 dB** below the unconstrained bound. HS-GS remains **4.365–4.850 dB** above the geometric CCRB.

These bounds now use exactly the same trial IDs and channel-power denominators as the estimator curves. The geometric tangent Jacobian was checked against finite differences; maximum checked absolute errors were below 5e-9. Scalar-information interpolation was checked against direct quadrature, with maximum relative error below 5.3e-7 on the probes. The projected Fisher matrices were inverted on an orthonormal tangent basis, and the constrained trace was checked positive and no larger than the unconstrained trace for every bound trial.

Retain the manuscript's unbiased-estimator caveat. Also call the geometric CCRB an ideal structural reference conditional on the local channel model and true path configuration. It is not a proof that a rank-selected biased estimator must lie above that curve.

## 3. Array size: supported, but keep both metrics

Each row below averages 12 operating-point summaries: two pilot counts and six SNRs. Each operating point contains 400 trials.

| N | New ratio-of-sums gain (dB) | New paired-median gain (dB) | Baseline NMSE (dB) | Projection inactive |
|---:|---:|---:|---:|---:|
| 8 | -0.095 | 0.236 | -8.666 | 42.94% |
| 16 | 0.866 | 1.044 | -8.786 | 6.38% |
| 32 | 2.938 | 3.264 | -8.733 | 0.31% |

The manuscript's qualitative conclusion reproduces. At N=8, the paired median is slightly positive while the ratio-of-sums measure is negative. Increasing N substantially improves the structural gain, while the averaged baseline NMSE spans only **0.120 dB**.

The N=8 inactivity rate is 42.94%, compared with the manuscript's 41.8%. Explain this as observed rank selection, not a guarantee that the selector recognizes every harmful constraint. It can select an active rank on a trial where that choice increases true estimation error.

## 4. Path count: supported

Configuration D: N=32, P=30, SNR=5 dB, RSR=12 dB, 50 final iterations, 20 selection iterations; 300 trials per L.

| Paths/user L | Ratio-of-sums gain (dB) | 95% CI | Paired median (dB) |
|---:|---:|---:|---:|
| 2 | 7.377 | [7.162, 7.607] | 7.850 |
| 4 | 3.592 | [3.400, 3.773] | 3.855 |
| 6 | 1.989 | [1.840, 2.140] | 2.197 |
| 8 | 1.131 | [1.008, 1.255] | 1.188 |
| 10 | 0.503 | [0.372, 0.618] | 0.603 |
| 12 | 0.089 | [-0.008, 0.183] | 0.209 |
| 14 | 0.022 | [-0.074, 0.115] | 0.098 |
| 16 | -0.125 | [-0.204, -0.046] | 0.000 |

The fresh endpoints are **7.377 and -0.125 dB**, compared with 7.04 and -0.12 dB. EM-GS varies by only **0.129 dB** over the sweep. The mechanism-level trend is convincing in this model.

At L=16, say that total normalized squared error becomes slightly worse, while the median paired gain is zero. “The method no longer helps” without identifying the metric loses this distinction.

**Use:** `path_count.pdf`.

## 5. Effective rank: useful descriptor; universal-threshold claim not established

The manuscript reports crossing values 0.588, 0.518 and 0.544 for N=16,32,64. Repeating the original analysis on fresh data gives:

| N | New crossing estimate | What the calculation actually establishes |
|---:|---:|---|
| 16 | 0.580 | Extrapolation beyond the original three positive-gain points |
| 32 | 0.509 | Interpolation between positive and negative ratio-of-sums point estimates |
| 64 | 0.526 | Extrapolation beyond the original three positive-gain points |

Do **not** describe all three as observed zero crossings. Extra high-complexity trials show median gains approaching a zero plateau, partly because full-rank fallback creates exactly zero trial differences. That does not identify a universal positive-to-negative boundary.

There is also a configuration mismatch in the original Figure 3: N32 uses fixed 5 dB, P30, RSR12, 50/20 iterations and ratio-of-sums gain; N16/N64 use random SNR in [-10,20], P20, RSR10, 100/25 iterations and pooled medians. The new `figure3_original_design` retains these differences explicitly. For a submission, prefer **`figure3_matched`**, which uses the latter design and pooled median throughout. In this matched experiment, N32 still has a positive median gain of **0.141 dB [0.090, 0.211] at rho=0.536**. Thus 0.5 is not a demonstrated universal cutoff.

### Comparison against simpler predictors

I calibrated a piecewise-linear predictor on the eight fresh, matched N32 cells and evaluated the original six N16/N64 test configurations without fitting to those targets. Results use the same design and same statistic throughout:

| Predictor | Held-out mean absolute error (dB) |
|---|---:|
| Raw path count L | 2.067 |
| Normalized path count L/r_max | **0.073** |
| Normalized effective rank rho | 0.317 |

The bootstrap interval for `MAE(rho) - MAE(L/r_max)` is **[0.086, 0.283] dB**, resampling calibration and target trials. This is conditional on this small test grid and interpolation scheme, not a universal ranking of predictors. Endpoint extrapolation is explicitly recorded: raw L extrapolates for N64/L29, and rho extrapolates slightly for N16/L7. All six target gains are positive, so sign accuracy here does not test failure detection.

The abstract's comparison against **raw path count** has support in this experiment. However, omitting normalized path count makes the effective-rank story appear stronger than the controlled comparison supports. Include this baseline and present rho as a useful descriptive coordinate, not the uniquely predictive quantity or a receiver-available switching rule.

## 6. Forced projection and alternative distributions: qualitative results reproduce

For N16/L14, the measured rho is **0.732**. Forcing rank 7 gives **-0.154 dB [-0.201, -0.111]** median gain, compared with the old -0.075 dB. Adaptive HS-GS has zero pooled median gain and falls back in 35.75% of these trials. This supports the risk of a forced structural constraint, but does not prove adaptive selection can never hurt.

For the two additional channel distributions, 400 new trials were run for each:

| Distribution | New rho | Frozen manuscript prediction (dB) | Fresh gain (dB) | 95% CI | Absolute prediction error (dB) |
|---|---:|---:|---:|---:|---:|
| Clustered, 4 clusters × 10 rays | 0.335 | 1.300 | 1.182 | [0.995, 1.398] | 0.118 |
| 10 independent paths | 0.407 | 0.648 | 0.831 | [0.710, 1.011] | 0.183 |

These are comparisons with the previously frozen predictions, not newly refitted predictions using the fresh measured rho. The two positive gains reproduce. The clustered prediction lies within the new interval; the 10-path prediction is below it. Report small prediction errors without claiming statistically exact calibration.

Both distributions still obey the geometric ULA measurement model. The implemented gains start with variance 1/D, and the whole channel matrix is then rescaled to the realized total power of a base channel draw. Specify this normalization, cluster counts, ±5° sub-ray offsets, and angle clipping. Do not describe these adapted models as exact reproductions of another paper's complete physical setup.

**Use:** `distribution_transfer.pdf` and panel (b) of `users_and_forced_constraint.pdf`.

## 7. User count and runtime: qualified support

For SNR ≥5 dB, the fresh median gains are **3.381, 3.509 and 3.552 dB** for K=2,3,4. Their range is **0.170 dB**, compared with the manuscript's 0.094 dB. The intervals overlap. This supports modest differences within the tested design, not invariance to K.

Disclose that P changes with K: `(K,P)=(2,13),(3,20),(4,27)`, with L=5, approximately fixed P/(2K), and aggregate SNR. The experiment does not vary only user count.

Serial timing used three independent trials at each N, P20, SNR5, RSR10, T100, selection25, four sweeps. Hardware: Apple M2, eight physical cores, 8 GiB RAM; each timed pipeline ran serially with single-thread environment settings. Values below are medians on this machine:

| N | Production interleaved total (s) | Production final-only total (s) | Production selection share | Vectorized interleaved total (s) |
|---:|---:|---:|---:|---:|
| 8 | 0.394 | 0.279 | 47.5% | 0.041 |
| 16 | 1.059 | 0.874 | 65.3% | 0.102 |
| 32 | 3.548 | 3.174 | 79.3% | 0.497 |
| 64 | 14.159 | 13.265 | 88.5% | 3.607 |

End-to-end final-only times include the same shared rank selection. Removing repeated projection does not remove that dominant cost. The old 57.7–85.5% range is a benchmark-specific observation, not a portable property. Describe these new timings with their settings and limited repetition count. Additional repetitions would be appropriate if runtime becomes a primary contribution.

**Use:** `runtime.pdf` and panel (a) of `users_and_forced_constraint.pdf`.

## 8. Projection placement: the controlled result changes the conclusion

All comparisons use 300 fresh paired trials per SNR and the same rank selected with four-sweep interleaved candidates. Positive values favor interleaving.

| SNR (dB) | Interleaved 4 vs final 1 (dB) | Interleaved 4 vs final 4 (dB) | Interleaved 1 vs final 1 (dB) |
|---:|---:|---:|---:|
| -5 | -0.090 | 0.275 | 0.129 |
| 0 | -0.182 | 0.191 | 0.130 |
| 5 | 0.004 | 0.267 | 0.310 |
| 10 | 0.020 | 0.133 | 0.239 |
| 15 | 0.071 | 0.148 | 0.299 |
| 20 | 0.103 | 0.210 | 0.326 |
| Mean of six medians | **-0.012** | **0.204** | **0.239** |
| Stratified-bootstrap 95% CI for that mean | [-0.049, 0.034] | [0.169, 0.242] | [0.204, 0.282] |

The original 4-versus-1 comparison remains close to zero, consistent with the manuscript's +0.003 dB. However, it does not isolate placement: it changes the structural operator's sweep count. Matching four sweeps per application reveals a modest positive advantage for interleaving. Matching one sweep also favors interleaving on average.

These are comparisons **conditional on a shared selected rank**, not fully tuned comparisons where each variant chooses its own rank. Equal sweeps per application do not make total computational budgets equal; repeated application is the intervention being tested. The results therefore support a modest placement effect under this design, not universal superiority.

The estimates themselves remain different: the median norm separation between interleaved-4 and final-1 is **0.532 times channel norm at -5 dB**, falling to **0.022 at 20 dB**. Replace any statement implying identical results or established equivalence.

**Use:** `projection_placement.pdf`.

## Required manuscript changes

1. **Correct SNR.** Use total multiuser signal power divided by complex noise variance. For three equal-power users, the old single-user wording differs by 4.771 dB. The new plots state “Aggregate SNR.”
2. **State the actual algorithm.** Give the EM update equations, spectral initialization, four Cadzow sweeps, shared scalar rank across users, deterministic 70/30 column split, candidate and final iteration budgets, and final refit on all pilots. The current description sounds as if 30% of pilots are permanently discarded.
3. **Describe finite Cadzow processing accurately.** Rank truncation followed by Hankel averaging is a low-rank structural heuristic; finite sweeps need not end at exactly rank r after re-lifting. Low Hankel rank alone does not enforce all physical unit-circle constraints of the ULA model.
4. **Choose a consistent statistical narrative.** Label paired medians, ratio-of-sums NMSE and their different uses. Add confidence intervals and exact counts. Pooled random-SNR numbers depend on the SNR sampling distribution.
5. **Replace the threshold language.** Distinguish measured interpolation, extrapolation and fallback plateaus. Prefer the matched Figure 3. Include normalized path count as a comparator.
6. **Rewrite the placement conclusion.** Report the original and matching-sweep comparisons. The effect of interleaving is modest but not absent in the controlled experiment.
7. **Update numerical summaries consistently.** Do not mix new figures with old exact numbers. Alternatively present the new run as a replication and identify the original numbers separately.
8. **Fully specify transfer and timing experiments.** State the alternative-channel normalization, changing pilot counts in the user experiment, total runtime including selection, hardware/software and timing repetitions.
9. **Repair equation (9).** The supplied PDF visibly starts that equation with an equals sign; its left-hand-side field variable is missing. Write `\mathbf E=\mathbf G\mathbf S+\mathbf B+\mathbf W` and then `\mathbf Z=|\mathbf E|`. Also remove or explain the unused configuration B in Table I when updating the experiment map.

Suggested replacement LaTeX is supplied in `suggested_replacements.tex`. It is an integration aid for the author's nine-page source, not an automatically edited version of the repository's different, longer manuscript.

## Submission assessment

The numerical foundation is now substantially stronger. The main improvement, array-capacity trend, complexity dependence and positive gains on two additional channel distributions survive independent trials. The matching-sweep comparison adds a useful result the original manuscript obscured.

I would **not submit the current PDF unchanged**. Its strongest effective-rank interpretation exceeds the evidence, two reported crossings are extrapolations, the placement statement omits a confound, and the SNR/algorithm descriptions need correction. After these changes, this can be presented as a reproducible empirical study of the benefits and limitations of structural channel estimation. A code-and-simulation replication cannot establish literature novelty or journal acceptance; the comparison against the closest prior estimator must still be defended in the paper.

## Files and reproducibility

- `verification_results.json`: all 73 cell summaries, intervals, bounds and array aggregates.
- `supplemental_statistics.json`: controlled-placement aggregates and matched predictor comparison.
- `cell_summary.csv`: scalar summaries; full intervals and bin details are in JSON.
- Eight plot sets, each in vector PDF, SVG and 300-dpi PNG.
- `results/trials/`: losslessly consolidated per-trial records; no accuracy data were discarded.
- `results/manifest.json`, `validation.json`, bound stores and test logs: provenance and verification.
- `README.md` and the campaign scripts: rerun commands.

Bootstrap intervals are pointwise percentile intervals, not simultaneous guarantees. Effective-rank resampling retains all users from a trial together. The six-SNR placement aggregate resamples within each SNR stratum. Predictor intervals resample calibration and target trials while retaining the specified test grid. Hardware/model robustness beyond the tested configurations remains outside this verification.
