# Paper 2: bounded reviewer diagnostics

Date: 2026-10-08. Status: complete; saved-output audit passed.

## Scope and what changed

These are **small exploratory checks of frozen estimators**, not a new confirmation study. No model was retrained, no hyperparameter was chosen from these cases, and no existing source, checkpoint, or result was changed. The original 400-case-per-cell study remains the main evidence. The present 32-case checks must not replace its headline numbers.

The primary EM-adaptive checkpoints from seeds 9101, 9102, and 9103 were fixed before generation of the new data. All three share their original training data; their predictions are not treated as independent channels. The run used one CPU thread and double precision. It finished in approximately 331 seconds, below the 850-second cap. This wall time is computation accounting, not a new controlled latency benchmark.

There are 288 new worlds: 32 cases at each of seven SNRs, 32 at weak reference, and 32 with two paths. Eight input-mismatch conditions reuse the 32 SNR=5 dB observations without regenerating them. Thus the evaluation has 17 conditions and 544 condition-cases, **not 544 independent worlds**. Every condition has N=32, K=3, and P=30.

Main SNRs are -10, -5, 0, 5, 10, 15, and 20 dB. Main RSR is 12 dB relative to one user; expected aggregate RSR is 12-10log10(3) dB. Main per-user path counts follow the frozen random 3–7-path generator. Weak-reference and two-path checks use SNR=5 dB, with RSR=0 dB or two paths per user respectively. These are the existing effective magnitude model, not an atomic-response or hardware simulation.

## Matched HS-GS rank-selection loss

Both HS-GS variants use exactly the same candidate channel estimates:

- Fit ranks 1 through 16 using the first 21 pilots, 25 EM iterations, and four Cadzow sweeps per iteration.
- Score predictions on the remaining nine pilots.
- Choose one rank and refit from the original initializer using all 30 pilots, 100 iterations, and four sweeps per iteration.

Only the held-out score changes: the original squared magnitude residual or the Rician negative log likelihood with the supplied noise variance. Terms depending only on observations and variance are omitted from the likelihood; they cannot change the selected rank. Ordinary EM-GS uses 100 iterations and no structural processing.

On a fresh SNR=5 trial, the paired implementation's residual-score HS-GS exactly reproduced the frozen `sahure.classical` estimate: maximum absolute output difference 0, identical rank, and identical score array. A separate unit test checks residual-score parity, and the likelihood is checked against `scipy.stats.rice` including the omitted normalization term.

Cell NMSE is 10log10(sum of squared channel errors / sum of channel energies). It is computed for each frozen learned seed and then averaged in dB. The main macro-average gives each of the seven SNR cells equal weight. Intervals are 95% paired case-bootstrap intervals with 2,000 resamples, sharing case indices across methods and seeds within each cell and using independent draws across SNR cells. They are conditional on these fixed checkpoints and are exploratory, unadjusted intervals, not simultaneous confirmation claims.

| Main seven-SNR comparison | Gain (dB; positive is better) | Conditional 95% interval |
|---|---:|---:|
| Rician-ranked HS-GS over residual-ranked HS-GS | 0.182 | [0.104, 0.267] |
| Adaptive ten-stage estimator over residual-ranked HS-GS | 1.465 | [1.271, 1.667] |
| Adaptive ten-stage estimator over Rician-ranked HS-GS | 1.282 | [1.092, 1.485] |
| Adaptive ten-stage estimator over EM-GS(100) | 4.081 | [3.942, 4.228] |
| Full ten-stage output over its five-stage prefix | 0.890 | [0.815, 0.964] |

Rician scoring changed the selected rank in 26 of 224 main cases, with 17 of those changes at -10 dB. Its gain was concentrated at low SNR. Therefore the different selection loss accounts for part, but not all, of the observed advantage in this small check. This does not establish superiority to all likelihood-tuned structural methods.

The adaptive-over-Rician-HS main gain was 1.383, 1.138, and 1.326 dB for the three fixed training seeds. This spread is separate from the conditional case-bootstrap intervals.

| Condition | EM-GS(100) NMSE | HS residual NMSE | HS Rician NMSE | Adaptive 10 NMSE | Prefix 5 NMSE |
|---|---:|---:|---:|---:|---:|
| SNR -10 | 5.569 | 3.035 | 1.949 | -1.197 | -1.217 |
| SNR -5 | 0.906 | -2.170 | -2.246 | -3.826 | -3.817 |
| SNR 0 | -4.984 | -7.811 | -7.857 | -9.035 | -8.720 |
| SNR 5 | -10.502 | -12.526 | -12.496 | -14.043 | -13.876 |
| SNR 10 | -15.744 | -18.613 | -18.713 | -19.195 | -18.701 |
| SNR 15 | -20.454 | -23.128 | -23.128 | -24.086 | -22.577 |
| SNR 20 | -25.925 | -28.235 | -28.235 | -28.317 | -24.563 |
| Weak reference, RSR 0 | -5.951 | -7.079 | -7.138 | -6.125 | -4.278 |
| Two paths per user | -10.335 | -17.930 | -17.930 | -16.477 | -16.775 |

All entries are dB; lower is better. The weak-reference and two-path failures remain. At SNR=20 the adaptive gain over HS-GS is small; no uniform-improvement claim follows.

## Reference and noise-input mismatch

Every mismatch is an **estimator-input error** on exactly the same saved observations, true reference, true channel, and true noise variance as the clean SNR=5 dB cell. No extra observations are introduced. This isolates sensitivity to the supplied calibration; it is not a calibrated model of real atomic hardware fluctuations.

- Amplitude error: supplied B is 0.9 B_true or 1.1 B_true. These are amplitude multipliers, not power percentages.
- Constant phase error: supplied B is exp(j delta) B_true, delta=-10 or +10 degrees, for every antenna and pilot.
- Phase-ramp error: supplied B[n,p] is B_true[n,p] exp(j delta p/(P-1)), p=0,...,29, ending at delta=-10 or +10 degrees. The phase error begins at zero, not at the opposite endpoint. The true observations are unchanged.
- Noise error: supplied total complex noise variance is sigma_true^2 times 10^(delta/10), delta=-6 or +6 dB. Actual noise and observations are unchanged.

| Supplied-input error | Adaptive degradation from clean (dB) | Conditional paired interval | Adaptive NMSE | Rician-HS NMSE |
|---|---:|---:|---:|---:|
| Reference amplitude x0.9 | 2.430 | [2.220, 2.665] | -11.613 | -9.799 |
| Reference amplitude x1.1 | 1.791 | [1.533, 2.076] | -12.252 | -11.328 |
| Reference phase -10 degrees | 2.358 | [2.114, 2.623] | -11.685 | -10.730 |
| Reference phase +10 degrees | 2.471 | [2.221, 2.743] | -11.572 | -10.564 |
| Phase ramp 0 to -10 degrees | 0.776 | [0.642, 0.909] | -13.267 | -12.029 |
| Phase ramp 0 to +10 degrees | 0.839 | [0.689, 0.995] | -13.204 | -11.774 |
| Noise variance -6 dB | 0.238 | [0.145, 0.332] | -13.806 | -12.567 |
| Noise variance +6 dB | 3.048 | [2.593, 3.494] | -10.995 | -12.404 |

Positive degradation means worse than the same estimator with clean inputs. **Noise overestimation by 6 dB reverses the adaptive estimator's advantage on this cell:** it is about 1.41 dB worse than Rician-ranked HS-GS. These results require a stronger calibration/noise-sensitivity limitation, not a robustness claim.

### Constant phase is a calibration ambiguity

For a common phase delta, |exp(j delta)(GS+B)|=|GS+B|. All tested estimators therefore rotate their estimated channel by exp(j delta) when the supplied reference is rotated. This equivariance held to a maximum relative discrepancy of 4.5e-15 in the saved outputs. With an unknown reference phase, the absolute common channel phase cannot be inferred from magnitudes alone without another anchor.

As an **oracle diagnostic only**, aligning each estimate to truth by one common phase makes the constant-phase degradation disappear to numerical precision. This alignment uses truth and is not available to the receiver. The raw channel NMSE above remains the primary metric; we do not hide calibration error by replacing it with aligned error. Phase ramps are not exactly a single common phase and do change the aligned error.

## Depth and controller behaviour

The five-stage estimate is taken directly from stage five of the unchanged ten-stage forward pass. It retains the trained t/10 stage feature. It is **not an independently trained five-stage model**, and no 20-stage model was trained or extrapolated. No depth hyperparameter was selected using these cases.

Full depth gained 0.890 dB on the main macro-average, largely from high-SNR refinement. It did not improve every condition: the prefix was slightly better at -10 dB and on the two-path cell. Therefore these checks support the usefulness of later updates in this frozen model, not optimality of exactly ten stages, diminishing returns beyond ten, or convergence.

The controller's dimensionless threshold tau/s_H and mixing eta were recorded after every stage for every case and seed. Here s_H=||H||_F/sqrt(min(H.shape)). Stage-wise means and empirical 10th/90th percentiles are in `summary.json`; raw trajectories are in the learned-model NPZ files.

| Condition | Mean tau/s_H, stage 1 | Mean tau/s_H, stage 10 | Mean eta range across stages |
|---|---:|---:|---:|
| SNR -10 | 1.992 | 2.213 | 0.999996–0.999998 |
| SNR 5 | 0.915 | 0.362 | 0.999956–0.999961 |
| SNR 20 | 0.809 | 0.147 | 0.999925–0.999979 |
| Weak reference | 1.122 | 0.380 | 0.999961–0.999971 |
| Two paths | 1.024 | 0.506 | 0.999967–0.999970 |
| Supplied noise +6 dB | 1.605 | 1.051 | 0.999974–0.999978 |

The controller changes the smoothing threshold; it does not meaningfully turn the filter off in these cases. Stronger smoothing with overstated noise is consistent with the new failure, but this is not a causal isolation because the same wrong variance also changes the measurement updates. These trajectory summaries do not establish that the controller detects a calibration fault.

## Verification and preserved evidence

- Four dedicated unit tests passed: Rician score normalization against SciPy, exact old rank-score parity, classical common-phase equivariance, and learned common-phase equivariance.
- Independent recomputation checked all 153 method-output files against the saved truth, with maximum saved-error discrepancy 0.
- Original expansion frozen-source hashes, all current source hashes, all three checkpoint hashes, and all new bank hashes remained unchanged.
- All 17 conditions and all 544 condition-cases completed with finite saved outputs.
- No prior experiment was overwritten. All new files are isolated under `paper2_review_checks/` and `results/paper2_review_checks/`.

Files: `config.json` gives fixed definitions; `manifest.json` records seeds, exact checkpoints, source/data hashes, and library versions; `summary.json` contains cell metrics, contrasts, intervals, and trajectories; `mismatch_summary.json` contains clean-versus-mismatch paired changes; `verification.json` records the independent audit; `cells/` stores per-case estimates and errors; `artifact_hashes.json` records output hashes.

Reproduce from the repository root with the existing paper-2 environment:

```
python -m unittest paper2_review_checks.test_checks -v
python -m paper2_review_checks.run
python -m paper2_review_checks.audit
```

## What remains unresolved

These checks do not supply independently trained 5/10/20-stage comparisons, hardware or atomic-response data, finite-shot or heavy-tailed measurement tests, retraining on calibration errors, a new shrinker comparison, a matched large learned baseline, or global identifiability/convergence guarantees. Separate reviewer-response items must not be marked complete solely because these bounded checks passed.
