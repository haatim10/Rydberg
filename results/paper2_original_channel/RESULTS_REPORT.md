# Paper 2: completed-experiment analysis

Analysis of the completed original-paper-style channel comparison, independent URformer reconstruction, and SNR-training diagnostic. This report does not change the manuscript, training code, dashboard, or saved experiment outputs. No new training was launched.

## Bottom line

Fixed rank-7 Hankel is not a general improvement to our URformer-based estimator. It consistently harms the independent-ray channel. In the clustered channel it provides a repeatable high-SNR gain at the final epoch, but not a consistent high-SNR gain under the predeclared validation-selected checkpoint rule. The ordinary URformer reconstruction still does not reproduce the published curve.

The defensible result is therefore conditional: benefit depends on channel structure and the training/checkpoint protocol. This is not evidence that we have improved the published URformer method.

## 1. Completion and verification

All three queues are complete:

- Main comparison: 12 runs, 600 epochs, two channel models × two estimators × three training seeds.
- Published-baseline reconstruction: three ordinary-model runs, 150 epochs.
- Training-SNR diagnostic: two ordinary-model runs, 20 epochs. These are ten-epoch prefixes of a 50-epoch learning-rate schedule, not full reproduction runs.

The independent audit checked all 17 result records and 770 epoch histories; all 17 validation-only checkpoint selections; selected-checkpoint finite weights and final-checkpoint agreement with resume weights; 24 main test files; all six 50-epoch U/H training-order matches; all 96 recorded main summary groups; 42 reproduction curve points; and 14 classical curve points. All checks passed.

Frozen source hashes matched all three manifests (49, 51 and 50 entries; many entries overlap). All 104 recorded cached-data file hashes matched. Training, validation and test IDs were disjoint within each experiment's dataset. The SNR diagnostic's starting-weight hashes and training orders matched; its latent training channels/pilots/reference/initial guesses matched, and its validation datasets were byte-identical.

Checkpoint config tuples were canonicalized to JSON lists before comparison; this is a serialization-format difference, not an experiment mismatch.

Audit scope: scores were independently recomputed from saved per-trial errors and stored validation errors. This was not a fresh forward-inference replay of all models, and hash agreement does not prove the scientific model is correct. The audit does not certify agreement with author code.

Machine-readable details: completed_experiments_audit.json. Re-run the read-only check with the project Python and scripts/paper2_completed_results_audit.py.

## 2. What the main comparison tests

U is our ordinary URformer-based implementation. H adds the existing fixed rank-7 Hankel step before the learned correction. Each run has 20,000 training cases and 2,000 validation/test cases, 32 antennas, four users, 20 pilots, 40 rays per user, and training SNR sampled across −10 to 20 dB.

The independent-ray interpretation has independent angles across the visible range. The clustered interpretation has four centers with ten rays each, with offsets uniform within ±5°. That angular spread is our stated assumption, not a confirmed parameter from the original authors.

These are not author-code reproductions. In particular, the U/H implementation differs from the separate printed-equation reconstruction in normalization, filter input, initialization and clipping. The two experiments must not be treated as interchangeable baselines.

Training seeds share the same fixed datasets. Three seeds are not three independently sampled channel populations. There are 2,000 test cases per channel mode, including 1,043 at SNR ≥5 dB; multiplying by seeds does not create 6,000 independent channels.

## 3. Main high-SNR results

Positive gain means Hankel reduces error. Negative gain means it increases error.

The primary planned statistic takes the median of paired per-case dB gains within each seed, then averages those three medians. The secondary pooled mean-NMSE gain first averages linear normalized squared errors across cases and seeds, then takes the dB ratio. These statistics answer different questions and should not be mixed.

| Channel | Saved-model rule | Primary median gain (dB) | Nominal 95% seed interval | Mean-NMSE gain (dB) | U mean NMSE (dB) | H mean NMSE (dB) |
|---|---|---:|---|---:|---:|---:|
| Independent rays | Validation-selected | -4.388 | [-4.719, -4.057] | -4.016 | -10.685 | -6.669 |
| Independent rays | Epoch 50 | -5.909 | [-6.233, -5.585] | -5.046 | -11.424 | -6.379 |
| Clustered ±5° | Validation-selected | -0.184 | [-0.872, 0.504] | +0.041 | -12.196 | -12.237 |
| Clustered ±5° | Epoch 50 | +0.821 | [0.516, 1.127] | +0.731 | -13.054 | -13.785 |

### Independent rays: a clear negative result

Hankel hurts high-SNR performance in all three seeds under both checkpoint rules. Average normalized error is 2.52 times U's error for selected checkpoints, and 3.20 times U's error at epoch 50. This is not an isolated seed failure.

For the full SNR range, mean-NMSE gain is −1.252 dB with selected checkpoints and −0.871 dB at the final epoch. Some very-low-SNR final-epoch cases benefit, but that does not offset the high-SNR damage or establish a general benefit.

### Clustered rays: a conditional positive result

At epoch 50, the three primary high-SNR median gains are +0.792, +0.957 and +0.716 dB. The average is +0.821 dB. Pooled mean-NMSE gain is +0.731 dB, corresponding to about 15.5% lower average normalized error. The positive final-epoch high-SNR result repeats across all three seeds.

With validation-selected models, the primary gains are −0.314, +0.134 and −0.372 dB. The average is −0.184 dB, and the nominal seed interval spans zero. Pooled mean-NMSE gain is only +0.041 dB (about 0.9% lower error). This is not evidence of a reliable selected-checkpoint high-SNR advantage, nor proof of exact equivalence.

Across the full SNR range, clustered mean-NMSE gains are +0.159 dB for selected models (3.6% lower error) and +0.276 dB for final models (6.2% lower error).

The epoch-50 gain is a valid preplanned secondary result. It should be reported, but cannot replace the unfavorable primary result simply because it looks better.

## 4. SNR does not give a universal benefit threshold

This table uses pooled mean-NMSE gain, not the primary paired-median statistic.

| SNR interval (dB) | Cases | Independent selected | Independent final | Clustered selected | Clustered final |
|---|---:|---:|---:|---:|---:|
| [-10,-5) | 317 | -0.230 | +0.428 | +0.140 | +0.309 |
| [-5,0) | 326 | -0.515 | -0.074 | +0.076 | +0.146 |
| [0,5) | 314 | -1.110 | -0.961 | +0.508 | +0.124 |
| [5,10) | 372 | -2.949 | -3.318 | +0.700 | +0.724 |
| [10,15) | 333 | -4.626 | -6.158 | -0.388 | +0.739 |
| [15,20) | 338 | -5.203 | -7.539 | -1.258 | +0.745 |

In the clustered selected-checkpoint comparison, the 5–10 dB bin improves by about 0.70 dB, but 10–15 dB worsens by 0.39 dB and 15–20 dB worsens by 1.26 dB. The latter two mean-NMSE losses occur in every seed. Thus the earlier suggestion that sufficiently high SNR alone ensures benefit is not supported.

At the final epoch, all clustered bins have positive mean-NMSE gain; the high-SNR bins are approximately +0.72 to +0.75 dB. These are bin-level observations, not a precisely located crossover threshold. A 5 dB cutoff was the predeclared reporting boundary, not a newly proven physical threshold.

## 5. Why this may be happening

The pretraining noiseless-channel diagnostic found that a rank-7 projection introduces mean normalized squared distortion of:

- Independent rays: 0.07978, about 8%.
- Clustered ±5° rays: 0.000978, about 0.10%.

Because these checks contain no measurement noise, the distortion is a change to genuine channel structure, not removal of measurement noise. This supports structural mismatch as a plausible explanation for the independent-ray damage.

It is not a complete causal proof. The learned correction can modify or restore components after projection; optimization, the identity-backward approximation through Hankel, and checkpoint selection can also affect the outcome. The noiseless distortion is not a strict lower bound on the final network error.

At high SNR, measurement noise is smaller, so imposed structural distortion can become comparatively more important. High SNR cannot make an incorrect rank assumption correct.

## 6. Checkpoint selection materially changes the comparison

The selection rule uses the earliest epoch within one standard error of the best full-range validation score. It was recomputed from saved validation errors without using test results.

| Run | Selected epoch | Best validation epoch | Best validation NMSE (dB) | Final validation NMSE (dB) |
|---|---:|---:|---:|---:|
| clustered_5deg/seed1_H_50ep | 5 | 12 | -5.635 | -5.188 |
| clustered_5deg/seed1_U_50ep | 14 | 18 | -5.556 | -4.944 |
| clustered_5deg/seed2_H_50ep | 4 | 8 | -5.712 | -5.187 |
| clustered_5deg/seed2_U_50ep | 11 | 20 | -5.414 | -4.913 |
| clustered_5deg/seed3_H_50ep | 5 | 10 | -5.635 | -5.199 |
| clustered_5deg/seed3_U_50ep | 15 | 18 | -5.498 | -4.933 |
| literal_independent/seed1_H_50ep | 4 | 5 | -4.009 | -3.411 |
| literal_independent/seed1_U_50ep | 4 | 6 | -5.200 | -4.122 |
| literal_independent/seed2_H_50ep | 3 | 6 | -4.049 | -3.418 |
| literal_independent/seed2_U_50ep | 4 | 6 | -5.232 | -4.229 |
| literal_independent/seed3_H_50ep | 4 | 5 | -4.008 | -3.403 |
| literal_independent/seed3_U_50ep | 4 | 7 | -5.211 | -4.141 |

For clustered channels, H selects epochs 5, 4 and 5, whereas U selects 14, 11 and 15. Selection is based on the entire SNR range, not just high SNR. Later training can improve high-SNR behavior while worsening average validation behavior, particularly at noisier SNRs.

Training loss falls while validation performance eventually worsens in all main runs, consistent with overfitting under the full-range objective. Finishing 50 epochs is not proof of convergence.

The two checkpoint rules describe two different, preplanned evaluation protocols. Their disagreement means the high-SNR advantage is protocol-dependent; it does not automatically make the final-epoch result invalid. Choosing a different high-SNR-specific selection rule now would be a new method requiring validation and untouched final-test data.

The nominal seed intervals have only two degrees of freedom, assume approximately normal seed effects, condition on the fixed data and pilot matrix, and are not corrected for the multiple bins/comparisons. They do not establish generality across realistic propagation environments.

## 7. Published-baseline reconstruction: still unresolved

The separate reconstruction compares ordinary URformer against Figure 3 of Xiao et al., arXiv:2509.12586v3. Targets were extracted from the downloaded EPS figure and are plot-derived, not author raw data, with a stated conservative digitization tolerance of 0.02 dB.

Mean NMSE is averaged in linear units over matched cases and three training seeds before conversion to dB. More negative NMSE is better.

| SNR (dB) | Published plot | Our epoch 50 | Our validation-selected | Epoch-50 gap to plot |
|---|---:|---:|---:|---:|
| -10 | -1.181 | 1.057 | -0.439 | +2.238 |
| -5 | -10.737 | -0.786 | -2.102 | +9.951 |
| 0 | -17.076 | -3.461 | -4.472 | +13.615 |
| 5 | -21.734 | -7.088 | -7.503 | +14.646 |
| 10 | -26.493 | -10.691 | -11.042 | +15.802 |
| 15 | -30.203 | -13.317 | -13.604 | +16.886 |
| 20 | -32.024 | -14.558 | -14.683 | +17.466 |

At 5 dB the final reconstruction is about 14.65 dB worse than the plot; at 20 dB it is about 17.47 dB worse. Selecting earlier checkpoints does not close the gap. All three reconstruction seeds select epoch 2.

Classical GS and EM-GS are much closer to their plotted curves, about 0.59–1.47 dB better in our reconstruction. This makes neural implementation/training assumptions an important investigation target, but does not prove the simulator matches the authors' setting.

A further warning: at 20 dB, our 100-iteration EM-GS has NMSE −21.970 dB, compared with −14.558 dB for our final reconstructed neural model. Thus this neural baseline is not even the most accurate method in that high-SNR reconstruction comparison. This does not directly compare EM-GS against H in the separate clustered experiment.

The discrepancy is not evidence that the published paper is wrong. Joint angle statistics, SNR-specific versus mixed-SNR training, scaling, initialization and unspecified architectural details remain possible sources of disagreement. No author-code execution or author clarification is available in this experiment.

## 8. The SNR-training diagnostic did not solve the gap

Both arms used the same latent training cases, starting weights and data order. Both were evaluated on the same fresh 2,000 validation cases at 5 dB.

| Saved-model rule | Mixed-SNR training NMSE | Fixed-5-dB training NMSE | Fixed-SNR gain |
|---|---:|---:|---:|
| Final epoch 10 | −7.480 dB | −7.687 dB | +0.207 dB |
| Validation-selected | −7.670 dB | −8.094 dB | +0.424 dB |

This short test suggests that concentrating training at 5 dB helps modestly, but it has not demonstrated anything close to closing the reconstruction gap. It is validation-only, single-seed, and only ten epochs into a 50-epoch schedule. It neither proves that SNR-specific training is unimportant nor predicts the result of fully training separate models at every SNR.

## 9. Relation to the earlier manuscript experiment

The earlier three-to-seven-path, three-user experiment retains its own evidence: at the 52-epoch budget its primary high-SNR median gain was +1.256 dB for selected models and +0.717 dB for final models.

The new findings limit extrapolation from that setting. They do not retroactively erase the earlier positive result. Changes included channel construction, number of users, reference/signal convention, initialization, sample count and training schedule. Therefore the difference from the earlier experiment cannot be attributed solely to channel geometry.

The two new channel modes, by contrast, deliberately use the same surrounding main-experiment protocol, with matched streams where possible. Their contrast supports the importance of angular structure within that protocol.

## 10. Implications for the manuscript

Supported:

- Fixed-rank Hankel can help a specific URformer-based implementation under compatible simulated channel structure and a specified training/evaluation protocol.
- The same addition can cause substantial harm when that rank assumption is inappropriate.
- Channel structure and checkpoint choice must be reported alongside SNR.
- The attempted independent reproduction has a large unresolved gap that is not explained by the short SNR-training diagnostic.

Not supported:

- A general improvement over published URformer.
- A universal high-SNR benefit or a universal 5 dB activation threshold.
- Robustness to general multipath channels.
- Convergence merely because the planned epochs finished.
- A claim that the published result is incorrect.
- A guarantee of publication or a demonstrated new method from these experiments alone.

My assessment: the broad improvement claim is not ready for submission. These experiments provide useful evidence about the limits of the addition, but a narrower paper would still need a clear contribution beyond Paper 1 and a credible baseline. We should not select only the clustered final-epoch gain for the abstract.

The immediate decision is to resolve or explicitly delimit the ordinary baseline before investing in another modification. This report launches no new experiments and makes no manuscript changes.

## Evidence locations

- reports/paper2_original_channel_plan.md
- results/paper2_original_channel/{literal_independent,clustered_5deg}/summary.json and seed*/test_{selected,final}.npz
- results/paper2_original_channel/smoke.json
- reports/paper2_reproduction_plan.md
- results/paper2_reproduction/summary.json, figure3_targets.json, classical.json and source/RAQR_CE_v1.tex
- reports/paper2_reproduction_diagnostic_plan.md and results/paper2_reproduction_diagnostic/summary.json
- results/paper2_long_training/RESULTS_REPORT.md
- completed_experiments_audit.json alongside this report

Published comparison source: https://arxiv.org/html/2509.12586v3 (the downloaded TeX/EPS content was used for this analysis).

