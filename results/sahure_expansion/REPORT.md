# Mixed-condition self-supervised Hankel evaluation

Completed 18 main models: two update cores, three structural variants, and three training seeds. Each model received 20 epochs on the same 16,384 training worlds. Checkpoints used 2,048 independent validation worlds.

The primary core selected by mean validation likelihood is **EM**. Its equally weighted mean gain over HS-GS across seven SNR values is **1.675 dB**, paired-world 95% interval **[1.618, 1.730]**. Median inference speedup at the declared reference condition is **63.80×**.

Combined accuracy–speed target met: **True**. Efficiency target met: **True**.

The improvement is not uniform. HS-GS is better on the two- and three-path tests and at RSR 0 dB. At SNR 20 dB, the positive mean gain has an interval that includes zero. The mean gate is almost one, so the evidence supports learned shrinkage strength rather than an on/off structural decision.

## What was tested

Training spans seven SNRs, three single-user RSRs, random 3–7 paths and fixed 2/4/8/12/16 paths per user. Half the training cases use random path counts. Validation gives equal weight to all 126 SNR/RSR/path-regime strata. RSR denotes reference power relative to one user; the expected reference-to-total-signal ratio is RSR minus 10 log10(K). For K=3, a single-user RSR of 12 dB corresponds to 7.229 dB relative to aggregate signal power. The three training seeds share worlds but change initialization, batch order and training masks. No true channels enter the training loss, estimator inputs or checkpoint selection.

Each epoch splits the 30 pilot columns into 21 context columns and nine held-out columns. The model uses only the context columns to estimate the channel; the loss measures its prediction of the held-out amplitudes. At deployment, weights are frozen and all available pilots are used in one pass. This changes the pilot-count feature from its training value, so transfer performance must be measured rather than assumed.

There are 32 test conditions, each with 400 paired trials. Two assumed-noise variants reuse the reference worlds, giving 12,000 distinct test worlds. Learned weights remain frozen for every condition. Classical baselines are evaluated once per world; their results are shared across training-seed comparisons.

## SNR results

| SNR (dB) | Adaptive NMSE (dB) | Gain vs HS-GS (dB), 95% interval | Adaptive gain vs global (dB), 95% interval |
|---:|---:|---:|---:|
| -10 | -1.254 | 4.696 [4.511, 4.889] | 5.693 [5.587, 5.797] |
| -5 | -4.122 | 1.687 [1.567, 1.804] | 3.289 [3.183, 3.396] |
| 0 | -9.191 | 1.620 [1.480, 1.758] | 2.316 [2.238, 2.401] |
| 5 | -14.348 | 1.422 [1.281, 1.556] | 1.342 [1.291, 1.392] |
| 10 | -19.171 | 1.101 [0.968, 1.231] | 0.316 [0.278, 0.354] |
| 15 | -24.260 | 1.062 [0.915, 1.209] | 0.103 [0.072, 0.136] |
| 20 | -28.563 | 0.139 [-0.017, 0.294] | 0.848 [0.783, 0.912] |

## Complexity and adaptation

Path counts 3 and 6 were not fixed-count training strata, but lie within the random 3–7 training distribution. Therefore this sweep tests complexity dependence; it is not evidence of generalization to wholly unseen path-count support.

| Paths per user | Gain vs HS-GS (dB) | Adaptive gain vs global (dB), 95% interval | Adaptive gain vs constant intervention (dB), 95% interval |
|---:|---:|---:|---:|
| 2 | -1.104 | 3.132 [3.046, 3.221] | 0.515 [0.408, 0.620] |
| 3 | -0.203 | 2.108 [2.037, 2.178] | 1.456 [1.341, 1.572] |
| 4 | 0.453 | 1.586 [1.531, 1.643] | 1.944 [1.833, 2.055] |
| 6 | 1.176 | 1.036 [0.989, 1.080] | 2.522 [2.424, 2.616] |
| 8 | 1.540 | 0.699 [0.661, 0.737] | 2.887 [2.787, 2.981] |
| 12 | 1.510 | 0.292 [0.261, 0.324] | 3.225 [3.139, 3.311] |
| 16 | 1.327 | -0.012 [-0.043, 0.019] | 3.299 [3.219, 3.381] |

The gate and actual correction answer different questions. These diagnostics hold SNR at 5 dB and RSR at 12 dB; values average over stages, users, worlds and training seeds.

| Paths per user | Mean gate | Mean relative threshold | Mean relative correction |
|---:|---:|---:|---:|
| 2 | 1.0000 | 0.7502 | 0.2318 |
| 3 | 1.0000 | 0.6355 | 0.2028 |
| 4 | 1.0000 | 0.6004 | 0.1960 |
| 6 | 1.0000 | 0.5768 | 0.1948 |
| 8 | 1.0000 | 0.5631 | 0.1946 |
| 12 | 1.0000 | 0.5660 | 0.2013 |
| 16 | 1.0000 | 0.5864 | 0.2140 |

## Transfer and mismatch

| Axis | Value | Adaptive gain vs HS-GS (dB), 95% interval |
|---|---|---:|
| rsr | 0 | -1.617 [-2.178, -1.031] |
| rsr | 5 | 1.720 [1.488, 2.038] |
| rsr | 10 | 1.376 [1.248, 1.505] |
| rsr | 12 | 1.422 [1.281, 1.556] |
| rsr | 15 | 1.194 [1.061, 1.325] |
| rsr | 20 | 1.094 [0.972, 1.217] |
| rsr | 25 | 1.154 [1.026, 1.285] |
| pilots | 10 | 2.334 [2.136, 2.531] |
| pilots | 20 | 1.427 [1.273, 1.577] |
| pilots | 30 | 1.422 [1.281, 1.556] |
| array | 8 | -0.106 [-0.286, 0.077] |
| array | 16 | 1.223 [1.086, 1.361] |
| array | 32 | 1.422 [1.281, 1.556] |
| array | 64 | 0.165 [0.024, 0.307] |
| users | 2 | 0.078 [-0.073, 0.228] |
| users | 3 | 1.422 [1.281, 1.556] |
| users | 4 | 1.529 [1.415, 1.639] |
| channel | ula | 1.422 [1.281, 1.556] |
| channel | clustered | 1.996 [1.890, 2.104] |
| channel | ten_path | 1.542 [1.444, 1.634] |
| channel | unequal | 1.776 [1.671, 1.880] |
| noise_offset | -3 | 1.226 [1.095, 1.359] |
| noise_offset | 0 | 1.422 [1.281, 1.556] |
| noise_offset | 3 | 1.050 [0.901, 1.193] |

Different P/N/K and alternative channel distributions are transfer tests. Reference ratios 0/10/15/25 dB were not training values. The unequal-power channel has ten paths with powers proportional to 0.5^ell; all alternative families are normalized to the base world’s realized total channel energy. The clustered family uses four centers with ten rays each and ±5° spread. The noise-offset axis changes the supplied variance only.

## Runtime and controls

| Method | Mean per-SNR gain vs HS-GS (dB) | Training-seed range (dB) | Median inference (ms) | Speedup |
|---|---:|---:|---:|---:|
| em_none | -2.335 | -2.335 to -2.334 | 2.98 | 176.42× |
| em_global | -0.311 | -0.312 to -0.311 | 5.61 | 93.74× |
| em_adaptive | 1.675 | 1.533 to 1.787 | 8.24 | 63.80× |
| em_constant_control | -3.546 | -4.028 to -2.779 | 5.61 | 93.64× |
| gd_none | -2.302 | -2.303 to -2.301 | 2.73 | 192.47× |
| gd_global | -0.174 | -0.177 to -0.169 | 5.38 | 97.66× |
| gd_adaptive | 1.848 | 1.788 to 1.965 | 8.00 | 65.73× |
| gd_constant_control | -2.787 | -3.263 to -2.190 | 5.36 | 98.15× |
| em_10 | -2.507 | -2.507 to -2.507 | 4.10 | 128.14× |
| em_100 | -2.461 | -2.461 to -2.461 | 17.01 | 30.90× |
| hs_gs | 0.000 | 0.000 to 0.000 | 525.62 | 1.00× |
| rician_100 | -2.448 | -2.448 to -2.448 | 17.40 | 30.20× |
| rician_converged | -2.470 | -2.470 to -2.470 | 45.37 | 11.59× |
| amplitude_100 | -3.093 | -3.093 to -3.093 | 8.87 | 59.26× |
| amplitude_converged | -3.129 | -3.129 to -3.129 | 20.93 | 25.11× |
| rician_hard | -1.679 | -1.679 to -1.679 | 47.39 | 11.09× |
| linearized_ls | -5.818 | -5.818 to -5.818 | 1.06 | 495.93× |
| rician_10 | -3.434 | -3.434 to -3.434 | 4.17 | 126.04× |
| amplitude_10 | -3.941 | -3.941 to -3.941 | 3.29 | 159.93× |

Fraction of reference runs meeting the stated stopping criterion before the 1,000-update cap:

| SNR (dB) | Rician gradient | Amplitude gradient |
|---:|---:|---:|
| -10 | 92.2% | 90.2% |
| -5 | 98.0% | 94.0% |
| 0 | 99.8% | 99.2% |
| 5 | 100.0% | 99.8% |
| 10 | 100.0% | 100.0% |
| 15 | 100.0% | 99.8% |
| 20 | 100.0% | 100.0% |

Timing uses CPU, one thread, double precision and batch size one, with initialization and HS-GS rank selection included. The timing pass runs after parallel accuracy workers exit. It uses 100 reference-condition worlds and three repeats per estimator and world. Speedup is specific to this implementation and machine; it is not established for every transfer condition.

The main parallel training span was 3.67 hours, plus 6.05 minutes for the retained first-epoch preflight. This span excludes data preparation. See `training_duration.json` for the timing definition.

## Interpretation and limits

Numbers average per-seed dB summaries; they are not results from averaging three predicted channels. Intervals use 4,000 paired resamples of entire worlds and are conditional on the three trained models. Training-seed variation is reported separately. Intervals are pointwise, without adjustment for multiple comparisons.

The gate is not an estimated rank. Threshold and gate can compensate for one another, so the figures also show the actual relative correction. Global controls learn fixed relative thresholds and gates per stage; their absolute threshold still scales with the current Hankel RMS singular value. A reduction in validation likelihood does not guarantee lower channel NMSE. The convergence-controlled classical runs have a 1,000-update cap; reaching that cap does not establish convergence.

These are simulations of the stated receiver model, not measurements from a physical Rydberg receiver. Low-reference failures can involve reduced identifiability as well as optimization difficulty; exact likelihood alone does not resolve either issue.

This stage does not include a matched supervised network, a separately trained final-only network, or the remaining architecture ablations. It cannot establish equivalence to supervised learning or prove that interleaving is necessary. Publication claims should follow the individual operating-point results, including harmed regions, rather than only the overall average.

## What this supports and what comes next

A supported provisional contribution is a small, self-supervised, measurement-conditioned Hankel shrinkage module within ten exact-Rician updates. It improves the declared average accuracy–runtime comparison against HS-GS without channel labels or deployment-time rank search. The adaptive-versus-global comparisons support using measurement-dependent outputs in this experiment. They do not establish that every feature or the gate is necessary.

The GD core has a larger test SNR-average gain, but EM remains the primary result because it was selected using validation likelihood before test generation. This difference also shows why validation likelihood and channel NMSE should not be treated as identical criteria.

The next focused checks are a gate fixed at one, feature ablations separating spectral information from noise/residual information, a matched supervised model, and a separately trained final-only model. The diagnostics show much stronger variation with SNR than with true effective rank within each SNR group; they do not establish discovery of a universal complexity threshold. This is substantial experimental evidence for developing a paper, but the remaining controls and a current literature comparison are still needed before calling it submission-ready.

## Figures

- [SNR and reference strength](figures/01_snr_reference.pdf): primary EM core, mean per-seed NMSE and pointwise intervals.
- [Channel complexity](figures/02_complexity.pdf): gain over HS-GS and adaptive-versus-global comparisons.
- [Accuracy and runtime](figures/03_accuracy_runtime.pdf): reference setting, serial batch-one timing.
- [Controls and training seeds](figures/04_controls_training_seeds.pdf): individual seeds and conditional paired-world intervals.
- [Transfer and correction diagnostics](figures/05_transfer_diagnostics.pdf): changes in system dimensions, channels and supplied noise.
- [Gate and threshold diagnostics](figures/05b_gate_threshold_diagnostics.pdf): the gate is nearly saturated; threshold varies with SNR.

## Reproduction

Configuration: `config.json`. Frozen selection and code/checkpoint hashes: `frozen_selection.json`. Training curves and optimizer checkpoints: `checkpoints/`. Per-world errors and learned predictions: `cells/`. Full numerical summaries: `summary.json`. Serial timings: `timings.jsonl`. Audit: `audit.json`. Figures are available as PDF and PNG in `figures/`. See the module README for commands.

## Numerical recovery

1 HS-GS trial required an alternative SVD driver after NumPy reported nonconvergence on finite inputs. The same trial was retained, with the same initialization, rank search, updates and projection equations. SciPy gesvd was used only for a failing matrix; reconstruction and orthogonality were checked. The recovery is recorded in `numerical_recovery/` and independently rerun by the audit. No trained checkpoint or original estimator source was changed.
