# Self-supervised adaptive Hankel: exploratory pilot

The primary adaptive model does not meet either predeclared accuracy–speed target at this pilot condition.

Completed six models, 1 epochs each, and 8 fresh paired test worlds. There is one training seed, one SNR (5 dB), and one single-user RSR (12 dB). This is an implementation and feasibility result, not a completed publication experiment.

The primary core selected by validation held-out likelihood is **GD**. Its adaptive model has NMSE **-10.269 dB**, gain over HS-GS **-1.833 dB** (pointwise paired 95% interval [-2.263, -1.480]), and **63.56×** median inference speedup.

## Results

| Estimator | NMSE (dB) | Gain vs HS-GS (dB), 95% CI | Median inference (ms) | Speedup vs HS-GS |
|---|---:|---:|---:|---:|
| HS-GS + rank search | -12.102 | 0.000 [0.000, 0.000] | 417.14 | 1.00× |
| Rician GD + hard Hankel | -11.820 | -0.282 [-1.503, 0.672] | 37.25 | 11.20× |
| Learned GD: adaptive shrinkage | -10.269 | -1.833 [-2.263, -1.480] | 6.56 | 63.56× |
| GD: adaptive replaced by constants | -10.269 | -1.833 [-2.263, -1.480] | 6.36 | 65.55× |
| Learned GD: global shrinkage | -10.268 | -1.834 [-2.264, -1.481] | 6.39 | 65.32× |
| Learned GD: no structure | -10.132 | -1.970 [-2.407, -1.626] | 2.22 | 187.93× |
| Learned EM: adaptive shrinkage | -9.834 | -2.268 [-2.669, -1.887] | 6.83 | 61.07× |
| EM: adaptive replaced by constants | -9.834 | -2.268 [-2.669, -1.887] | 6.93 | 60.20× |
| Learned EM: global shrinkage | -9.833 | -2.268 [-2.670, -1.888] | 6.73 | 62.01× |
| Learned EM: no structure | -9.720 | -2.382 [-2.776, -2.003] | 2.52 | 165.28× |
| EM-GS (10) | -9.719 | -2.382 [-2.776, -2.003] | 3.30 | 126.28× |
| Rician GD (100) | -9.605 | -2.497 [-2.987, -2.134] | 13.44 | 31.03× |
| Rician GD (convergence) | -9.605 | -2.497 [-2.988, -2.134] | 36.67 | 11.37× |
| EM-GS (100) | -9.605 | -2.497 [-2.988, -2.134] | 13.12 | 31.80× |
| Amplitude GD (100) | -9.498 | -2.604 [-3.050, -2.282] | 7.02 | 59.45× |
| Amplitude GD (convergence) | -9.497 | -2.605 [-3.051, -2.283] | 15.26 | 27.33× |
| Linearized LS | -8.047 | -4.055 [-5.280, -3.037] | 0.81 | 512.32× |

Positive gain means lower total squared channel error than the reference. NMSE is 10 log10(sum error / sum true-channel energy). Intervals use 4,000 paired resamples of whole test worlds. They do not measure variation across training seeds. Timing is the median across worlds of three batch-one repeats.

## Does adaptation add value?

| Adaptive model compared with | Gain (dB) | Paired 95% CI |
|---|---:|---:|
| em_adaptive_vs_em_global | 0.001 | [0.001, 0.001] |
| em_adaptive_vs_em_none | 0.115 | [0.096, 0.133] |
| em_adaptive_vs_em_constant_control | 0.000 | [0.000, 0.000] |
| gd_adaptive_vs_gd_global | 0.001 | [0.001, 0.001] |
| gd_adaptive_vs_gd_none | 0.137 | [0.116, 0.159] |
| gd_adaptive_vs_gd_constant_control | 0.000 | [0.000, 0.000] |

The constant-output intervention averages the adaptive network outputs on validation context observations, then freezes them per stage. It keeps the learned update scalars. This differs from separately training the global-shrinkage model. A small gap to this intervention weakens the case that sample-dependent decisions are necessary. All these comparisons are within the same 3–7-path distribution; held-out-complexity adaptation has not yet been established.

## Training and measurement protocol

- Train/validation/test sizes: 32 / 8 / 8; disjoint master seeds in `config.json`.
- N=32, K=3, P=30, independent per-user path counts uniform on 3–7. Paper 1 channel, Gaussian pilots, reference and aggregate-SNR calibration reused.
- Each epoch redraws context/held-out pilot sets: 21 context and 9 loss pilots. Initialization and every model feature use context only.
- Validation uses a fixed pilot split and held-out Rician likelihood only. Test inference uses all 30 pilots. No channel-label loss or selection.
- Six models: EM/GD × none/global/adaptive structure. Parameter counts: 10 / 30 / 188. Adam 0.001, batches 32, 10 stages, 20 epochs, float32 training.
- Total measured training plus per-epoch validation: 0.01 minutes. Data generation, preflight and test evaluation are additional.
- All inference timings include spectral initialization, feature extraction and structural operations. HS-GS timing includes its exhaustive rank search. CPU single-thread, double precision for every estimator.
- Classical comparisons use the vectorized NumPy/SciPy implementation; learned models use PyTorch. Speed is implementation-specific, not an operation-count theorem.
- Before test evaluation, unused spectral-feature computations were removed from the global controls. All six arms gave bit-identical outputs and parameter gradients on the equivalence fixture. The original training source and both hashes are saved; the live training run retained its original implementation.
- Global hard-Hankel Rician rank is 4, selected by held-out validation NLL (25 candidate updates, one sweep). Final run uses 100 updates. HS-GS retains Paper 1 selection (magnitude residual, 25 candidate updates, four sweeps) and 100 final updates.
- Rician and amplitude gradient references share initialization and 100-step budgets; additional convergence-controlled runs cap at 1,000 and report actual convergence.
- Xu comparison solves the 1-D linearized least-squares objective; it still receives the same exact nonlinear noisy observations. It is not a reproduction of Xu’s 2-D rank-constrained estimator.

Rician GD (convergence) reached the declared stopping tolerance in 8/8 trials. Remaining trials hit the cap; no global optimum is claimed.

Amplitude GD (convergence) reached the declared stopping tolerance in 8/8 trials. Remaining trials hit the cap; no global optimum is claimed.

## Numerical verification

The test suite checks density normalization and SciPy agreement, total complex noise variance, Wirtinger/autograd/finite-difference agreement, zero and large fields, the EM preconditioner identity, repository initializer/EM/GD parity, Hankel inversion, solve/SVD shrinkage equivalence, repeated singular values, zero-gate behavior, user permutation, float32/64 agreement, held-out isolation, truth-file independence, mask behavior, checkpoint reload, and the fixed-output intervention. See the saved test log.

## Publication scope and next decision

The primary adaptive model does not meet either predeclared accuracy–speed target at this pilot condition.
The full multi-SNR success criterion cannot be evaluated by this fixed-SNR pilot. No claim of transfer, hardware validation, supervised equivalence, superiority to URformer, or a learned rank is supported here. The architecture retains exact Rician data processing and learns a smooth spatial correction without channel labels. Its empirical contribution must be qualified by the comparison table above.

The larger balanced training campaign, three-seed repetition, SNR/RSR and complexity sweeps, transfer tests, supervised comparator, separately trained final-only model and remaining ablations are deferred to the next decision point. The five pilot figure groups are marked as fixed-condition diagnostics; they are not substitutes for the planned SNR/RSR and transfer figures.

## Files

- `config.json`, phase manifests: configuration, seeds, source hashes, software and hardware metadata.
- `frozen_selection.json`: checkpoint hashes, validation decisions and fixed classical settings, saved before test generation.
- `checkpoints/`: best and last models, optimizer state, training histories and constant controls.
- `data/*_observations.npz`: label-free training inputs. `data/*_truth.npz`: evaluation-only labels/diagnostics.
- `trials.jsonl`: every method/world error, energy, repeated timing, rank or per-stage diagnostics.
- `summary.json`: aggregate values and paired uncertainty.
- `figures/`: five pilot figure groups in PDF and PNG.
- `../../sahure/MATHEMATICS.md`, `LITERATURE.md`, and `README.md`: derivation, overlap assessment and reproduction commands.
