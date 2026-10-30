# Self-supervised adaptive Hankel: exploratory pilot

Promising accuracy and speed at the pilot condition.

Completed six models, 20 epochs each, and 400 fresh paired test worlds. There is one training seed, one SNR (5 dB), and one single-user RSR (12 dB). This is an implementation and feasibility result, not a completed publication experiment.

The primary core selected by validation held-out likelihood is **GD**. Its adaptive model has NMSE **-14.791 dB**, gain over HS-GS **1.789 dB** (pointwise paired 95% interval [1.656, 1.920]), and **67.93×** median inference speedup.

## Results

| Estimator | NMSE (dB) | Gain vs HS-GS (dB), 95% CI | Median inference (ms) | Speedup vs HS-GS |
|---|---:|---:|---:|---:|
| Learned GD: adaptive shrinkage | -14.791 | 1.789 [1.656, 1.920] | 5.32 | 67.93× |
| Learned EM: adaptive shrinkage | -14.739 | 1.737 [1.607, 1.864] | 5.52 | 65.43× |
| Learned GD: global shrinkage | -14.540 | 1.538 [1.406, 1.674] | 3.67 | 98.46× |
| GD: adaptive replaced by constants | -14.496 | 1.493 [1.356, 1.635] | 3.66 | 98.77× |
| Learned EM: global shrinkage | -14.430 | 1.427 [1.294, 1.562] | 3.83 | 94.24× |
| EM: adaptive replaced by constants | -14.402 | 1.400 [1.262, 1.539] | 3.84 | 94.06× |
| Rician GD + hard Hankel | -13.319 | 0.316 [0.159, 0.472] | 31.92 | 11.32× |
| HS-GS + rank search | -13.003 | 0.000 [0.000, 0.000] | 361.20 | 1.00× |
| Learned GD: no structure | -10.927 | -2.075 [-2.223, -1.930] | 1.85 | 195.63× |
| Learned EM: no structure | -10.844 | -2.159 [-2.305, -2.011] | 2.02 | 179.11× |
| EM-GS (10) | -10.825 | -2.178 [-2.323, -2.033] | 2.77 | 130.37× |
| Rician GD (100) | -10.651 | -2.352 [-2.496, -2.209] | 11.43 | 31.60× |
| EM-GS (100) | -10.650 | -2.352 [-2.497, -2.210] | 11.21 | 32.22× |
| Rician GD (convergence) | -10.650 | -2.352 [-2.497, -2.210] | 29.36 | 12.30× |
| Amplitude GD (100) | -10.538 | -2.464 [-2.609, -2.319] | 5.87 | 61.53× |
| Amplitude GD (convergence) | -10.536 | -2.466 [-2.611, -2.321] | 12.59 | 28.70× |
| Linearized LS | -8.946 | -4.057 [-4.242, -3.869] | 0.69 | 523.33× |

Positive gain means lower total squared channel error than the reference. NMSE is 10 log10(sum error / sum true-channel energy). Intervals use 4,000 paired resamples of whole test worlds. They do not measure variation across training seeds. Timing is the median across worlds of three batch-one repeats.

## Does adaptation add value?

| Adaptive model compared with | Gain (dB) | Paired 95% CI |
|---|---:|---:|
| em_adaptive_vs_em_global | 0.309 | [0.264, 0.357] |
| em_adaptive_vs_em_none | 3.896 | [3.782, 4.013] |
| em_adaptive_vs_em_constant_control | 0.337 | [0.290, 0.387] |
| gd_adaptive_vs_gd_global | 0.251 | [0.206, 0.295] |
| gd_adaptive_vs_gd_none | 3.864 | [3.751, 3.981] |
| gd_adaptive_vs_gd_constant_control | 0.295 | [0.251, 0.339] |

The constant-output intervention averages the adaptive network outputs on validation observations using all 30 pilots, matching deployment, then freezes them per stage. It keeps the learned update scalars. This differs from separately training the global-shrinkage model. A small gap to this intervention weakens the case that sample-dependent decisions are necessary. All these comparisons are within the same 3–7-path distribution; held-out-complexity adaptation has not yet been established.

The simpler GD global-shrinkage model already improves on HS-GS by 1.538 dB. The adaptive network adds 0.251 dB, with interval [0.206, 0.295]. Therefore, the full gain over HS-GS should not be attributed to measurement-dependent adaptation alone.

The learned gates stay almost fully open in this pilot: EM minimum/mean 0.9964/0.9979; GD minimum/mean 0.9980/0.9989. Both exceed 0.99 at every measured user/stage. This does **not** demonstrate learning when to turn structure off. A fixed-gate ablation and higher-complexity channels remain necessary to assess that claim.

## Training and measurement protocol

- Train/validation/test sizes: 4,096 / 512 / 400; disjoint master seeds in `config.json`.
- N=32, K=3, P=30, independent per-user path counts uniform on 3–7. Paper 1 channel, Gaussian pilots, reference and aggregate-SNR calibration reused.
- Each epoch redraws context/held-out pilot sets: 21 context and 9 loss pilots. Initialization and every model feature use context only.
- Validation uses a fixed pilot split and held-out Rician likelihood only. Test inference uses all 30 pilots. No channel-label loss or selection.
- Six models: EM/GD × none/global/adaptive structure. Parameter counts: 10 / 30 / 188. Adam 0.001, batches 32, 10 stages, 20 epochs, float32 training.
- Total measured training plus per-epoch validation: 27.26 minutes. Data generation, preflight and test evaluation are additional.
- All inference timings include spectral initialization, feature extraction and structural operations. HS-GS timing includes its exhaustive rank search. CPU single-thread, double precision for every estimator. Hardware: Apple M2, 8 physical cores, 8 GiB RAM; see `hardware.json`.
- Classical comparisons use the vectorized NumPy/SciPy implementation; learned models use PyTorch. Speed is implementation-specific, not an operation-count theorem.
- Before test evaluation, unused spectral-feature computations were removed from the global controls. All six arms gave bit-identical outputs and parameter gradients on the equivalence fixture. The original training source and both hashes are saved; the live training run retained its original implementation.
- Global hard-Hankel Rician rank is 5, selected by held-out validation NLL (25 candidate updates, one sweep). Final run uses 100 updates. HS-GS retains Paper 1 selection (magnitude residual, 25 candidate updates, four sweeps) and 100 final updates.
- Rician and amplitude gradient references share initialization and 100-step budgets; additional convergence-controlled runs cap at 1,000 and report actual convergence.
- Xu comparison solves the 1-D linearized least-squares objective; it still receives the same exact nonlinear noisy observations. It is not a reproduction of Xu’s 2-D rank-constrained estimator.

Rician GD (convergence) reached the declared stopping tolerance in 399/400 trials. Remaining trials hit the cap; no global optimum is claimed.

Amplitude GD (convergence) reached the declared stopping tolerance in 399/400 trials. Remaining trials hit the cap; no global optimum is claimed.

## Numerical verification

The test suite checks density normalization and SciPy agreement, total complex noise variance, Wirtinger/autograd/finite-difference agreement, zero and large fields, the EM preconditioner identity, repository initializer/EM/GD parity, Hankel inversion, solve/SVD shrinkage equivalence, repeated singular values, zero-gate behavior, user permutation, float32/64 agreement, held-out isolation, truth-file independence, mask behavior, checkpoint reload, and the fixed-output intervention. See the saved test log.

## Publication scope and next decision

Promising accuracy and speed at the pilot condition.
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
