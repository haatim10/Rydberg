# Controller-input check

Declared 23 September 2026, before feature-study training or fresh test generation. This is a small-budget mechanism diagnostic, not a replacement for the completed broad experiment.

## Question

Does the controller benefit from channel-spectrum information, noise/residual information, or their combination?

| Arm | Hankel-spectrum features | Residual and noise-proxy features | Size and iteration features |
|---|---|---|---|
| full | yes | yes | yes |
| spectral | yes | no | yes |
| measurement | no | yes | yes |
| context | no | no | yes |

Inputs are masked at the shared controller, and each version is trained from scratch. All models store the same 188 parameters and use the same hidden width. Masked input weights do not affect the output, so equal stored parameter counts do not mean equal effective input flexibility. The underlying Rician update, Hankel operation and channel-dependent threshold scale remain present in every arm. This does not remove all physical noise or channel-structure information from the estimator.

## Matched design

- The first 4,096 examples from the old development training set and all 2,048 old validation examples. No old test examples used for training or checkpoint selection.
- Four arms, three matched seeds (9101–9103), 12 epochs: 12 training jobs and 144 complete epoch records.
- Same initial parameters within each seed; same 21/9 context/held-out split seeds, batch order, batch size 32, Adam rate 0.001 and ten EM stages.
- Select the lowest equal-stratum held-out Rician NLL, including epoch zero. Retain final checkpoints as a sensitivity check. Do not pick an arm, seed or epoch by test performance.
- Float32 training and float64 evaluation on one CPU thread. No GPU requirement.

The reduced budget is deliberate. The full-input arm is retrained at the same reduced budget rather than compared using its much larger old training run. Failure to distinguish arms at this budget will not establish equivalence or final-scale redundancy.

## Fresh evaluation

Generate tests only after all selected checkpoints are frozen. Two independent fresh banks contain seven primary SNR cells (−10 through 20 dB in 5 dB steps), plus weak-reference and two-path stress cells. Each cell has 128 paired worlds: 2,304 worlds in total. Use N=32, K=3, P=30, RSR=12 except RSR=0 in the weak-reference cell. The two-path cell uses SNR=5; other primary channels have random 3–7 paths per user.

Evaluate selected and final checkpoints on identical observations. Save predictions, errors, channel energies, trial IDs and observation/checkpoint hashes. There are 24 checkpoint evaluations per cell and 432 expected result files. Old HS-GS results are not reused on new observations; this experiment isolates feature access and does not make a new classical-baseline superiority claim.

Primary score: equal average of seven per-SNR ratio-of-sums NMSE dB scores, then mean of seed scores. Report each seed, cell, stress case, severe-error frequency (per-world NMSE > 1) and final-checkpoint sensitivity.

Three primary contrasts: full versus spectral, full versus measurement, full versus context. Use 4,000 paired-world bootstrap resamples with Bonferroni adjustment across three contrasts within each bank. Intervals are conditional on trained weights; they do not measure uncertainty across training-data populations. The two test banks do not provide independent training datasets.

A practical screening flag requires at least 0.2 dB average gain, an adjusted interval above zero, and no primary-cell point degradation above 0.25 dB. This is not a simultaneous safety confidence bound or a publication success rule. Show both banks even if they disagree. Do not hide stress-case harm behind the primary average or infer equivalence from a nonsignificant difference.

## Verification and resource controls

Before launch, test exact full-model forward/backward equivalence, masked-feature invisibility, disjoint pilot subsets, finite losses and gradients, deterministic epoch restart, and a tiny end-to-end training/evaluation/report audit. Rerun the earlier numerical and pipeline tests. Verify protected code and input hashes.

The local runner has an exclusive lock and rejects duplicates. It trains in round-robin epochs, saves complete-epoch checkpoints and logs progress. It stops at a two-hour budget instead of silently extending into a larger study. A partial epoch is discarded and can be recomputed deterministically. Completion requires saved-output audit and representative checkpoint replays. If a failure occurs, inspect it; do not silently modify the frozen protocol.

Preflight timing is in `preflight.json`; live state is in `status.json`, and epoch updates are in `run.log`. These do not update the old dashboard. Existing manuscripts and completed studies are unchanged.

## Interpretation

If spectral-only matches the full controller, do not claim the noise/residual inputs are essential. If measurement-only matches it, do not claim spectral features drive the improvement. If context-only matches it, the evidence for measurement-dependent control is weaker at this budget. An apparent advantage should survive seed inspection, both test banks and final-checkpoint sensitivity before it informs the paper. A larger run is a separate decision.
