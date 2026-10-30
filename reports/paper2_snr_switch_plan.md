# Exploratory SNR-switch experiment

Written before running this new experiment, 19 September 2026. This is a
follow-up motivated by already-seen results, not an externally registered study.
Do not modify the manuscript, existing weights, or longer-training experiment.

## Question and frozen choices

Can routing a sample to the conventional-trained unstructured model (U) at low
SNR and the Hankel model (H) at high SNR preserve H's high-SNR benefit without
its low-SNR penalty? This is a two-model selector, not a newly trained network.
Also test the literal alternative of disabling Hankel inside the H-trained
network below the same threshold. Its weights were never trained for that
disabled path; it is a diagnostic, not an equivalent replacement for U.

Use all three seeds' validation-selected checkpoints from the 52-epoch-budget
experiment, fixed before evaluation. No training, architecture changes, or
checkpoint changes. Only conventional-loss checkpoints are covered.

Select one common threshold on the existing 2,000-case validation set by
minimizing full-range mean normalized squared error, averaged across the three
seeds. Candidates: -11 (always H), 0, 2.5, 5, 7.5, 10, 21 (always U) dB.
Exact ties prefer the larger threshold (fewer H calls). Use H at SNR >= threshold.
Freeze this selection to disk before generating/evaluating the fresh test set.
Report the previously suggested fixed 5 dB rule as a secondary comparator.
The validation set was previously used for model selection; the fresh test set
is not used for threshold/model selection. Do not adjust any design after test.

Generate 2,000 fresh test cases from the same physical distribution with master
seed 20260919 and trial IDs [3000000, 3002000), test range [3000000, 4000000).
Retain fixed pilots, channel parameters, SNR range [-10,20], initialization,
numeric precision, and all model settings. Verify separation from existing
train/validation/test trial IDs. Three training seeds share this fresh dataset;
they are not three independent test datasets.

## Available SNR information

The estimator already receives simulator-provided noise variance. In this
normalized simulation, SNR = 10 log10(K beta c^2 / sigma2). Validate this
identity. Route using this calibrated input, not channel ground truth or the
per-sample estimation error. This is still an ideal known-noise/known-calibration
assumption, NOT a demonstrated practical SNR estimator.

Secondary routing-only sensitivity tests: add fixed-seed zero-mean Gaussian
SNR errors with standard deviations 1 and 3 dB, and constant biases -3 and +3 dB.
Do not retune thresholds. These perturb only the router; the estimators retain
the correct noise variance. They do NOT test end-to-end noise-estimation error.

## Outcomes and checks

Primary: full-range normalized-error-sum gain of the selected router versus U
and versus H, per seed and average. Also report low/high SNR and six 5 dB bins,
paired median gains, raw-error-sum gains, absolute NMSE, and fraction routed H.
Report three-seed SD and nominal t intervals as limited training-seed uncertainty
on one fresh dataset; these do not cover simulator/hardware uncertainty.

Verify route endpoints and exact per-trial branch identities. Independently
execute routing (only one model per sample) and compare to stored branch outputs.
Test same-weight hard toggling separately. Finite-value checks on every array.
Save per-trial results and source checkpoint/code/data hashes.

Measure warmed single-thread CPU inference for U, H, two-model routing and
same-weight toggling on the first 256 fresh cases, batch size 32, three repeats
with rotated execution order. Exclude data generation and common spectral
initialization; include routing and tensor splitting. State these are local
batched timings, not hardware deployment latency, and report two-model storage.
No significance or novelty claim follows automatically from a positive result.
