# Compact reproducibility package

This package supports checking the paper's saved numerical results and replaying its primary models without new training. It is not a public archive or a software-license grant. All simulations are synthetic; no real receiver measurements are claimed.

## Quick check

Use Python 3.12 and the recorded dependencies in `sahure/requirements.txt`. From this extracted directory:

```sh
python reproduce_saved.py
python -m unittest sahure.test_numerics sahure.test_pipeline sahure_expansion.test_expansion sahure_checks.test_checks sahure_features.test_features
```

The first command recalculates all 19 main-study SNR averages from saved per-case errors and replays all three primary checkpoints on eight saved observations. It does not train anything. Tests include tiny temporary training/resume fixtures; those are implementation checks, not research experiments. Only load model checkpoints from a source you trust.

For the primary estimator, use `sahure.train.load_model` and `model.estimate(Z, S, B, noise_variance)`. See `sahure/README.md` and `sahure/MATHEMATICS.md` for shapes, complex-noise conventions and equations. The manuscript averages the three seed scores, not the three predicted channels.

## Included

- Unchanged estimator, simulator, training, evaluation and regression-test source modules.
- Frozen configurations, selected/final checkpoints, histories and recorded software versions for the three manuscript studies.
- Compact per-case errors and channel energies for all 2,254 result files. Trial identifiers and observation/checkpoint hashes are preserved where originally recorded.
- Saved summary statistics, timing records, protocol documents and audits.
- An eight-case observation subset and saved predictions for the three primary EM models.
- `SHA256SUMS.json` hashes every packaged payload. These hashes describe the actual compact package.

## Omitted and scope

Full training/validation/test observation arrays, full predicted-channel arrays, large stage diagnostics and irrelevant runtime logs are omitted. The original local results retain those arrays. The compact error arrays suffice to recalculate aggregate errors; they cannot independently verify every saved prediction against every true channel. The new full-result audit was run on the complete original data before packaging.

Original files such as `artifact_hashes.json`, `protocol.json` and `test_hashes.json` describe the full historical experiment, not this compact copy. Some paths refer to omitted files or the original workspace. Do not expect original full-data audits to pass against the compact archive alone. Never edit those recorded hashes to hide omissions.

To regenerate full experiments, inspect the frozen configuration and the module READMEs in an isolated fresh working copy. The drivers use fixed relative output directories. Do not launch them over existing results, and do not start multiple controllers. Training and full evaluation are intentionally not run by `reproduce_saved.py`. Regeneration needs the recorded dependencies and seeds; floating-point differences on other library/device versions are possible.

The feature comparison is a smaller-budget diagnostic that failed its cell-protection condition. Bootstrap intervals describe paired test-world uncertainty conditional on saved models, not independent training populations. No novelty certification or hardware validation is provided by this package.
