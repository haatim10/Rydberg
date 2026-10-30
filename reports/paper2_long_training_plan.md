# Paper 2: conventional-loss training-budget control

Written before new experimental results on 17 September 2026. This is a local
prospective plan, not an externally registered study. Existing manuscript
results are already known. User authorized execution in this conversation.

## Question and scope

How does the measured Hankel gain change when conventional wide-SNR training
has a larger optimization budget? This tests sensitivity to training budget.
It does not by itself prove convergence or separate every effect of loss
weighting and sample allocation. Establishing that the difference between
training designs persists at long budgets would also require long-budget
focused and balanced comparisons, which are outside this initial experiment.

## Frozen design

- Fresh runs for each of 13, 26 and 52 epochs, both U (without Hankel) and H
  (fixed rank 7, one step at each layer, identity backward approximation).
- Three seed pairs: labels 1/2/3 mean 20260827/20262827/20263827.
- 18 runs, 546 total training epochs, 43,680,000 sample presentations.
- Existing geometric ULA model, N=32, K=3, P=20, 3--7 paths, RSR=10 dB,
  spectral initialization, 80,000 training worlds, 2,000 validation worlds
  and the original 2,000 test worlds. Training SNR is [-10,20] dB.
- Original model, conventional mean normalized squared error, batch size 32,
  Adam starting at 0.001, clipping at 1.0. Cosine schedule reaches zero at
  the respective budget. Thus the intervention changes training duration AND
  schedule duration, not duration alone. Continuing the old zero-rate schedule
  is invalid. Do not describe this as an exact checkpoint extension.
- CPU float32; numerical library versions and platform are recorded. Fresh
  13-epoch controls distinguish within-platform changes from old checkpoints.
- The same learned initial parameters and per-epoch sample order within each
  pair. Cached arrays are exact copies of the original dataset tensors; record
  hashes and compare representative entries against the original generator.
- One-standard-error selection using validation only, matching the existing
  rule. Also report the final epoch separately; this prevents early selection
  from concealing later behavior. Never choose using test scores.

## Saved evidence and analysis

Store each epoch's weights, training and validation curves, per-trial validation
errors, batch-order hashes, optimizer/scheduler state and random-generator
states. Interrupted runs resume from the last complete epoch. Save test errors
for both selected and final checkpoints, including numerator and denominator.

Primary readout: per-seed median paired gain 10log10(error_U/error_H) above
5 dB, its mean across seeds and paired change from the fresh 13-epoch control.
Report all six SNR bins and the full-range result. Separately report true
ratio-of-squared-error-sums and ratio-of-normalized-error-sums, naming them
accurately. Three-seed t intervals assume approximate normality; no claim of
absence or equivalence based merely on a non-significant result.

Do not stop, add seeds or choose a budget based on test results. Numerical
failure stops the queue and records an error. A successful 52-epoch run does
not establish convergence automatically: inspect the training/validation
curves and report any continuing trend. Manuscript figures are not replaced
until all runs and matched-order checks are complete.

## Execution

From the repository root, use the project-root `.venv-paper2/bin/python` with
`PYTHONPATH=.`. The driver is `scripts/paper2_long_training.py`, with `smoke`,
`launch`, `status`, and `run` commands. All outputs are isolated under
`results/paper2_long_training/`. The launcher starts a persistent local process;
the computer must remain powered on and awake for training to progress.
