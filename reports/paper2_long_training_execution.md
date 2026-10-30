# Longer-training experiment: execution record

Started locally on 17 September 2026. Outputs are in
`results/paper2_long_training/`. This record describes setup and checks;
it contains no conclusions from the new full training runs.

## Checks completed before the full queue

- Installed the repository's recorded versions in an isolated project
  environment: PyTorch 2.13.0, NumPy 2.4.6 and SciPy 1.17.1. CPU float32.
- Verified identical learned initial parameters for the two network variants.
- Verified that restoring model, optimizer and random state reproduces the
  next optimizer update exactly on a diagnostic batch.
- Exercised data caching, both training variants, validation checkpoint
  selection, selected/final test scoring, completed-run skipping and checkpoint
  recovery using a disposable small dataset. These diagnostic outputs were
  removed and are not included in scientific results.
- Re-evaluated the original six conventional-loss checkpoints on the original
  2,000 test cases. The three high-SNR median gains reproduced as 1.20878673,
  1.35138880 and 1.29836706 dB. The maximum absolute discrepancy across these
  and all per-bin values was approximately 0.00001013 dB, below the predetermined
  0.005 dB tolerance. Full details: `original_reproduction.json` in the output
  directory. This is a reproduction check, not a new experiment result.

## Queue and timing

18 runs: three seeds, two variants and separate 13/26/52-epoch schedules.
One run at a time limits memory pressure on this 8 GB Mac. Shared data arrays
are stored on disk and mapped into memory. No paid compute service is used.

A brief warm timing test estimated about 30.4 hours for training updates alone.
Validation, checkpoint writes and thermal throttling can increase this. Data
generation precedes training. No completion time is guaranteed by this estimate.

The laptop must remain powered on and awake for the process to progress.
Sleep pauses computation; an interrupted process can resume from its last
completed epoch. Each run records its own learning curves and checkpoints.

## Progress and follow-up

- `status.json`: current phase, run, epoch and batch.
- `run.log`: detailed progress and any traceback.
- `manifest.json`: configuration, versions, code hash and repository commit.
- `summary.json`: paired results as each complete pair becomes available.
- `pid.json`: local process identifier.
- `smoke.json`, `pipeline_check.json`, `original_reproduction.json`: checks.

An hourly follow-up is configured in this task as `paper-2-training-follow-up`.
It remains quiet during normal progress and reports completion, failures or
issues requiring action. It is instructed to verify the full experiment before
drawing conclusions and to pause itself after delivering the results.

The frozen scientific plan is `reports/paper2_long_training_plan.md`.
Do not replace manuscript numbers with incomplete pairs or early results.
