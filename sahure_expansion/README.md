# Mixed-condition follow-up to the completed SAHURE pilot

This module preserves `sahure/` and `results/sahure_pilot/`. It imports the
audited estimator and physical simulator without changing their equations.
Outputs go to `results/sahure_expansion/`.

## Scope

This campaign implements the next validation stage: six main learned arms,
three training seeds, mixed-condition training, frozen one-axis tests, and
separate serial inference timing. It does not train the later supervised,
final-only, stage-count, fixed-gate, fixed-step, amplitude-loss or fixed-mask
ablations. The report explicitly limits claims accordingly.

- Training: 16,384 worlds; validation: 2,048 independent worlds.
- Shared datasets across seeds 9101/9102/9103; independent initialization,
  random pilot masks and batch order. All six arms within a seed share masks.
- 20 epochs, Adam 0.001, batch 32, 10 stages, float32/complex64.
- Training mixture: half random per-user 3–7 paths, half fixed per-user
  counts 2/4/8/12/16. SNR -10/-5/0/5/10/15/20 dB; RSR 5/12/20 dB.
- Validation selection: equal means over 126 SNR × RSR × path-regime
  strata. These weights differ intentionally from the training mixture.
- Classical hard-rank selection uses the same equal-stratum validation
  rule with 25 Rician updates and one sweep; final estimation uses 100.
  For new array sizes, the selected rank is clipped to the new capacity.
- Test: 400 paired worlds per condition. Duplicate reference points share
  worlds. The two noise-offset points reuse reference data and only change
  the variance supplied to every estimator.
- All learned arms and constant-output controls use each of three seeds.
  Classical baselines are shared across seeds, not counted as fresh trials.
- Evaluation is float64/complex128 for all methods. Parallel accuracy
  workers do not supply latency measurements. Serial batch-one timing uses
  100 reference-condition worlds and three repeats after workers exit.
- Classical EM, Rician-gradient and amplitude-gradient methods include
  fixed ten-update references to match the unfolded depth, plus their
  stronger 100-update references. Rician/amplitude convergence-controlled
  runs use a 1,000-update cap. These controls were declared before testing.

The mean gain averages per-seed, per-SNR dB gains. It is not an ensemble
prediction. Paired bootstrap resampling preserves the entire test world and
all its method/seed predictions. Its intervals are conditional on the three
trained models. Seed variation is reported separately.

## Reproduce

Run from the repository root with the requirements in `sahure/requirements.txt`.
The local interpreter used here is `../../.venv-paper2/bin/python`.

```sh
python -m unittest sahure.test_numerics sahure.test_pipeline sahure_expansion.test_expansion
python -c 'from sahure_expansion.protocol import initialize; from sahure_expansion.data import prepare; initialize(); prepare()'
python -m sahure_expansion.train --seed 9101 --epochs 1
python -m sahure_expansion.campaign --workers 3
```

The one-epoch preflight is retained; campaign training resumes from it.
Completed epochs are checkpointed with optimizer state. A restart repeats
an interrupted epoch from the last complete checkpoint. Do not run two
campaign controllers simultaneously. `--phase train` or `--phase evaluate`
can resume a specific phase. Estimator/checkpoint hash changes after freeze
cause an error rather than silently mixing experiments.

One recorded finite-input NumPy SVD nonconvergence occurred in HS-GS at
`cell_24`, trial 211 (N=64). The original failure was reproduced. The
explicit recovery below retains the same world and equations, retries
NumPy first, and uses SciPy's `gesvd` only for a failing matrix. It checks
reconstruction and orthogonality, records the recovery and appends that
one missing result. The original estimator files remain unchanged.

```sh
python -m sahure_expansion.recover_svd --cell cell_24 --trial 211
python -m sahure_expansion.campaign --workers 3 --phase evaluate
```

Use the recovery command only after that recorded failure on a fresh
reproduction. To verify the already recovered result, add `--verify`.
The final audit performs this verification automatically.

The training module never reads the truth files. Condition IDs are used
only for the predefined validation weights, never as estimator features.
The generator records truths separately for evaluation and diagnostics.
The frozen selection is written before test generation.

## Artifacts

`REPORT.md`, `summary.json`, `figures/*.pdf`, `figures/*.png`, saved models,
optimizer checkpoints, validation histories, source hashes, observation
datasets, per-world errors/predictions, inference timings and `audit.json`
are under the result directory. Alternative channel details and inference
timing scope appear in the report. The prior pilot ZIP remains unchanged.
