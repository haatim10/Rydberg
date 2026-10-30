# Small baseline-learning checks: results

## Main conclusion

The baseline learns, but fixed unit normalization is not a useful fix in these
checks. The more important finding is an SNR-dependent weakness: the learned
estimator is better at low SNR, while classical EM-GS is much better at high SNR.
This is evidence for investigating the learned update and training objective,
not evidence that the published baseline has been reproduced.

All four planned runs finished, with 224 total epochs and 3,872 optimizer
updates. Training, data preparation and summary evaluation took about 8.8
minutes locally on one CPU thread, excluding the subsequent audit. No final
test set, Hankel arm, manuscript or dashboard was used or changed.

## What was compared

Both arms use the same independent URformer reconstruction, initial learned
weights, training examples, batch order, optimizer and within-study schedule.
The ordinary arm uses the existing physical units. The normalized arm divides
channel, reference and magnitude amplitudes by sqrt(32), divides noise variance
by 32, and converts the predicted channel back to physical units. Pilots are
unchanged. This uses a fixed known ensemble scale, not each example's true
channel energy. It preserves the statistical measurement problem and NMSE;
it can change neural optimization.

The channel has 32 antennas, four users, 20 pilots and 40 independent rays per
user, with the existing fixed Gaussian pilot and known reference. All examples
are fresh development data. This is a one-seed diagnostic, not a large benchmark.
The settings and decision criteria were written in
reports/paper2_baseline_learning_plan.md before the runs.

All table entries are mean per-example NMSE converted to dB. Lower is better.

## 1. Can the network fit nearly noiseless data?

At 60 dB SNR, each arm trained on 128 examples for 100 epochs (400 updates),
with 256 separate validation examples.

| Measure | Ordinary units | Normalized units |
|---|---:|---:|
| Initial training NMSE | -0.078 | -0.068 |
| Final training NMSE | -14.628 | -13.644 |
| Training improvement | 14.550 | 13.576 |
| Final validation NMSE | -5.891 | -4.298 |
| Validation-selected checkpoint NMSE | -5.893 | -4.822 |
| Selected epoch | 50 | 30 |

Both networks improve substantially on their training examples, so learning is
not completely blocked. Neither meets the full pre-set sanity-check target:
at least 10 dB training improvement AND final training NMSE at most -20 dB.
The improvement requirement passes; the absolute fitting requirement fails.
This limited 400-update check is not proof of a bug, inability to fit with more
optimization, or convergence. It also shows a substantial generalization gap.

Normalization worsens final held-out error by 1.593 dB, and the
validation-selected comparison by 1.071 dB.

Classical EM-GS reaches -19.511 dB with 10 iterations and -61.850 dB with
100 iterations on these same validation examples. These methods have different
computation costs, so this is not a speed comparison. Nevertheless, the
near-noiseless data are recoverable much more accurately than these small-data
neural runs recover them. Missing information alone does not explain their
large errors in this particular check.

## 2. Does normalization improve mixed-SNR learning?

Each arm trained on 4,096 examples for 12 epochs (1,536 updates). The 512
validation examples have SNR uniformly sampled from -10 to 20 dB.

| Final validation range | Cases | Ordinary | Normalized | EM-GS 10 | EM-GS 100 |
|---|---:|---:|---:|---:|---:|
| All | 512 | -4.955 | -4.948 | -1.298 | -0.306 |
| [-10,-5) dB | 79 | -0.727 | -0.768 | 4.973 | 5.959 |
| [-5,0) dB | 79 | -2.620 | -2.722 | 0.472 | 1.653 |
| [0,5) dB | 99 | -5.690 | -5.713 | -5.416 | -4.512 |
| [5,10) dB | 83 | -8.108 | -8.061 | -9.486 | -8.986 |
| [10,15) dB | 88 | -10.411 | -9.949 | -13.627 | -13.963 |
| [15,20) dB | 84 | -11.458 | -10.780 | -16.851 | -19.326 |

The overall normalization gain is -0.007 dB: essentially no difference in this
run, not a formal equivalence claim. The pre-set +0.5 dB screening criterion
is not met. Normalization slightly helps several lower-SNR bins but worsens
the highest two bins by 0.461 and 0.678 dB. Those are descriptive single-seed
results, not established population effects.

The secondary validation-selection rule chooses epoch 4 for both arms.
Their full-range scores are -4.885 and -4.827 dB, respectively; normalization
is 0.059 dB worse. Thus changing between the two pre-specified reporting rules
does not rescue the proposed normalization fix.

Final training NMSE is -6.470 dB for ordinary units and -6.038 dB for normalized
units. The different training set, SNR distribution and schedule mean these
numbers must not be interpreted as a direct extension of the nearly noiseless
study or of the earlier 50-epoch reconstruction.

## What the SNR split means

The learned model is not uniformly poor: it beats these two fixed-iteration
classical methods on the low-SNR bins and on the full-range average. However,
at 15-20 dB, ordinary units score -11.458 dB versus -19.326 dB for EM-GS 100,
a 7.868 dB disadvantage. Even EM-GS 10 is substantially better there.

Mean-linear NMSE can put more weight on high-error, low-SNR cases. A good
full-range score therefore need not imply good high-SNR reconstruction. This
is a plausible training trade-off, not a causal explanation established by
these four runs. Longer classical iteration is also not universally better:
EM-GS 100 has worse low-SNR errors than EM-GS 10 here.

## Verification

- All five diagnostic unit/integration tests pass: physical and guarded-kappa
  scaling, loss/gradient equivalence, identical initialization and disjoint
  IDs, exact agreement of the ordinary arm with the unchanged reconstruction,
  and exact interrupted-versus-uninterrupted training resumption.
- All four results contain the expected epoch count and checkpoint files.
  All recorded validation errors and gradient norms are finite.
- Initial learned weights and every epoch's batch-order hash match across arms.
- All four cached-data hashes and 51 current source/plan hashes verify.
- Selected and final checkpoints replay their complete validation-error arrays
  exactly on this environment. Final training errors replay exactly too.
- Checkpoint selection and all reported paired per-bin gains recompute from
  the saved per-example errors. Both classical references replay exactly.
- The 150 source-manifest entries across the three earlier completed experiments
  remain unchanged. See audit.json for machine-readable checks.

A command-line import-path issue was corrected before any real experiment
started or manifest was written; no numerical intervention or setting change
was made during training.

## Recommended next experiment, not launched

Keep ordinary units as the reference. Do not spend a large training budget on
this normalization alone. A more targeted next diagnostic would start a
separate learned model from a working classical EM-GS update, initially making
the neural correction zero, and test whether training preserves its high-SNR
behavior while improving its low-SNR behavior. Compare against the unchanged
reconstruction and validation-tuned classical stopping, with matched data and
a fixed budget. An SNR-balanced objective should be a separate ablation, not
changed at the same time if we want to identify the cause.

This would be an independently specified model, not a reconstruction of the
published architecture. It would need multiple training seeds and an untouched
final test after development before a performance claim. Previously tested
SNR-routing ideas should not be presented as new. Hankel comparisons should
follow a credible baseline, not use its high-SNR weakness as a source of gain.

These checks do not resolve the published-model discrepancy, establish novelty,
or establish a publishable Hankel advantage. They narrow the problem: the model
can learn, the proposed fixed rescaling does not fix it, and preserving accurate
high-SNR inversion while learning low-SNR denoising is a concrete next target.
