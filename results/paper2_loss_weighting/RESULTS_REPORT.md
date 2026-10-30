# High-SNR loss weighting: completed diagnostic

## Finding

Giving high-SNR examples more weight helps their accuracy, but this particular
change does not meet either pre-set success criterion. It is a partial recovery,
not a resolved baseline or a validated new contribution.

Two 12-epoch runs completed in about three minutes, including data preparation
and checkpoint replay. The control exactly reproduces the earlier classical-start
run at every epoch. No previous experiment, manuscript or dashboard was changed.
No additional weighting search or experiment was launched.

## What changed, and what did not

The architecture remains ten exact EM-GS layers with trainable Transformer
corrections initially equal to zero. The two runs have identical initial weights,
examples, batch orders, optimizer and schedule. Both use physical units.

Only the training objective changes: SNR >=15 dB examples receive ten times the
weight of other examples. We divide by the mean raw weight on the full training
set, 2.46337890625, to make the mean weight one. There are 666 high-SNR cases among
4,096 training examples. This weighting is fixed before training, not selected
from a search. Training SNR is simulator metadata; inference gets no new input.

All reported NMSE values below are UNWEIGHTED mean per-example errors, converted
to dB. Lower is better. The training-log NMSE is also unweighted; gradients use
the weighted objective. The checkpoint rule remains the same unweighted
full-range validation rule.

Data and seed are deliberately reused from the classical-start diagnostic:
4,096 training cases, 512 development-validation cases, seed2026092501,12epochs,
batch32,Adam.001 and cosine decay. This is not a new independent test. Classical
stopping is selected on the same separate256 development cases as before.

## Primary result: final epoch

| SNR range | Cases | Ordinary reconstruction | Classical-start, standard loss | Classical-start, weighted loss |
|---|---:|---:|---:|---:|
| All | 512 | -4.594 | -4.623 | -4.688 |
| [-10,-5) dB | 96 | -0.625 | -0.586 | -0.659 |
| [-5,0) dB | 90 | -2.707 | -2.619 | -2.684 |
| [0,5) dB | 93 | -5.293 | -5.191 | -5.150 |
| [5,10) dB | 69 | -8.872 | -9.471 | -9.462 |
| [10,15) dB | 89 | -11.572 | -12.500 | -12.766 |
| [15,20) dB | 75 | -13.024 | -14.748 | -15.610 |

The weighting recovers 0.862 dB in the 15-20 dB bin relative to standard-loss
classical-start training. The weighted model is 2.586 dB better than the ordinary
reconstruction in that bin. Overall, however, gains are only 0.065 dB versus
standard-loss classical-start and 0.093 dB versus the ordinary reconstruction.
Those small overall differences are descriptive, not evidence of significance.

Before training, the classical-start model exactly equals EM-GS10. Its high-SNR
NMSE is -16.920 dB, so the weighted trained model still loses 1.310 dB. EM-GS100
reaches -19.504 dB in the same bin; the weighted model remains 3.894 dB worse.

## Pre-set decision criteria

| Criterion | Required | Observed final epoch | Decision |
|---|---:|---:|---|
| Overall gain versus ordinary reconstruction | >=0.5 dB | 0.093 dB | Fail |
| High-SNR deterioration versus initial EM-GS10 | <=0.5 dB | 1.310 dB | Fail |

Neither threshold was met. These thresholds are development screening choices,
not statistical tests or publication standards. We did not adjust them afterward.

## Secondary checkpoint selection

The one-standard-error rule selects epoch3 for both new arms and for the previous
ordinary model. Full-range scores are -4.799 dB for ordinary, -4.625 dB for
standard-loss classical-start, and -4.588 dB for weighted classical-start.
Thus weighting is worse overall under this secondary rule: by0.037 dB versus
the standard-loss model and by0.211 dB versus ordinary.

Its selected high-SNR score is -16.046 dB, still 0.874 dB worse than initial
EM-GS10. Both success conditions also fail under this reporting rule. We did
not select a special high-SNR checkpoint after inspecting the results.

## Classical comparison and interpretation

The separate stopping-tuning set chooses one global EM-GS iteration. On the512
validation cases this scores -2.916 dB overall but only -4.666 dB at15-20dB.
EM-GS10 and100 score -0.642 and+0.324 dB overall, respectively, while being much
better at high SNR. Their poor low-SNR cases dominate their overall means.
The learned models still have a low-SNR/full-range advantage over these classical
references; they do not have a uniform advantage across SNR. Different iteration
counts have different processing costs; no speed superiority is claimed.

The controlled comparison supports a narrow statement: the training loss affects
the high-SNR trade-off in this implementation. It does not establish that the
loss is the only cause, that weighting alone can fix it, or that no other weight
could work. One weight and one seed were examined. Existing development results
motivated the intervention, so confirmation requires independent data later.

## Verification

- Three new test methods passed: objective/boundary/gradient formula and
  factor-one equivalence; unchanged architecture, initialization and control
  update; exact weighted-run checkpoint resumption.
- Both arms completed all12epochs with finite errors and gradient norms.
- Initial full-model hashes and all12batch-order hashes match between arms.
- The complete12x512 control error array exactly matches the previous run.
- Selected/final checkpoint replay is exact for both new arms.
- Saved checkpoint errors, checkpoint selection, all SNR-bin scores and success
  calculations were recomputed by the separate audit.
- Current source/data hashes and the prior comparison-artifact hashes verify.

See summary.json for scores, audit.json for verification, and
reports/paper2_loss_weighting_plan.md for the pre-run plan.

## Decision

Do not promote this weighting setting as the solution or launch a large benchmark
on its strength. Preserve it as a negative development result with a useful
partial effect. Further work needs a deliberate choice of performance objective
or a mechanism to preserve the classical solution, not an open-ended search for
a favorable curve. Any such follow-up is a separate decision; none was started.
