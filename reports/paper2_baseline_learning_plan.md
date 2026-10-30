# Baseline learning and unit-normalization diagnostic

Authorized by the user's “go ahead”. Written before observing these runs.
This is an independent implementation study, not a published reproduction.
No Hankel arm, manuscript edit, author contact, or final-test evaluation.
The completed experiments and cancelled calibration project remain untouched.

## Questions and fixed comparison

1. Can the ordinary reconstruction learn a small nearly noiseless training set?
2. Does expressing the same physical problem in fixed normalized units improve
   training and independent development-validation performance?

Two arms: `physical` (unchanged reconstruction) and `normalized` (divide channel,
initial guess, reference and magnitude observations by sqrt(32); divide noise
variance by 32; leave pilots unchanged; multiply the estimated channel back by
sqrt(32)). The factor is known ensemble channel RMS, never realized true energy.
Kappa, physical SNR/RSR and target physical NMSE are preserved. Adjust the layer's
epsilon by the same amplitude factor. Neural features and Adam optimization
need not be scale invariant; that is precisely the intervention being tested.
Raw-kappa filtering, model architecture, initial learned parameter values,
optimizer, data and batch order are matched. No clipping or silent stabilization.

Retain N=32, K=4, P=20, independent 40-ray angles, independent CN(0,1) gains,
fixed Gaussian pilots, known reference, ensemble multiuser RSR=10 dB. Do not
select channel assumptions or settings for agreement with a published curve.

## Small studies

- `memorization`: 128 train, 256 validation, SNR=60 dB; 100 epochs, 400 optimizer
  steps per arm. Fresh IDs 20,000,000+ and 21,000,000+. This is a learning sanity
  check, not a realistic performance claim. Fit training examples and separately
  measure held-out behavior. Record initial, final and per-epoch NMSE. A useful
  diagnostic target is at least a 10 dB training improvement and final training
  NMSE at most −20 dB. Missing this target is NOT by itself proof of a code bug.
- `development`: 4,096 train, 512 validation, SNR sampled uniformly from −10 to
  20 dB; 12 epochs, 1,536 steps per arm. IDs 22,000,000+ and 23,000,000+. Compare
  mean-linear NMSE overall and in all six 5 dB bins. All cases are development
  data, not a final confirmation test. A ≥0.5 dB final validation gain is a
  screening signal only, not a publication threshold or proof of improvement.

Single model seed 2026092401 in both arms. Batch 32, Adam lr=0.001, no weight
decay, cosine decay to zero over each study's own fixed budget. Compare arms
within a study only; the two studies differ in sample count, SNR and schedule.
Initial learned weights and per-epoch order hashes must match exactly.

Primary comparison: final epoch. Also report the existing one-standard-error
validation-selected checkpoint, chosen without reference to the paper's curve.
Archive per-epoch validation errors and every epoch checkpoint. A new final
test must remain untouched until an eventual larger experiment is frozen.

Evaluate 10- and 100-step classical EM-GS on the same development validation
sets, as diagnostic references with different computational costs. A broader
Paper-1/classical benchmark and SNR-balancing experiments are not part of this
initial check and will depend on its findings.

## Safeguards and interpretation

One CPU thread. New output directory results/paper2_baseline_learning only.
Exclusive run lock, source/data provenance, epoch-boundary resumption with
optimizer/scheduler/RNG, progress status and log. Refuse changed-source resumes;
stop on non-finite loss/gradients. Do not launch duplicate workers.

Unit tests: normalization preserves physical classical outputs and kappa; loss
and gradients agree when expressed in either unit system; matched initial
weights/order; new disjoint IDs; tiny run and exact epoch-boundary resumption.
Observed speed determines ETA, not a change to the fixed budgets.

Do not claim that this resolves the published-model discrepancy, establishes
convergence, or supports Hankel. Report negative findings and seed/sample limits.
