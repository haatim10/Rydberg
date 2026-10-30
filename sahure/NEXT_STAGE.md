# Conditional next campaign — not run by the pilot command

First read `results/sahure_pilot/REPORT.md`. If neither core gives a useful
accuracy/runtime tradeoff, report the negative result before increasing
model size or simulation budget. A pilot pass concerns only SNR 5 dB and
RSR 12 dB; it is not the full seven-SNR success criterion.

## Mixed training campaign

Before launch, time one epoch and estimate the full cost. Keep N=32, K=3,
P=30. Use 16,384 training and 2,048 validation worlds, three independent
training seeds, balanced SNR {-10,-5,0,5,10,15,20} and RSR {5,12,20} dB.
Half use random per-user 3–7 paths, half fixed per-user L drawn from
{2,4,8,12,16}. Define condition counts before generating data. Select models
using equally weighted validation-condition means of held-out Rician NLL,
not pooled sample counts or channel NMSE. Keep the same six-arm controls.

This mixed sampler and weighted selector are not implemented by the fixed
pilot runner. They need an explicitly versioned campaign configuration;
do not modify the pilot's data files or reuse its test set for tuning.

## Frozen evaluation axes

Each point uses 400 fresh paired worlds; no Cartesian cross-product:

| Axis | Values | Fixed settings / interpretation |
|---|---|---|
| SNR | -10,-5,0,5,10,15,20 dB | RSR12, random 3–7 paths |
| RSR | 0,5,10,12,15,20,25 dB | SNR5; record single-user and aggregate ratios |
| Paths | 2,3,4,6,8,12,16 | SNR5; test complexity beyond training support explicitly |
| Pilots | 10,20,30 | Other dimensions fixed |
| Array | 8,16,64 | No retraining; label transfer |
| Users | 2,4 | P=30 fixed initially; label transfer |
| Channel distribution | clustered, ten-path, declared unequal path powers | Preserve calibration and label distribution shift |
| Assumed noise | true variance times 10^(±3/10) | Change estimator input only, not generated data |

## Remaining ablations

Select the primary core by validation NLL and keep both reported. Train
separate models for Rician versus amplitude-MSE processing/loss, gate fixed
at one, fixed step scalars, T=5/10/15, fixed versus random masks, matched
supervised loss, and a separately trained final-only structural model.
`ModelConfig` already exposes fixed steps, fixed gate and final-only
placement, but the six-arm pilot does not train these ablations. Supervised
training must be a separate explicit label-using path, not an option that
silently feeds truth to the self-supervised loader.

## Acceptance criteria and reporting

- Combined target: at least 0.3 dB equally weighted mean per-SNR ratio-of-sums
  gain over HS-GS, paired interval above zero, and at least 2× median inference
  speedup.
- Efficiency target: within 0.2 dB of HS-GS with at least 2× speedup; describe
  it as efficiency evidence rather than improved estimation accuracy.
- Adaptation claim: held-out-complexity benefit over global shrinkage, plus
  the validation-constant intervention. If absent, simplify the claim.
- Self-supervision: report the measured gap to the matched supervised
  architecture; do not call that architecture an upper bound.

Use 4,000 paired trial resamples and show training-seed variation separately.
Plot all individual conditions; never hide harmed regions in a pooled mean.
The five final figure groups should cover SNR/RSR, complexity, runtime,
supervision/ablations, and transfer/structural diagnostics. This requires new
experiments; the pilot figures alone cannot fill that scope.
