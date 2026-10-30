# Paper 2: longer-training results

19 September 2026. All 18 planned runs finished. This report concerns the
conventional-loss experiment only. The manuscript has not been changed.

## Main finding

Giving both models a larger training budget did not remove the Hankel model's
high-SNR advantage in this experiment. However, the size and direction of the
change in that advantage depended on which saved model was evaluated.

When validation data selected an early saved model, the average high-SNR gain
was 1.14 dB at the 13-epoch budget and 1.26 dB at the 52-epoch budget. When the
last model was evaluated instead, the gain fell from 1.10 to 0.72 dB. All three
seeds showed the latter decrease, and all still had a positive high-SNR gain.
These are relative comparisons: a smaller gap does not establish that the
ordinary model improved in absolute terms.

Both models fit their training examples better as training continued, while
their validation performance eventually became worse. This is consistent with
overfitting. It is not evidence that 52 epochs established convergence.

## What was compared

The frozen local plan specified three training seeds, two models (U: without
Hankel; H: with a fixed rank-7 Hankel step), and three separate training budgets
(13, 26 and 52 epochs). This gives 18 fresh runs and 546 completed epochs.
Each epoch uses 80,000 training examples, for 43,680,000 sample presentations.
The validation and test sets each contain 2,000 examples. The same datasets
are shared across the runs; seeds vary training randomness, not the test set.

The setup uses the simulated geometric channel with N=32, K=3, P=20,
3–7 paths, RSR=10 dB, spectral initialization, fixed pilots, and training
SNR from -10 to 20 dB. Models use CPU float32, Adam, batch size 32 and an
initial learning rate of 0.001. U/H pairs have matched starting learned
parameters (checked before execution) and matched per-epoch sample order.

Important: the cosine learning-rate schedule reaches zero at the respective
budget. Thus both training duration and schedule duration changed. These are
not extensions of a single 13-epoch checkpoint, nor a duration-only experiment.
The plan was written locally before these runs, not externally registered.

## Main scores and uncertainty

For each test case, gain = 10 log10(squared error U / squared error H).
Positive means H has lower error. The primary result takes the median gain
over the 983 test cases with SNR >=5 dB, then averages those medians over the
three training seeds. It is not the gain in mean error and not a pooled median.

| Budget | Saved-model rule | Seed 1 | Seed 2 | Seed 3 | Mean gain | Seed SD | 95% seed t interval |
|---|---|---:|---:|---:|---:|---:|---|
| 13 | Validation-selected | 1.161 | 1.128 | 1.117 | 1.136 | 0.023 | [1.078, 1.193] |
| 26 | Validation-selected | 1.216 | 1.417 | 1.108 | 1.247 | 0.157 | [0.858, 1.636] |
| 52 | Validation-selected | 1.282 | 1.218 | 1.269 | 1.256 | 0.034 | [1.172, 1.340] |
| 13 | Final epoch | 1.138 | 1.124 | 1.050 | 1.104 | 0.047 | [0.987, 1.221] |
| 26 | Final epoch | 0.839 | 0.917 | 0.787 | 0.848 | 0.065 | [0.686, 1.010] |
| 52 | Final epoch | 0.757 | 0.707 | 0.687 | 0.717 | 0.036 | [0.627, 0.807] |

All entries are dB. Validation selection uses the earliest epoch within one
standard error of the best validation score. It never uses test scores.

Changes are paired within seed against the fresh 13-epoch runs:

| Comparison | Rule | Mean change in gain (dB) | 95% paired seed t interval |
|---|---|---:|---|
| 26 minus 13 | Validation-selected | +0.111 | [-0.278, +0.501] |
| 52 minus 13 | Validation-selected | +0.121 | [+0.043, +0.198] |
| 26 minus 13 | Final epoch | -0.256 | [-0.372, -0.141] |
| 52 minus 13 | Final epoch | -0.387 | [-0.454, -0.319] |

The completed third seed changes the earlier provisional interpretation:
the selected-model increase at 52 epochs is positive in all three seeds, and
its nominal paired interval excludes zero. This remains a small observed
increase, not evidence of a universal improvement. The final-model reduction
also repeats across all seeds.

These intervals have only two degrees of freedom and assume approximately
normal seed effects. They quantify training-seed variation on this fixed
dataset, not uncertainty over receiver designs or channel populations.
Intervals are pointwise and not corrected for the multiple comparisons.
Three seeds cannot establish equivalence or reliable tail behaviour; an
interval spanning zero does not prove no effect.

## Where the advantage occurs

The following are averages of per-seed median gains, in dB. S means
validation-selected; F means final epoch. Bin counts are identical for all
runs and sum to 2,000.

| SNR bin | Cases | 13 S | 26 S | 52 S | 13 F | 26 F | 52 F |
|---|---:|---:|---:|---:|---:|---:|---:|
| [-10,-5) | 346 | -0.169 | -0.130 | -0.138 | -0.139 | -0.021 | +0.040 |
| [-5,0) | 354 | -0.499 | -0.357 | -0.327 | -0.426 | -0.250 | -0.254 |
| [0,5) | 317 | -0.401 | -0.145 | -0.146 | -0.334 | -0.241 | -0.205 |
| [5,10) | 331 | +0.411 | +0.566 | +0.544 | +0.301 | +0.182 | +0.108 |
| [10,15) | 336 | +1.256 | +1.385 | +1.337 | +1.156 | +0.945 | +0.835 |
| [15,20) | 316 | +1.948 | +2.044 | +2.115 | +2.041 | +1.582 | +1.462 |

The high-SNR benefit is not a universal advantage. Selected models have
negative mean gains in every bin below 5 dB. At the final epoch the lowest
bin can change sign, but the two adjacent noisier bins remain negative on
average. The per-seed evidence, rather than these averages alone, is retained
in summary.json and the per-trial test files.

## Why the error summary matters

Below, every entry is a mean across three seeds of that seed's dB score.
Error-sum gain is 10 log10(sum raw squared errors U / sum raw squared errors H).
Normalized-error-sum gain first divides each case's squared error by its true
channel energy. These two measures are distinct from the paired median.

| Budget/rule | High-SNR error sums | High-SNR normalized-error sums | Full-range paired median | Full-range error sums | Full-range normalized-error sums |
|---|---:|---:|---:|---:|---:|
| 13 selected | +0.738 | +0.672 | +0.134 | -0.217 | -0.220 |
| 26 selected | +0.876 | +0.813 | +0.271 | -0.125 | -0.132 |
| 52 selected | +0.915 | +0.850 | +0.290 | -0.121 | -0.128 |
| 13 final | +0.716 | +0.649 | +0.180 | -0.179 | -0.182 |
| 26 final | +0.548 | +0.492 | +0.224 | -0.062 | -0.065 |
| 52 final | +0.378 | +0.352 | +0.196 | -0.038 | -0.036 |

Thus the positive full-range median must not be reported as a reduction in
full-range total squared error: the latter has the opposite sign on average.

## Training and validation behaviour

The 52-epoch runs selected early models:

| Seed/model | Selected epoch | Best validation epoch | Best validation NMSE (dB) | Final validation NMSE (dB) |
|---|---:|---:|---:|---:|
| 1 U | 7 | 11 | -6.769 | -6.222 |
| 1 H | 8 | 11 | -6.613 | -6.075 |
| 2 U | 8 | 11 | -6.772 | -6.150 |
| 2 H | 9 | 13 | -6.636 | -6.109 |
| 3 U | 8 | 11 | -6.796 | -6.237 |
| 3 H | 8 | 11 | -6.587 | -6.072 |

Lower validation NMSE is better. In all six runs, final validation performance
is about 0.52–0.62 dB worse than its best value, despite lower training loss.
Over epochs 48–52, validation NMSE still worsens by roughly 0.01–0.02 dB.
The flattening near the end also occurs while the learning rate is being
forced to zero. Neither completion nor flattening establishes convergence.

## Verification and provenance

The accompanying completion_audit.json records the post-run checks. The
independent audit script does not modify the frozen training driver or scores.
It verifies all 18 run records and configurations, all 546 epoch-file paths,
saved validation arrays and validation-only selection, and the final weights
against each final training checkpoint. It reloads and re-evaluates all 36
selected/final model choices on the cached test set and compares every saved
per-trial array (tolerance: relative 1e-6, absolute 1e-7).
All checks passed. The maximum observed absolute difference in replayed
squared-error numerators was exactly zero across the 36 evaluations.

It also checks all nine matched U/H data-order histories, paired test IDs,
SNRs and denominators, finite positive errors, and independently recalculates
all eight evaluation groups' three metrics for every pair and selection rule,
including all six SNR bins. Seed means, SDs, t intervals and paired changes
are checked against summary.json. Dataset file hashes and representative
samples from the original generator are rechecked; the frozen driver hash
matches manifest.json. The replay uses the same model implementation, so
this is a reproducibility/consistency audit, not an independent validation of
the physical receiver model or of every underlying model equation.

Before execution, the recorded pipeline test passed and replay of the six
original manuscript checkpoints matched their original scores to within
0.0000102 dB. Fresh training need not reproduce the old training trajectory:
these new 13-epoch runs are the correct within-experiment controls.

Evidence files are in this report's directory: summary.json,
completion_audit.json, manifest.json, original_reproduction.json,
pipeline_check.json, and each seed/model/budget folder's result.json,
history.json, checkpoint.pt, epoch files and test_selected/final.npz files.

## What the results do and do not establish

The results weaken the narrow explanation that the observed high-SNR gain
would disappear merely by giving conventional training a larger budget under
the tested schedules. They show sensitivity to the evaluation checkpoint and
signal region. They do not show that a structural rule is always helpful,
that the ordinary model is fully optimized, or that the cause of the original
focused/balanced contrast has been isolated.

Only conventional-loss training was extended. Focused and balanced designs
were not tested at 26 and 52 epochs here. Therefore this experiment cannot
establish that the contrast between training designs survives at matched
long budgets. Architecture, simulator, pilots and dataset were also fixed;
no hardware or second-architecture validation was performed.

For the manuscript, present this as a training-budget-and-schedule control,
report both selected and final models, and retain the per-SNR and metric
qualifications. Do not claim that Hankel's benefit disappears with longer
training or that these results prove a causal training-allocation mechanism.
Review the full evidence before deciding whether a further matched
focused/balanced comparison or a second setting is needed. No additional
training is launched by this report.
