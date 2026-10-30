# Can SNR-dependent switching make Hankel more useful?

19 September 2026. Completed inference-only follow-up. No retraining and no
changes to existing models, the longer-training experiment, or the manuscript.

## Short answer

Yes, selecting between two separately trained models works in this simulation:
use the ordinary URformer-based model (U) below 5 dB and the Hankel-trained
model (H) at or above 5 dB. This exactly retains U's low-SNR performance and
H's high-SNR performance. However, simply disabling Hankel inside the H-trained
network is not equivalent and performed worse at low SNR.

The two-model switch has a real but small full-range average-error improvement
over U, while taking about twice as much model inference time and storing two
sets of weights. Its main attraction is better high-SNR accuracy, not a large
overall error reduction or a free computational improvement.

## What was tested

Three matched seed pairs, using the validation-selected checkpoints from the
52-epoch-budget conventional-loss experiment. These selected checkpoints are
early saved models, not the final epoch-52 weights. The frozen follow-up plan is
`reports/paper2_snr_switch_plan.md`.

The threshold was selected using only the existing 2,000-case validation set.
We minimized mean normalized squared error over all validation SNRs, averaged
across three seeds. Candidate thresholds were -11, 0, 2.5, 5, 7.5, 10 and 21 dB;
the endpoints permit choosing always H or always U. Validation selected 5 dB,
which was saved before fresh test generation. Nearby thresholds had similar
validation scores; this is not evidence that precisely 5 dB is universally best.

We then generated 2,000 fresh cases using master seed 20260919 and trial IDs
3000000–3001999. They are separate from the previous training, validation and
test cases. The physical distribution, fixed pilots, initialization and other
settings were unchanged: N=32, K=3, P=20, 3–7 geometric paths, RSR=10 dB and
SNR spread over [-10,20] dB. The same fresh dataset is shared by all three seeds.
There are 1,011 low-SNR and 989 high-SNR cases.

The model selector routes each case before inference and executes only one
model for that case. It does not calculate both answers and select the one
closest to the true channel. The previous fixed 5 dB suggestion was retained
as a secondary comparison; it coincided with the validation-selected rule.

## Accuracy results

Gains are relative to U; positive means lower error. High/low columns are the
average across seeds of each seed's median paired gain. The full-range column
instead uses the ratio of sums of normalized squared errors, in dB, averaged
across seeds. These are deliberately different summaries, not interchangeable.

| Approach | Low-SNR paired median | High-SNR paired median | Full-range normalized-error gain |
|---|---:|---:|---:|
| Always U | 0.000 | 0.000 | 0.000 |
| Always H | -0.200 | +1.232 | -0.138 |
| Select U below 5 / H above | 0.000 | +1.232 | +0.050 |
| Same H weights, disable Hankel below 5 | -0.528 | +1.232 | -0.311 |

All entries are dB. The two-model router's full-range gain versus U was
0.0532, 0.0460 and 0.0513 dB for seeds 1, 2 and 3. Mean 0.0502 dB, seed SD
0.0037 dB, nominal 95% seed t interval [0.0409,0.0594] dB. Its full-range gain
versus always H was 0.1877 dB, interval [0.0492,0.3262] dB.

The router's high-SNR median gains were 1.261, 1.162 and 1.271 dB; mean
1.232 dB, interval [1.081,1.382] dB. Its high-SNR normalized-error-sum gain
was smaller, 0.741 dB. The full-range paired median was exactly zero because
more than half the cases use U unchanged. That zero does not contradict the
positive average-error improvement.

Per-bin router median gains: 0 dB in all three bins below 5 dB; +0.413 dB
at 5–10; +1.358 dB at 10–15; +2.215 dB at 15–20. For the ideal switch,
zero low-SNR difference is guaranteed by reusing U, not a newly discovered
learning effect. The fresh data test whether the high-SNR H advantage persists.

The small full-range improvement occurs because noisy cases dominate the
total error, and those cases are unchanged. This matters when deciding what
performance objective would justify the added computational cost.

## Why literal on/off switching is different

The H network's learned weights were trained with Hankel active inside each
layer. Turning that operation off at inference does not recover the separately
trained U network. On the fresh test set, this same-weight toggle produced a
low-SNR median penalty of 0.528 dB and a full-range normalized-error penalty
of 0.311 dB versus U. This result does not rule out training a new network
specifically for both paths, but that would be a separate experiment.

## Available SNR and robustness limits

The existing estimator already receives the simulator's correct noise variance.
With known normalization, SNR = 10 log10(3 / sigma2). We verified that mapping
and used it for routing, without using true channel values or estimation errors.
This is an ideal known-noise/known-calibration experiment, not a demonstrated
measurement-based SNR estimator or deployable receiver.

Without retuning, artificial routing errors produced these full-range gains
versus U:

| Routing information | Full-range normalized-error gain |
|---|---:|
| Correct SNR | +0.0502 dB |
| Gaussian error, SD 1 dB | +0.0503 dB |
| Gaussian error, SD 3 dB | +0.0468 dB |
| Constant bias -3 dB | +0.0428 dB |
| Constant bias +3 dB | +0.0440 dB |

These use one fixed Gaussian error realization, shared across seeds and scaled
for the two SDs. They are sensitivity checks, not independent replicated SNR
estimation experiments. With SD 3 dB the low-SNR normalized-error gain became
-0.0009 dB; with +3 dB bias it became -0.0066 dB. Thus perfect low-SNR protection
does not persist automatically when the routing information is imperfect.
The estimator itself still received the correct noise variance: combined
errors in routing, model inputs and calibration remain untested.

## Computation and storage

Warmed CPU, one thread, batch size 32, first 256 fresh cases, three repeated
passes per seed with rotated method order. Values below are the average of
the three seeds' median milliseconds per case. They are batched throughput
measurements, not isolated single-case latency or a hardware benchmark.

| Approach | Model inference time per case |
|---|---:|
| U | 0.541 ms |
| H | 1.530 ms |
| Two-model router | 1.128 ms |
| Same-weight toggle | 1.121 ms |

The router was about 2.09 times as slow as U and 26% faster than always H in
this measurement. Timing includes routing/tensor splitting but excludes common
spectral initialization, data generation, loading, and practical SNR estimation.
It does not establish end-to-end latency. About 49.45% of fresh cases used H.

Each model has 1,586,900 learned parameters, occupying 6,347,600 bytes at the
tested precision. Storing both doubles learned-weight storage to 12,695,200
bytes (about 12.1 MiB), excluding buffers and runtime overhead. Both models
also had to be trained; this is not a single-model training-cost saving.

## Verification and interpretation

Four independent logic tests passed. All six selected checkpoint hashes and
the frozen source/plan hashes match. Direct routed inference exactly reproduced
the branch-combined squared errors (maximum absolute difference 0.0) for all
three seeds and both switching approaches. An independent arithmetic audit
recomputed validation threshold selection, branch identities, every saved
per-bin/aggregate score and every seed interval. See `audit.json`.

The three-seed intervals quantify training-randomness variation on one fresh
dataset, with only two degrees of freedom and a normality assumption. They
do not establish population-wide confidence, hardware performance, or robustness
to different channels. Multiple comparisons are not multiplicity-adjusted.

Conclusion: selecting between two conventional-trained models is a promising
way to retain the observed high-SNR benefit while avoiding the observed low-SNR
penalty under known SNR. It is not evidence of a cost-free universal improvement,
a successful same-network on/off switch, or a novel switching principle. It
also does not establish an advantage over focused/balanced alternatives.
Before deployment claims, test realistic noise/SNR estimation and measure
end-to-end cost. Before broader scientific claims, evaluate another setting
and compare against appropriate training alternatives. No further training or
manuscript edits were performed by this follow-up.
