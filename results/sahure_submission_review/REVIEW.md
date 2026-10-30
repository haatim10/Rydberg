# Paper 2: submission review

22 September 2026. This is a read-only review of completed experiments, not a new training study. The manuscript draft and this review do not alter the frozen protocols.

## Decision

Keep the original 188-parameter EM model as the primary method. Keep all three validation-selected training seeds; do not choose the best test seed. Present the fixed-open gate as an ablation, not a replacement that has passed the full transfer study.

The supported result is a small channel estimator trained without channel labels, using measurement-dependent soft Hankel shrinkage. It has a useful measured accuracy–processing-cost trade-off against the implemented HS-GS baseline. It is not evidence that self-supervision always beats supervision, that every channel benefits, or that we have reproduced and surpassed published URformer.

## What was checked

The new review recalculated matched comparisons from saved per-world errors, examined checkpoint histories, replayed six selected models on validation observations, and compared 21-pilot and 30-pilot inputs. All 149 frozen input/source hashes and all 42 checkpoint references matched their recorded hashes. The existing follow-up audit reports 702 result files; its 15 training jobs and 300 epochs are complete. The follow-up contains 3,600 fresh worlds in two banks.

Machine-readable calculations are in `review.json`. Reproduce from the repository root with `PYTHONPATH=. ../../.venv-paper2/bin/python scripts/sahure_submission_review.py`. This script writes only to the new review directory. It does not retrain models. The new gate intervals are exploratory, conditional on the saved weights, and are not additional preregistered tests.

## 1. The supervised comparison needs careful wording

Positive gain means lower error for the original self-supervised model. Each bank has seven primary SNR cells with 200 worlds per cell. A macro score gives equal weight to the seven cell scores in dB.

| Comparison or aggregation | Bank 1 gain (dB) | Bank 2 gain (dB) |
|---|---:|---:|
| Supervised checkpoint selected by its validation channel loss; macro score | 1.015 | 1.012 |
| Supervised checkpoint selected by held-out likelihood; macro score | 0.735 | 0.730 |
| Supervised final checkpoint; macro score | 1.105 | 1.104 |
| Channel-loss-selected supervised checkpoint; pooled ratio-of-sums NMSE | 0.361 | 0.355 |
| Channel-loss-selected supervised checkpoint; mean per-world NMSE | 0.338 | 0.333 |

Checkpoint choice and reporting weights change the size of the advantage. Neither reverses the aggregate comparison here. However, these comparisons do not isolate an inherent benefit of self-supervision.

The supervised training objective is mean per-world linear channel NMSE. In the selected supervised models, approximately 92.6–92.7% of the equal-stratum validation version of that objective comes from SNR at or below 0 dB. This is a decomposition of loss values, not a measurement of gradient contributions. Equal numbers of examples across conditions do not make this objective equivalent to an equal-SNR dB objective.

The supervised models have lower validation channel loss than the self-supervised models when both receive the training-style 21-pilot inputs. Their training curves improve; the results do not look like a complete failure to optimize. Some best checkpoints are at the final epoch, so optimization saturation has not been established.

| Training seed | Self-supervised, 21 pilots | Supervised, 21 pilots | Self-supervised, 30 pilots | Supervised, 30 pilots |
|---:|---:|---:|---:|---:|
| 9101 | 0.27430 | 0.25555 | 0.22263 | 0.23553 |
| 9102 | 0.27853 | 0.25517 | 0.22293 | 0.26158 |
| 9103 | 0.28226 | 0.25509 | 0.21779 | 0.21699 |

These are equal-stratum mean per-world validation NMSE values in linear units, not the primary test ratio-of-sums metric. The same validation worlds were used for each paired input-length diagnostic. The supervised model's transfer from 21 training inputs to all 30 deployment inputs is inconsistent across seeds. One seed gets worse even on its own channel metric. This observed behavior does not prove which feature or optimization choice caused it.

The original model's advantage is largest at high SNR. Supervision is slightly better at 5 dB and approximately 0.80–0.82 dB better on the two-path stress cases. We must report those counterexamples.

Safe wording: “The label-free model outperformed this matched supervised training recipe on the declared primary average.” Unsafe wording: “Labels hurt performance” or “self-supervision is superior to supervision.”

## 2. Do not promote the fixed-open gate after seeing test results

Fixing the gate at one improves the primary average by 0.064 and 0.066 dB. The average does not describe every seed or condition.

| Two-path stress test | Seed 9101 | Seed 9102 | Seed 9103 | Mean |
|---|---:|---:|---:|---:|
| Bank 1 fixed-gate gain (dB) | -0.040 | -0.131 | +0.603 | +0.144 |
| Bank 2 fixed-gate gain (dB) | -0.042 | -0.122 | +0.602 | +0.146 |

Two seeds worsen; the third drives the positive average. Narrow paired-world intervals do not remove this training-seed uncertainty. On weak-reference cases, the gate also worsens some seeds. Severe-error frequency in bank 1 changes from 0.0050 to 0.0067; that small count difference is descriptive, not proof of a population-level increase.

Even with a fixed-open gate, HS-GS remains better on the weak-reference and two-path stress tests. The tested gate-one implementation still stores 188 parameters and has no measured speed advantage. It is not a pruned network.

The broad study's nearly saturated gate and this ablation both weaken a claim about learned on/off decisions. They support focusing on learned shrinkage strength instead. They do not establish that removing the gate is uniformly harmless.

## 3. Repeated correction has a cost

Repeated structural correction improves the primary average over a separately trained final-only correction by 0.697 and 0.694 dB. In the new timing sample, median processing times are approximately 4.65 ms and 2.05 ms respectively: about 2.27 times the cost. This supports an accuracy–cost choice, not a claim that final-only processing is unusable.

The broader study's original timing is a separate measurement: 8.24 ms for the primary model and 525.62 ms for HS-GS, or 63.8 times faster, under its declared serial CPU conditions. Do not mix timings from the two studies to create a larger speedup. Neither measurement is a hardware latency guarantee or a comparison with an optimized GPU implementation.

## 4. Replication is positive but limited

On the two independent training-data replicates, the adaptive model improves the primary average over global learned shrinkage by 2.252 and 2.185 dB, over no structure by 4.250 and 4.191 dB, and over HS-GS by 1.939 and 1.837 dB. This is useful evidence that the original result is not restricted to one dataset.

These are two new datasets with one fixed training seed each, not the earlier proposed five-dataset/two-initialization confirmation. Conditional test-world intervals cannot be presented as uncertainty across a population of training datasets. The main study's three seeds share training worlds and are not three independent datasets.

## 5. Literature and publication status

The updated comparison is in `NOVELTY.md`. Self-supervised Rydberg sensing, self-supervised low-rank reconstruction, learned Hankel processing, and unfolded Rydberg estimators all have prior art. The plausible distinction is the particular label-free nonlinear channel-inversion method and its demonstrated accuracy–cost trade-off, not any of those ingredients in isolation.

The exact overlap with the recent XL-MIMO DeepTensor conference paper is unresolved because its full text was not obtained. No “first” or “never done before” claim is defensible from this search. The method does not implement the earlier proposed Fisher-direction filtering mechanism; that idea must not be attributed to these results.

## Remaining work before submission

1. Check the full XL-MIMO paper and trace the nearest structured-low-rank references. Resolve the distinction from the companion Paper 1 in the introduction.
2. Run a small, separately declared feature ablation: spectral features versus noise/residual features versus the full controller. The existing results do not identify which information provides the benefit. Do not retune against the already inspected test banks.
3. If a superiority-to-supervised claim is retained, strengthen that comparison using deployment-matched 30-pilot supervised training and a predeclared equal-condition objective, on fresh evaluation worlds. Otherwise narrow the wording and retain the current supervised recipe as an ablation.
4. Decide the publication scope. A broad robustness claim needs more independent dataset/pilot replicates and stronger regularized, noise-tuned classical controls. The current narrower simulation paper must explicitly show the weak-reference and simple-channel failures.
5. Complete figures, reproducibility packaging, bibliography and manuscript review. Write the qualitative abstract and conclusion after fixing the final claim. Do not launch a large matrix merely to obtain a positive result.

The current evidence supports writing a paper draft. It does not yet justify calling that draft submission-ready.
