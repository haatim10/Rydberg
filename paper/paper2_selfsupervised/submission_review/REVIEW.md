# Paper 2: focused submission review

23 September 2026. General IEEE journal layout; Abdullah Haatim is the sole author, as requested. No venue has been selected. No new research training was launched, and no frozen experiment source, checkpoint or result was edited. Small temporary fixtures were used by the regression tests.

## Decision

The revised document is a reviewed general IEEE research manuscript, not an unconditional submission clearance. Its narrow contribution is a compact, label-free Rician channel estimator with measurement-conditioned soft Hankel filtering and a measured accuracy/processing-cost trade-off. The experiments do not establish a new self-supervision principle, a uniformly protective gate, or superiority over published URformer.

The remaining scientific release condition is the detailed comparison with the XL-MIMO DeepTensor conference paper. Its title and presentation are verified from the official program, but its equations and experiments remain unavailable. The revised paper discloses this limitation and does not invent a distinction. The user must also approve the scientific content, affiliations, companion-paper status and AI-use disclosure before any submission. Venue-specific formatting and declarations are intentionally deferred.

## Checks completed

- All 43 existing numerical and pipeline tests passed. They include likelihood/derivative checks, noise scaling, finite gradients, pilot isolation, repeated singular values, precision checks, deterministic resume and temporary end-to-end smoke tests.
- Rechecked 86 protected source/input hashes and 90 frozen checkpoint references.
- Independently recalculated saved-prediction errors and every reported cell NMSE across 1,120 main-study result files, 702 follow-up files and 432 feature-study files.
- Recalculated the main seven-SNR averages, descriptive severe-error rates and timing aggregation for all 19 comparator variants.
- Replayed the three primary checkpoints on eight common saved observations each. Maximum difference from saved predictions: 0.0.
- Existing bootstrap confidence intervals were retained; this pass did not regenerate them or interpret them as training-dataset uncertainty.
- Reviewed the paper equations against `sahure/math.py`, `sahure/model.py`, training split/selection code and the classical solver. Preserved the original estimator and model selection.

`evidence_audit.json` contains machine-readable results. Run `python paper/paper2_selfsupervised/submission_review/check_evidence.py` from the complete original repository to repeat the saved-result audit. It only writes this review's outputs and the generated manuscript table. The compact reproducibility ZIP does not contain every original observation/prediction array required by this full audit.

## Changes made to the paper

1. Narrowed the contributions to the actual implemented method and tests; split-pilot learning and Rician loss are explicitly not claimed as new principles.
2. Added more specific, source-backed distinctions from learned Hankel factorization, structured-low-rank MRI, original DeepTensor and Rician self-supervision.
3. Added all 19 main-study comparators to one table. The secondary GD model's slightly better test NMSE is visible; it does not replace the validation-selected EM primary model.
4. Added severe-error rates alongside mean accuracy and latency. These rates are descriptive and do not establish reliability outside the tested mixture.
5. Explained HS-GS rank selection, validation-selected hard rank, solver budgets, different software implementations and the meaning of constant-output controls.
6. Clarified that simulated path-setting labels weight validation strata but are not inference inputs. The method is channel-label-free, not free from a designed simulation population.
7. Specified the limited alternative-channel tests and their energy matching. They are not full robustness sweeps.
8. Disclosed the recorded classical SVD recovery without deleting its trial or changing the comparison.
9. Added reproducibility and AI-use statements. The latter must remain truthful and be reviewed by the author; it does not claim that the author has already independently checked everything.

## What is not claimed to be fixed

- No verified, exhaustive novelty result or completed full-text XL-MIMO comparison.
- No new 30-pilot supervised-training comparison. The existing supervised recipe stays an ablation, not a universal superiority benchmark.
- No new noise-tuned regularized classical baseline, hardware test or broad multi-dataset robustness confirmation.
- The feature study still fails its predefined 0.25 dB cell-protection condition. Passing code audits does not override that failure.
- Paper 1 remains an unpublished companion unless the author confirms a different status. The new manuscript must not be represented as independent evidence for Paper 1's older results.
- No public code archive, DOI, copyright transfer, exclusive-submission declaration or funding statement has been invented or submitted.

## Minimum remaining actions

1. Obtain a legitimate full text of the XL-MIMO DeepTensor paper and compare the measurement model, loss, fitting procedure and claimed contribution. If access stays unavailable, make an explicit author decision about submitting with the disclosed limitation; do not describe the novelty audit as complete.
2. Read and approve the revised paper, author details, companion-paper citation and AI-use statement.
3. Once a venue is chosen, check its current scope, length, anonymity rules, required declarations and supplementary-file rules. Adapt this general version then.

No large new experiment is automatically required by the narrow current claim. If the closest-work comparison exposes a missing distinction, decide the smallest discriminating comparison before committing more compute.
