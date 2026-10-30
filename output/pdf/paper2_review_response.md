# Paper 2: response to the supplied review

Revision date: 8 October 2026. This is a record of changes and evidence, not a claim that every requested experiment has been completed. The five-page source and historical experiments are preserved. The revised main paper is accompanied by a technical supplement.

## Main outcome

The review motivated useful new comparisons and exposed narrower operating limits. The paper now distinguishes the complete learned estimator from the cubic filter itself. In a matched single-final-filter experiment, OptShrink is 2.042 dB better than cubic filtering. On the same small fresh test set, the complete frozen learned method remains 0.552 dB better on the seven-SNR average, but is worse at 20 dB SNR and with a weak reference. Supplying noise variance 6 dB too high also makes the learned method worse than likelihood-ranked HS-GS. All these adverse results are retained in the main paper and supplement.

New evidence consists of 288 unique fresh worlds, 32 per SNR/stress cell. Eight mismatch conditions reuse the 32 reference worlds, so there are 544 condition-cases, not 544 independent worlds. Three existing learned seeds share these cases. No model was retrained. A separate 72-world development bank chose final-filter settings using withheld likelihood only. Intervals are exploratory, unadjusted, and conditional on fixed weights and development choices; they do not replace the original 400-case study.

## Responses to the ten questions

### 1. Identifiability, reference strength, pilot diversity and path richness

**Addressed with a qualified local result, not a global theorem.** The main paper now states the real-coordinate amplitude Jacobian and its full-column-rank local recovery condition. The supplement supplies the inverse-function argument, Rician Fisher matrix, a conditioning interpretation, and counterexamples. P >= 2K is necessary for regular full-rank recovery of an unrestricted row, not sufficient. Full row rank of the pilot matrix is not a phase-retrieval guarantee. Known nonzero reference alone is not sufficient either: a reference in the pilot row space admits a shifted phase ambiguity. These conditions are not a path-count theorem.

A constant reference-phase error can be absorbed into a rotated channel with identical magnitude likelihood and Hankel singular values. The manuscript no longer leaves the impression that withheld measurements can detect or correct this ambiguity without extra assumptions. RSR 0 dB is explicitly distinguished from B = 0. Standard affine phase-retrieval literature is cited; these elementary local results are not claimed as a new identifiability theorem.

### 2. Why ten stages? What about five and twenty?

**Partly addressed; independently trained depths remain open.** Ten is now clearly described as a fixed computation budget, not a demonstrated optimum. The exact stage-five prefix of the frozen ten-stage model was evaluated. Completing ten stages improves its seven-SNR average by 0.890 dB [0.815, 0.964], but not every cell. The stage feature remains t/10, so this must not be presented as five-stage retraining. No twenty-stage experiment was run. A trained 5/10/20 comparison would require matched data, search and training budgets. The parameter formula 178 + T and operation scaling with T are documented.

### 3. Cubic shrinkage versus OptShrink or alternative filters

**A bounded matched component comparison was added; a full iterative LRHD comparison remains open.** Ordinary EM100 supplies the same input to no correction, hard rank truncation, cubic filtering, and one OptShrink operation, each followed by the same Hankel averaging. Independent development data and withheld Rician likelihood select global rank six, relative cubic threshold 0.3, and OptShrink rank seven; every method has a no-op candidate. Search grids and counts are explicitly reported, not called an exhaustive equal-search-budget comparison.

Seven-SNR NMSE is -10.162 dB without filtering, -11.811 with hard truncation, -11.648 with cubic filtering, and -13.691 with OptShrink. The learned estimator is -14.243 dB on exactly the same worlds: a 0.552 dB conditional paired advantage [0.432, 0.673], but a 1.040 dB loss at 20 dB SNR and 2.408 dB loss with RSR 0 dB. Its two-path improvement over globally selected OptShrink does not erase its two-path loss against per-case rank-selected HS-GS.

OptShrink uses the published empirical D-transform with reduced tail dimensions, rectangular zero directions and an explicit tied-boundary policy. Numerical checks passed. This is not full LRHD, a same-architecture trained shrinker replacement, a universal ranking, or a new runtime benchmark. The main contribution is narrowed accordingly.

### 4. Reference amplitude/phase bias and variation

**Small frozen-model stress tests added; no calibration method is claimed.** Assumed reference amplitudes 0.9/1.1 times truth, constant phases +/-10 degrees, and phase-error ramps from 0 to +/-10 degrees were tested on unchanged observations. At the SNR 5 reference cell, amplitude errors cost the learned model 2.430/1.791 dB; phase ramps cost 0.776/0.839 dB. Constant phase errors rotate every estimate and cost about 2.36-2.47 dB in the fixed true frame; truth-aligned errors remain unchanged. Truth alignment is diagnostic only, never an inference correction. These are assumed-input errors relative to a fixed physical reference, not a validated time-varying atomic receiver or a revived joint-calibration project.

### 5. Align HS-GS rank selection with the Rician likelihood

**Implemented and tested.** Both scores use exactly the same 25-update, four-sweep rank candidates, the same 21 input and nine withheld pilots, and the same 100-update full-data refit. Rician scoring improves the seven-SNR average by 0.182 dB [0.104, 0.267]. The learned method retains 1.282 dB [1.092, 1.485] on this small check. Therefore the scoring difference matters, but does not explain the whole gap here. The new residual implementation replays the original solver exactly; the likelihood is independently checked against SciPy's Rician density. No historical rank selection or results were overwritten.

### 6. More N, K and P values; near 2K; scaling

**Existing tests made explicit; a near-boundary experiment remains open.** The original study already includes N = 8,16,32,64; K = 2,3,4; P = 10,20,30. The supplement now tabulates these results and intervals. They vary one factor at a time, not the full product. With K = 3, the shortest tested P is ten, not six. No near-2K recovery claim is made. A leading-order complexity analysis includes initialization, pilot Gram construction, EM products/solves, the spectral-feature SVD, dense shrinkage solve, and HS-GS rank search. No unmeasured transfer-device speedups or exact FLOP counts are asserted.

### 7. Larger variance errors and non-Gaussian/finite-shot noise

**Variance range extended; physical statistics remain untested.** +/-6 dB supplied-variance errors supplement the earlier +/-3 dB tests. Underestimating variance by 6 dB costs 0.238 dB here; overestimating it costs 3.048 dB and makes the method 1.409 dB worse than likelihood-ranked HS-GS. This is disclosed as a failure, not silently repaired by retuning. We did not equate finite shot count with heavy tails: the suggested multi-shot Rydberg paper uses Rician shots and concerns detection. Hardware readout and other noise laws need a justified separate model. No physical-receiver result is claimed.

### 8. Threshold and mixing trajectories

**Added.** A new supplementary figure plots relative threshold, mixing, and actual relative correction over all ten stages for low/medium/high SNR, weak reference and two-path cells. At the SNR 5 reference condition, mean relative threshold decreases about 0.915 to 0.362; mixing stays around 0.99996. This supports interpreting adaptation as smoothing strength, not reliable on/off protection. Curves average cases, users and fixed seeds; they are descriptions, not causal feature attribution or uncertainty guarantees.

### 9. Rebalancing/curriculum for weak references and few paths

**Not run; factual training coverage clarified.** Two-path and RSR 5 dB examples already occur in training, while RSR 0 dB does not. No curriculum, calibration-aware retraining, or test-driven reweighting was introduced. The adverse feature-protection result remains reported. A new training recipe needs separate development and untouched final-test data; it cannot be judged by repeatedly tuning to these inspected cases.

### 10. Two-dimensional arrays

**Scope clarified, not implemented.** A block/multilevel-Hankel extension is plausible but needs a new geometry model, normalization, cost analysis and transfer evaluation. The present paper is for a one-dimensional uniform array. We do not describe 2D extension as automatic or tested.

## Other changes

- Added verified primary references for OptShrink, LRHD, structured ALS, robust Hankel completion, affine phase retrieval, sparse noncoherent channel estimation, partially coherent estimation, and multi-shot detection. The supplement explains observation-model differences instead of declaring all methods directly interchangeable. Partially coherent matching pursuit retains within-packet complex phase information; it is not a magnitude-only baseline.
- Added a bound on one filter's antidiagonal-weighted energy and correction size. It is not a bound on estimation error, a pairwise contraction theorem, or full-estimator convergence.
- Proved the quadratic relaxed-EM surrogate decrease for 0 <= gamma <= 2. Contrary to a blanket over-relaxation warning, gamma > 1 does not itself invalidate this step's generalized-EM property. The following structural correction has no matching likelihood guarantee.
- Clarified that the rational filter is smooth at repeated singular values, but ordered spectral controller features need not be globally differentiable. Finite-gradient checks do not prove global smoothness.
- Preserved independent-validation selection and explained its limits: no test-driven checkpoint selection, but no equivalence between likelihood and NMSE and no broad training-dataset uncertainty estimate.
- Moved the full original input-ablation figure and follow-up table to the supplement, preserving their negative results and numerical values while keeping the main paper concise.

## Verification and remaining release conditions

The 43 original focused tests and nine new theory checks passed together (52 tests). Four dedicated rank-score/calibration checks also passed. The new audit recalculated all 153 estimator outputs exactly and verified original source/checkpoint/data hashes. OptShrink primitive checks passed and its independently computed EM100 errors match the other pipeline exactly. None of these checks proves novelty or a hardware guarantee.

Outstanding scientific work: matched independently trained stage counts; a current-protocol large learned baseline; fully adapted iterative low-rank and sparse phase-retrieval comparisons; full-text comparison with the previously identified XL-MIMO DeepTensor conference paper; broader independently generated training/pilot replication if a generality claim is sought; hardware-informed statistics or measurements if a physical deployment claim is sought. These are not all mandatory for the narrow current scope, but must not be marked completed.

Venue, author approval, companion-paper accessibility, required declarations and AI-use disclosure still need review before submission. This revision is not unconditional submission clearance.
