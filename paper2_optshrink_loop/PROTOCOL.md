# OptShrink inside the Rician estimation loop

Frozen before development selection or new test generation, 8 October 2026.

## Question

Does the existing ten-stage, 188-parameter learned estimator still offer an
accuracy or processing-time advantage over an independently tuned Rician EM
estimator that applies Hankel OptShrink after every update?

This is a targeted comparison, not independent retraining confirmation, a full
LRHD reproduction, or a claim that OptShrink's matrix-noise optimality theorem
applies to errors in lifted channel estimates. Previous sources and results are
read-only. No neural training and no manuscript edits are part of this run.

## Model and data

Use the unchanged simulator: 32 antennas, three users, 30 magnitude observations,
known complex reference and noise variance. Main cells have random 3–7 paths per
user, nominal reference-to-one-user power ratio 12 dB, and aggregate SNR
−10, −5, 0, 5, 10, 15, 20 dB. Two separately reported stress cells use SNR 5 dB
with reference ratio 0 dB, or two paths per user and reference ratio 12 dB.

Development: 32 independent cases in each of the seven main cells. Seed
202610081101 + 100 times cell index. No development channel truth is saved or
read by selection. Fit using the first 21 pilots; score the remaining nine using
the exact Rician negative log likelihood, omitting observation-only constants.

Test: 128 new cases in each of nine cells (1,152 total), generated only after
selection is frozen. Seed 202610081901 + 100 times cell index. The observation
and truth files are separate. Methods see identical pilots, reference and noisy
observations within each case. Case identifiers include seed and trial number.
Fresh pilots and channels are generated per world by the existing simulator.

## Methods and tuning

Start every classical trajectory from the existing spectral initializer. Use
the unchanged exact Rician EM update and one Hankel filtering/anti-diagonal
averaging operation per iteration. Each user channel is lifted to a 17 × 16
matrix. OptShrink uses the reduced noise-tail dimensions in Algorithm 1 of
Nadakuditi (2014), as in the already checked final-filter primitive.

Candidate ranks: 1 through 15. A separate no-filter candidate is included;
rank 16 is not treated as OptShrink because it leaves no noise tail. Mix the
filtered and unfiltered update using strengths 0.25, 0.5, or 1. Iteration budgets
are 10, 20, 50 and 100. Use ordinary, unrelaxed EM updates. Ties choose the first
candidate (no filter first, then rank and strength in ascending order), and the
lowest iteration budget. No true rank, path count or test error enters tuning.

Primary comparator: independently choose rank, strength and iteration budget
for each development noise level using mean held-out Rician likelihood. At
inference, select the nearest development level using the supplied noise
variance and known three-user unit-power convention, SNR = 10 log10(3/v).
Do not use realized channel power or channel-family labels. The stress cells
use the same SNR-5 settings as the main cell. This noise-aware rule is an
adapted, validation-tuned comparator, not a published LRHD implementation.

Also report a single global development-selected in-loop configuration, all
four fixed-iteration noise-aware configurations, and a final-only OptShrink
control tuned on exactly the same grid and development cases. For the final
control, ordinary EM is followed by one mixed filtering step. This controls
placement; it does not make the learned and classical training/search budgets
identical.

Retain ordinary EM100 and Rician-selected HS-GS: ranks 1–16, 25 iterations and
four Hankel sweeps on the 21-pilot context; select by held-out likelihood and
refit for 100 iterations with four sweeps using all 30 observations. Evaluate
all three existing adaptive checkpoints (seeds 9101/9102/9103) unchanged.

No test-driven tuning, selection of the best test seed, or longer runs after
seeing outcomes. All deployed methods use all 30 observations. Classical
offline tuning costs and online selection costs are reported separately.

## Outcomes and uncertainty

Primary cell metric: 10 log10(sum squared channel errors / sum channel energies).
Average this dB metric equally across the seven SNR cells. For the learned
method compute the metric separately for each training seed, then average in
dB; these repeated predictions are not independent test channels. Also report
per-case normalized error, the fraction with normalized error >1, every cell,
and the two stress cells separately.

Primary contrast is learned minus noise-aware selected in-loop performance,
reported as a positive dB gain when the learned method has lower error.
Use 4,000 paired, within-cell case bootstrap resamples (seed 202610082901),
retaining all predictions for each sampled case. Give a two-sided 95% interval.
Other contrasts and per-cell intervals are descriptive, unadjusted diagnostics.
These intervals are conditional on the development selection and three frozen
models; they do not measure new-training-dataset variability.

Runtime: CPU, one numerical-library thread, double precision, batch one,
including initialization, online selection, and all iterative work. Separate
warmed serial timing on 24 fresh SNR-5 worlds (seed 202610083901), three repeats,
round-robin method order. Model loading and file I/O are excluded. Never time
an ensemble and call it the cost of one model. Give distributions and per-seed
times; timing is implementation/device specific, not an asymptotic guarantee.

## Verification and resource limit

Check vectorized weights against the existing independently checked OptShrink
primitive, zero and repeated singular values, scaling, no-filter equivalence
to EM, and deterministic prefixes. Hash sources, checkpoints, development data
and the frozen selection before generating test data; check again at the end.
Save progress per development case and test case. Resumption requires identical
configuration and sources and may only skip complete files. An exclusive lock
prevents duplicate launches. Stop cleanly after at most two hours per invocation
and preserve partial work. Benchmark actual timings before giving an ETA.

If the stronger comparator removes the learned accuracy advantage, report that
directly. A computation advantage is a separate finding. Do not relabel an
accuracy failure as success, claim universal superiority, or expand the training
study automatically.

Reference: Nadakuditi, IEEE Transactions on Information Theory 60(5), 2014,
doi:10.1109/TIT.2014.2311661; https://arxiv.org/html/1306.6042v4.
