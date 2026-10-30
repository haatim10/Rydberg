# Paper 2: original-paper-style channel comparison

19 September 2026. User authorized starting this separate experiment. This plan
is frozen locally before training, not externally preregistered. Existing
manuscript and results remain unchanged.

Source: Xiao et al., arXiv:2509.12586v2, equations (1), (2), (7), (20), Table I
and numerical-results setup: https://arxiv.org/html/2509.12586v2 .

## Scope and unresolved publication details

This is an original-paper-style channel test with documented assumptions, NOT
an exact reproduction of the published curves or an author-code reproduction.
Use K=4, M=32, P=20, L=4 clusters, 10 rays per cluster; positive-sign ULA
steering, half-wavelength spacing. Implement equation (1) literally with factor
sqrt(M/40) and the unnormalized steering vector printed in equation (2).
Assume independent CN(0,1) ray gains (gain distribution/normalization is not
fully specified in the cited setup). Thus expected per-element channel power
is M=32, NOT one. Do not normalize individual realized channels to unit power.

Two frozen angle interpretations, neither chosen using performance:

1. **literal_independent** (primary): all 40 ray angles independent uniform
   [-pi/2,pi/2], following the marginal distribution in Table I and explicitly
   assuming independence. Cluster labels are bookkeeping only in this reading.
2. **clustered_5deg** (sensitivity): four independent uniform cluster centers,
   independent uniform ray offsets in [-5,+5] degrees, clipped to the visible
   angular range. This spread and boundary rule are OUR assumptions, not claims
   about the original authors' unpublished implementation.

Use matched gain/noise/SNR/initialization random streams across interpretations
where possible; each U/H pair uses byte-identical datasets. Independent streams
are keyed by master seed 2026091940 and disjoint train/val/test trial ranges
[4000000,4020000), [5000000,5002000), [6000000,6002000).

Fixed CN(0,1) pilots from seed 777000001, full row rank. This pilot distribution
and reuse are explicit implementation assumptions. For these fixed pilots set
ensemble signal power q=M*sum(abs(S)^2)/P. Set sigma2=q/10^(SNR/10), reference
amplitude sqrt(10*q), constant known reference phase, repeated over pilots.
This gives paper-convention (multi-user) RSR=10 dB. Verify ensemble rather than
per-realization calibration; do not use true sample channel power to set noise.
Known atomic gain is divided out (set to one in calibrated units).

## Training design

20,000 train / 2,000 validation / 2,000 test examples per interpretation.
Training and evaluation SNR uniform [-10,20] dB on a millidB grid, an explicit
assumption because the paper does not specify the training-SNR distribution.
Each channel interpretation: three seeds, U/H pair, 50 epochs. Total 12 runs,
600 epochs, 12 million training-example presentations. Fresh models, no transfer
of prior trained weights. Random CN(0,1) initial channel estimates, rather than
the earlier experiment's spectral initialization, following the paper's stated
random normalized-variance initialization convention. Same initial estimate
for both arms of each sample, fixed across epochs.

Use existing URformer-based implementation: 10 untied unrolled layers,
3 Transformer encoders, width 64, 4 attention heads, FilterNet hidden size 32,
log1p(kappa) input, zero dropout, initial update gate sigmoid(-2). Unspecified
architecture details remain our implementation choices; this is not identical
author software. Adam lr=0.001, batch=32, cosine schedule to zero over 50 epochs,
existing clipping, CPU float32, one thread. H adds the existing fixed rank-7,
one-pass Hankel step with identity-backward approximation. No rank tuning or
SNR switching in this comparison. Record mismatched-rank cost separately.

## Controls, outcomes and reporting

Check exact initial learned-weight matching, batch-order hashes, finite loss
and gradients, deterministic generation, disjoint IDs and exact checkpoint
resume. Retain every epoch and validation curve; select earliest epoch within
one standard error of best validation, and separately report final epoch.
Tests are used for reporting only; no choices made from test scores.

Primary: paired median high-SNR (>=5 dB) H-over-U gain per seed, averaged over
three seeds, as in the current paper. Report all six SNR bins and the full
range, raw-error-sum and normalized-error-sum gains, absolute errors, seed SD
and nominal t intervals. A high-SNR benefit is not a full-range improvement.
Three seeds on one fixed dataset do not establish generality or convergence.

Before training run a channel diagnostic: empirical mean power, SNR and RSR
calibration, and the rank-7 noiseless-channel projection distortion for both
interpretations. This diagnostic does not select/tune ranks or architectures.
Use timing measurements to estimate duration, not to change the budgets.

Run primary literal pair first, followed by clustered pair, for each seed.
Use an exclusive process lock and resume only from completed epochs. Stop and
report numerical errors; do not silently change the frozen experiment. On
completion independently verify all 12 results, 600 epoch records, six matched
data orders, selected/final replays and per-bin metrics, then write a report.

Neither a positive nor a negative outcome is permission to rewrite the
manuscript automatically. Results test both the channel assumptions and other
paper-style setup changes (K, RSR, initialization, sample/epoch budget); they
are not a one-factor comparison against the earlier geometric experiment.
