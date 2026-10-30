# Independent URformer reproduction attempt — 19 September 2026

## Purpose and limits

First reproduce the ordinary baseline's NMSE-versus-SNR curve, then investigate
disagreement before using it to claim an improvement over the published method.
This is an independent reconstruction, not an execution of the authors' code.
No Hankel training is included in this attempt. The ongoing original-channel
U/H experiment remains separate and unchanged.

Target: Xiao et al., arXiv:2509.12586v3 (11 May 2026), Figure 3.
https://arxiv.org/html/2509.12586v3
https://arxiv.org/src/2509.12586v3
The downloaded source archive contains TeX, bibliography and EPS figures, not
training code. Targeted title/arXiv-ID/URformer GitHub searches did not locate
an author implementation. This is not proof that no implementation exists.

## Audit and frozen first attempt

The current local U/H implementation uses log1p(kappa), normalization before
the encoder FFN and before the output projection, and a zero-initialized output.
These are not all specified by the paper. Matching its channel alone therefore
does not constitute reproducing its published baseline.

The new isolated implementation uses raw kappa, the printed encoder sequence
A = MHSA(LN(V)) + V; V_next = FFN(A) + A, and no extra final normalization.
The embedding is bias-free as in Eq. (19). No gradient clipping is applied.
Each layer uses the printed gated magnitude reconstruction, fixed least squares,
and a learned residual. Initial channel guesses are independent unit-variance
complex Gaussian, not ground-truth-based. The known atomic gain is divided out.

Paper-stated settings: 4 users, 32 antennas, 4 groups of 10 rays, uniform
ray-angle marginal on [-pi/2, pi/2], 10 unrolled layers, 3 encoders per layer,
width 64, 20 pilots, multiuser RSR 10 dB, 20,000 training samples, batch size 32,
Adam starting at 0.001, cosine schedule lasting 50 epochs, mean per-sample NMSE.

Unspecified choices, fixed before training (NOT claims about the authors' code):

- Independent ray angles (the paper names clusters but does not give joint
  angle statistics or an angular spread); independent CN(0,1) gains;
  half-wavelength antenna spacing; literal unnormalized steering and Eq. (1)
  scaling. Expected power per channel element is 32.
- One fixed CN(0,1) Gaussian pilot matrix, seed 777000001; positive constant LO.
  Noise and LO use ensemble multiuser signal power for that pilot matrix.
- One model trained across a uniform [-10,20] dB SNR range, not one model per
  SNR. The paper does not state which training protocol generated its curve.
- Four attention heads; FFN 64->256->64 with GELU; output FFN also
  64->256->64 with GELU; FilterNet 1->32->1 with ReLU and sigmoid.
  The output FFN depth/hidden width, head count and activations are unspecified.
- Untied unrolled weights, no dropout, neutral gate logit zero, position
  initialization N(0,0.02^2), otherwise PyTorch default initialization.
  No zero output initialization, filter pretraining, weight decay or clipping.
- Fixed finite datasets, single-thread float32 CPU, training seeds
  2026092101/2026092102/2026092103. Adam defaults beta=(0.9,0.999), eps=1e-8.

These choices can materially affect results. Failure to match is a diagnostic
finding, not evidence that the paper is incorrect. Subsequent changes must be
separate, named experiments; no silent tuning to the published test curve.

## Data, measurements and reporting

Use the verified original-channel generator in read-only mode, with fresh trial
IDs: training 7,000,000..7,019,999; validation 8,000,000..8,001,999;
test base channels 9,000,000..9,001,999. Master seed 2026091940. No sample from
the earlier experiment's training/validation/test sets is reused.
Training and validation SNR are independently generated uniform [-10,20] dB.
Test each of the 2,000 base channels at exactly -10,-5,0,5,10,15,20 dB, with
matched unit-noise realizations across SNR and training seeds. This produces
14,000 test cases but only 2,000 independent channel/noise draws per seed.

Primary checkpoint is epoch 50. Also report a validation-only one-standard-error
selected checkpoint as a separate diagnostic (not a replacement for epoch 50).
Metric: 10 log10(mean over samples of squared channel error / channel energy),
not median improvements or mean dB errors. Preserve every test numerator and
denominator and every completed epoch. Never use test scores to select epochs.

Extract all five Figure 3 curves from the EPS polylines and calibrate using the
0 and -30 dB axis ticks. These are plot-derived values, not raw author outputs;
use a conservative 0.02 dB digitization tolerance. Compare trained ordinary
URformer against its curve and 100-iteration GS/EM-GS against their curves.
Show signed discrepancies at every SNR. Flag differences over 1 dB for review;
this is a practical screening threshold, not a statistical reproduction proof.
Three training seeds only support limited seed-variability assessment.

## Execution safeguards

New source files and results directory only. Do not edit the frozen earlier
drivers, shared model modules or manuscript. Lock against duplicate launches,
record source and dataset hashes, and restore the last complete epoch on resume.
Stop and record non-finite loss/gradient/result errors; do not silently stabilize
or change the model. Run numerical and tiny end-to-end checks first. The full
baseline queue is three 50-epoch runs, sequential, one CPU thread, low priority;
it may modestly slow the existing separate training process.
