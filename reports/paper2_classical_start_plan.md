# Classical-start learning diagnostic

Authorized by “launch it”; fixed before training. No prior experiment changes.

Question: can a learned estimator starting exactly at EM-GS 10 improve low-SNR
error without losing high-SNR accuracy? This is a new independent model, NOT a
claim of reproducing the published URformer. No Hankel or SNR-balancing change.

Two arms: ordinary independent reconstruction, and ten fixed exact EM-GS layers
each followed by a trainable PaperFormer correction. In the latter, every
correction's final output weight and bias start at zero. The Bessel filter is
fixed; no unused learned filters/gates are retained. All other Transformer
initial weights match the ordinary model. Consequently this is a bundled
fixed-filter/initialization intervention, not pure initialization attribution.
Do not claim identical parameter counts or identical initial full weights.

Physical units, N32/K4/P20, independent 40-ray channel, same fixed pilots,
known reference/noise, ensemble multiuser RSR10 dB. Mixed SNR uniform [-10,20].
Fresh 4096 train IDs24,000,000+, 512 validation IDs25,000,000+.
Additional 256 classical-stopping tuning cases IDs26,000,000+, disjoint from
both. No final test. Seed2026092501; batch32; Adam.001;12epochs cosine tozero;
1536 optimizer steps per arm. Same examples/order, no clipping or silent fixes.
One CPU thread. Epoch checkpoints, finite-gradient guards, resumable state,
exclusive lock, source/data hashes. No manuscript/dashboard changes.

Primary comparisons: final validation mean-linear NMSE overall and all six
5dB bins. Secondary: existing one-standard-error validation selection. Save
initial classical-start validation errors to measure any loss of initial
high-SNR accuracy. Candidate screen: >=0.5dB overall gain vs ordinary AND at
most0.5dB deterioration vs initial EM-GS10 in [15,20). This is an arbitrary
development screen, not significance or publication evidence.

Classical references: EM-GS10,100 and one global stopping count chosen from
{1,3,5,10,20,50,100} by full-range mean-linear error on the separate256 tuning
cases. Evaluate the selected count on the512 validation cases. Different
iteration/compute costs must be disclosed. No validation-per-case oracle or
SNR-routing selection. Report all negative findings and one-seed uncertainty.

Before launching: exact EM-GS equivalence at initialization, unchanged ordinary
arm, matching non-output Transformer weights, usable gradients through all
ten layers, exact interrupted-training resumption, disjoint data IDs. Check
source hashes and matched orders on completion; replay final/selected weights.
Launch locally with this driver's `launch` command; resume same command only
after the worker stops. A Mac shutdown stops computation; tmux would not fix it.
