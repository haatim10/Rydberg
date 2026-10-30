# Noise and clustered-channel audits

These two directories are frozen records. This archive adds navigation outside
them and preserves their existing code, configurations and results.

## Noise-convention audit

- [Full report](../../../experiments/urformer_snr_audit/REPORT.md)
- [Summary](../../../experiments/urformer_snr_audit/summary.json)
- [Saved final verification](../../../experiments/urformer_snr_audit/final_verification.json)

The audit compared six noise conventions, classical GS/EM-GS curves, phase-known
genie estimators and independent confirmation data. It rejected the proposed
common roughly 15 dB noise shift. The `reproduction_ensemble` convention remains
the validated convention for the follow-up. It did not reproduce the published
neural accuracy, and it did not launch new neural training.

## Clustered-ray audit

- [Full report](../../../experiments/urformer_cluster_audit/REPORT.md)
- [Summary](../../../experiments/urformer_cluster_audit/summary.json)
- [Selection made before confirmation](../../../experiments/urformer_cluster_audit/validation_selection.json)
- [Saved final verification](../../../experiments/urformer_cluster_audit/final_verification.json)

The follow-up retained `reproduction_ensemble` and compared independent rays with
Laplacian and Gaussian subrays at 0.5, 1, 2, 5 and 10 degrees. Each generator used
a separate 20,000-draw covariance estimate and 2,000 validation draws. A selected
negative case and the independent-ray control received 2,000 confirmation draws.

All 11 generators passed the earlier classical-curve tolerance. None of the 10
clustered generators passed the required global-covariance genie LMMSE screen.
Small spreads lower the effective dimension within a channel, but random
cluster positions keep the population covariance broad. Independent physical
angles also do not imply spatially white antenna channels.

This is **not an impossibility result for a nonlinear learned estimator**. Global
LMMSE does not know each new channel's cluster positions. The prescribed gate
failed, so the 64-sample overfit test, new three-seed v3 training and v1 encoder
sensitivity training were **not run**.

## What the archive contains

Reports, configurations, code, per-cell metrics, covariance matrices, rank
summaries and figures are included where listed in the archive inventory. Large
generated input banks remain local. Their paths, sizes and hashes are recorded.
Original manifests describe the full local experiments, so a missing local-only
file in a GitHub clone is an intentional archive omission, not a changed result.

Reference PDFs and full-text copies are kept local. Read the authors' original
[URformer v1](https://arxiv.org/html/2509.12586v1) and
[v3 paper](https://arxiv.org/pdf/2509.12586v3); the reports preserve version notes
and approximate curve readings.
