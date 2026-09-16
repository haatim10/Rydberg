# Independent replication of ThesisProj-2.pdf

This directory is the Codex verification campaign on branch `codex`, based on
`739dba4`. It does not modify the production estimator or overwrite earlier
results. The reference manuscript is the user's nine-page `ThesisProj-2.pdf`.

## Scope and decisions

| Manuscript result | Fresh simulation |
|---|---|
| Figure 2(a), NMSE and CRLB/CCRB | 400 paired worlds per SNR; N=32, P=30, RSR=12, T=100, selection=25 |
| Figure 2(b), array size table, inactivity | N=8/16/32 × P=10/30 × six SNRs × 400 trials; includes Figure 2(a) |
| Path-count trend | N=32, P=30, SNR=5, RSR=12, T=50, selection=20, L=2:2:16, 300 each |
| Figure 3, effective rank | Original N16/N64 cells and extra boundary cells; random SNR in [-10,20], P=20, RSR=10, T=100, selection=25; 400 per N16 cell, 300 per N64 cell |
| Matched Figure 3 control | N32, L=2:2:16, with the same random-SNR design and pooled median as N16/N64; 400 per cell |
| Forced structural projection | Same N16 worlds at L=7/9/14; fixed rank cap−1 compared with adaptive and EM-GS |
| Alternative distributions | Clustered 4×10-ray and 10-path adaptations, 400 fresh trials each, original power normalization |
| User count | K/P=2/13, 3/20, 4/27, L=5, 400 random-SNR trials each |
| Projection placement | Six SNRs × 300 trials; original interleaved-4 versus final-1 plus matching 4/4 and 1/1 comparisons conditional on shared rank |
| Runtime | Separately timed runs; report local hardware/implementation, not a portable percentage |

Figure 1 is a system illustration, not a numerical experiment.

SNR is **aggregate user signal power / complex noise variance**, matching the
generator, despite the PDF's single-user wording. RSR retains its single-user
reference. This convention must be corrected in the manuscript.

All main simulation seeds are new and disjoint from the existing campaigns.
The new curves must not reproduce old values exactly: they are independent
Monte Carlo estimates. Per-trial squared errors, channel power, selected rank,
held-out residuals, effective ranks, SNR and trial identifiers are saved. Trial
counts are fixed beforehand; no early stopping based on observed accuracy.

The new solver evaluates the same equations in double precision, batches rank
candidates, reuses least-squares factors, and omits unused diagnostics. It is
not a new estimator. `validate.py` checks outputs, selection scores, selected
ranks, full-rank fallback, Bessel ratios and Cadzow outputs against production.
The results record source fingerprints; resumption rejects changed code.

## Environment and reproduction

Core dependencies: NumPy 2.4.6, SciPy 1.17.1, pytest 9.1.1. Plotting uses
Matplotlib 3.11.2; thread control uses threadpoolctl 3.7.0. No neural training or
PyTorch dependency is required for this Paper 1 campaign.

From the repository root, with dependencies available:

```sh
export OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OMP_NUM_THREADS=1
python -m codex_verification.validate
python -m codex_verification.run --workers 6
python -m codex_verification.bounds
python -m codex_verification.analyze
python -m codex_verification.supplement
python -m codex_verification.timing
python -m codex_verification.consolidate
```

Run and bound outputs are checkpointed under `results/`. The driver never
loads the earlier simulations as input. Original manuscript numbers may be
included only as explicitly labeled comparison targets in analysis.

The completed main campaign contains **27,300 paired trials across 73 cells**.
Run `timing` only after the simulation campaign has finished. Three timing
trials per array size characterize this machine and implementation; they are
not an accuracy experiment or a universal runtime guarantee.

`results/trials/` contains losslessly consolidated per-trial arrays. Analysis
can run from these files without the local chunked restart stores. The initial
dense-averaging pilot was preserved locally and excluded from every reported
result after the solver was optimized and revalidated. All published cells
use one final source fingerprint.

## Interpretation rules

- Report the paired-median gain and ratio-of-sums NMSE separately.
- Bootstrap paired trials; preserve all users from one trial as a cluster
  when quantifying effective-rank uncertainty.
- Do not claim a zero crossing is measured if it is extrapolated beyond the
  measured positive-gain points. Exact zero due to estimator fallback is not
  evidence of a negative side of a crossing.
- The original Figure 3 combines different settings and statistics: N32 is a
  fixed-SNR ratio-of-sums curve; N16/N64 are random-SNR pooled medians. Label
  this explicitly and provide a matched-design control before claiming collapse.
- A truth-derived effective rank is retrospective, not an observable receiver
  control input.
- A near-zero average placement difference does not establish equal estimates
  or statistical equivalence. Timing includes rank selection where applicable.
- CRLB/CCRB curves are local unbiased references, not unconditional lower
  limits for these biased, rank-selected estimators.
