# Baseline reproduction gap: first diagnostic

## What is authorized and unchanged

The user asked to investigate and fix the reproduction discrepancy. Start with
one controlled baseline-only training comparison. Do not change the active
original-channel U/H experiment, its shared modules, the dashboard, or manuscript.
The cancelled adaptive-calibration project is not involved.

This is NOT a certified reproduction or a new proposed estimator. No authors
have been contacted and no messages sent on the user's behalf.

## Evidence and unknowns

The completed reconstruction has three 50-epoch seeds. The pooled mean-linear
NMSE at 5 dB is -7.088 dB versus a plot-derived -21.734 dB. Validation-selected
checkpoints give -7.503 dB. GS/EM-GS discrepancies are about 0.6–1.5 dB; this
does not fully validate channel statistics or rule out simulation differences.
Training error falls while validation error worsens after early epochs.

A small read-only diagnostic of seed 1, epochs 2 and 50, used 32 validation
samples nearest 0 and 15 dB. At high SNR many filter outputs are near 0 or 1.
Saturation near 1 can be physically appropriate: saturation alone is NOT a bug
diagnosis. No model or checkpoint was modified by this inspection.

Primary source: https://arxiv.org/html/2509.12586v3 and its already-downloaded
TeX/EPS source under results/paper2_reproduction/source. Searches on 20 September
2026 (Oslo time) for URformer, its arXiv identifier, GitHub and author/code terms
did not locate author training code. This is not proof that none exists.
The paper does not resolve per-SNR versus mixed-SNR training, joint cluster-angle
statistics, or several architectural/initialization details.

## Frozen first comparison

- Two ordinary, non-Hankel arms: mixed training SNR uniform [-10,20] dB;
  fixed training SNR 5 dB. One seed, 2026092201.
- 20,000 training samples. Identical channels, pilots, reference, initial guesses
  and normalized noise draws in both arms; only training noise amplitude changes.
- Both arms use the same 2,000 fresh validation examples, all at 5 dB.
- Fresh training trial IDs 10,000,000..10,019,999 and validation IDs
  11,000,000..11,001,999. No final test set generated or accessed.
- Identical printed-equation reconstruction and initial weights, raw kappa,
  no gradient clipping or numerical changes, Adam lr .001, batch size 32.
- Train the first **10 epochs of a 50-epoch cosine schedule**. The schedule
  duration is not compressed. Same optimizer-step count and exact batch order.
- Evaluate final epoch and the pre-existing validation-only one-SE selection
  rule. Preserve per-sample validation errors, every epoch, and source hashes.
- Single CPU thread, low priority, sequential arms. Live status and log live
  under results/paper2_reproduction_diagnostic; the existing dashboard is unchanged.
- Duplicate-worker locks and epoch-boundary checkpoints. Resume must retain
  source hashes, config, optimizer, scheduler and RNG. Stop on numerical failure.

Primary comparison: 10 log10(mean normalized error mixed / mean normalized
error fixed5), using matched validation examples. Positive favors fixed-SNR.
It is an exploratory validation result, not an independent generalization test.
Ten epochs and one seed cannot establish convergence, reproducibility, or the
authors' protocol. A favorable result needs a new frozen multi-seed experiment
and untouched test data; an unfavorable result does not rule out later benefits.

## Next checks, not automatically queued

1. Seek author clarification about SNR-specific training, ray correlation,
   scaling and initialization. Contact requires a user-directed message.
2. Inspect learned filters by separating near-zero from near-one saturation and
   compare with the exact Bessel kernel; check gradients before changing features.
3. Test channel clustering separately. The current U/H experiment's clustered
   branch has a different implementation and cannot settle reproduction alone.
4. Audit scale transformations for physical equivalence and different optimization
   effects. A numerically useful change is a named variant, not retroactively
   asserted to be what the paper used.
5. Do not retune to the published test curve or silently replace failed results.

## Commands

From the repository root, with the project Python environment:

```text
../../.venv-paper2/bin/python -m unittest discover -s tests -p 'test_paper2_reproduction*.py' -v
PYTHONPATH=. ../../.venv-paper2/bin/python scripts/paper2_reproduction_diagnostic.py launch
```

The launch command starts only this diagnostic. It never restarts or changes
the existing experiment. Shutting down the Mac still stops local computation.
