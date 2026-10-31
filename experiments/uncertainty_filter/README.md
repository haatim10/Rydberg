# Uncertainty-scaled Hankel filter study

Tests whether the EM phase posterior gives a better data-dependent scale for the
receiver's Hankel filter. Start with [REPORT.md](REPORT.md); the math is in
[DERIVATION.md](DERIVATION.md); every reported number is in `summary.json`.

Status: **closed at Gate 1 (failed); reported as a negative result, not adopted.**
Nothing in the frozen studies was modified.

```bash
.venv/bin/python -m pytest experiments/uncertainty_filter -o addopts=""
```

```bash
.venv/bin/python -m experiments.uncertainty_filter.phase1_calibration --workers 4
```

```bash
.venv/bin/python -m experiments.uncertainty_filter.phase1_analysis
```

Run from the repository root with numpy 2.4.6, scipy 1.17.1 and torch 2.13.0
(the versions pinned in `sahure/requirements.txt`).
