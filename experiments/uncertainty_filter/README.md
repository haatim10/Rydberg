# Uncertainty-scaled Hankel filter study

Tests whether the EM phase posterior gives a better data-dependent scale for the
receiver's Hankel filter. Start with [REPORT.md](REPORT.md); the math is in
[DERIVATION.md](DERIVATION.md); every reported number is in `summary.json`.

Status: **closed at Gate 1 (failed); reported as a negative result, not adopted.**
Nothing in the frozen studies was modified. Read [REPORT.md](REPORT.md) for the
original pre-registered audit and [CORRECTION.md](CORRECTION.md) for the corrected
relaxed-output calibration. The correction uses the saved raw records only and
does not change the decision.

```bash
.venv/bin/python -m pytest experiments/uncertainty_filter -o addopts=""
```

```bash
.venv/bin/python -m experiments.uncertainty_filter.phase1_calibration --workers 4
```

```bash
.venv/bin/python -m experiments.uncertainty_filter.phase1_analysis
```

```bash
.venv/bin/python -m experiments.uncertainty_filter.relaxed_audit
```

The last command writes a separate `results/phase1_relaxed_audit/summary.json`.
It leaves `results/phase1/` unchanged, because that folder is the original
pre-registered analysis.

`HASHES.sha256` is the preserved manifest for the original study. The corrected
files and saved-data result are listed separately in `CORRECTION_HASHES.sha256`.

Run from the repository root with numpy 2.4.6, scipy 1.17.1 and torch 2.13.0
(the versions pinned in `sahure/requirements.txt`).
