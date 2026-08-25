"""Frozen configuration for the Hankel-vs-EM-GS study.

Every value here is INHERITED from the audited Track-B implementation. None
was chosen or tuned for this study. The provenance column in AUDIT.md names
the file each value comes from.

Nothing in this file may be changed in response to a result. If an
alternative is worth testing it goes in an ablation script and is reported
separately (see README, "Ablations").
"""
from __future__ import annotations

# --- system model (rydberg_sim/track_b_drivers.py) -------------------------
MASTER_SEED = 20250820      # TRACK_B_MASTER_SEED
K           = 3             # TRACK_B_K
N_DEFAULT   = 8             # TRACK_B_N
L_MIN, L_MAX = 3, 7         # TRACK_B_L_MIN / _MAX  (U{3..7}, i.i.d. per user)
RSR_DB      = 12.0          # TRACK_B_RSR_DB, single-user denominator (Cui eq. 37)
P_DEFAULT   = 30            # frozen B1 panel

# --- estimator hyperparameters (rydberg_sim/track_b_proposed.py) -----------
GS_MAX_ITER   = 50          # Cui t0; identical for both estimators
EXACT_STEP    = "em_gs"     # the measurement update both estimators share
CADZOW_ITER   = 4           # cadzow_project(n_iter=...) default
PROJECT_EVERY = 1           # Cadzow after EVERY EM-GS iteration (interleaved)
SELECT_ITER   = 20          # iterations per candidate during order selection
VAL_FRAC      = 0.3         # held-out pilot fraction for order selection
RIDGE         = 0.0         # no regularisation anywhere

# --- sweeps ----------------------------------------------------------------
# The repository's established sweep is (-5..20). -10 is added because the
# study brief asked for it; it is an EXTENSION, not a replacement.
SNR_GRID_DB = (-10.0, -5.0, 0.0, 5.0, 10.0, 15.0, 20.0)
N_GRID      = (8, 16, 32)
L_GRID      = (2, 4, 6, 8, 10, 12, 14, 16)   # experiment C: L fixed, not drawn
EXP_C_N     = 32            # cap(32) = 16, so L_GRID reaches the ceiling exactly
EXP_C_SNR   = 5.0

# --- Monte Carlo -----------------------------------------------------------
# Trials per operating point. Cost is dominated by rank selection, which runs
# the estimator once per candidate rank (1..r_max), so one trial at N=32 costs
# ~8.5x one at N=8. Measured throughput on 4 cores made a uniform 600 a ~5.7 h
# sweep, so the budget is tiered: 600 where trials are cheap, 400 elsewhere. At
# 400 trials the paired-bootstrap CI half-width is ~0.15 dB, well below the
# effects being resolved. Per-point trial counts and CIs are reported for every
# point, so nothing depends on the reader assuming a uniform budget.
N_TRIALS        = 600       # N = 8 grid (experiment A)
N_TRIALS_LARGE  = 400       # N = 16, 32 grid (experiment B)
N_TRIALS_PATH   = 400       # experiment C
CHUNK      = 50             # checkpoint flush interval
NBOOT      = 2000           # paired bootstrap resamples
BOOT_SEED  = 987654321      # matches the Track-B analysis scripts

ESTIMATORS = ("em_gs", "hankel_em_gs")
