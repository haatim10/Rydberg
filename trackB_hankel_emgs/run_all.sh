#!/usr/bin/env bash
# Full Monte Carlo sweep, ordered so the decisive experiments finish first:
#   A  (N=8 grid)   -- cheap, and is also experiment B's N=8 column
#   C  (path count) -- the mechanism test; runs before the expensive N=32 grid
#   B  (full grid)  -- adds N=16 then N=32
# All points are checkpointed every 50 trials and resume where they left off,
# so this script is safe to interrupt and re-run.
set -euo pipefail
cd "$(dirname "$0")"

# Single-instance lock. Two concurrent sweeps compete for cores AND can write
# the same store, so refuse to start if one is already running.
LOCK="results/.run_all.lock"
mkdir -p results
if ! mkdir "$LOCK" 2>/dev/null; then
    echo "A sweep is already running (lock: $LOCK)." >&2
    echo "If that is stale, remove the directory and retry." >&2
    exit 1
fi
trap 'rmdir "$LOCK" 2>/dev/null' EXIT
echo "=== $(date -u +%H:%M:%S) Experiment A (N=8 grid; also B's N=8 column) ==="
python3 experiment_snr.py
echo "=== $(date -u +%H:%M:%S) Experiment C (mechanism test) ==="
python3 experiment_path_count.py
echo "=== $(date -u +%H:%M:%S) Experiment B (adds N=16, N=32) ==="
python3 experiment_array_size.py
echo "=== $(date -u +%H:%M:%S) ALL SWEEPS DONE ==="
# Re-running is cheap and idempotent: completed points are skipped. Always
# confirm completeness rather than trusting that the stages ran to the end.
python3 - <<'EOF'
import glob, os, sys, numpy as np, config as cfg

def count(path):
    return np.load(path)["trial"].size if os.path.exists(path) else 0

miss = []

# --- Experiments A and B: every (N, SNR) point on the grid -----------------
# Expected budget per N, matching experiment_array_size.points().
want = {cfg.N_DEFAULT: int(os.environ.get("N_TRIALS", cfg.N_TRIALS)),
        16: int(os.environ.get("N_TRIALS_LARGE", cfg.N_TRIALS_LARGE)),
        32: cfg.N_TRIALS_N32}
for N in cfg.N_GRID:
    need = want.get(N, cfg.N_TRIALS_LARGE)
    for snr in cfg.SNR_GRID_DB:
        f = f"results/grid/N{N:02d}_P{cfg.P_DEFAULT}_snr{snr:+05.1f}.npz"
        n = count(f)
        if n == 0:
            miss.append(f"A/B N={N} SNR={snr:+.0f}: MISSING")
        elif n < need:
            miss.append(f"A/B N={N} SNR={snr:+.0f}: {n}/{need}")

# --- Experiment C: every L on the path-count sweep -------------------------
needC = int(os.environ.get("N_TRIALS", cfg.N_TRIALS_PATH))
for L in cfg.L_GRID:
    f = f"results/pathcount/L{L:02d}.npz"
    n = count(f)
    if n == 0:
        miss.append(f"C L={L}: MISSING")
    elif n < needC:
        miss.append(f"C L={L}: {n}/{needC}")

if miss:
    print(f"INCOMPLETE — {len(miss)} point(s):")
    for m in miss:
        print("  " + m)
    sys.exit(1)
n_pts = len(cfg.N_GRID) * len(cfg.SNR_GRID_DB) + len(cfg.L_GRID)
print(f"all {n_pts} points complete at the expected trial counts")
EOF
