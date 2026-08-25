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
import glob, numpy as np, config as cfg
miss = []
for f in glob.glob("results/pathcount/L*.npz"):
    n = np.load(f)["trial"].size
    if n < cfg.N_TRIALS_PATH: miss.append(f"{f}: {n}/{cfg.N_TRIALS_PATH}")
have = {int(f.split("L")[-1][:2]) for f in glob.glob("results/pathcount/L*.npz")}
for L in cfg.L_GRID:
    if L not in have: miss.append(f"experiment C L={L}: MISSING")
print("INCOMPLETE:" if miss else "all experiment-C points complete")
for m in miss: print("  " + m)
EOF
