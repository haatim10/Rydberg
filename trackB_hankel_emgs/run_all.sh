#!/usr/bin/env bash
# Full Monte Carlo sweep, ordered so the decisive experiments finish first:
#   A  (N=8 grid)   -- cheap, and is also experiment B's N=8 column
#   C  (path count) -- the mechanism test; runs before the expensive N=32 grid
#   B  (full grid)  -- adds N=16 then N=32
# All points are checkpointed every 50 trials and resume where they left off,
# so this script is safe to interrupt and re-run.
set -u
cd "$(dirname "$0")"
echo "=== $(date -u +%H:%M:%S) Experiment A (N=8 grid; also B's N=8 column) ==="
python3 experiment_snr.py
echo "=== $(date -u +%H:%M:%S) Experiment C (mechanism test) ==="
python3 experiment_path_count.py
echo "=== $(date -u +%H:%M:%S) Experiment B (adds N=16, N=32) ==="
python3 experiment_array_size.py
echo "=== $(date -u +%H:%M:%S) ALL SWEEPS DONE ==="
