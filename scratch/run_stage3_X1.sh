#!/usr/bin/env bash
# Chain the remaining Part B work so there is no idle gap behind H1.
#
#   1. wait for the three EM-GS train shards (~2.5 h, already running)
#   2. build the val EM-GS cache (n_shards=1, ~11 min, matches stage3's loader)
#   3. train X1
#
# H1 runs on its own core throughout; this uses one more, leaving two free.
set -euo pipefail
cd /home/user/Rydberg
export PYTHONPATH=. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

C=results/track_d/emgs_cache
until [ -f "$C/train_n80000_T100_shard0of3.npy" ] \
   && [ -f "$C/train_n80000_T100_shard1of3.npy" ] \
   && [ -f "$C/train_n80000_T100_shard2of3.npy" ]; do
  if ! pgrep -f "emgs_cache --split train" > /dev/null; then
    echo "FATAL: train precompute died before all shards were written" >&2
    exit 1
  fi
  sleep 60
done
echo "train shards complete: $(date -u +%H:%M:%SZ)"

# n_shards=1 so the filename matches stage3.train_run's loader exactly.
python3 -m trackD_urformer.emgs_cache --split val --n 2000 --shard 0 \
        --n-shards 1 --T-GS 100
echo "val cache complete: $(date -u +%H:%M:%SZ)"

python3 -m trackD_urformer.stage3 --i-have-approval --runs X1_emgs_plus_former
echo "X1 complete: $(date -u +%H:%M:%SZ)"
