#!/usr/bin/env bash
# One worker: its share of the SNR grid, then its share of the pilot grid.
set -eu
cd /home/user/Rydberg
export PYTHONPATH=. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python3 scratch/trackD_sweeps.py --mode snr    --shard "$1" --n-shards 4 --n-trials 400
python3 scratch/trackD_sweeps.py --mode pilots --shard "$1" --n-shards 4 --n-trials 400
echo "worker $1 done"
