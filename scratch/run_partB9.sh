#!/usr/bin/env bash
# Wait for the stage-5 training runs to release the cores, then sweep Part B
# across four shards of the priority-ordered cell list.
set -u
cd /home/user/Rydberg
export PYTHONPATH=. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
while pgrep -f "trackD_urformer.stage5" > /dev/null; do sleep 60; done
echo "stage-5 done, starting Part B: $(date -u +%H:%M:%SZ)"
for s in 0 1 2 3; do
  python3 scratch/trackD_partB9_sweeps.py --group all --shard "$s" --n-shards 4 \
    > "logs/partB9_shard$s.log" 2>&1 &
done
wait
echo "Part B complete: $(date -u +%H:%M:%SZ)"
