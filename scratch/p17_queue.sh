#!/usr/bin/env bash
# PROMPT 17 Part C batch launcher. Takes ONE batch name and runs only that
# batch, so the gate between batches stays where the pre-registration puts it:
# C2 is not launched by a script, it is launched after C1 has been scored.
#
#   scratch/p17_queue.sh C1
#
# Training is single-threaded by config (trackD_urformer/config.py, num_threads
# = 1) and Tier 0.5 cost 8800-10700 s per run at four concurrent jobs on four
# cores, so a four-run batch is about three hours wall clock.
#
# Every run checkpoints per epoch and resumes, and a run whose result.json
# already exists is skipped, so this is safe to re-run.
set -u
cd /home/user/Rydberg
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.

BATCH="${1:?usage: p17_queue.sh <C1|C2|C3|C4>}"
MAXJOBS=4
LOG=logs/p17_queue.log
mkdir -p logs results/p17/tier1

running() { pgrep -cf "scratch/p17_tier1.py --run" || true; }

for run in $(python3 scratch/p17_tier1.py --list-batch "$BATCH"); do
  if [ -f "results/p17/tier1/$run/result.json" ]; then
    echo "$(date -u +%H:%M:%S) $BATCH $run already complete, skipping" >> "$LOG"
    continue
  fi
  while [ "$(running)" -ge "$MAXJOBS" ]; do sleep 20; done
  echo "$(date -u +%H:%M:%S) launching $run" >> "$LOG"
  nohup python3 scratch/p17_tier1.py --run "$run" > "logs/p17_$run.log" 2>&1 &
  sleep 5
done

echo "$(date -u +%H:%M:%S) batch $BATCH launched; waiting" >> "$LOG"
wait
echo "$(date -u +%H:%M:%S) BATCH $BATCH COMPLETE" >> "$LOG"
