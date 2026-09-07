#!/usr/bin/env bash
# PROMPT 12 job queue. Keeps at most MAXJOBS of my jobs running, launching the
# remaining Part B item (B4) and then the Part C training runs in the brief's
# priority order as slots free up.
#
# Training is single-threaded by config (trackD_urformer/config.py:322,
# num_threads = 1), and the historical runs cost 7900-11100 s each at that
# setting, so four concurrent runs on four cores go at close to full speed.
#
# Every job checkpoints per epoch and skips if already complete, so this is
# safe to re-run.
set -u
cd /home/user/Rydberg
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.

MAXJOBS=4

running() { pgrep -cf "scratch/p12_part[BC]" || true; }

wait_slot() {
  while [ "$(running)" -ge "$MAXJOBS" ]; do sleep 20; done
}

launch() {  # launch <logname> <command...>
  local log="logs/p12_$1.log"; shift
  wait_slot
  echo "$(date -u +%H:%M:%S) launching $log" >> logs/p12_queue.log
  nohup "$@" > "$log" 2>&1 &
  sleep 5
}

# Remaining Part B item.
if [ ! -f results/p12/B4.json ]; then
  launch B4 python3 scratch/p12_partB4.py
fi

# B2 addendum: the forced-order diagnostic. B2 established that at N=16 the
# selector abstains in essentially every trial once rho is high, so Delta_HS is
# identically zero and the crossing cannot be bracketed from the negative side.
# This separates "the effect is absent" from "abstention hides it".
if [ ! -f results/p12/B2_forced.json ]; then
  launch B2forced python3 scratch/p12_partB2_forced.py
fi

# Part C, in the brief's priority order: the focused pair carries the small
# number (+0.078 dB) and is where seed noise could be fatal, so it goes first.
for run in C1_U1_seed2 C1_H1_seed2 C1_U1_seed3 C1_H1_seed3 \
           C2_H1_balanced C3_U1_logdb \
           C1x_U1_mixed_seed2 C1x_H1_mixed_seed2; do
  if [ ! -f "results/p12/partC/$run/result.json" ]; then
    launch "$run" python3 scratch/p12_partC.py --run "$run"
  fi
done

echo "$(date -u +%H:%M:%S) all jobs launched; waiting for completion" >> logs/p12_queue.log
wait
echo "$(date -u +%H:%M:%S) QUEUE COMPLETE" >> logs/p12_queue.log
