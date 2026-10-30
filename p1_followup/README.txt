Focused second-review checks (8 October 2026)

Preserves p1_revision, the historical implementation, and all P2 files.
The main campaign freezes its protocol and source hashes before simulation.
600 independent worlds: L=3,7,14 crossed with SNR=0,10 dB, 100 each.
N32/K3/P30/RSR12; 25 candidate and 100 final updates; full-pilot refitting.

Sweep sensitivity: Q=1,2,4,8 for hard and smooth processing. Each setting
selects its own rank or factor. This is sensitivity of the complete tuned
procedure, not a same-rank projection-placement test. The smooth-filter
threshold scale is recomputed on every sweep, as in the historical code.

Split stability: fixed trailing 30% and three independently drawn random
30% subsets per world, four sweeps, hard rank only. They are alternative
single-split runs, not a new multi-fold selection method. Final fitting uses
original-order full pilots and the same initializer. When repeated masks
select the same rank, the deterministic final-fit error is reused. This is
only an experiment cache, not a claimed deployment speedup.

Intervals resample worlds 4000 times. For random-vs-fixed performance,
first average squared errors over the three random partitions within each
world, then compare ratio-of-sums errors and bootstrap worlds. Do not count
three random partitions as three independent trials. Rank-disagreement rates
and within-world ranges are descriptive. No simultaneous coverage claimed.

Historical rank/harm analysis reuses 2400 old baseline worlds. The rank is
shared across users, and these cells have the same L for every user.
Below-true-rank selection is a diagnostic, not automatically a mistake.

Timing is a separate serial experiment on 12 new worlds at L7/SNR5 with
all eight hard/smooth sweep settings, randomized order within each world.
It includes selection and initialization. Four-worker campaign times are
not inference benchmarks. Timing assumes a quiet machine, but no isolation
from unrelated host activity is guaranteed.

Commands from repo root, with numpy/scipy/matplotlib available:
Set OPENBLAS_NUM_THREADS, OMP_NUM_THREADS, MKL_NUM_THREADS,
VECLIB_MAXIMUM_THREADS to 1 before Python starts. Use a writable MPLCONFIGDIR.
python -m unittest discover -s p1_followup -t . -p 'test_*.py'
python -m p1_followup.run --preflight
python -m p1_followup.run
python -m p1_followup.analyze
python -m p1_followup.timing

The runner resumes matching saved trials; mismatched sources are rejected.
Analysis writes new summaries; preserve originals before reanalysis.
Timing refuses to overwrite completed raw timing rows. Its protocol is saved
before collecting times. edit_paper.py is a one-time editorial patch against
the preserved pre-follow-up manuscript, not an idempotent replay command.
