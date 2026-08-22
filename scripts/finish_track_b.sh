#!/usr/bin/env bash
# Everything that must happen once the last extended point lands.
# Idempotent: safe to re-run after an interruption.
set -euo pipefail
cd /home/user/rydberg-trackb
export PYTHONPATH=.

# 1. make sure every flagged point actually reached its budget
python3 scripts/run_extend.py results/track_b/b3

# 2. aggregate both experiments from the raw per-trial stores
python3 scripts/analyze_b3.py results/track_b/b3 > logs/b3_summary.txt 2>&1
python3 scripts/analyze_b3.py results/track_b/b4 > logs/b4_summary.txt 2>&1

# 3. re-apply the adaptive rule to record which CIs actually resolved
python3 scripts/extend_rule.py results/track_b/b3 > logs/b3_rule_final.txt 2>&1

# 4. interpretation tests A-H and the B5 scaling summary
python3 scripts/interpret_b3.py > logs/interpretation.txt 2>&1
python3 scripts/b5_scaling.py  > logs/b5_scaling.txt 2>&1

# 5. figures (Track-B worktree only; never the frozen Track-A tree)
python3 scripts/plot_b3_b4_b5.py > logs/plots.txt 2>&1

echo "FINISH PIPELINE COMPLETE"
