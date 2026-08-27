#!/usr/bin/env bash
# Track D (URformer) - full pipeline. SCAFFOLD ONLY: NOT RUN in the build phase.
#
# PROMPT 2 sec. 11 requires this to exist and stay unexecuted. Every stage
# refuses without TRACKD_APPROVED=1, so an accidental invocation cannot start a
# multi-hour job.
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH="${PYTHONPATH:-.}"

if [[ "${TRACKD_APPROVED:-0}" != "1" ]]; then
  cat <<'MSG'
REFUSING to run: Track D training and experiments require explicit approval.

This script is scaffolding built during the implementation phase and left
deliberately unrun. Nothing here has been executed.

To run once approved:
    TRACKD_APPROVED=1 ./trackD_urformer/run_all.sh

To inspect the plan without running anything:
    PYTHONPATH=. python3 -m trackD_urformer.runner plan
MSG
  exit 2
fi

# Single-instance lock: two concurrent runs would corrupt the result store.
LOCK=/tmp/trackD_urformer.lock
exec 9>"$LOCK"
flock -n 9 || { echo "another Track D run holds $LOCK; aborting"; exit 1; }

echo "== verification gates =="
python3 -m trackD_urformer.verify

echo "== training matrix =="
python3 -m trackD_urformer.runner plan

echo "== D1 / D2 / D3 =="
python3 -m trackD_urformer.runner d1 --i-have-approval
python3 -m trackD_urformer.runner d2 --i-have-approval
python3 -m trackD_urformer.runner d3 --i-have-approval

echo "done"
