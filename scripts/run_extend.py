"""Run only the B3 points the adaptive rule flagged, up to EXTEND_TO trials.

Resumes each flagged point from its existing 400-trial checkpoint and
computes only the missing trial indices. Completed trials are never rerun.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import run_b3 as rb3
from extend_rule import EXTEND_TO


def main() -> None:
    import multiprocessing as mp
    store = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "results/track_b/b3"
    rb3.OUT = store
    plan = json.loads((store / "extension_plan.json").read_text())
    jobs = [(r["N"], r["P"], r["snr_db"], EXTEND_TO) for r in plan["extend"]]
    if not jobs:
        print("nothing to extend")
        return
    print(f"extending {len(jobs)} points to {EXTEND_TO} trials", flush=True)
    for r in plan["extend"]:
        print(f"  N={r['N']} P={r['P']} SNR={r['snr_db']:+.1f}: {r['reason']}",
              flush=True)
    with mp.Pool(int(os.environ.get("EXT_PROCS", "4"))) as pool:
        for msg in pool.imap_unordered(rb3.run_point, jobs):
            print(" ", msg, flush=True)
    print("extension done", flush=True)


if __name__ == "__main__":
    main()
