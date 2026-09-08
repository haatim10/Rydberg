"""PROMPT 15 A1 -- is evaluation deterministic given a checkpoint?

Runs the discriminating tests the brief names, cheapest first, and prints
bitwise comparisons rather than summary statistics, because a difference of
1e-16 and a difference of 0.6 dB have different causes and a rounded table
hides the first.

  1. stage4.evaluate() twice in ONE process, same checkpoints.
  2. The same, but through p12_partC_eval.eval_models(), which is the path that
     produced +0.682 against a stored +0.100.
  3. The two paths against each other, per trial.

Run:  PYTHONPATH=. python3 scratch/p15_a1_probe.py [n_test]
"""
from __future__ import annotations

import sys
from dataclasses import replace

import numpy as np
import torch

from trackD_urformer.config import TrackDConfig
from trackD_urformer.stage4 import HIGH_SNR, PART_C, evaluate

N = int(sys.argv[1]) if len(sys.argv) > 1 else 100


def arr(d, k):
    return np.asarray(d["per_trial_nmse"][k], dtype=np.float64)


def cmp(tag, a, b):
    same = np.array_equal(a, b)
    d = np.abs(a - b)
    print(f"  {tag:<34} bitwise={'YES' if same else 'NO '}  "
          f"max|diff|={d.max():.3e}  n_diff={(d > 0).sum()}/{len(a)}")
    return same


def main():
    cfg = TrackDConfig()
    print(f"system.P={cfg.system.P}  n_test(cfg)={cfg.data.n_test}  "
          f"master_seed={cfg.system.master_seed}")
    print(f"probe n_test={N}, HIGH_SNR={HIGH_SNR}, arms={PART_C}\n")

    print("TEST 1 -- stage4.evaluate() twice in one process")
    r1 = evaluate(cfg, PART_C, HIGH_SNR, N, False)
    r2 = evaluate(cfg, PART_C, HIGH_SNR, N, False)
    ok = True
    for k in PART_C:
        ok &= cmp(f"stage4 run1 vs run2: {k}", arr(r1, k), arr(r2, k))
    ok &= cmp("stage4 run1 vs run2: snr_db",
              np.asarray(r1["snr_db"]), np.asarray(r2["snr_db"]))
    print(f"  => stage4.evaluate is {'DETERMINISTIC' if ok else 'NOT deterministic'}"
          " in-process\n")

    print("TEST 2 -- p12_partC_eval.eval_models() on the SAME checkpoints")
    from pathlib import Path
    from scratch.p12_partC_eval import eval_models, load_path
    S4 = Path("results/track_d/stage4")
    models = {
        "C_U1_snr5_20": load_path(S4 / "C_U1_snr5_20" / "best.pt", hankel=False),
        "C_H1_snr5_20": load_path(S4 / "C_H1_snr5_20" / "best.pt", hankel=True),
    }
    p1, s1 = eval_models(models, HIGH_SNR, n_test=N)
    p2, s2 = eval_models(models, HIGH_SNR, n_test=N)
    ok2 = True
    for k in PART_C:
        ok2 &= cmp(f"partC run1 vs run2: {k}",
                   np.asarray(p1[k]), np.asarray(p2[k]))
    ok2 &= cmp("partC run1 vs run2: snr_db", s1, s2)
    print(f"  => eval_models is {'DETERMINISTIC' if ok2 else 'NOT deterministic'}"
          " in-process\n")

    print("TEST 3 -- stage4.evaluate() vs eval_models(), same checkpoints")
    cmp("snr_db", np.asarray(r1["snr_db"]), s1)
    for k in PART_C:
        cmp(f"per-trial NMSE: {k}", arr(r1, k), np.asarray(p1[k]))

    # The contrast each path reports, on the same 100 worlds.
    def med(per_u, per_h):
        u = 10 * np.log10(np.asarray(per_u))
        h = 10 * np.log10(np.asarray(per_h))
        return float(np.median(u - h))
    print(f"\n  stage4 path   U1-H1 median = {med(arr(r1, PART_C[0]), arr(r1, PART_C[1])):+.4f} dB")
    print(f"  eval_models   U1-H1 median = {med(p1[PART_C[0]], p1[PART_C[1]]):+.4f} dB")
    print(f"  (stored stage-4 value over 2000 trials: +0.0778 dB)")

    print(f"\ntorch {torch.__version__}  threads={torch.get_num_threads()}")


if __name__ == "__main__":
    main()
