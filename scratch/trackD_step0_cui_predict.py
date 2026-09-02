"""PROMPT 10 Step 0, PART ONE: predict Delta_HS on the Cui channel.

Protocol identical to B6, and run in two separate commits so the prediction is
timestamped BEFORE the measurement exists:

    1. this script      -> r_eff/cap and the predicted Delta_HS  [COMMIT]
    2. the measurement  -> one classical cell, EM-GS vs hs_gs_auto

Channel, exactly as the brief specifies it:

    L = 10 paths per user, alpha_l ~ CN(0,1), incident angles ~ U(-90, 90) deg

NOTE ON THE CITATION, and it must survive into the paper. The brief attributes
this configuration to arXiv:2408.14366v2. That PDF is **not in this repository
and I cannot verify it**. The repository's own Cui reference is a DIFFERENT
paper -- arXiv:2404.04864, "Towards Atomic MIMO Receivers"
(`rydberg_sim/channel_cui.py:1`) -- whose Table I specifies 23 clusters x 20
rays, not L = 10 independent paths. So what is run here is *the configuration
the brief states*, and the paper must cite it as such with the reference marked
unverified rather than asserting it is Cui et al.'s.

Run:  PYTHONPATH=. python3 scratch/trackD_step0_cui_predict.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from rydberg_sim.track_b_proposed import hankel_rank_cap
from rydberg_sim.track_b_structure import hankel_matrix

OUT = Path("reports/trackD_step0_cui_prediction.json")
N, K, L_PATHS = 32, 3, 10
N_COLS = 400
SEED = 20260910

# The A2 re-indexed reference relation: (r_eff/cap, Delta_HS in dB), from
# trackB_hankel_emgs/results/experiment_C_path_count.csv re-indexed in
# reports/trackD_normalization.md. This is the SAME curve B6 was predicted
# from; it is not re-fitted here and must not be re-fitted afterwards.
A2_CURVE = [(0.119, 7.043), (0.212, 3.556), (0.285, 1.792), (0.356, 1.038),
            (0.408, 0.577), (0.460, 0.266), (0.507, 0.046), (0.546, -0.117)]


def eff_rank(g: np.ndarray) -> float:
    """Roy-Vetterli effective rank of the column's Hankel embedding."""
    s = np.linalg.svd(hankel_matrix(g), compute_uv=False)
    s = s[s > 0]
    p = s / s.sum()
    return float(np.exp(-np.sum(p * np.log(np.maximum(p, 1e-300)))))


def cui_column(rng, n_arr):
    """One user column: L independent paths, CN(0,1) gains, U(-90,90) angles."""
    th = rng.uniform(-np.pi / 2, np.pi / 2, L_PATHS)
    a = (rng.standard_normal(L_PATHS)
         + 1j * rng.standard_normal(L_PATHS)) / np.sqrt(2 * L_PATHS)
    return (a[None, :] * np.exp(
        -1j * (np.pi * np.sin(th))[None, :] * n_arr[:, None])).sum(1)


def main() -> int:
    rng = np.random.default_rng(SEED)
    n_arr = np.arange(N)
    vals = [eff_rank(cui_column(rng, n_arr)) for _ in range(N_COLS)]
    r_eff = float(np.median(vals))
    cap = hankel_rank_cap(N)
    x = r_eff / cap
    xs = np.array([a for a, _ in A2_CURVE])
    ys = np.array([b for _, b in A2_CURVE])
    pred = float(np.interp(x, xs, ys))
    # Prediction interval: propagate the interquartile spread of r_eff through
    # the same curve, so the interval reflects channel variability rather than
    # being asserted.
    q1, q3 = np.percentile(vals, [25, 75])
    lo, hi = (float(np.interp(v / cap, xs, ys)) for v in (q3, q1))

    res = {
        "channel": {"source_as_stated_in_brief": "arXiv:2408.14366v2",
                    "citation_verified": False,
                    "note": "PDF not in repository; the repository's own Cui "
                            "reference is arXiv:2404.04864 with a different "
                            "(23 cluster x 20 ray) configuration",
                    "L_paths": L_PATHS, "gains": "CN(0,1)",
                    "angles": "U(-90,90) deg", "N": N, "K": K},
        "n_columns": N_COLS, "seed": SEED,
        "r_eff_median": r_eff, "r_eff_iqr": [float(q1), float(q3)],
        "cap": cap, "r_eff_over_cap": x,
        "PREDICTED_delta_hs_db": pred,
        "prediction_interval_db": [lo, hi],
        "interpolated_from": "A2 re-indexed Track B experiment C; not refitted",
        "falsifier": "measured Delta_HS outside the stated interval, or on the "
                     "wrong side of zero",
    }
    OUT.write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
    print(f"  r_eff (median of {N_COLS} columns) = {r_eff:.3f}"
          f"   IQR [{q1:.3f}, {q3:.3f}]")
    print(f"  cap = floor({N}/2) = {cap}     r_eff/cap = {x:.3f}")
    print(f"  PREDICTED Delta_HS = {pred:+.3f} dB   "
          f"interval [{lo:+.3f}, {hi:+.3f}]")
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
