"""PROMPT 22 Part D1 -- how different are the two Cadzow placements, per trial?

The manuscript reports that interleaved and post-hoc placement differ by
0.003 dB in the mean and concludes the placement does not matter. That number
establishes only that two MEANS nearly coincide. Two estimators can produce
visibly different estimates and still land on the same mean squared error, and
the six-point mean here is an average of values running from -0.164 to +0.168
which happen to cancel.

This re-runs exactly the B5 cell of scratch/p12_partB.py -- same worlds, same
seeds, same shared L_hat, same operator -- and additionally records, per trial,

    q_i = || G_interleaved - G_posthoc ||_F / || G_true ||_F

so the question "are these the same estimate?" is answered directly rather than
inferred from a difference of means. Per-trial dB differences are stored too,
so a paired interval on the six-point mean can be formed instead of averaging
six medians.

Nothing about the estimator changes. This is the SAME measurement with two more
quantities written down.

Run:  PYTHONPATH=. python3 scratch/p22_partD_placement.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from rydberg_sim.track_b_proposed import (cadzow_project, hs_gs,
                                          select_order_heldout)
from scratch.p12_partB import (B3_SNR, TD_KW, boot_ci_median, em_gs_only,
                               nmse_parts, trackb_world)

OUT = REPO / "results/p22"
N_TRIALS = 300          # identical to the committed B5 cell
SEED_BASE = 610_000     # identical to the committed B5 cell


def one_snr(snr):
    t0 = time.time()
    num_i, num_p, den, q, lhat = [], [], [], [], []
    for t in range(N_TRIALS):
        w = trackb_world(SEED_BASE + t, N=32, P=30, snr_db=snr, rsr_db=12.0)
        G = np.asarray(w.G)
        # One shared L_hat, so the contrast isolates PLACEMENT and nothing else.
        L_hat, _ = select_order_heldout(
            w.S, w.Z, w.B, w.sigma2, exact_step="em_gs",
            max_iter=TD_KW["select_iter"], cadzow_iter=4)
        inter = hs_gs(w.S, w.Z, w.B, w.sigma2, L_hat=L_hat,
                      exact_step="em_gs", max_iter=TD_KW["max_iter"],
                      project_every=1, cadzow_iter=4).G_hat
        base = em_gs_only(w, TD_KW["max_iter"])
        post = np.array(base, copy=True)
        for k in range(post.shape[1]):
            post[:, k] = cadzow_project(post[:, k], L_hat, n_iter=1)

        a, b = nmse_parts(inter, G)
        num_i.append(a)
        den.append(b)
        num_p.append(nmse_parts(post, G)[0])
        # THE NEW QUANTITY: how far apart are the two ESTIMATES themselves,
        # normalised by the channel they are both estimating.
        q.append(float(np.linalg.norm(inter - post, "fro")
                       / np.linalg.norm(G, "fro")))
        lhat.append(int(L_hat) if np.isscalar(L_hat) else int(np.mean(L_hat)))

    num_i = np.asarray(num_i)
    num_p = np.asarray(num_p)
    den = np.asarray(den)
    d = 10 * np.log10(num_p / den) - 10 * np.log10(num_i / den)
    ci = boot_ci_median(d)
    qa = np.asarray(q)
    return {
        "snr_db": snr, "n": N_TRIALS, "seconds": round(time.time() - t0, 1),
        "interleaved_minus_posthoc_db": float(np.median(d)),
        "boot_ci95_median": list(ci),
        "ci_excludes_zero": bool(ci[0] > 0 or ci[1] < 0),
        "per_trial_db": [float(v) for v in d],
        "q_median": float(np.median(qa)),
        "q_mean": float(qa.mean()),
        "q_p75": float(np.percentile(qa, 75)),
        "q_p90": float(np.percentile(qa, 90)),
        "q_p99": float(np.percentile(qa, 99)),
        "q_max": float(qa.max()),
        "per_trial_q": [float(v) for v in qa],
        "mean_L_hat": float(np.mean(lhat)),
    }


def main():
    import multiprocessing as mp
    OUT.mkdir(parents=True, exist_ok=True)
    procs = int(os.environ.get("CC_PROCS", "6"))
    with mp.Pool(procs) as pool:
        rows = pool.map(one_snr, list(B3_SNR))
    rows.sort(key=lambda r: r["snr_db"])

    # Paired interval on the six-point mean, over the pooled per-trial values.
    allv = np.concatenate([np.asarray(r["per_trial_db"]) for r in rows])
    allq = np.concatenate([np.asarray(r["per_trial_q"]) for r in rows])
    pooled_ci = boot_ci_median(allv)
    six = np.array([r["interleaved_minus_posthoc_db"] for r in rows])
    from scipy import stats
    half = float(stats.t.ppf(0.975, six.size - 1)) * six.std(ddof=1) / np.sqrt(six.size)

    out = {
        "cell": "P22_D1_placement_divergence",
        "config": {"N": 32, "K": 3, "P": 30, "rsr_db": 12.0,
                   "n_trials": N_TRIALS, "seed_base": SEED_BASE,
                   "operator": TD_KW,
                   "note": ("positive = interleaved better; shared L_hat. "
                            "Same worlds and seeds as the committed B5 cell "
                            "of scratch/p12_partB.py.")},
        "points": rows,
        "six_point_mean_db": float(six.mean()),
        "six_point_mean_ci95_over_points": [float(six.mean() - half),
                                            float(six.mean() + half)],
        "pooled_per_trial_median_db": float(np.median(allv)),
        "pooled_per_trial_ci95": list(pooled_ci),
        "q_pooled": {
            "median": float(np.median(allq)),
            "p75": float(np.percentile(allq, 75)),
            "p90": float(np.percentile(allq, 90)),
            "p99": float(np.percentile(allq, 99)),
            "max": float(allq.max()),
            "n": int(allq.size),
        },
    }
    (OUT / "p22_d1_placement.json").write_text(json.dumps(out, indent=2) + "\n")

    print(f"{'SNR':>6}{'n':>5}{'inter-post dB':>15}{'CI95':>22}"
          f"{'q med':>9}{'q p90':>9}{'q max':>9}")
    for r in rows:
        print(f"{r['snr_db']:+6.1f}{r['n']:5d}"
              f"{r['interleaved_minus_posthoc_db']:+15.4f}"
              f"  [{r['boot_ci95_median'][0]:+.4f},{r['boot_ci95_median'][1]:+.4f}]"
              f"{r['q_median']:9.4f}{r['q_p90']:9.4f}{r['q_max']:9.4f}")
    print()
    print("six-point mean %+.4f dB, 95%% CI over the six points [%+.4f, %+.4f]"
          % (out["six_point_mean_db"], *out["six_point_mean_ci95_over_points"]))
    print("pooled per-trial median %+.4f dB, CI [%+.4f, %+.4f]"
          % (out["pooled_per_trial_median_db"], *out["pooled_per_trial_ci95"]))
    print(f"\nwrote {OUT/'p22_d1_placement.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
