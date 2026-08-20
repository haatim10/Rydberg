"""Track-B structure diagnostic summary for N = 8, 16, 32.

Adds what the first smoke run did not record: true L_k, effective degrees
of freedom / structural redundancy, convergence behaviour, and outlier
behaviour. Identical CRN worlds across all three estimators.
"""
import json
from pathlib import Path

import numpy as np

from rydberg_sim.gs import biased_gs_channel_rows, em_gs_channel_rows
from rydberg_sim.metrics import channel_nmse
from rydberg_sim.monte_carlo import generate_channel_estimation_trial
from rydberg_sim.track_b_drivers import TRACK_B_RSR_DB, draw_L_k, track_b_spec
from rydberg_sim.track_b_proposed import hankel_rank_cap, hs_gs_auto, magnitude_residual

NT = 60
K = 3
OUT = Path(__file__).resolve().parent.parent / "results" / "track_b"
POINTS = ((10, 0.0), (10, 10.0), (30, 0.0), (30, 10.0), (30, 20.0))


def world(t, P, snr, N):
    L = draw_L_k(t, K)
    sp = track_b_spec(P=P, n_trials=t + 1, N=N, K=K, L=L, experiment="tb_diag")
    return generate_channel_estimation_trial(sp, t, float(snr), TRACK_B_RSR_DB)


def run(N, P, snr, n_trials=NT):
    per = {k: [] for k in ("biased_gs", "em_gs", "hs_gs")}
    acc = {k: 0.0 for k in per}
    tot = 0.0
    trueL, Lhat, active, conv, mono = [], [], [], [], []
    for t in range(n_trials):
        w = world(t, P, snr, N)
        tr = float(np.linalg.norm(w.G, ord="fro") ** 2)
        tot += tr
        out = {
            "biased_gs": biased_gs_channel_rows(w.S, w.Z, w.B, max_iter=50).G_hat,
            "em_gs": em_gs_channel_rows(w.S, w.Z, w.B, w.sigma2, max_iter=50).G_hat,
        }
        r = hs_gs_auto(w.S, w.Z, w.B, w.sigma2, max_iter=50, select_iter=20)
        out["hs_gs"] = r.G_hat
        for k, G in out.items():
            e = channel_nmse(G, w.G).error_energy
            acc[k] += e
            per[k].append(e / tr)
        trueL.append(float(np.mean(w.L_k)))
        Lhat.append(r.L_hat)
        active.append(bool(r.constraint_active))
        h = r.residual_history
        conv.append(float(h[-1] / h[0]))
        mono.append(bool(np.all(np.diff(h) <= 1e-6 * max(1.0, abs(h[0])))))
    per = {k: np.asarray(v) for k, v in per.items()}
    sumL = float(np.mean(trueL)) * K
    return dict(
        N=N, P=P, snr=snr,
        pooled={k: 10 * np.log10(acc[k] / tot) for k in acc},
        median={k: 10 * np.log10(np.median(per[k])) for k in per},
        p90={k: 10 * np.log10(np.percentile(per[k], 90)) for k in per},
        worst_share={k: float(per[k].max() / per[k].sum()) for k in per},
        gain_vs_emgs=10 * np.log10(acc["em_gs"] / acc["hs_gs"]),
        frac_hs_better=float(np.mean(per["hs_gs"] < per["em_gs"])),
        mean_true_L=float(np.mean(trueL)),
        mean_Lhat=float(np.mean(Lhat)),
        active_frac=float(np.mean(active)),
        rank_cap=hankel_rank_cap(N),
        dof_unstructured=2 * N * K,
        dof_geometric=3 * sumL,
        redundancy=2 * N * K / (3 * sumL),
        resid_ratio=float(np.median(conv)),
        monotone_frac=float(np.mean(mono)),
    )


if __name__ == "__main__":
    rows = []
    for N in (8, 16, 32):
        print(f"\n=== N = {N}  (rank cap {hankel_rank_cap(N)}) ===")
        print(f"{'P':>3}{'SNR':>6} | {'GS':>7} {'EM-GS':>7} {'HS-GS':>7} {'gain':>6} "
              f"{'better':>7} | {'trueL':>6} {'Lhat':>5} {'act':>5} {'redund':>7} "
              f"{'J50/J1':>7} {'mono':>5} | {'HS p90':>7} {'HSworst%':>8}")
        for P, snr in POINTS:
            r = run(N, P, snr)
            rows.append(r)
            print(f"{P:3d}{snr:6.1f} | {r['pooled']['biased_gs']:7.2f} "
                  f"{r['pooled']['em_gs']:7.2f} {r['pooled']['hs_gs']:7.2f} "
                  f"{r['gain_vs_emgs']:+6.2f} {r['frac_hs_better']:7.0%} | "
                  f"{r['mean_true_L']:6.2f} {r['mean_Lhat']:5.2f} "
                  f"{r['active_frac']:5.0%} {r['redundancy']:7.2f} "
                  f"{r['resid_ratio']:7.3f} {r['monotone_frac']:5.0%} | "
                  f"{r['p90']['hs_gs']:7.2f} {r['worst_share']['hs_gs']:8.1%}")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "structure_diagnostic.json").write_text(json.dumps(rows, indent=2))
    print(f"\nwrote {OUT/'structure_diagnostic.json'}")
