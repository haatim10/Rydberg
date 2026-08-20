"""Smoke comparison: frozen GS/EM-GS baselines vs the proposed HS-GS.

Identical worlds (same CRN) for every method at every point.
"""
import numpy as np

from rydberg_sim.gs import biased_gs_channel_rows, em_gs_channel_rows
from rydberg_sim.metrics import channel_nmse
from rydberg_sim.track_b_drivers import TRACK_B_RSR_DB, draw_L_k, track_b_spec
from rydberg_sim.monte_carlo import generate_channel_estimation_trial
from rydberg_sim.track_b_proposed import (
    hankel_rank_cap,
    hs_gs,
    hs_gs_auto,
    magnitude_residual,
)

NT = 60


def world(t, P, snr, N, K=3):
    L = draw_L_k(t, K)
    sp = track_b_spec(P=P, n_trials=t + 1, N=N, K=K, L=L, experiment="tb_prop")
    return generate_channel_estimation_trial(sp, t, float(snr), TRACK_B_RSR_DB)


def compare(N, P, snr, n_trials=NT, verbose=False):
    acc = {k: 0.0 for k in ("biased_gs", "em_gs", "hs_gs_auto")}
    tot = 0.0
    Ls, trueL, active = [], [], []
    for t in range(n_trials):
        w = world(t, P, snr, N)
        tot += float(np.linalg.norm(w.G, ord="fro") ** 2)
        acc["biased_gs"] += channel_nmse(
            biased_gs_channel_rows(w.S, w.Z, w.B, max_iter=50).G_hat, w.G).error_energy
        acc["em_gs"] += channel_nmse(
            em_gs_channel_rows(w.S, w.Z, w.B, w.sigma2, max_iter=50).G_hat,
            w.G).error_energy
        r = hs_gs_auto(w.S, w.Z, w.B, w.sigma2, max_iter=50, select_iter=20)
        acc["hs_gs_auto"] += channel_nmse(r.G_hat, w.G).error_energy
        Ls.append(r.L_hat)
        trueL.append(float(np.mean(w.L_k)))
        active.append(r.constraint_active)
    db = {k: 10 * np.log10(v / tot) for k, v in acc.items()}
    return db, float(np.mean(Ls)), float(np.mean(trueL)), float(np.mean(active))


print("Frozen baselines vs proposed HS-GS, identical CRN worlds, "
      f"{NT} trials/point, RSR={TRACK_B_RSR_DB} dB, t0=50")
print("NMSE_G (dB), ratio of sums\n")

for N in (8, 16, 32):
    cap = hankel_rank_cap(N)
    print(f"=== N = {N} (Hankel rank cap {cap}; L_k ~ U{{3..7}}) ===")
    print(f"{'P':>4}{'SNR':>6} | {'biased GS':>10} {'EM-GS':>10} {'HS-GS':>10} "
          f"| {'vs EM-GS':>9} {'mean Lhat':>10} {'active':>7}")
    for P, snr in ((10, 0.0), (10, 10.0), (30, 0.0), (30, 10.0), (30, 20.0)):
        db, mL, mtL, act = compare(N, P, snr)
        print(f"{P:4d}{snr:6.1f} | {db['biased_gs']:10.2f} {db['em_gs']:10.2f} "
              f"{db['hs_gs_auto']:10.2f} | {db['em_gs']-db['hs_gs_auto']:+9.2f} "
              f"{mL:10.2f} {act:7.0%}")
    print()

print("Convergence of the residual (N=16, P=30, SNR=10 dB, one world):")
w = world(0, 30, 10.0, 16)
for L in (2, 4, 6, 8):
    r = hs_gs(w.S, w.Z, w.B, w.sigma2, L_hat=L, max_iter=50)
    h = r.residual_history
    nm = 10 * np.log10(channel_nmse(r.G_hat, w.G).nmse_linear)
    print(f"  L={L}: J[1]={h[0]:9.3f} J[10]={h[9]:9.3f} J[50]={h[-1]:9.3f} "
          f"monotone={bool(np.all(np.diff(h) <= 1e-9))} NMSE={nm:7.2f} dB "
          f"active={r.constraint_active}")
gs = em_gs_channel_rows(w.S, w.Z, w.B, w.sigma2, max_iter=50).G_hat
print(f"  EM-GS baseline: J={magnitude_residual(gs, w.S, w.B, w.Z):9.3f} "
      f"NMSE={10*np.log10(channel_nmse(gs, w.G).nmse_linear):7.2f} dB")
