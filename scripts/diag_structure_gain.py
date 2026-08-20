"""Does ULA structure help at all? Direct projection diagnostic.

Isolates the question from the alternating scheme: apply each projection
directly to a converged exact estimate and see whether NMSE_G improves.
Also projects the TRUE G as an upper bound on what the projection can do.
"""
from __future__ import annotations

import numpy as np

from rydberg_sim.config import SimulationConfig
from rydberg_sim.gs import em_gs_channel_rows
from rydberg_sim.metrics import channel_nmse
from rydberg_sim.monte_carlo import ExperimentSpec, generate_channel_estimation_trial
from rydberg_sim.track_b_structure import project_matrix

METHODS = ("hankel", "angular", "esprit")


def spec(N=16, K=3, P=32, L=(3, 5, 7), seed=20250820):
    cfg = SimulationConfig.create(N=N, K=K, L=L, beta=1.0, master_seed=seed, c=1.0)
    return ExperimentSpec(
        experiment="track_b_diag", track="B", cfg=cfg, P=P, vartheta=0.0,
        snr_db_grid=(0.0,), rsr_db_grid=(30.0,), n_trials=1,
        algorithms=("em_gs",), max_iter=50, qam_M=4,
        channel_model="ula_geometric", write_ber=False,
    )


def db(e, t):
    return 10 * np.log10(e / t)


def run(snr, P, n_paths, n_trials=12):
    sp = spec(P=P)
    acc = {k: 0.0 for k in ("raw", *METHODS)}
    acc.update({f"oracle_{m}": 0.0 for m in METHODS})
    tot = 0.0
    for t in range(n_trials):
        w = generate_channel_estimation_trial(sp, t, snr, 30.0)
        Gh = em_gs_channel_rows(w.S, w.Z, w.B, w.sigma2, max_iter=50).G_hat
        r = channel_nmse(Gh, w.G)
        acc["raw"] += r.error_energy
        tot += r.true_energy
        for m in METHODS:
            acc[m] += channel_nmse(
                project_matrix(Gh, m, n_paths), w.G).error_energy
            # upper bound: project the TRUE channel (uses truth -- diagnostic only)
            acc[f"oracle_{m}"] += channel_nmse(
                project_matrix(w.G, m, n_paths), w.G).error_energy
    return {k: db(v, tot) for k, v in acc.items()}


print("Direct projection of a converged EM-GS estimate (N=16, K=3, L=(3,5,7))")
print("'oracle_*' projects the TRUE G -- the best the projection could ever do.\n")
for P in (32, 64):
    for snr in (0.0, 10.0, 20.0, 30.0):
        r = run(snr, P, n_paths=5)
        print(f"P={P:3d} SNR={snr:+5.1f} | raw={r['raw']:7.2f} | "
              + " ".join(f"{m}={r[m]:7.2f}" for m in METHODS))
        print(f"{'':17s}| oracle:      "
              + " ".join(f"{m}={r['oracle_'+m]:7.2f}" for m in METHODS))
print()
print("Effect of the assumed path count (P=64, SNR=20 dB, true L=(3,5,7)):")
for npath in (3, 5, 7, 9, 12):
    r = run(20.0, 64, n_paths=npath)
    print(f"  n_paths={npath:2d} | raw={r['raw']:7.2f} | "
          + " ".join(f"{m}={r[m]:7.2f}" for m in METHODS))
