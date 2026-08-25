"""Mechanism diagnostic: Hankel singular values, true vs EM-GS vs Hankel-EM-GS.

ONE representative example, fully specified so it can be reproduced and is
demonstrably not cherry-picked:

    N = 32, K = 3, P = 30, SNR = 5 dB, RSR = 12 dB
    trial index 0, master seed 20250820 (config.MASTER_SEED)
    L = 3 for all users, column k = 0
    L_hat forced to the true L = 3 so the figure shows what the projection
    does at the correct rank, not what the selector happens to pick

The trial index is 0 -- the first trial of the sweep -- chosen before looking
at any spectrum. Change SEED_TRIAL below to inspect any other realisation;
diagnostic_spectrum.json records exactly which one produced the figure.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import config as cfg
import em_gs
import hankel_projection as hp
from system_model import make_world

SEED_TRIAL = 0
N, L, K_COL, SNR = 32, 3, 0, 5.0
RES = Path(__file__).resolve().parent / "results"


def main() -> None:
    w = make_world(SEED_TRIAL, N=N, P=cfg.P_DEFAULT, snr_db=SNR, L=L)
    G_em = em_gs.em_gs(w.S, w.Z, w.B, w.sigma2)
    g_true = np.asarray(w.G[:, K_COL])
    g_em = np.asarray(G_em[:, K_COL])
    g_cad = hp.project(g_em, L, n_iter=cfg.CADZOW_ITER)

    norm = lambda g: (lambda s: s / s[0])(hp.singular_values(g))
    out = {
        "config": dict(N=N, K=cfg.K, P=cfg.P_DEFAULT, L=L, column=K_COL,
                       snr_db=SNR, rsr_db=cfg.RSR_DB, trial=SEED_TRIAL,
                       master_seed=cfg.MASTER_SEED,
                       cadzow_iter=cfg.CADZOW_ITER, L_hat="forced to true L",
                       hankel_shape=list(hp.lift(g_true).shape),
                       r_max=hp.rank_cap(N)),
        "true_channel": norm(g_true).tolist(),
        "em_gs_estimate": norm(g_em).tolist(),
        "after_cadzow": norm(g_cad).tolist(),
    }
    RES.mkdir(parents=True, exist_ok=True)
    (RES / "diagnostic_spectrum.json").write_text(json.dumps(out, indent=1))

    t, e, c = norm(g_true), norm(g_em), norm(g_cad)
    print(f"N={N} L={L} trial={SEED_TRIAL} SNR={SNR} dB  Hankel {hp.lift(g_true).shape}")
    print(f"  true    sv[{L}]/sv[0] = {t[L]:.3e}   <- exact low rank")
    print(f"  EM-GS   sv[{L}]/sv[0] = {e[L]:.3e}   <- noise fills the spectrum")
    print(f"  Cadzow  sv[{L}]/sv[0] = {c[L]:.3e}   <- cliff restored, approximately")
    print(f"  wrote {RES / 'diagnostic_spectrum.json'}")


if __name__ == "__main__":
    main()
