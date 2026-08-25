"""Ablations. Reported SEPARATELY -- these never replace the baseline config.

The baseline configuration (config.py) is inherited from the audited Track-B
implementation and is not tuned in response to any result. This script varies
one setting at a time so the effect of each can be attributed, at a single
operating point, with everything else held fixed.

  1. schedule    -- interleaved (PROJECT_EVERY=1, baseline) vs post-hoc
                    (project once after the last iteration).
  2. oracle rank -- L_hat = true L instead of the held-out selection, to bound
                    how much of the gap is rank misspecification. NOT a
                    deployable estimator; it uses ground truth.
  3. cadzow      -- 1, 4 (baseline), 8 sweeps.

    python ablations.py                 # 200 trials, N=32, SNR=5 dB
    ABL_TRIALS=400 python ablations.py
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

import config as cfg
import em_gs
import hankel_em_gs as hem
from system_model import channel_nmse_parts, make_world

RES = Path(__file__).resolve().parent / "results"
N, P, SNR, L = 32, cfg.P_DEFAULT, cfg.EXP_C_SNR, 4


def main() -> None:
    n = int(os.environ.get("ABL_TRIALS", 200))
    acc: dict[str, list] = {k: [] for k in
                            ("den", "em", "base", "posthoc", "oracle", "cz1", "cz8")}
    lhat: list[int] = []
    for t in range(n):
        w = make_world(t, N=N, P=P, snr_db=SNR, L=L)
        G_em = em_gs.em_gs(w.S, w.Z, w.B, w.sigma2)
        e_em, den = channel_nmse_parts(G_em, w.G)
        acc["den"].append(den); acc["em"].append(e_em)

        base = hem.hankel_em_gs(w.S, w.Z, w.B, w.sigma2)
        lhat.append(base.L_hat)
        acc["base"].append(channel_nmse_parts(base.G_hat, w.G)[0])

        post = hem.hankel_em_gs(w.S, w.Z, w.B, w.sigma2,
                                project_every=cfg.GS_MAX_ITER, L_hat=base.L_hat)
        acc["posthoc"].append(channel_nmse_parts(post.G_hat, w.G)[0])

        orc = hem.hankel_em_gs(w.S, w.Z, w.B, w.sigma2, L_hat=L)
        acc["oracle"].append(channel_nmse_parts(orc.G_hat, w.G)[0])

        for tag, ci in (("cz1", 1), ("cz8", 8)):
            r = hem.hankel_em_gs(w.S, w.Z, w.B, w.sigma2, cadzow_iter=ci,
                                 L_hat=base.L_hat)
            acc[tag].append(channel_nmse_parts(r.G_hat, w.G)[0])

    A = {k: np.asarray(v, float) for k, v in acc.items()}
    db = lambda x: float(10 * np.log10(x.sum() / A["den"].sum()))
    gain = lambda x: float(10 * np.log10(A["em"].sum() / x.sum()))
    rows = [("EM-GS baseline", db(A["em"]), 0.0),
            ("Hankel, interleaved (BASELINE CONFIG)", db(A["base"]), gain(A["base"])),
            ("Hankel, post-hoc (project once at end)", db(A["posthoc"]), gain(A["posthoc"])),
            ("Hankel, ORACLE rank L_hat=L (not deployable)", db(A["oracle"]), gain(A["oracle"])),
            ("Hankel, 1 Cadzow sweep", db(A["cz1"]), gain(A["cz1"])),
            ("Hankel, 8 Cadzow sweeps", db(A["cz8"]), gain(A["cz8"]))]

    print(f"\nABLATIONS  N={N} P={P} SNR={SNR} dB L={L} (fixed), {n} paired trials")
    print(f"  selected rank: mean L_hat = {np.mean(lhat):.2f} vs true L = {L}")
    print(f"{'variant':<45} {'NMSE dB':>9} {'gain dB':>9}")
    for name, d, g in rows:
        print(f"{name:<45} {d:>9.3f} {g:>+9.3f}")
    print("  Baseline config is row 2. The others are diagnostics and do NOT")
    print("  replace it, whichever way they come out.")

    RES.mkdir(parents=True, exist_ok=True)
    (RES / "ablations.json").write_text(json.dumps(
        {"config": dict(N=N, P=P, snr_db=SNR, L=L, trials=n,
                        mean_L_hat=float(np.mean(lhat))),
         "rows": [{"variant": a, "nmse_db": b, "gain_db": c} for a, b, c in rows]},
        indent=1))
    print(f"  wrote {RES / 'ablations.json'}")


if __name__ == "__main__":
    main()
