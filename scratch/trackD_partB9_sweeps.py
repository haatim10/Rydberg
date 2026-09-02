"""PROMPT 9 Part B -- classical scaling sweeps. No training.

EM-GS vs ADAPTIVE-RANK HS-EM-GS (`hs_gs_auto`, held-out pilot residual -- not
fixed r=7, which is a Track D artifact). SNR is drawn per trial across the
usual range and binned POST HOC, so each cell yields a whole Delta_HS(SNR)
curve rather than one scalar.

  B1  array-size collapse: N in {16,64} at L/cap in {0.25,0.45,0.90}
  B2  K in {2,3,4} at FIXED pilot adequacy P/2K = 3.33  (P = 13,20,27)
  B3  Delta_HS(SNR) under adaptive rank at the default configuration
  B4  the unstructured-LS oracle on every cell
  B6  Xiao's Saleh-Valenzuela channel -- tests the A2 prediction of +1.30 dB
  B7  pilot sweep at the default configuration -- the pre-registration's P15
      CLASSICAL half, which the original B1/B2/B3/B6 design left unscored

Trial budgeting, and why it is not the flat 1000 the brief asked for
--------------------------------------------------------------------
`hs_gs_auto` reruns the estimator once per candidate rank (1..cap), so a cell's
cost scales as roughly cap x N^2. At N=64 (cap 32) that is ~30x an N=32 trial,
which would put three N=64 cells alone at ~20 h -- past tonight's budget.

So each cell gets a TIME budget and reports the trial count it achieved, with
the realized paired SE. 1000 trials was justified in the brief by "SE ~ 0.01 dB
against effects of order 1 dB"; at 250 trials the SE is ~0.02-0.04 dB, still an
order of magnitude below the effects being measured. The achieved n and SE are
reported per cell so the reader can see exactly what each number rests on.

Run one cell group:
  PYTHONPATH=. python3 scratch/trackD_partB9_sweeps.py --group B1 --shard 0 --n-shards 4
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from rydberg_sim.track_b_proposed import hankel_rank_cap, hs_gs_auto
from trackD_urformer.baselines import nmse_parts, run_em_gs
from trackD_urformer.config import TrackDConfig
from trackD_urformer.dataset import make_world
from trackD_urformer.torch_forward import least_squares_G

OUT = Path("results/track_d/partB9")
T_GS = 100
SNR_RANGE = (-10.0, 20.0)
CELL_SECONDS = 2400          # 40 min per cell
# The floor must be low enough that an N=64 cell (cap 32, ~30x an N=32
# trial) still terminates inside its budget. At 60 trials the paired SE is
# ~0.05-0.10 dB, still well under the ~1 dB effects being measured; the
# achieved n and SE are reported per cell so nothing is hidden.
N_MIN, N_MAX = 60, 1000
CUI_RAY_OFFSET_DEG = 5.0


def cells(group: str) -> list[dict]:
    """Cell definitions. L chosen to hit L/cap in {0.25, 0.45, 0.90}."""
    if group == "B1":
        out = []
        for N, Ls in ((16, (2, 4, 7)), (64, (8, 14, 29))):
            for L in Ls:
                out.append({"tag": f"B1_N{N}_L{L}", "N": N, "L": L, "K": 3,
                            "P": 20, "channel": "ula"})
        return out
    if group == "B2":
        # FIXED pilot adequacy P/2K = 3.33 -- sweeping K at fixed P would move
        # adequacy at the same time and confound the two effects.
        return [{"tag": f"B2_K{K}_P{P}", "N": 32, "L": 5, "K": K, "P": P,
                 "channel": "ula"}
                for K, P in ((2, 13), (3, 20), (4, 27))]
    if group == "B3":
        return [{"tag": "B3_default", "N": 32, "L": None, "K": 3, "P": 20,
                 "channel": "ula"}]
    if group == "B7":
        # The pre-registration's P15 CLASSICAL half -- "Delta_HS grows as P
        # falls" -- had no cell in the original B1/B2/B3/B6 design. That was
        # an omission, not a scoping decision: it leaves a pre-registered
        # prediction unscored. B3_default already supplies the P = 20 point at
        # this configuration, so only the other pilot counts are run here.
        return [{"tag": f"B7_P{P}", "N": 32, "L": None, "K": 3, "P": P,
                 "channel": "ula"} for P in (10, 15, 35)]
    if group == "B6":
        return [{"tag": f"B6_xiao_{m}", "N": 32, "L": None, "K": 3, "P": 20,
                 "channel": f"sv_{m}"} for m in ("clustered", "literal")]
    if group == "B8":
        # PROMPT 10 Step 0: the second out-of-model prediction. The predicted
        # value was committed in 060205b BEFORE this cell was ever run.
        return [{"tag": "B8_cui", "N": 32, "L": None, "K": 3, "P": 20,
                 "channel": "sv_cui"}]
    if group == "all":
        # Priority order: B6 and B3 first. B6 tests the A2 prediction on a
        # channel specified by someone else, and B3 is the axis genuinely
        # missing for the classical paper's own claim -- so if the night runs
        # short, those are the two that must already be done.
        return cells("B6") + cells("B3") + cells("B2") + cells("B1")
    raise ValueError(group)


def sv_channel(N, K, rng, *, mode, n_clusters=4, rays=10, cui_paths=10):
    """Saleh-Valenzuela column sets: both Xiao readings, plus the Cui config.

    ``mode="cui"`` is the configuration named in the PROMPT 10 brief -- L = 10
    independent paths, CN(0,1) gains, incident angles U(-90, 90) deg. See
    scratch/trackD_step0_cui_predict.py for why the attribution of that
    configuration is recorded as unverified.
    """
    n = np.arange(N)
    G = np.empty((N, K), dtype=np.complex128)
    for k in range(K):
        if mode == "cui":
            th = rng.uniform(-np.pi / 2, np.pi / 2, cui_paths)
        elif mode == "literal":
            th = rng.uniform(-np.pi / 2, np.pi / 2, n_clusters * rays)
        else:
            ctr = rng.uniform(-np.pi / 2, np.pi / 2, n_clusters)
            off = np.deg2rad(rng.uniform(-CUI_RAY_OFFSET_DEG,
                                         CUI_RAY_OFFSET_DEG,
                                         (n_clusters, rays)))
            th = np.clip(ctr[:, None] + off, -np.pi / 2, np.pi / 2).ravel()
        D = th.size
        a = (rng.standard_normal(D) + 1j * rng.standard_normal(D)) / np.sqrt(2 * D)
        G[:, k] = (a[None, :] * np.exp(
            -1j * (np.pi * np.sin(th))[None, :] * n[:, None])).sum(1)
    return G


def sv_world(cfg, trial, *, N, K, P, snr_db, mode):
    """A world whose channel is Xiao's, but whose forward model is ours.

    Only G is replaced; pilots, reference beam and noise all come from the
    repository generator so the comparison stays like-for-like.
    """
    from rydberg_sim.forward import exact_forward
    from rydberg_sim.rng import get_operating_point_rngs

    base = make_world(trial, sysc=replace(cfg.system, K=K), N=N, P=P,
                      snr_db=snr_db)
    rng = np.random.default_rng(np.random.SeedSequence([trial, 0x5A17]))
    G = sv_channel(N, K, rng, mode=mode)
    G *= np.sqrt(np.mean(np.abs(np.asarray(base.G_true)) ** 2)
                 / np.mean(np.abs(G) ** 2))         # match total channel power
    rngs = get_operating_point_rngs(cfg.system.master_seed, trial, snr_db,
                                    base.rsr_db)
    ex = exact_forward(G, base.S, base.B, base.sigma2, rng_noise=rngs.noise)
    return replace(base, Z=np.asarray(ex.Z), G_true=G)


def one_cell(cfg, cell: dict, *, seconds: int, seed0: int) -> dict:
    """EM-GS, adaptive HS-EM-GS and the oracle on shared realizations."""
    keys = ["EM-GS", "HS-EM-GS-auto", "oracle"]
    num = {k: [] for k in keys}
    den = {k: [] for k in keys}
    snr, lhat = [], []
    N, K, P, L = cell["N"], cell["K"], cell["P"], cell["L"]
    rng = np.random.default_rng(np.random.SeedSequence([seed0, N, K, P]))
    t0 = time.time()
    t = 0
    while t < N_MAX and (t < N_MIN or time.time() - t0 < seconds):
        s_db = float(np.round(rng.uniform(*SNR_RANGE), 3))
        if cell["channel"].startswith("sv_"):
            s = sv_world(cfg, seed0 + t, N=N, K=K, P=P, snr_db=s_db,
                         mode=cell["channel"][3:])
        else:
            s = make_world(seed0 + t, sysc=replace(cfg.system, K=K), N=N, P=P,
                           snr_db=s_db, L=L)
        est = {"EM-GS": run_em_gs(s, max_iter=T_GS, init="spectral",
                                  seed=s.trial)}
        r = hs_gs_auto(s.S, s.Z, s.B, s.sigma2, exact_step="em_gs",
                       max_iter=T_GS, select_iter=25)
        est["HS-EM-GS-auto"] = r.G_hat
        lhat.append(int(r.L_hat))

        from rydberg_sim.forward import exact_forward
        from rydberg_sim.rng import get_operating_point_rngs
        rngs = get_operating_point_rngs(cfg.system.master_seed, s.trial,
                                        s.snr_db, s.rsr_db)
        ex = exact_forward(s.G_true, s.S, s.B, s.sigma2, rng_noise=rngs.noise)
        of = s.Z * np.exp(1j * np.angle(np.asarray(ex.E)))
        import torch
        T_ = lambda a: torch.as_tensor(np.array(a, copy=True)[None],
                                       dtype=torch.complex128)
        est["oracle"] = least_squares_G(T_(of) - T_(s.B), T_(s.S))[0].numpy()

        for k in keys:
            a, b = nmse_parts(est[k], s.G_true)
            num[k].append(a)
            den[k].append(b)
        snr.append(s_db)
        t += 1
    return {**cell, "n": t, "seconds": round(time.time() - t0, 1),
            "cap": hankel_rank_cap(N), "snr_db": snr, "L_hat": lhat,
            "num": num, "den": den}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="PROMPT 9 Part B classical sweeps")
    ap.add_argument("--group", required=True, choices=("B1", "B2", "B3", "B6", "B7", "B8", "all"))
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--seconds", type=int, default=CELL_SECONDS)
    a = ap.parse_args(argv)

    cfg = TrackDConfig()
    OUT.mkdir(parents=True, exist_ok=True)
    cs = cells(a.group)
    for i in range(a.shard, len(cs), a.n_shards):
        cell = cs[i]
        f = OUT / f"{cell['tag']}.json"
        if f.exists():
            print(f"skip (done): {f.name}", flush=True)
            continue
        r = one_cell(cfg, cell, seconds=a.seconds, seed0=940_000 + 1000 * i)
        f.write_text(json.dumps(r) + "\n", encoding="utf-8")
        e = np.array(r["num"]["EM-GS"]) / np.array(r["den"]["EM-GS"])
        h = np.array(r["num"]["HS-EM-GS-auto"]) / np.array(r["den"]["HS-EM-GS-auto"])
        d = 10 * np.log10(e) - 10 * np.log10(h)
        print(f"[{cell['tag']}] n={r['n']:4d} in {r['seconds']:6.0f}s  "
              f"Delta_HS {np.median(d):+6.3f} dB  SE {d.std(ddof=1)/np.sqrt(d.size):.3f}"
              f"  mean L_hat {np.mean(r['L_hat']):.2f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
