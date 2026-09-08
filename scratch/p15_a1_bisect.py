"""PROMPT 15 A1 -- bisect WHERE the divergence lives.

The first probe established that evaluation is deterministic (bitwise, in and
across code paths) and that the test worlds are bitwise identical to the ones
stored. So the stored rows and today's rows disagree for some reason other than
nondeterminism. This narrows it.

Stage 3 stored per-trial NMSE for arms that use NO trained weights at all --
`U0_em_gs` (pure NumPy EM-GS) and `oracle_phase` (a least-squares solve) --
alongside the learned arms, on the same worlds. That is the discriminator:

  * classical arms reproduce, learned arms do not  ->  the change is in torch
    or in the loaded weights, not in the world or the initialiser;
  * classical arms also fail  ->  the change is upstream of the network, in
    the world, the spectral initialiser, or NumPy/LAPACK itself.

Run:  PYTHONPATH=. python3 scratch/p15_a1_bisect.py [n]
"""
from __future__ import annotations

import json
import sys
from dataclasses import replace

import numpy as np
import torch

from rydberg_sim.forward import exact_forward
from rydberg_sim.rng import get_operating_point_rngs
from trackD_urformer.baselines import nmse_parts, run_em_gs
from trackD_urformer.torch_forward import least_squares_G
from trackD_urformer.config import TrackDConfig
from trackD_urformer.dataset import TrackDDataset
from rydberg_sim.track_b_proposed import hs_gs

N = int(sys.argv[1]) if len(sys.argv) > 1 else 40
T_GS, HANKEL_RANK = 100, 7


def rep(tag, a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = np.abs(a - b)
    rel = d / np.maximum(np.abs(a), 1e-300)
    print(f"  {tag:<16} bitwise={'YES' if np.array_equal(a, b) else 'NO '}  "
          f"n_diff={(d > 0).sum():>3}/{len(a)}  max|rel|={rel.max():.3e}  "
          f"median_db stored={10*np.log10(np.median(a)):+8.4f} "
          f"now={10*np.log10(np.median(b)):+8.4f}")


def main():
    st = json.load(open("reports/trackD_stage3_results.json"))["test"]
    cfg = TrackDConfig()
    print(f"stage-3 cfg snr_range_db = {cfg.data.snr_range_db}, "
          f"stored n_test = {st['n_test']}, probing first {N}\n")

    ds = TrackDDataset("test", sysc=cfg.system, datac=cfg.data,
                       numeric=cfg.numeric, init=cfg.train.init)
    cd = cfg.numeric.complex_dtype

    # 0. Do the worlds match?
    snr_now = np.array([ds.sample(i).snr_db for i in range(N)])
    snr_st = np.asarray(st["snr_db"][:N], float)
    print(f"worlds: snr bitwise={np.array_equal(snr_st, snr_now)}   "
          f"L_k bitwise={st['L_k'][:N] == [ds.sample(i).L_k for i in range(N)]}\n")

    got = {"U0_em_gs": [], "H0_hs_em_gs": [], "oracle_phase": []}
    for i in range(N):
        s = ds.sample(i)
        e = run_em_gs(s, max_iter=T_GS, init="spectral", seed=s.trial)
        got["U0_em_gs"].append(np.divide(*nmse_parts(e, s.G_true)))
        h = hs_gs(s.S, s.Z, s.B, s.sigma2, L_hat=HANKEL_RANK,
                  exact_step="em_gs", max_iter=T_GS).G_hat
        got["H0_hs_em_gs"].append(np.divide(*nmse_parts(h, s.G_true)))
        rngs = get_operating_point_rngs(cfg.system.master_seed, s.trial,
                                        s.snr_db, s.rsr_db)
        ex = exact_forward(s.G_true, s.S, s.B, s.sigma2, rng_noise=rngs.noise)
        of = s.Z * np.exp(1j * np.angle(np.asarray(ex.E)))
        T_ = lambda a: torch.as_tensor(np.array(a, copy=True)[None],
                                       dtype=torch.complex128)
        o = least_squares_G(T_(of) - T_(s.B), T_(s.S))[0].numpy()
        got["oracle_phase"].append(np.divide(*nmse_parts(o, s.G_true)))

    print("CLASSICAL arms (no trained weights):")
    for k in got:
        rep(k, st["per_trial_nmse"][k][:N], got[k])

    print(f"\nnumpy {np.__version__}  torch {torch.__version__}")


if __name__ == "__main__":
    main()
