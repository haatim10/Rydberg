"""Paper-style operating-point sweeps: NMSE vs SNR and NMSE vs pilots.

Produces the data behind the two figures the URformer paper reports (its
Fig. 3 and Fig. 4), for every method this project has trained or implemented:

    classical      GS, EM-GS (both spectral init, T_GS = 100)
    classical+prior HS-EM-GS (Track B, L_hat = 7, Cadzow x4)
    learned        URformer (U1), HS-URformer (H1), G1 gated, G2 SNR-gated,
                   X1 = EM-GS + one Transformer post-processor
    reference      unstructured-LS oracle (perfect phase, then LS)

Two caveats that travel with the numbers
----------------------------------------
1. **The pilot sweep is a GENERALIZATION test, not a retrained curve.** Every
   learned arm was trained at ``P = 20`` and is evaluated unchanged at other
   pilot counts. The paper's Fig. 4 does not say whether its networks were
   retrained per ``P``; if they were, its curve answers a different question
   than this one. Nothing here is retrained -- that would be four more
   training runs per pilot count.
2. **The channel model is Track B's**, ``L_k ~ U{3..7}`` discrete paths, not
   the paper's Table I ``L = 4`` clusters x ``C_l = 10`` subrays. Under Table
   I as printed (independent DoA per subray) a channel column is full Hankel
   rank at ``N = 32``, so the HS-* arms would have no low-rank structure to
   exploit. These curves describe the sparse-path channel.

Per-trial numerator and denominator are stored separately, so ratio-of-sums,
median and any other pooling can be reconstructed without re-running.

Run one shard::

    PYTHONPATH=. python3 scratch/trackD_sweeps.py --mode snr --shard 0 --n-shards 4
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from trackD_urformer.baselines import nmse_parts, run_em_gs, run_gs
from trackD_urformer.config import TrackDConfig
from trackD_urformer.dataset import TrackDDataset
from trackD_urformer.stage1 import build_model
from trackD_urformer.stage3 import PostProcessor
from trackD_urformer.torch_forward import least_squares_G

OUT = Path("results/track_d/sweeps")
N_TRIALS = 400
T_GS = 100
L_HAT = 7

SNR_GRID = [-10.0, -7.5, -5.0, -2.5, 0.0, 2.5, 5.0, 7.5, 10.0, 12.5, 15.0,
            17.5, 20.0]
PILOT_GRID = [10, 12, 15, 20, 25, 30, 35]
PILOT_SNR = 5.0          # the paper evaluates its pilot sweep at a fixed 5 dB

CKPT = {
    "URformer":    ("results/track_d/stage2/B3_80k_13ep/best.pt", False, "none"),
    "HS-URformer": ("results/track_d/stage3/H1_hs_urformer_80k/best.pt", True, "none"),
    "G1 gated":    ("results/track_d/stage4/G1_gate_scalar/best.pt", True, "scalar"),
    "G2 SNR-gated": ("results/track_d/stage4/G2_gate_snr/best.pt", True, "snr"),
}
METHODS = ["GS", "EM-GS", "HS-EM-GS", "X1 EM-GS+former", "URformer",
           "HS-URformer", "G1 gated", "G2 SNR-gated", "unstructured-LS oracle"]


def load_models(cfg):
    models = {}
    for name, (path, hk, gate) in CKPT.items():
        rc = replace(cfg, model=replace(cfg.model, filter_init="random",
                                        use_transformer=True, use_hankel=hk,
                                        hankel_rank=L_HAT, hankel_mode="fixed",
                                        hankel_gate=gate))
        m, _ = build_model(rc, "arm1b_full_random")
        m.load_state_dict(torch.load(path, map_location="cpu",
                                     weights_only=False)["model"])
        models[name] = m.eval()
    x1 = PostProcessor(cfg.system.N, cfg.system.K, cfg.model)
    x1.load_state_dict(torch.load(
        "results/track_d/stage3/X1_emgs_plus_former/best.pt",
        map_location="cpu", weights_only=False)["model"])
    models["X1 EM-GS+former"] = x1.eval()
    return models


@torch.no_grad()
def one_point(cfg, models, *, snr_db: float, P: int, n: int) -> dict:
    """All methods at one operating point, on one shared set of trials."""
    from rydberg_sim.forward import exact_forward
    from rydberg_sim.rng import get_operating_point_rngs
    from rydberg_sim.track_b_proposed import hs_gs

    ds = TrackDDataset("test", sysc=cfg.system,
                       datac=replace(cfg.data, n_test=n), numeric=cfg.numeric,
                       P=P, snr_db=snr_db, init=cfg.train.init)
    cd, rd = cfg.numeric.complex_dtype, cfg.numeric.real_dtype
    T = lambda a, d: torch.as_tensor(np.array(a, copy=True)[None], dtype=d)
    num = {k: [] for k in METHODS}
    den = {k: [] for k in METHODS}

    for i in range(n):
        s = ds.sample(i)
        est = {}
        est["GS"] = run_gs(s, max_iter=T_GS, init="spectral", seed=s.trial)
        est["EM-GS"] = run_em_gs(s, max_iter=T_GS, init="spectral", seed=s.trial)
        est["HS-EM-GS"] = hs_gs(s.S, s.Z, s.B, s.sigma2, L_hat=L_HAT,
                                exact_step="em_gs", max_iter=T_GS).G_hat

        rngs = get_operating_point_rngs(cfg.system.master_seed, s.trial,
                                        s.snr_db, s.rsr_db)
        ex = exact_forward(s.G_true, s.S, s.B, s.sigma2, rng_noise=rngs.noise)
        of = s.Z * np.exp(1j * np.angle(np.asarray(ex.E)))
        est["unstructured-LS oracle"] = least_squares_G(
            T(of, torch.complex128) - T(s.B, torch.complex128),
            T(s.S, torch.complex128))[0].numpy()

        G0, Z = T(ds.g0(i), cd), T(s.Z, rd)
        S, B = T(s.S, cd), T(s.B, cd)
        s2 = torch.tensor([s.sigma2], dtype=rd)
        for name in CKPT:
            est[name] = models[name](G0, Z, S, B, s2)[0].numpy()
        est["X1 EM-GS+former"] = models["X1 EM-GS+former"](
            T(est["EM-GS"], cd))[0].numpy()

        for k in METHODS:
            a, b = nmse_parts(est[k], s.G_true)
            num[k].append(a)
            den[k].append(b)
    return {"snr_db": snr_db, "P": P, "n": n, "num": num, "den": den}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="NMSE vs SNR / vs pilots sweeps")
    ap.add_argument("--mode", choices=("snr", "pilots"), required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--n-trials", type=int, default=N_TRIALS)
    a = ap.parse_args(argv)

    torch.set_num_threads(1)
    cfg = TrackDConfig()
    cfg = replace(cfg, train=replace(cfg.train, init="spectral"))
    models = load_models(cfg)

    pts = ([(s, 20) for s in SNR_GRID] if a.mode == "snr"
           else [(PILOT_SNR, p) for p in PILOT_GRID])
    mine = [pts[i] for i in range(a.shard, len(pts), a.n_shards)]
    OUT.mkdir(parents=True, exist_ok=True)

    for snr_db, P in mine:
        f = OUT / f"{a.mode}_snr{snr_db:+.1f}_P{P}_n{a.n_trials}.json"
        if f.exists():
            print(f"skip (done): {f.name}", flush=True)
            continue
        t0 = time.time()
        r = one_point(cfg, models, snr_db=snr_db, P=P, n=a.n_trials)
        r["seconds"] = round(time.time() - t0, 1)
        f.write_text(json.dumps(r) + "\n", encoding="utf-8")
        med = {k: 10 * np.log10(np.median(np.array(r["num"][k])
                                          / np.array(r["den"][k])))
               for k in METHODS}
        print(f"[{a.mode}] SNR {snr_db:+5.1f} P {P:2d}  {r['seconds']:6.1f}s  "
              + "  ".join(f"{k}={med[k]:.1f}" for k in
                          ("EM-GS", "HS-EM-GS", "URformer", "G2 SNR-gated")),
              flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
