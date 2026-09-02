"""B5 cost table: parameters, training cost, inference cost per trial.

Inference is timed single-threaded on an otherwise idle machine, one method at
a time on shared realizations, so the numbers are comparable to each other.
They are wall-clock on this container's CPU and are not a claim about any
other hardware; the RATIOS are the portable part.

Run:  PYTHONPATH=. python3 scratch/trackD_cost_table.py
"""
from __future__ import annotations

import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from rydberg_sim.track_b_proposed import hs_gs, hs_gs_auto
from trackD_urformer.baselines import run_em_gs, run_gs
from trackD_urformer.config import TrackDConfig
from trackD_urformer.dataset import TrackDDataset
from trackD_urformer.stage1 import build_model

OUT = Path("reports/trackD_cost_table.json")
N_TIME = 40
T_GS = 100

# Wall clock for the full 13-epoch run, all single-machine CPU. U1 is summed
# from its own curves.csv `seconds` column (it predates the "done in" log
# line); the rest are that line verbatim.
TRAIN_SECONDS = {
    "URformer (U1, P=20)": 6_570.9,           # stage2/B3_80k_13ep/curves.csv
    "HS-URformer (H1, P=20)": 7_904.4,        # logs/stage3_H1.log:18
    "G1 gated (P=20)": 10_162.5,              # logs/stage4_G1_gate_scalar.log:16
    "C1 SNR-balanced (P=20)": 9_294.4,        # logs/stage5_C1_snr_balanced_P20.log
    "C2 URformer (P=10)": 7_902.9,            # logs/stage5_C2_urformer_P10.log
    "C3 URformer (P=35)": 11_140.6,           # logs/stage5_C3_urformer_P35.log
    "C4 G1 gated (P=10)": 10_975.0,           # logs/stage5_C4_g1_P10.log
}


def n_params(hankel: bool, gate: str) -> int:
    cfg = TrackDConfig()
    rc = replace(cfg, model=replace(cfg.model, filter_init="random",
                                    use_transformer=True, use_hankel=hankel,
                                    hankel_rank=7, hankel_mode="fixed",
                                    hankel_gate=gate))
    m, _ = build_model(rc, "arm1b_full_random")
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


@torch.no_grad()
def main() -> int:
    torch.set_num_threads(1)
    cfg = replace(TrackDConfig(),
                  train=replace(TrackDConfig().train, init="spectral"))
    ds = TrackDDataset("test", sysc=cfg.system,
                       datac=replace(cfg.data, n_test=N_TIME),
                       numeric=cfg.numeric, P=20, snr_db=5.0,
                       init=cfg.train.init)
    samples = [ds.sample(i) for i in range(N_TIME)]
    cd, rd = cfg.numeric.complex_dtype, cfg.numeric.real_dtype
    T = lambda a, d: torch.as_tensor(np.array(a, copy=True)[None], dtype=d)

    rc = replace(cfg, model=replace(cfg.model, filter_init="random",
                                    use_transformer=True, use_hankel=False,
                                    hankel_rank=7, hankel_mode="fixed",
                                    hankel_gate="none"))
    model, _ = build_model(rc, "arm1b_full_random")
    model.load_state_dict(torch.load("results/track_d/stage2/B3_80k_13ep/best.pt",
                                     map_location="cpu",
                                     weights_only=False)["model"])
    model.eval()

    def timeit(fn):
        fn(samples[0])                                   # warm up
        t0 = time.perf_counter()
        for s in samples:
            fn(s)
        return (time.perf_counter() - t0) / len(samples)

    inf = {
        "GS (100 it)": timeit(lambda s: run_gs(s, max_iter=T_GS,
                                               init="spectral", seed=s.trial)),
        "EM-GS (100 it)": timeit(lambda s: run_em_gs(s, max_iter=T_GS,
                                                     init="spectral",
                                                     seed=s.trial)),
        "HS-EM-GS (fixed r=7)": timeit(
            lambda s: hs_gs(s.S, s.Z, s.B, s.sigma2, L_hat=7,
                            exact_step="em_gs", max_iter=T_GS)),
        "HS-EM-GS (adaptive r)": timeit(
            lambda s: hs_gs_auto(s.S, s.Z, s.B, s.sigma2, exact_step="em_gs",
                                 max_iter=T_GS, select_iter=25)),
        "URformer (10 layers)": timeit(
            lambda s, i=[0]: model(T(ds.g0(i[0] % N_TIME), cd), T(s.Z, rd),
                                   T(s.S, cd), T(s.B, cd),
                                   torch.tensor([s.sigma2], dtype=rd))),
    }
    params = {"URformer / C1 / C2 / C3": n_params(False, "none"),
              "G1 gated (adds one scalar per layer)": n_params(True, "scalar"),
              "classical estimators": 0}
    res = {"n_timed": N_TIME, "threads": 1, "P": 20, "snr_db": 5.0,
           "inference_seconds_per_trial": inf,
           "trainable_parameters": params,
           "training_seconds_80k_13ep": TRAIN_SECONDS}
    base = inf["EM-GS (100 it)"]
    res["inference_relative_to_em_gs"] = {k: v / base for k, v in inf.items()}
    OUT.write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
    print(f"{'method':26s} {'s/trial':>9s} {'xEM-GS':>8s}")
    for k, v in inf.items():
        print(f"{k:26s} {v:9.4f} {v/base:8.2f}")
    print("\nparameters:", params)
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
