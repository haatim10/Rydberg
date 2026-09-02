"""PROMPT 9 Part C evaluation: per-bin Delta(SNR) for every matched-training run.

Contrasts, each on identical paired realizations:

  P13  C1 (SNR-balanced, P=20)  vs  U1 (uniform loss, P=20)
  P14  C2 (matched P=10)        vs  U1 evaluated OOD at P=10
       C3 (matched P=35)        vs  U1 evaluated OOD at P=35
  P15  C4 (G1, P=10)            vs  C2 (URformer, P=10) -- both matched-trained
  C5   every model, trained at L_k ~ U{3,7}, evaluated unchanged at L_k ~ U{5,10}

The unstructured-LS oracle is carried in every condition (B4), so results read
as "captures y% of achievable" and not only "beats B by x dB".

Run:  PYTHONPATH=. python3 scratch/trackD_stage5_eval.py
"""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from trackD_urformer.baselines import nmse_parts
from trackD_urformer.config import TrackDConfig
from trackD_urformer.dataset import TrackDDataset
from trackD_urformer.stage1 import build_model
from trackD_urformer.stage3 import paired
from trackD_urformer.stage4 import BINS, by_bin
from trackD_urformer.torch_forward import least_squares_G

OUT = Path("reports/trackD_stage5_eval.json")
N_TEST = 2000

# name -> (checkpoint, hankel, gate, P it was trained at)
MODELS = {
    "U1_P20_uniform":  ("results/track_d/stage2/B3_80k_13ep/best.pt", False, "none", 20),
    "C1_P20_balanced": ("results/track_d/stage5/C1_snr_balanced_P20/best.pt", False, "none", 20),
    "C2_P10_matched":  ("results/track_d/stage5/C2_urformer_P10/best.pt", False, "none", 10),
    "C3_P35_matched":  ("results/track_d/stage5/C3_urformer_P35/best.pt", False, "none", 35),
    "C4_G1_P10":       ("results/track_d/stage5/C4_g1_P10/best.pt", True, "scalar", 10),
}


def load(cfg, name):
    path, hk, gate, _ = MODELS[name]
    rc = replace(cfg, model=replace(cfg.model, filter_init="random",
                                    use_transformer=True, use_hankel=hk,
                                    hankel_rank=7, hankel_mode="fixed",
                                    hankel_gate=gate))
    m, _ = build_model(rc, "arm1b_full_random")
    m.load_state_dict(torch.load(path, map_location="cpu",
                                 weights_only=False)["model"])
    return m.eval()


@torch.no_grad()
def evaluate(cfg, names, *, P, L_range=None, n_test=N_TEST):
    """One pass; every named model plus the oracle on identical worlds."""
    from rydberg_sim.forward import exact_forward
    from rydberg_sim.rng import get_operating_point_rngs

    sysc = cfg.system if L_range is None else replace(
        cfg.system, L_min=L_range[0], L_max=L_range[1])
    ds = TrackDDataset("test", sysc=sysc,
                       datac=replace(cfg.data, n_test=n_test),
                       numeric=cfg.numeric, P=P, init=cfg.train.init)
    cd, rd = cfg.numeric.complex_dtype, cfg.numeric.real_dtype
    models = {n: load(cfg, n) for n in names}
    keys = list(names) + ["oracle"]
    per = {k: [] for k in keys}
    snr = []
    T = lambda a, d: torch.as_tensor(np.array(a, copy=True)[None], dtype=d)

    for i in range(n_test):
        s = ds.sample(i)
        snr.append(s.snr_db)
        G0, Z = T(ds.g0(i), cd), T(s.Z, rd)
        S, B = T(s.S, cd), T(s.B, cd)
        s2 = torch.tensor([s.sigma2], dtype=rd)
        for n, m in models.items():
            gh = m(G0, Z, S, B, s2)[0].numpy()
            a, b = nmse_parts(gh, s.G_true)
            per[n].append(a / b)
        rngs = get_operating_point_rngs(cfg.system.master_seed, s.trial,
                                        s.snr_db, s.rsr_db)
        ex = exact_forward(s.G_true, s.S, s.B, s.sigma2, rng_noise=rngs.noise)
        of = s.Z * np.exp(1j * np.angle(np.asarray(ex.E)))
        gh = least_squares_G(T(of, torch.complex128) - T(s.B, torch.complex128),
                             T(s.S, torch.complex128))[0].numpy()
        a, b = nmse_parts(gh, s.G_true)
        per["oracle"].append(a / b)
    db = lambda x: 10 * np.log10(x)
    return {"P": P, "L_range": list(L_range) if L_range else [3, 7],
            "n_test": n_test, "snr_db": snr, "per_trial_nmse": per,
            "median_db": {k: db(np.median(np.array(v))) for k, v in per.items()}}


def main() -> int:
    torch.set_num_threads(1)
    cfg = replace(TrackDConfig(),
                  train=replace(TrackDConfig().train, init="spectral"))
    res = {}

    print("=== P13: SNR-balanced vs uniform loss, P=20 ===", flush=True)
    e = evaluate(cfg, ["U1_P20_uniform", "C1_P20_balanced"], P=20)
    snr = np.asarray(e["snr_db"])
    e["contrast"] = by_bin(e["per_trial_nmse"], "U1_P20_uniform",
                           "C1_P20_balanced", snr)
    e["vs_oracle"] = {k: by_bin(e["per_trial_nmse"], k, "oracle", snr)
                      for k in ("U1_P20_uniform", "C1_P20_balanced")}
    res["P13_balanced_vs_uniform_P20"] = e
    for r in e["contrast"]["bins"]:
        print(f"  [{r['bin'][0]:+3d},{r['bin'][1]:+3d})  {r['median_diff_db']:+7.3f} "
              f"[{r['boot_ci95_median'][0]:+.3f},{r['boot_ci95_median'][1]:+.3f}]")
    print(f"  SNR>=5 {e['contrast']['high_snr_ge5']['median_diff_db']:+.3f}   "
          f"SNR<5 {e['contrast']['low_snr_lt5']['median_diff_db']:+.3f}")

    for tag, mm, P in (("P14_P10", "C2_P10_matched", 10),
                       ("P14_P35", "C3_P35_matched", 35)):
        print(f"\n=== {tag}: matched vs OOD at P={P} ===", flush=True)
        names = ["U1_P20_uniform", mm] + (["C4_G1_P10"] if P == 10 else [])
        e = evaluate(cfg, names, P=P)
        snr = np.asarray(e["snr_db"])
        e["contrast"] = by_bin(e["per_trial_nmse"], "U1_P20_uniform", mm, snr)
        if P == 10:
            e["p15_learned"] = by_bin(e["per_trial_nmse"], mm, "C4_G1_P10", snr)
        e["vs_oracle"] = {k: by_bin(e["per_trial_nmse"], k, "oracle", snr)
                          for k in names}
        res[tag] = e
        c = e["contrast"]
        print(f"  matched - OOD: SNR>=5 {c['high_snr_ge5']['median_diff_db']:+.3f}  "
              f"SNR<5 {c['low_snr_lt5']['median_diff_db']:+.3f}  "
              f"pooled {c['pooled_SAMPLING_DESIGN_DEPENDENT']['median_diff_db']:+.3f}")
        if P == 10:
            g = e["p15_learned"]
            print(f"  P15 learned (C2->G1): SNR>=5 "
                  f"{g['high_snr_ge5']['median_diff_db']:+.3f}  SNR<5 "
                  f"{g['low_snr_lt5']['median_diff_db']:+.3f}")

    print("\n=== C5: path-richness OOD, trained U{3,7} -> tested U{5,10} ===",
          flush=True)
    ood = {}
    for P, names in ((20, ["U1_P20_uniform", "C1_P20_balanced"]),
                     (10, ["C2_P10_matched", "C4_G1_P10"])):
        a = evaluate(cfg, names, P=P, L_range=(3, 7), n_test=1000)
        b = evaluate(cfg, names, P=P, L_range=(5, 10), n_test=1000)
        ood[f"P{P}"] = {"in_dist": a["median_db"], "ood": b["median_db"],
                        "degradation_db": {k: b["median_db"][k] - a["median_db"][k]
                                           for k in a["median_db"]}}
        print(f"  P={P}: " + "  ".join(
            f"{k} {ood[f'P{P}']['degradation_db'][k]:+.2f}"
            for k in ood[f"P{P}"]["degradation_db"]))
    res["C5_path_richness_ood"] = ood

    OUT.write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
