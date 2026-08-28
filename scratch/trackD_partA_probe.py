"""Track D PROMPT 4 Part A pre-flight.

A2  classical headroom + the oracle-phase bound at the stage-1 operating point
A3  warm-start refit under a criterion that means something (max abs err < 0.01)
A4  SNR / kappa coverage
plus an empirical check of the RSR convention direction.

Writes reports/trackD_partA.json. No training. Throwaway.

Run:  PYTHONPATH=. python3 scratch/trackD_partA_probe.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from trackD_urformer.baselines import (
    nmse_parts, run_em_gs, run_gs, run_linearised_ls,
)
from trackD_urformer.config import D1_SNR_GRID_DB, TrackDConfig
from trackD_urformer.dataset import TrackDDataset, make_world
from trackD_urformer.filter_net import (
    FilterNet, measure_kappa_range, warmstart_filternet,
)
from trackD_urformer.torch_forward import (
    bessel_ratio_torch, em_kappa, forward_field, least_squares_G,
)

OUT = Path("reports/trackD_partA.json")
C128, F64 = torch.complex128, torch.float64
cfg = TrackDConfig()
torch.set_num_threads(2)
res: dict = {}

N_TRIALS = 400          # paired, per SNR point
db = lambda x: 10.0 * np.log10(max(float(x), 1e-30))


def _t(a, dtype=C128):
    return torch.as_tensor(np.array(a, copy=True)[None], dtype=dtype)


# ---------------------------------------------------------------------------
# RSR convention direction - measured, not asserted
# ---------------------------------------------------------------------------
print("== RSR convention direction (empirical) ==")
accB = accSingle = accAll = 0.0
for t in range(300):
    s = make_world(9_000_000 + t, sysc=cfg.system, N=cfg.system.N,
                   P=cfg.system.P, snr_db=5.0)
    accB += np.sum(np.abs(s.B) ** 2) / s.B.size
    accSingle += np.sum(np.abs(s.G_true[:, 0:1] @ s.S[0:1, :]) ** 2) / s.B.size
    accAll += np.sum(np.abs(s.G_true @ s.S) ** 2) / s.B.size
rsr_ours = db(accB / accSingle)
rsr_paper = db(accB / accAll)
res["rsr_convention"] = {
    "target_rsr_ours_db": cfg.system.rsr_db,
    "measured_rsr_ours_db": rsr_ours,
    "measured_rsr_paper_db": rsr_paper,
    "measured_difference_db": rsr_ours - rsr_paper,
    "ten_log10_K": float(10 * np.log10(cfg.system.K)),
    "config_rsr_paper_equiv_db": cfg.system.rsr_paper_equiv_db,
    "config_is_correct": bool(abs(cfg.system.rsr_paper_equiv_db - rsr_paper) < 0.2),
    "correct_value": rsr_paper,
}
print(json.dumps(res["rsr_convention"], indent=2))

# ---------------------------------------------------------------------------
# A2 - classical headroom and the ORACLE-PHASE BOUND
# ---------------------------------------------------------------------------
print("\n== A2: classical headroom + oracle-phase bound ==")
print("oracle-phase estimate = LS(Z (+) e^{j angle(GS+B+W)} - B, S).")
print("Since Z = |GS+B+W|, that product IS GS+B+W exactly, so the oracle")
print("estimate is G + LS(W, S): the pure LS noise floor. Exact ceiling for")
print("any magnitude-only estimator that recovers phase perfectly.\n")

a2_rows = []
for snr in D1_SNR_GRID_DB:
    acc = {k: [0.0, 0.0] for k in
           ("gs_spectral", "em_gs_spectral", "linearised_ls", "oracle_phase")}
    per_trial = {k: [] for k in acc}
    for i in range(N_TRIALS):
        trial = 2_000_000 + i          # TEST seed range
        s = make_world(trial, sysc=cfg.system, N=cfg.system.N,
                       P=cfg.system.P, snr_db=snr)

        ests = {
            "gs_spectral": run_gs(s, max_iter=cfg.baseline.T_GS,
                                  init="spectral", seed=trial),
            "em_gs_spectral": run_em_gs(s, max_iter=cfg.baseline.T_GS,
                                        init="spectral", seed=trial),
            "linearised_ls": run_linearised_ls(s),
        }
        # oracle phase: exact noisy field reconstructed from its own magnitude
        # and phase, then the SAME repository LS step.
        E = s.G_true @ s.S + s.B                      # noiseless field
        # rebuild the noisy field from Z and its true phase
        W = None
        # recover W by re-deriving the world's own noise: E_noisy = G S + B + W,
        # and Z = |E_noisy|. We have Z; the true phase comes from E_noisy, which
        # we reconstruct exactly from the generator.
        from rydberg_sim.rng import get_operating_point_rngs
        from rydberg_sim.forward import exact_forward
        rngs = get_operating_point_rngs(cfg.system.master_seed, trial,
                                        float(snr), s.rsr_db)
        ex = exact_forward(s.G_true, s.S, s.B, s.sigma2, rng_noise=rngs.noise)
        E_noisy = np.asarray(ex.E)
        assert np.allclose(np.abs(E_noisy), s.Z, atol=1e-10), "Z mismatch"
        oracle_field = s.Z * np.exp(1j * np.angle(E_noisy))
        ests["oracle_phase"] = least_squares_G(
            _t(oracle_field) - _t(s.B), _t(s.S)
        )[0].numpy()

        for k, gh in ests.items():
            e, d = nmse_parts(gh, s.G_true)
            acc[k][0] += e
            acc[k][1] += d
            per_trial[k].append(e / d)

    row = {"snr_db": float(snr)}
    for k in acc:
        row[k + "_db"] = db(acc[k][0] / acc[k][1])
        row[k + "_median_db"] = db(np.median(per_trial[k]))
    row["headroom_emgs_minus_oracle_db"] = (
        row["em_gs_spectral_db"] - row["oracle_phase_db"])
    a2_rows.append(row)
    print(f"  SNR {snr:6.1f}  GS {row['gs_spectral_db']:7.2f}  "
          f"EM-GS {row['em_gs_spectral_db']:7.2f}  "
          f"linLS {row['linearised_ls_db']:7.2f}  "
          f"ORACLE {row['oracle_phase_db']:7.2f}  "
          f"headroom {row['headroom_emgs_minus_oracle_db']:6.2f} dB")

res["A2"] = {"n_trials": N_TRIALS, "rows": a2_rows,
             "oracle_definition": "LS(Z*exp(j*angle(GS+B+W)) - B, S) = G + LS(W,S)"}

headrooms = [r["headroom_emgs_minus_oracle_db"] for r in a2_rows]
res["A2"]["headroom_min_db"] = float(np.min(headrooms))
res["A2"]["headroom_max_db"] = float(np.max(headrooms))
res["A2"]["headroom_at_snr5_db"] = float(
    [r for r in a2_rows if r["snr_db"] == 5.0][0]["headroom_emgs_minus_oracle_db"])

# paper comparison at P=15, SNR=5 (paper Fig. 4 reports about -20 dB)
print("\n  paper comparison: P=15, SNR=5 dB")
accp = {"em_gs_spectral": [0.0, 0.0], "oracle_phase": [0.0, 0.0]}
for i in range(N_TRIALS):
    trial = 2_500_000 + i
    s = make_world(trial, sysc=cfg.system, N=cfg.system.N, P=15, snr_db=5.0)
    e, d = nmse_parts(run_em_gs(s, max_iter=cfg.baseline.T_GS, init="spectral",
                                seed=trial), s.G_true)
    accp["em_gs_spectral"][0] += e; accp["em_gs_spectral"][1] += d
    from rydberg_sim.rng import get_operating_point_rngs
    from rydberg_sim.forward import exact_forward
    rngs = get_operating_point_rngs(cfg.system.master_seed, trial, 5.0, s.rsr_db)
    ex = exact_forward(s.G_true, s.S, s.B, s.sigma2, rng_noise=rngs.noise)
    of = s.Z * np.exp(1j * np.angle(np.asarray(ex.E)))
    gh = least_squares_G(_t(of) - _t(s.B), _t(s.S))[0].numpy()
    e, d = nmse_parts(gh, s.G_true)
    accp["oracle_phase"][0] += e; accp["oracle_phase"][1] += d
res["A2"]["paper_point_P15_SNR5"] = {
    "em_gs_spectral_db": db(accp["em_gs_spectral"][0] / accp["em_gs_spectral"][1]),
    "oracle_phase_db": db(accp["oracle_phase"][0] / accp["oracle_phase"][1]),
    "paper_reported_db": -20.0,
    "K_ours": cfg.system.K, "K_paper": 4,
}
print("   ", json.dumps(res["A2"]["paper_point_P15_SNR5"], indent=2))

# ---------------------------------------------------------------------------
# A3 - warm-start refit with max-abs-error criterion
# ---------------------------------------------------------------------------
print("\n== A3: warm-start refit, criterion max abs err < 0.01 ==")
ds = TrackDDataset("train", sysc=cfg.system, datac=cfg.data, numeric=cfg.numeric)
kstats = measure_kappa_range(ds, eps=1e-12, n_samples=512)
res["A3_kappa_stats"] = kstats

a3 = []
for width in (32, 64, 128):
    for variant in ("R", "one_minus_R"):
        torch.manual_seed(0)
        net = FilterNet(hidden=width, filter_input=cfg.model.filter_input,
                        predict_one_minus_R=(variant == "one_minus_R"))
        info = warmstart_filternet(
            net, kstats, cache_path=None, max_steps=40000, target_mse=1e-12,
            target_max_abs=0.01,
        )
        a3.append({"hidden": width, "variant": variant, **{
            k: v for k, v in info.items() if k != "kappa_stats"}})
        print(f"  hidden={width:4d} variant={variant:12s} "
              f"mse={info['achieved_mse']:.3e} "
              f"max_abs={info['achieved_max_abs']:.4f} "
              f"{'PASS' if info['achieved_max_abs'] < 0.01 else 'fail'}")
res["A3"] = a3

passing = [r for r in a3 if r["achieved_max_abs"] < 0.01]
if passing:
    best = min(passing, key=lambda r: (r["hidden"], r["variant"] != "R"))
    res["A3_adopted"] = best
    print(f"\n  adopt: hidden={best['hidden']} variant={best['variant']}")
else:
    res["A3_adopted"] = None
    print("\n  NO width/variant reached max abs err < 0.01")

# residual vs kappa for the adopted fit
if passing:
    torch.manual_seed(0)
    net = FilterNet(hidden=best["hidden"], filter_input=cfg.model.filter_input,
                    predict_one_minus_R=(best["variant"] == "one_minus_R"))
    warmstart_filternet(net, kstats, cache_path=None, max_steps=40000,
                        target_mse=1e-12, target_max_abs=0.01)
    kg = torch.logspace(np.log10(kstats["grid_lo"]), np.log10(kstats["grid_hi"]),
                        60, dtype=torch.float64)
    with torch.no_grad():
        pred = net(kg.view(1, 1, -1)).view(-1)
    tgt = bessel_ratio_torch(kg)
    res["A3_residual_profile"] = [
        {"kappa": float(k), "R_exact": float(a), "R_fit": float(b),
         "abs_err": float(abs(a - b))}
        for k, a, b in zip(kg, tgt, pred)
    ]
    worst = max(res["A3_residual_profile"], key=lambda d: d["abs_err"])
    print(f"  worst residual at kappa={worst['kappa']:.2f}: "
          f"{worst['abs_err']:.5f}")

# ---------------------------------------------------------------------------
# A4 - coverage
# ---------------------------------------------------------------------------
print("\n== A4: coverage ==")
train_lo, train_hi = cfg.data.snr_range_db
eval_lo, eval_hi = min(D1_SNR_GRID_DB), max(D1_SNR_GRID_DB)
snr_ok = train_lo <= eval_lo and train_hi >= eval_hi

# kappa across the EVALUATION sweep, not only training
eval_k_min, eval_k_max = np.inf, -np.inf
for snr in D1_SNR_GRID_DB:
    for i in range(40):
        s = make_world(2_000_000 + i, sysc=cfg.system, N=cfg.system.N,
                       P=cfg.system.P, snr_db=snr)
        G0 = torch.zeros((1, s.N, s.K), dtype=C128)
        Y = forward_field(G0, _t(s.S), _t(s.B))
        k = em_kappa(_t(s.Z, F64), Y, torch.tensor([s.sigma2], dtype=F64), 1e-12)
        eval_k_min = min(eval_k_min, float(k.min()))
        eval_k_max = max(eval_k_max, float(k.max()))
kappa_ok = (kstats["grid_lo"] <= eval_k_min and kstats["grid_hi"] >= eval_k_max)
res["A4"] = {
    "snr_train_range_db": [train_lo, train_hi],
    "snr_eval_range_db": [float(eval_lo), float(eval_hi)],
    "snr_covers_eval": bool(snr_ok),
    "kappa_grid": [kstats["grid_lo"], kstats["grid_hi"]],
    "kappa_eval_observed": [eval_k_min, eval_k_max],
    "kappa_grid_covers_eval": bool(kappa_ok),
}
print(json.dumps(res["A4"], indent=2))

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
print(f"\nwrote {OUT}")
