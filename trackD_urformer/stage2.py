"""Track D stage 2 - matched-compute data scaling (PROMPT 5 Part B).

Total sample-passes held ~constant so each run costs roughly the same wall
clock, isolating DATA QUANTITY from COMPUTE:

    run   samples   epochs   sample-passes
    B1     20,000       50   1,000,000   <- already have this (stage 1 arm 1b)
    B2     40,000       25   1,000,000
    B3     80,000       13   1,040,000

Architecture: full URformer, filter_init="random" (the stage-1 TEST winner).
Plus arm 2 (filter-only, 980 params) at the 80,000 budget only, so attribution
is measured at the BEST data budget rather than only the worst.

The bar does not move: >= 2 dB median paired improvement over EM-GS-spectral
with a bootstrap CI excluding zero.

Validation and test sets are FIXED and IDENTICAL across every run - only
n_train changes, and the val/test seed ranges are independent of it. Asserted
at startup.

Fix carried in from PROMPT 5 A4
-------------------------------
Stage 1 kept only best.pt and the final checkpoint.pt, so the exact
"every epoch vs the selected epoch, paired" test was not computable. Stage 2
stores the PER-TRIAL validation NMSE at every epoch (2000 float64 = 16 KB per
epoch), which makes that test exact without storing 50 sets of weights.

Run:  PYTHONPATH=. python3 -m trackD_urformer.stage2 --i-have-approval
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import TrackDConfig
from .dataset import TrackDDataset, collate
from .stage1 import build_model, db
from .train import make_initial_batch, nmse_loss
from .urformer import URformer, count_parameters

RESULTS = Path("results") / "track_d" / "stage2"
REPORTS = Path("reports")

# (n_train, epochs, use_transformer)
RUNS: dict[str, dict] = {
    "B2_40k_25ep": {"n_train": 40_000, "epochs": 25, "use_transformer": True},
    "B3_80k_13ep": {"n_train": 80_000, "epochs": 13, "use_transformer": True},
    "B3_80k_13ep_filteronly": {"n_train": 80_000, "epochs": 13,
                               "use_transformer": False},
}

# Pre-registered selection rule for stage 2 (see reports/trackD_stage2_prereg.md).
# Applied to stage 2 ONLY; never retro-applied to stage 1.
SELECTION_RULE = "one_se"      # "best" | "one_se"


@torch.no_grad()
def validation_per_trial(model, loader, cfg) -> np.ndarray:
    """Per-trial validation NMSE. Enables the exact A4 test in stage 2."""
    model.eval()
    out = []
    for batch in loader:
        G0 = make_initial_batch(batch, cfg.train.init, cfg)
        est = model(G0, batch["Z"], batch["S"], batch["B"], batch["sigma2"])
        num = torch.sum(torch.abs(est - batch["G_true"]) ** 2, dim=(1, 2))
        den = torch.sum(torch.abs(batch["G_true"]) ** 2, dim=(1, 2))
        out.extend((num / den).tolist())
    model.train()
    return np.asarray(out, dtype=np.float64)


def select_epoch(val_curve: np.ndarray, per_trial: np.ndarray | None,
                 rule: str) -> tuple[int, dict]:
    """Choose the reported epoch from the VALIDATION curve only.

    ``one_se``: the EARLIEST epoch whose validation metric is within one
    standard error of the best. Pre-registered before stage 2 ran.
    """
    best = int(np.argmin(val_curve))
    if rule == "best" or per_trial is None:
        return best, {"rule": "best", "best_epoch": best}
    rng = np.random.default_rng(20260830)
    b = np.array([rng.choice(per_trial, per_trial.size, replace=True).mean()
                  for _ in range(2000)])
    se_db = float(np.std(10 * np.log10(b), ddof=1))
    thresh = val_curve[best] + se_db
    chosen = int(np.argmax(val_curve <= thresh))
    return chosen, {"rule": "one_se", "best_epoch": best, "se_db": se_db,
                    "threshold_db": float(thresh), "chosen_epoch": chosen,
                    "n_within_1se": int(np.sum(val_curve <= thresh))}


def train_run(cfg: TrackDConfig, name: str, spec: dict, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(cfg.train.num_threads)
    torch.manual_seed(cfg.train.seed)
    np.random.seed(cfg.train.seed)

    rcfg = replace(cfg, data=replace(cfg.data, n_train=spec["n_train"]),
                   model=replace(cfg.model,
                                 use_transformer=spec["use_transformer"],
                                 filter_init="random"))
    epochs = int(spec["epochs"])

    train_ds = TrackDDataset("train", sysc=rcfg.system, datac=rcfg.data,
                             numeric=rcfg.numeric, init=rcfg.train.init)
    val_ds = TrackDDataset("val", sysc=rcfg.system, datac=rcfg.data,
                           numeric=rcfg.numeric, init=rcfg.train.init)
    # The val set must be byte-identical across runs.
    assert len(val_ds) == cfg.data.n_val
    assert val_ds.sample(0).trial == cfg.data.val_seed_range[0]

    train_ld = DataLoader(train_ds, batch_size=rcfg.train.batch_size,
                          shuffle=True, collate_fn=collate, num_workers=0)
    val_ld = DataLoader(val_ds, batch_size=rcfg.train.batch_size, shuffle=False,
                        collate_fn=collate, num_workers=0)

    model, meta = build_model(rcfg, "arm1b_full_random"
                              if spec["use_transformer"] else
                              "arm2_filteronly_warmstart")
    if not spec["use_transformer"]:
        # arm 2 at this budget uses RANDOM filter init too, matching B2/B3.
        model, meta = build_model(
            replace(rcfg, model=replace(rcfg.model, filter_init="random")),
            "arm2_filteronly_warmstart")
        model.mcfg = replace(model.mcfg, filter_init="random")

    opt = torch.optim.Adam(model.parameters(), lr=rcfg.train.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    ckpt = out_dir / "checkpoint.pt"
    csv_path = out_dir / "curves.csv"
    vpt_path = out_dir / "val_per_trial.npy"
    start, history, vpt = 0, [], []

    if ckpt.exists():
        blob = torch.load(ckpt, map_location="cpu", weights_only=False)
        model.load_state_dict(blob["model"])
        opt.load_state_dict(blob["optimizer"])
        sched.load_state_dict(blob["scheduler"])
        start = blob["epoch"] + 1
        history = blob.get("history", [])
        if vpt_path.exists():
            vpt = list(np.load(vpt_path))
        print(f"  [{name}] resumed at epoch {start}")

    if start == 0:
        with csv_path.open("w", newline="", encoding="utf-8") as fh:
            csv.writer(fh).writerow(["epoch", "train_loss_db", "val_nmse_db",
                                     "lr", "seconds", "mean_alpha"])

    for epoch in range(start, epochs):
        t0 = time.time()
        run_loss, nb = 0.0, 0
        for batch in train_ld:
            opt.zero_grad()
            G0 = make_initial_batch(batch, rcfg.train.init, rcfg)
            loss = nmse_loss(model(G0, batch["Z"], batch["S"], batch["B"],
                                   batch["sigma2"]), batch["G_true"])
            loss.backward()
            if rcfg.train.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(),
                                               rcfg.train.grad_clip)
            opt.step()
            run_loss += float(loss.detach())
            nb += 1
        sched.step()

        v = validation_per_trial(model, val_ld, rcfg)
        vpt.append(v)
        np.save(vpt_path, np.asarray(vpt))

        rec = {"epoch": epoch, "train_loss_db": db(run_loss / max(nb, 1)),
               "val_nmse_db": db(v.mean()), "lr": sched.get_last_lr()[0],
               "seconds": round(time.time() - t0, 1),
               "mean_alpha": float(np.mean(model.initial_alphas()))}
        history.append(rec)
        with csv_path.open("a", newline="", encoding="utf-8") as fh:
            csv.writer(fh).writerow([rec[k] for k in
                                     ("epoch", "train_loss_db", "val_nmse_db",
                                      "lr", "seconds", "mean_alpha")])
        # Every epoch's weights are kept for the selection rule to pick from.
        torch.save({"model": model.state_dict()}, out_dir / f"ep{epoch:03d}.pt")
        torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(),
                    "scheduler": sched.state_dict(), "epoch": epoch,
                    "history": history, "config": rcfg.to_dict(),
                    "run": name}, ckpt)
        print(f"  [{name}] epoch {epoch:3d}  train {rec['train_loss_db']:7.2f} dB"
              f"  val {rec['val_nmse_db']:7.2f} dB  {rec['seconds']:5.1f}s"
              f"  alpha {rec['mean_alpha']:.3f}")

    val_curve = np.array([r["val_nmse_db"] for r in history])
    vpt_arr = np.asarray(vpt)
    chosen, sel = select_epoch(val_curve, vpt_arr[int(np.argmin(val_curve))],
                               SELECTION_RULE)
    # Promote the chosen epoch to best.pt for the single test evaluation.
    blob = torch.load(out_dir / f"ep{chosen:03d}.pt", map_location="cpu",
                      weights_only=False)
    torch.save({"model": blob["model"], "epoch": chosen,
                "config": rcfg.to_dict()}, out_dir / "best.pt")

    return {**meta, "run": name, "spec": spec,
            "n_train": spec["n_train"], "epochs": epochs,
            "sample_passes": spec["n_train"] * epochs,
            "history": history, "selection": sel,
            "chosen_epoch": chosen,
            "chosen_val_db": float(val_curve[chosen]),
            "best_val_db": float(val_curve.min()),
            "best_path": str(out_dir / "best.pt"),
            "early_stopping": "none (fixed budget)"}


@torch.no_grad()
def evaluate_test_stage2(cfg: TrackDConfig, runs: dict, n_test: int) -> dict:
    """ONE pass over the FIXED test set. Every method sees identical worlds.

    Stage 1's evaluate_test_once builds models from its own ARMS table, which
    does not know the stage-2 run names, so stage 2 carries its own builder.
    The test data, baselines and pairing are the same shared realizations.
    """
    from rydberg_sim.forward import exact_forward
    from rydberg_sim.rng import get_operating_point_rngs

    from .baselines import nmse_parts, run_em_gs, run_gs, run_linearised_ls
    from .torch_forward import least_squares_G

    ds = TrackDDataset("test", sysc=cfg.system, datac=cfg.data,
                       numeric=cfg.numeric, init=cfg.train.init)
    cd, rd = cfg.numeric.complex_dtype, cfg.numeric.real_dtype

    models = {}
    for name, info in runs.items():
        spec = RUNS[name]
        rcfg = replace(cfg, model=replace(cfg.model,
                                          use_transformer=spec["use_transformer"],
                                          filter_init="random"))
        m, _ = build_model(rcfg, "arm1b_full_random" if spec["use_transformer"]
                           else "arm2_filteronly_warmstart")
        blob = torch.load(info["best_path"], map_location="cpu",
                          weights_only=False)
        m.load_state_dict(blob["model"])
        m.eval()
        models[name] = m

    keys = ["gs_spectral", "em_gs_spectral", "linearised_ls",
            "oracle_phase"] + list(models)
    per = {k: [] for k in keys}
    en = {k: [0.0, 0.0] for k in keys}

    T = lambda a, d: torch.as_tensor(np.array(a, copy=True)[None], dtype=d)
    for i in range(n_test):
        s = ds.sample(i)
        est = {
            "gs_spectral": run_gs(s, max_iter=cfg.baseline.T_GS,
                                  init="spectral", seed=s.trial),
            "em_gs_spectral": run_em_gs(s, max_iter=cfg.baseline.T_GS,
                                        init="spectral", seed=s.trial),
            "linearised_ls": run_linearised_ls(s),
        }
        rngs = get_operating_point_rngs(cfg.system.master_seed, s.trial,
                                        s.snr_db, s.rsr_db)
        ex = exact_forward(s.G_true, s.S, s.B, s.sigma2, rng_noise=rngs.noise)
        of = s.Z * np.exp(1j * np.angle(np.asarray(ex.E)))
        est["oracle_phase"] = least_squares_G(
            T(of, torch.complex128) - T(s.B, torch.complex128),
            T(s.S, torch.complex128))[0].numpy()

        G0, Z = T(ds.g0(i), cd), T(s.Z, rd)
        S, B = T(s.S, cd), T(s.B, cd)
        s2 = torch.tensor([s.sigma2], dtype=rd)
        for name, m in models.items():
            est[name] = m(G0, Z, S, B, s2)[0].detach().numpy()

        for k, gh in est.items():
            e, d_ = nmse_parts(gh, s.G_true)
            en[k][0] += e
            en[k][1] += d_
            per[k].append(e / d_)

    out = {"n_test": n_test, "methods": {}, "paired_vs_em_gs": {}}
    for k in keys:
        a = np.array(per[k])
        out["methods"][k] = {
            "nmse_ratio_of_sums_db": db(en[k][0] / en[k][1]),
            "nmse_median_db": db(np.median(a)),
            "nmse_mean_of_ratios_db": db(a.mean()),
        }
    base = 10 * np.log10(np.array(per["em_gs_spectral"]))
    rng = np.random.default_rng(20260830)
    for k in list(models) + ["gs_spectral", "linearised_ls", "oracle_phase"]:
        d_db = 10 * np.log10(np.array(per[k])) - base
        b = np.array([np.median(rng.choice(d_db, d_db.size, replace=True))
                      for _ in range(4000)])
        lo, hi = float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))
        out["paired_vs_em_gs"][k] = {
            "median_diff_db": float(np.median(d_db)),
            "mean_diff_db": float(d_db.mean()),
            "std_diff_db": float(d_db.std(ddof=1)),
            "boot_ci95_median": [lo, hi],
            "ci_excludes_zero": bool(lo > 0 or hi < 0),
            "clears_2db_bar": bool(hi < -2.0 or float(np.median(d_db)) <= -2.0),
            "win_rate": float(np.mean(d_db < 0)),
            "percentiles_db": {p: float(np.percentile(d_db, p))
                               for p in (5, 25, 50, 75, 95)},
        }
    out["per_trial_nmse"] = {k: per[k] for k in keys}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Track D stage 2")
    ap.add_argument("--i-have-approval", action="store_true")
    ap.add_argument("--runs", type=str, default=",".join(RUNS))
    args = ap.parse_args(argv)
    if not args.i_have_approval:
        print("REFUSING: stage 2 requires --i-have-approval (PROMPT 5 Part B).")
        return 2

    cfg = TrackDConfig()
    cfg = replace(cfg, train=replace(cfg.train, init="spectral"))
    print(f"initializer: {cfg.train.init}   selection rule: {SELECTION_RULE}")
    print(f"val/test FIXED: n_val={cfg.data.n_val} from {cfg.data.val_seed_range[0]}, "
          f"n_test={cfg.data.n_test} from {cfg.data.test_seed_range[0]}")

    RESULTS.mkdir(parents=True, exist_ok=True)
    path = REPORTS / "trackD_stage2_results.json"
    summary = json.loads(path.read_text()) if path.exists() else {
        "config": cfg.to_dict(), "runs": {}}

    for name in args.runs.split(","):
        name = name.strip()
        print(f"\n=== {name} ===")
        t0 = time.time()
        info = train_run(cfg, name, RUNS[name], RESULTS / name)
        info["train_seconds"] = round(time.time() - t0, 1)
        summary["runs"][name] = info
        path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(f"  [{name}] done in {info['train_seconds']}s, "
              f"chosen epoch {info['chosen_epoch']} "
              f"val {info['chosen_val_db']:.3f} dB")

    print("\n=== single test evaluation ===")
    test = evaluate_test_stage2(cfg, summary["runs"], cfg.data.n_test)
    summary["test"] = test
    path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    for k, v in test["methods"].items():
        print(f"  {k:28s} {v['nmse_ratio_of_sums_db']:8.3f} dB "
              f"(median {v['nmse_median_db']:8.3f})")
    print("\n  paired vs EM-GS-spectral (negative = better):")
    for k, v in test["paired_vs_em_gs"].items():
        bar = "  <-- CLEARS 2 dB" if v["median_diff_db"] <= -2.0 else ""
        print(f"  {k:28s} median {v['median_diff_db']:+7.3f} dB  "
              f"CI [{v['boot_ci95_median'][0]:+.3f}, "
              f"{v['boot_ci95_median'][1]:+.3f}]  "
              f"excl0={v['ci_excludes_zero']}{bar}")
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
