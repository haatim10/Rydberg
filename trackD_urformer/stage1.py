"""Track D stage 1 - three arms at N=32, sequential, resumable.

PROMPT 4 Part B. Runs only when Part A's four go/no-go conditions passed.

    arm 1a  URformer full           filter_init = emgs_warmstart   the best shot
    arm 1b  URformer full           filter_init = random           paper control
    arm 2   use_transformer=False   filter_init = emgs_warmstart   attribution

Discipline (B3), enforced in code, not by intention:

* checkpoint selection on VALIDATION NMSE only; ``best.pt`` is restored before
  the single test evaluation
* the test set is touched EXACTLY ONCE per arm, at the end
* every epoch is checkpointed and every run is resumable -- this container is
  ephemeral
* curves go to CSV on disk, not only stdout
* arms run sequentially and each writes its result the moment it finishes, so a
  later failure cannot bury an earlier result
* NO early stopping: a fixed 50-epoch budget, identical for every arm, so the
  comparison is never confounded by different stopping points

Run:  PYTHONPATH=. python3 -m trackD_urformer.stage1 --i-have-approval
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

from .baselines import make_initial_G, nmse_parts, run_em_gs, run_gs, run_linearised_ls
from .config import TrackDConfig
from .dataset import TrackDDataset, collate
from .train import evaluate_split, make_initial_batch, nmse_loss
from .urformer import URformer, count_parameters

RESULTS = Path("results") / "track_d" / "stage1"
REPORTS = Path("reports")

ARMS = {
    "arm1a_full_warmstart": {"use_transformer": True, "filter_init": "emgs_warmstart"},
    "arm1b_full_random": {"use_transformer": True, "filter_init": "random"},
    "arm2_filteronly_warmstart": {"use_transformer": False,
                                  "filter_init": "emgs_warmstart"},
}

db = lambda x: 10.0 * float(np.log10(max(float(x), 1e-30)))


# ---------------------------------------------------------------------------
def build_model(cfg: TrackDConfig, arm: str) -> tuple[URformer, dict]:
    spec = ARMS[arm]
    mcfg = replace(cfg.model, use_transformer=spec["use_transformer"],
                   filter_init=spec["filter_init"])
    torch.manual_seed(cfg.train.seed)
    model = URformer(cfg.system.N, cfg.system.K, mcfg, cfg.numeric)
    if cfg.numeric.dtype == "float64":
        model = model.double()

    ws_info = None
    if spec["filter_init"] == "emgs_warmstart":
        ws_info = model.apply_filter_warmstart(mcfg.filter_warmstart_cache)
    return model, {"arm": arm, "spec": spec,
                   "params": count_parameters(model), "warmstart": ws_info}


def train_arm(cfg: TrackDConfig, arm: str, *, epochs: int, out_dir: Path,
              log_every: int = 1) -> dict:
    """Train one arm. Resumable; checkpoints every epoch; CSV curves."""
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(cfg.train.num_threads)
    torch.manual_seed(cfg.train.seed)
    np.random.seed(cfg.train.seed)

    train_ds = TrackDDataset("train", sysc=cfg.system, datac=cfg.data,
                             numeric=cfg.numeric)
    val_ds = TrackDDataset("val", sysc=cfg.system, datac=cfg.data,
                           numeric=cfg.numeric)
    train_ld = DataLoader(train_ds, batch_size=cfg.train.batch_size, shuffle=True,
                          collate_fn=collate, num_workers=0)
    val_ld = DataLoader(val_ds, batch_size=cfg.train.batch_size, shuffle=False,
                        collate_fn=collate, num_workers=0)

    model, meta = build_model(cfg, arm)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.train.lr,
                           weight_decay=cfg.train.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    ckpt = out_dir / "checkpoint.pt"
    best_path = out_dir / "best.pt"
    csv_path = out_dir / "curves.csv"
    start_epoch, best_val, best_epoch = 0, float("inf"), -1
    history: list[dict] = []

    if ckpt.exists():
        blob = torch.load(ckpt, map_location="cpu", weights_only=False)
        model.load_state_dict(blob["model"])
        opt.load_state_dict(blob["optimizer"])
        sched.load_state_dict(blob["scheduler"])
        start_epoch = blob["epoch"] + 1
        best_val = blob["best_val_nmse"]
        best_epoch = blob.get("best_epoch", -1)
        history = blob.get("history", [])
        print(f"  [{arm}] resumed at epoch {start_epoch}, "
              f"best val {db(best_val):.3f} dB")

    if start_epoch == 0:
        with csv_path.open("w", newline="", encoding="utf-8") as fh:
            csv.writer(fh).writerow(
                ["epoch", "train_loss_db", "val_nmse_db", "lr", "seconds",
                 "mean_alpha"])

    for epoch in range(start_epoch, epochs):
        t0 = time.time()
        model.train()
        run_loss, nb = 0.0, 0
        for batch in train_ld:
            opt.zero_grad()
            G0 = make_initial_batch(batch, cfg.train.init, cfg)
            out = model(G0, batch["Z"], batch["S"], batch["B"], batch["sigma2"])
            loss = nmse_loss(out, batch["G_true"])
            loss.backward()
            if cfg.train.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(),
                                               cfg.train.grad_clip)
            opt.step()
            run_loss += float(loss.detach())
            nb += 1
        sched.step()

        val = evaluate_split(model, val_ld, cfg)
        rec = {
            "epoch": epoch,
            "train_loss_db": db(run_loss / max(nb, 1)),
            "val_nmse_db": val["nmse_db"],
            "lr": sched.get_last_lr()[0],
            "seconds": round(time.time() - t0, 1),
            "mean_alpha": float(np.mean(model.initial_alphas())),
        }
        history.append(rec)
        with csv_path.open("a", newline="", encoding="utf-8") as fh:
            csv.writer(fh).writerow([rec[k] for k in
                                     ("epoch", "train_loss_db", "val_nmse_db",
                                      "lr", "seconds", "mean_alpha")])

        # Selection on VALIDATION only. Test is never consulted here.
        if val["nmse_ratio_of_sums"] < best_val:
            best_val = val["nmse_ratio_of_sums"]
            best_epoch = epoch
            torch.save({"model": model.state_dict(), "epoch": epoch,
                        "val_nmse": best_val, "config": cfg.to_dict()},
                       best_path)

        torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(),
                    "scheduler": sched.state_dict(), "epoch": epoch,
                    "best_val_nmse": best_val, "best_epoch": best_epoch,
                    "config": cfg.to_dict(), "history": history,
                    "arm": arm}, ckpt)

        if epoch % log_every == 0 or epoch == epochs - 1:
            print(f"  [{arm}] epoch {epoch:3d}  train {rec['train_loss_db']:7.2f} dB"
                  f"  val {rec['val_nmse_db']:7.2f} dB  {rec['seconds']:5.1f}s"
                  f"  alpha {rec['mean_alpha']:.3f}")

    return {**meta, "history": history, "best_val_nmse": best_val,
            "best_val_nmse_db": db(best_val), "best_epoch": best_epoch,
            "epochs": epochs, "early_stopping": "none (fixed budget)",
            "checkpoint_selection": "best validation ratio-of-sums NMSE",
            "best_path": str(best_path)}


# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate_test_once(cfg: TrackDConfig, arms: dict[str, dict],
                       n_test: int) -> dict:
    """ONE pass over the shared test set. Every method sees the same worlds."""
    ds = TrackDDataset("test", sysc=cfg.system, datac=cfg.data,
                       numeric=cfg.numeric)
    cd, rd = cfg.numeric.complex_dtype, cfg.numeric.real_dtype

    models = {}
    for arm, info in arms.items():
        m, _ = build_model(cfg, arm)
        blob = torch.load(info["best_path"], map_location="cpu",
                          weights_only=False)
        m.load_state_dict(blob["model"])
        m.eval()
        models[arm] = m

    methods = ["gs_spectral", "em_gs_spectral", "linearised_ls",
               "oracle_phase"] + list(arms)
    per_trial = {k: [] for k in methods}
    energies = {k: [0.0, 0.0] for k in methods}

    from rydberg_sim.forward import exact_forward
    from rydberg_sim.rng import get_operating_point_rngs
    from .torch_forward import least_squares_G

    for i in range(n_test):
        s = ds.sample(i)
        trial = s.trial
        ests = {
            "gs_spectral": run_gs(s, max_iter=cfg.baseline.T_GS,
                                  init="spectral", seed=trial),
            "em_gs_spectral": run_em_gs(s, max_iter=cfg.baseline.T_GS,
                                        init="spectral", seed=trial),
            "linearised_ls": run_linearised_ls(s),
        }
        rngs = get_operating_point_rngs(cfg.system.master_seed, trial,
                                        s.snr_db, s.rsr_db)
        ex = exact_forward(s.G_true, s.S, s.B, s.sigma2, rng_noise=rngs.noise)
        of = s.Z * np.exp(1j * np.angle(np.asarray(ex.E)))
        T = lambda a, d: torch.as_tensor(np.array(a, copy=True)[None], dtype=d)
        ests["oracle_phase"] = least_squares_G(
            T(of, torch.complex128) - T(s.B, torch.complex128),
            T(s.S, torch.complex128))[0].numpy()

        # MUST be the initializer the model was TRAINED with. Hardcoding
        # "spectral" here while train.init defaulted to "random" produced a
        # silent train/evaluate mismatch (caught at epoch 0 of the first run):
        # the model would have been evaluated out of distribution.
        G0np = make_initial_G(cfg.train.init, S=s.S, Z=s.Z, B=s.B, seed=trial)
        G0 = T(G0np, cd)
        Z, S, B = T(s.Z, rd), T(s.S, cd), T(s.B, cd)
        s2 = torch.tensor([s.sigma2], dtype=rd)
        for arm, m in models.items():
            ests[arm] = m(G0, Z, S, B, s2)[0].detach().numpy()

        for k, gh in ests.items():
            e, d = nmse_parts(gh, s.G_true)
            energies[k][0] += e
            energies[k][1] += d
            per_trial[k].append(e / d)

    out = {"n_test": n_test, "methods": {}}
    for k in methods:
        arr = np.array(per_trial[k])
        out["methods"][k] = {
            "nmse_ratio_of_sums_db": db(energies[k][0] / energies[k][1]),
            "nmse_median_db": db(np.median(arr)),
            "nmse_mean_of_ratios_db": db(arr.mean()),
        }

    # Paired differences vs EM-GS-spectral, in dB, per trial.
    base = np.array(per_trial["em_gs_spectral"])
    rng = np.random.default_rng(20260828)
    out["paired_vs_em_gs"] = {}
    for k in list(arms) + ["gs_spectral", "linearised_ls", "oracle_phase"]:
        d_db = 10 * np.log10(np.array(per_trial[k])) - 10 * np.log10(base)
        boot = np.array([np.median(rng.choice(d_db, d_db.size, replace=True))
                         for _ in range(2000)])
        lo, hi = float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))
        out["paired_vs_em_gs"][k] = {
            "median_diff_db": float(np.median(d_db)),
            "mean_diff_db": float(d_db.mean()),
            "std_diff_db": float(d_db.std(ddof=1)),
            "boot_ci95_median": [lo, hi],
            "ci_excludes_zero": bool(lo > 0 or hi < 0),
            "win_rate": float(np.mean(d_db < 0)),
            "percentiles_db": {p: float(np.percentile(d_db, p))
                               for p in (5, 25, 50, 75, 95)},
        }
    out["per_trial_nmse"] = {k: per_trial[k] for k in methods}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Track D stage 1")
    ap.add_argument("--i-have-approval", action="store_true", required=False)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--n-train", type=int, default=None)
    ap.add_argument("--n-test", type=int, default=None)
    ap.add_argument("--arms", type=str, default=",".join(ARMS))
    args = ap.parse_args(argv)

    if not args.i_have_approval:
        print("REFUSING: stage 1 requires --i-have-approval (PROMPT 4 Part B).")
        return 2

    cfg = TrackDConfig()
    # Stage 1 is SPECTRAL INIT THROUGHOUT (PROMPT 4 B1, and the pre-registered
    # criterion compares URformer-spectral against EM-GS-spectral). The package
    # default is "random" (paper-faithful), so stage 1 must override it
    # explicitly rather than inherit it.
    cfg = replace(cfg, train=replace(cfg.train, init="spectral"))
    if args.n_train or args.n_test:
        cfg = replace(cfg, data=replace(
            cfg.data,
            n_train=args.n_train or cfg.data.n_train,
            n_test=args.n_test or cfg.data.n_test))
    epochs = args.epochs or cfg.train.epochs
    assert cfg.train.init == "spectral", "stage 1 requires spectral init"
    print(f"initializer: {cfg.train.init} (train AND evaluate)")
    print(f"SNR training range: {cfg.data.snr_range_db} dB")
    print(f"RSR: {cfg.system.rsr_db} dB ours "
          f"= {cfg.system.rsr_paper_equiv_db:.2f} dB paper convention")

    RESULTS.mkdir(parents=True, exist_ok=True)
    summary_path = REPORTS / "trackD_stage1_results.json"
    summary = {"config": cfg.to_dict(), "epochs": epochs, "arms": {}}
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())

    for arm in args.arms.split(","):
        arm = arm.strip()
        print(f"\n=== {arm} ===")
        t0 = time.time()
        info = train_arm(cfg, arm, epochs=epochs, out_dir=RESULTS / arm)
        info["train_seconds"] = round(time.time() - t0, 1)
        summary["arms"][arm] = info
        summary_path.write_text(json.dumps(summary, indent=2) + "\n",
                                encoding="utf-8")
        print(f"  [{arm}] done in {info['train_seconds']}s, "
              f"best val {info['best_val_nmse_db']:.3f} dB "
              f"@ epoch {info['best_epoch']}")

    print("\n=== single test evaluation ===")
    test = evaluate_test_once(cfg, summary["arms"], cfg.data.n_test)
    summary["test"] = test
    summary_path.write_text(json.dumps(summary, indent=2) + "\n",
                            encoding="utf-8")
    for k, v in test["methods"].items():
        print(f"  {k:28s} {v['nmse_ratio_of_sums_db']:8.3f} dB "
              f"(median {v['nmse_median_db']:8.3f})")
    print("\n  paired vs EM-GS-spectral (negative = better):")
    for k, v in test["paired_vs_em_gs"].items():
        print(f"  {k:28s} median {v['median_diff_db']:+7.3f} dB  "
              f"CI [{v['boot_ci95_median'][0]:+.3f}, "
              f"{v['boot_ci95_median'][1]:+.3f}]  "
              f"excl0={v['ci_excludes_zero']}")
    print(f"\nwrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
