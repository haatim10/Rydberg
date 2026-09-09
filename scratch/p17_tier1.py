"""PROMPT 17 Part C -- Tier 1, fourteen runs in four gated batches.

One driver for every batch, so that within any contrast both arms are trained
by the same code. That is what the PROMPT 17 Part B conditional exists to
detect the absence of, and running C1-C4 through separate scripts would
reintroduce exactly the confound Tier 0.5 hit.

  C1  focused Delta_H, seeds 2-3      4 runs   seed variance on +0.078
  C2  mixed   Delta_H, seeds 2-3      4 runs   seed variance on +1.209
  C3  H1-balanced,     seeds 1-3      3 runs   P30
  C4  U1-log,          seeds 1-3      3 runs   P31

All arms init="spectral", 80k / 13 epochs, N=32, K=3, L_k ~ U{3..7}, P=20,
RSR 10 dB, matched on data order and schedule. Only the seed, the prior, the
training SNR range and the loss change.

Seed 1 exists already for C1 and C2 (stage 4 and stage 2/3 respectively) and is
NOT retrained by those batches; C3 and C4 have no valid seed 1, because the
only arms that ever tested them were the init='random' ones PROMPT 15
invalidated, so all three seeds are trained here.

  PYTHONPATH=. python3 scratch/p17_tier1.py --run C1_H1_seed2
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

from trackD_urformer.config import TrackDConfig
from trackD_urformer.dataset import TrackDDataset, collate
from trackD_urformer.stage1 import build_model, db
from trackD_urformer.stage2 import SELECTION_RULE, select_epoch
from trackD_urformer.stage5 import (
    EPOCHS, HANKEL_RANK, N_TRAIN, snr_weights, weighted_nmse_loss,
)
from trackD_urformer.train import make_initial_batch

RESULTS = Path("results/p17/tier1")
REPORT = Path("reports/p17/tier1_runs.json")

FOCUSED = (5.0, 20.0)
MIXED = (-10.0, 20.0)
P = 20

# batch -> run name -> spec. `loss` is one of ratio | balanced | logdb.
BATCHES: dict[str, dict[str, dict]] = {
    "C1": {  # focused-training Delta_H; seed 1 is the stage-4 pair
        f"C1_{arm}_seed{s}": {"hankel": arm == "H1", "snr": FOCUSED,
                              "loss": "ratio", "seed": s}
        for arm in ("U1", "H1") for s in (2, 3)},
    "C2": {  # mixed-SNR Delta_H; seed 1 is stage2/B3 and stage3/H1
        f"C2_{arm}_seed{s}": {"hankel": arm == "H1", "snr": MIXED,
                              "loss": "ratio", "seed": s}
        for arm in ("U1", "H1") for s in (2, 3)},
    "C3": {  # H1 under the balanced loss; no valid seed 1 exists
        f"C3_H1bal_seed{s}": {"hankel": True, "snr": MIXED,
                              "loss": "balanced", "seed": s}
        for s in (1, 2, 3)},
    "C4": {  # U1 under the log-domain loss; no valid seed 1 exists
        f"C4_U1log_seed{s}": {"hankel": False, "snr": MIXED,
                              "loss": "logdb", "seed": s}
        for s in (1, 2, 3)},
}
RUNS = {k: v for b in BATCHES.values() for k, v in b.items()}


# Seed label -> multiplier on the 1000-step offset. Irregular on purpose:
# label 1 must reproduce the published arms' seed exactly (offset 0), and
# labels 2 and 3 must match what scratch/p12_partC.py and
# scratch/p16_tier05.py already used, so that "seed 2" names the SAME random
# seed in Tier 0.5 and Tier 1. A tidy 0,1,2 mapping here would silently make
# Tier 1's seed 2 a different draw from Tier 0.5's, which is precisely the
# kind of mismatch that cost PROMPT 15 a turn.
SEED_OFFSET = {1: 0, 2: 2, 3: 3}


def run_cfg(cfg: TrackDConfig, spec: dict) -> TrackDConfig:
    """Seed label 1 reproduces the published seed (20260827) exactly."""
    return replace(
        cfg,
        system=replace(cfg.system, P=P),
        data=replace(cfg.data, n_train=N_TRAIN, snr_range_db=spec["snr"]),
        train=replace(cfg.train,
                      seed=cfg.train.seed + 1000 * SEED_OFFSET[spec["seed"]],
                      init="spectral"),
        model=replace(cfg.model, filter_init="random", use_transformer=True,
                      use_hankel=spec["hankel"], hankel_rank=HANKEL_RANK,
                      hankel_mode="fixed", hankel_gate="none"))


def loss_fn(mode: str):
    """ratio is exactly nmse_loss; balanced is stage 5's; logdb is per-sample
    error taken in decibels before averaging."""
    if mode in ("ratio", "balanced"):
        def f(G_hat, G, snr_db):
            w = snr_weights(snr_db) if mode == "balanced" else None
            return weighted_nmse_loss(G_hat, G, w)
        return f
    if mode == "logdb":
        def f(G_hat, G, snr_db):
            num = torch.sum(torch.abs(G_hat - G) ** 2, dim=(1, 2))
            den = torch.sum(torch.abs(G) ** 2, dim=(1, 2))
            return torch.mean(10.0 * torch.log10(num / den + 1e-30))
        return f
    raise ValueError(mode)


@torch.no_grad()
def validation_per_trial(model, loader, cfg) -> np.ndarray:
    """Always the UNWEIGHTED per-trial NMSE, whatever the training loss, so
    epoch selection is comparable across arms."""
    model.eval()
    out = []
    for b in loader:
        G0 = make_initial_batch(b, cfg.train.init, cfg)
        est = model(G0, b["Z"], b["S"], b["B"], b["sigma2"])
        num = torch.sum(torch.abs(est - b["G_true"]) ** 2, dim=(1, 2))
        den = torch.sum(torch.abs(b["G_true"]) ** 2, dim=(1, 2))
        out.extend((num / den).tolist())
    model.train()
    return np.asarray(out, dtype=np.float64)


def train_run(name: str) -> dict:
    spec = RUNS[name]
    out_dir = RESULTS / name
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = TrackDConfig()
    rcfg = run_cfg(cfg, spec)
    assert rcfg.train.init == "spectral", "initialiser not set"
    torch.set_num_threads(rcfg.train.num_threads)
    torch.manual_seed(rcfg.train.seed)
    np.random.seed(rcfg.train.seed)
    lossf = loss_fn(spec["loss"])

    train_ds = TrackDDataset("train", sysc=rcfg.system, datac=rcfg.data,
                             numeric=rcfg.numeric, P=P, init=rcfg.train.init)
    val_ds = TrackDDataset("val", sysc=rcfg.system, datac=rcfg.data,
                           numeric=rcfg.numeric, P=P, init=rcfg.train.init)
    train_ld = DataLoader(train_ds, batch_size=rcfg.train.batch_size,
                          shuffle=True, collate_fn=collate, num_workers=0)
    val_ld = DataLoader(val_ds, batch_size=rcfg.train.batch_size, shuffle=False,
                        collate_fn=collate, num_workers=0)

    model, meta = build_model(rcfg, "arm1b_full_random")
    opt = torch.optim.Adam(model.parameters(), lr=rcfg.train.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

    ckpt, csv_path = out_dir / "checkpoint.pt", out_dir / "curves.csv"
    vpt_path = out_dir / "val_per_trial.npy"
    start, history, vpt = 0, [], []
    if ckpt.exists():
        blob = torch.load(ckpt, map_location="cpu", weights_only=False)
        model.load_state_dict(blob["model"])
        opt.load_state_dict(blob["optimizer"])
        sched.load_state_dict(blob["scheduler"])
        start, history = blob["epoch"] + 1, blob.get("history", [])
        if vpt_path.exists():
            vpt = list(np.load(vpt_path))
        print(f"  [{name}] resumed at epoch {start}", flush=True)
    if start == 0:
        with csv_path.open("w", newline="", encoding="utf-8") as fh:
            csv.writer(fh).writerow(["epoch", "train_loss", "val_nmse_db",
                                     "lr", "seconds"])

    t_start = time.time()
    for epoch in range(start, EPOCHS):
        t0, run_loss, nb = time.time(), 0.0, 0
        for b in train_ld:
            opt.zero_grad()
            G0 = make_initial_batch(b, rcfg.train.init, rcfg)
            est = model(G0, b["Z"], b["S"], b["B"], b["sigma2"])
            loss = lossf(est, b["G_true"], b["snr_db"])
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
        # logdb's loss is already in dB; the other two are ratios.
        tl = run_loss / max(nb, 1)
        rec = {"epoch": epoch,
               "train_loss": tl if spec["loss"] == "logdb" else db(tl),
               "val_nmse_db": db(v.mean()), "lr": sched.get_last_lr()[0],
               "seconds": round(time.time() - t0, 1)}
        history.append(rec)
        with csv_path.open("a", newline="", encoding="utf-8") as fh:
            csv.writer(fh).writerow([rec[k] for k in
                                     ("epoch", "train_loss", "val_nmse_db",
                                      "lr", "seconds")])
        torch.save({"model": model.state_dict()}, out_dir / f"ep{epoch:03d}.pt")
        torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(),
                    "scheduler": sched.state_dict(), "epoch": epoch,
                    "history": history, "config": rcfg.to_dict(),
                    "run": name, "spec": spec}, ckpt)
        print(f"  [{name}] epoch {epoch:3d}  train {rec['train_loss']:8.2f}"
              f"  val {rec['val_nmse_db']:7.2f} dB  {rec['seconds']:6.1f}s",
              flush=True)

    val_curve = np.array([r["val_nmse_db"] for r in history])
    vpt_arr = np.asarray(vpt)
    chosen, sel = select_epoch(val_curve, vpt_arr[int(np.argmin(val_curve))],
                               SELECTION_RULE)
    blob = torch.load(out_dir / f"ep{chosen:03d}.pt", map_location="cpu",
                      weights_only=False)
    torch.save({"model": blob["model"], "epoch": chosen,
                "config": rcfg.to_dict(), "spec": spec}, out_dir / "best.pt")
    return {**meta, "run": name, "spec": spec, "P": P,
            "train_seed": rcfg.train.seed, "init": rcfg.train.init,
            "n_train": N_TRAIN, "epochs": EPOCHS, "history": history,
            "selection": sel, "chosen_epoch": chosen,
            "chosen_val_db": float(val_curve[chosen]),
            "best_val_db": float(val_curve.min()),
            "train_seconds": round(time.time() - t_start, 1)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=str, choices=list(RUNS))
    ap.add_argument("--list-batch", type=str, choices=list(BATCHES))
    a = ap.parse_args(argv)
    if a.list_batch:
        print(" ".join(BATCHES[a.list_batch]))
        return 0
    if not a.run:
        ap.error("--run or --list-batch required")
    RESULTS.mkdir(parents=True, exist_ok=True)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    info = train_run(a.run)
    # Each run owns one file; the aggregate is rebuilt by scanning them. Four
    # concurrent runs of near-identical length can finish inside the same
    # read-modify-write of a shared JSON and silently drop an entry, which
    # Tier 0.5 escaped only because its queue staggered the launches.
    (RESULTS / a.run / "result.json").write_text(
        json.dumps(info, indent=1) + "\n", encoding="utf-8")
    cur = {p.parent.name: json.loads(p.read_text())
           for p in sorted(RESULTS.glob("*/result.json"))}
    REPORT.write_text(json.dumps(cur, indent=1) + "\n", encoding="utf-8")
    print(f"  [{a.run}] done in {info['train_seconds']}s, epoch "
          f"{info['chosen_epoch']}, val {info['chosen_val_db']:.3f} dB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
