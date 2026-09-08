"""PROMPT 16 Part B -- Tier 0.5: seed the +1.902 dB headline. Four runs.

Two extra seeds on each arm of the balanced-versus-unbalanced contrast at
P = 20. Nothing else: not the Hankel arms, not the log arm, not the mixed-SNR
design.

Everything is stage 5's procedure, reused rather than reimplemented: the same
model builder, the same bin weights, the same weighted loss, the same optimiser
and cosine schedule, the same one-SE epoch selection. Only the seed and the
`balanced` flag change between runs.

`init="spectral"` is set explicitly here. That is the defect PROMPT 15 Part A
found in scratch/p12_partC.py: TrackDConfig().train.init defaults to "random"
(config.py:324) and inheriting it silently trains the arm from a different G0
than every arm it will be compared against.

  PYTHONPATH=. python3 scratch/p16_tier05.py --run U1_seed2
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
    EPOCHS, FULL_SNR, HANKEL_RANK, N_TRAIN, snr_weights, weighted_nmse_loss,
)
from trackD_urformer.train import make_initial_batch

RESULTS = Path("results/p16/tier05")
REPORT = Path("reports/p16/tier05_runs.json")

# seed 1 is the existing pair (stage2/B3_80k_13ep, stage5/C1_snr_balanced_P20),
# both at cfg.train.seed = 20260827. The offsets below match the convention
# scratch/p12_partC.py used, so seed labels stay comparable across turns.
RUNS: dict[str, dict] = {
    "U1_seed2": {"balanced": False, "seed": 2},
    "U1_seed3": {"balanced": False, "seed": 3},
    "C1_seed2": {"balanced": True, "seed": 2},
    "C1_seed3": {"balanced": True, "seed": 3},
}
P = 20


def run_cfg(cfg: TrackDConfig, spec: dict) -> TrackDConfig:
    """Stage 5's config for P=20, with the seed offset and the initialiser."""
    return replace(
        cfg,
        system=replace(cfg.system, P=P),
        data=replace(cfg.data, n_train=N_TRAIN, snr_range_db=FULL_SNR),
        train=replace(cfg.train, seed=cfg.train.seed + 1000 * spec["seed"],
                      init="spectral"),
        model=replace(cfg.model, filter_init="random", use_transformer=True,
                      use_hankel=False, hankel_rank=HANKEL_RANK,
                      hankel_mode="fixed", hankel_gate="none"))


@torch.no_grad()
def validation_per_trial(model, loader, cfg) -> np.ndarray:
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
            csv.writer(fh).writerow(["epoch", "train_loss_db", "val_nmse_db",
                                     "lr", "seconds"])

    t_start = time.time()
    for epoch in range(start, EPOCHS):
        t0, run_loss, nb = time.time(), 0.0, 0
        for b in train_ld:
            opt.zero_grad()
            G0 = make_initial_batch(b, rcfg.train.init, rcfg)
            est = model(G0, b["Z"], b["S"], b["B"], b["sigma2"])
            w = snr_weights(b["snr_db"]) if spec["balanced"] else None
            loss = weighted_nmse_loss(est, b["G_true"], w)
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
               "seconds": round(time.time() - t0, 1)}
        history.append(rec)
        with csv_path.open("a", newline="", encoding="utf-8") as fh:
            csv.writer(fh).writerow([rec[k] for k in
                                     ("epoch", "train_loss_db", "val_nmse_db",
                                      "lr", "seconds")])
        torch.save({"model": model.state_dict()}, out_dir / f"ep{epoch:03d}.pt")
        torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(),
                    "scheduler": sched.state_dict(), "epoch": epoch,
                    "history": history, "config": rcfg.to_dict(),
                    "run": name}, ckpt)
        print(f"  [{name}] epoch {epoch:3d}  train {rec['train_loss_db']:7.2f}"
              f"  val {rec['val_nmse_db']:7.2f} dB  {rec['seconds']:6.1f}s",
              flush=True)

    val_curve = np.array([r["val_nmse_db"] for r in history])
    vpt_arr = np.asarray(vpt)
    chosen, sel = select_epoch(val_curve, vpt_arr[int(np.argmin(val_curve))],
                               SELECTION_RULE)
    blob = torch.load(out_dir / f"ep{chosen:03d}.pt", map_location="cpu",
                      weights_only=False)
    torch.save({"model": blob["model"], "epoch": chosen,
                "config": rcfg.to_dict()}, out_dir / "best.pt")
    return {**meta, "run": name, "balanced": spec["balanced"], "P": P,
            "seed_label": spec["seed"], "train_seed": rcfg.train.seed,
            "init": rcfg.train.init, "n_train": N_TRAIN, "epochs": EPOCHS,
            "history": history, "selection": sel, "chosen_epoch": chosen,
            "chosen_val_db": float(val_curve[chosen]),
            "best_val_db": float(val_curve.min()),
            "train_seconds": round(time.time() - t_start, 1)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=str, required=True, choices=list(RUNS))
    a = ap.parse_args(argv)
    RESULTS.mkdir(parents=True, exist_ok=True)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    info = train_run(a.run)
    cur = json.loads(REPORT.read_text()) if REPORT.exists() else {}
    cur[a.run] = info
    REPORT.write_text(json.dumps(cur, indent=1) + "\n", encoding="utf-8")
    print(f"  [{a.run}] done in {info['train_seconds']}s, epoch "
          f"{info['chosen_epoch']}, val {info['chosen_val_db']:.3f} dB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
