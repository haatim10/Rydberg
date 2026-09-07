"""PROMPT 12 Part C -- training runs. Scores P19, P20, P21.

Reuses the stage-4/5 machinery unchanged. The ONLY new variables are the
training seed and the loss mode; architecture, data budget, schedule, epoch
selection and evaluation are inherited so the new arms are matched to the
existing ones by construction.

  C1  seeds on the collapse.  C_U1_snr5_20 and C_H1_snr5_20 on two extra
      seeds -> three seeds per arm. Scores P19.
  C2  the missing arm.  H1 trained with the SNR-balanced loss over the FULL
      range, matched to the existing balanced U1 (C1_snr_balanced_P20).
      Scores P20.
  C3  log-domain loss control.  U1 trained with the per-sample loss taken in
      decibels. Scores P21.

Priority order is the brief's: C1 focused pair first, then C2, then C3.

Run one arm:
  PYTHONPATH=. python3 scratch/p12_partC.py --run C1_U1_seed2
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
from trackD_urformer.stage4 import BINS
from trackD_urformer.stage5 import (
    BIN_EDGES, BIN_WEIGHTS, snr_weights, weighted_nmse_loss,
)
from trackD_urformer.train import make_initial_batch

RESULTS = Path("results") / "p12" / "partC"
N_TRAIN, EPOCHS, HANKEL_RANK = 80_000, 13, 7
FULL_SNR = (-10.0, 20.0)
HIGH_SNR = (5.0, 20.0)

# arm -> (hankel, snr_range, loss mode, seed offset)
#   loss: "ratio"    per-sample normalised NMSE (the established loss)
#         "balanced" the same, weighted by the static per-bin table
#         "logdb"    the per-sample loss taken in decibels
RUNS: dict[str, dict] = {
    # ---- C1: seeds on the collapse (highest priority) ----
    "C1_U1_seed2": {"hankel": False, "snr": HIGH_SNR, "loss": "ratio", "seed": 2},
    "C1_H1_seed2": {"hankel": True,  "snr": HIGH_SNR, "loss": "ratio", "seed": 2},
    "C1_U1_seed3": {"hankel": False, "snr": HIGH_SNR, "loss": "ratio", "seed": 3},
    "C1_H1_seed3": {"hankel": True,  "snr": HIGH_SNR, "loss": "ratio", "seed": 3},
    # ---- C2: the missing arm, H1 under the recommended fix ----
    "C2_H1_balanced": {"hankel": True, "snr": FULL_SNR, "loss": "balanced",
                       "seed": 1},
    # ---- C3: log-domain loss control ----
    "C3_U1_logdb": {"hankel": False, "snr": FULL_SNR, "loss": "logdb",
                    "seed": 1},
    # ---- C1 extension: mixed-SNR seeds, only if budget remains ----
    "C1x_U1_mixed_seed2": {"hankel": False, "snr": FULL_SNR, "loss": "ratio",
                           "seed": 2},
    "C1x_H1_mixed_seed2": {"hankel": True,  "snr": FULL_SNR, "loss": "ratio",
                           "seed": 2},
}

PRIORITY = ["C1_U1_seed2", "C1_H1_seed2", "C1_U1_seed3", "C1_H1_seed3",
            "C2_H1_balanced", "C3_U1_logdb",
            "C1x_U1_mixed_seed2", "C1x_H1_mixed_seed2"]


def run_cfg(cfg: TrackDConfig, spec: dict) -> TrackDConfig:
    """Identical to the stage-4/5 config except for the SNR range and seed."""
    return replace(
        cfg,
        data=replace(cfg.data, n_train=N_TRAIN, snr_range_db=spec["snr"]),
        train=replace(cfg.train, seed=cfg.train.seed + 1000 * spec["seed"]),
        model=replace(cfg.model, filter_init="random", use_transformer=True,
                      use_hankel=spec["hankel"], hankel_rank=HANKEL_RANK,
                      hankel_mode="fixed", hankel_gate="none"))


def loss_fn(mode):
    """The three loss modes. 'ratio' is exactly the established nmse_loss."""
    if mode in ("ratio", "balanced"):
        def f(G_hat, G, snr_db):
            w = snr_weights(snr_db) if mode == "balanced" else None
            return weighted_nmse_loss(G_hat, G, w)
        return f
    if mode == "logdb":
        def f(G_hat, G, snr_db):
            num = torch.sum(torch.abs(G_hat - G) ** 2, dim=(1, 2))
            den = torch.sum(torch.abs(G) ** 2, dim=(1, 2))
            # 10 log10 of the per-sample normalised error, averaged.
            return torch.mean(10.0 * torch.log10(num / den + 1e-30))
        return f
    raise ValueError(mode)


@torch.no_grad()
def validation_per_trial(model, loader, cfg) -> np.ndarray:
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


def gradient_shares(model, loader, cfg, mode, n_batches=24) -> dict:
    """Realised share of gradient norm per SNR bin, for P21's first clause."""
    g = np.zeros(len(BIN_WEIGHTS))
    lf = loss_fn(mode)
    seen = 0
    for b in loader:
        if seen >= n_batches:
            break
        snr = b["snr_db"]
        idx = np.clip(np.digitize(snr.detach().cpu().numpy(), BIN_EDGES[1:-1]),
                      0, len(BIN_WEIGHTS) - 1)
        for j in range(len(BIN_WEIGHTS)):
            m = idx == j
            if not m.any():
                continue
            sub = {k: (v[m] if torch.is_tensor(v) and v.shape[:1] == snr.shape[:1]
                       else v) for k, v in b.items()}
            model.zero_grad(set_to_none=True)
            G0 = make_initial_batch(sub, cfg.train.init, cfg)
            est = model(G0, sub["Z"], sub["S"], sub["B"], sub["sigma2"])
            lf(est, sub["G_true"], sub["snr_db"]).backward()
            n = sum(float(p.grad.norm()) ** 2 for p in model.parameters()
                    if p.grad is not None)
            g[j] += np.sqrt(n) * float(m.sum())
        seen += 1
    model.zero_grad(set_to_none=True)
    tot = g.sum()
    share = (g / tot).tolist() if tot > 0 else g.tolist()
    below5 = float(sum(share[:3]))
    span = float(max(share) / min(share)) if min(share) > 0 else None
    return {"bins": [list(b) for b in BINS], "grad_share": share,
            "grad_share_below_5dB": below5, "span_max_over_min": span}


def train_one(name: str) -> dict:
    spec = RUNS[name]
    cfg = TrackDConfig()
    rcfg = run_cfg(cfg, spec)
    out_dir = RESULTS / name
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.set_num_threads(rcfg.train.num_threads)
    torch.manual_seed(rcfg.train.seed)
    np.random.seed(rcfg.train.seed)

    train_ds = TrackDDataset("train", sysc=rcfg.system, datac=rcfg.data,
                             numeric=rcfg.numeric, init=rcfg.train.init)
    val_ds = TrackDDataset("val", sysc=rcfg.system, datac=rcfg.data,
                           numeric=rcfg.numeric, init=rcfg.train.init)
    train_ld = DataLoader(train_ds, batch_size=rcfg.train.batch_size,
                          shuffle=True, collate_fn=collate, num_workers=0)
    val_ld = DataLoader(val_ds, batch_size=rcfg.train.batch_size, shuffle=False,
                        collate_fn=collate, num_workers=0)

    model, meta = build_model(rcfg, "arm1b_full_random")
    opt = torch.optim.Adam(model.parameters(), lr=rcfg.train.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    lf = loss_fn(spec["loss"])

    ckpt = out_dir / "checkpoint.pt"
    csv_path = out_dir / "curves.csv"
    vpt_path = out_dir / "val_per_trial.npy"
    start, history, vpt = 0, [], []
    if ckpt.exists():
        blob = torch.load(ckpt, map_location="cpu", weights_only=False)
        model.load_state_dict(blob["model"]); opt.load_state_dict(blob["optimizer"])
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
        for batch in train_ld:
            opt.zero_grad()
            G0 = make_initial_batch(batch, rcfg.train.init, rcfg)
            est = model(G0, batch["Z"], batch["S"], batch["B"], batch["sigma2"])
            loss = lf(est, batch["G_true"], batch["snr_db"])
            loss.backward()
            if rcfg.train.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(),
                                               rcfg.train.grad_clip)
            opt.step()
            run_loss += float(loss.detach()); nb += 1
        sched.step()
        v = validation_per_trial(model, val_ld, rcfg)
        vpt.append(v); np.save(vpt_path, np.asarray(vpt))
        rec = {"epoch": epoch, "train_loss": run_loss / max(nb, 1),
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
                    "history": history, "config": rcfg.to_dict(), "run": name},
                   ckpt)
        print(f"  [{name}] epoch {epoch:3d}  loss {rec['train_loss']:9.4f}  "
              f"val {rec['val_nmse_db']:7.2f} dB  {rec['seconds']:6.1f}s",
              flush=True)

    val_curve = np.array([r["val_nmse_db"] for r in history])
    vpt_arr = np.asarray(vpt)
    chosen, sel = select_epoch(val_curve, vpt_arr[int(np.argmin(val_curve))],
                               SELECTION_RULE)
    blob = torch.load(out_dir / f"ep{chosen:03d}.pt", map_location="cpu",
                      weights_only=False)
    torch.save({"model": blob["model"], "epoch": chosen,
                "config": rcfg.to_dict()}, out_dir / "best.pt")
    model.load_state_dict(blob["model"])

    out = {**meta, "run": name, "spec": spec, "n_train": N_TRAIN,
           "epochs": EPOCHS, "seed": rcfg.train.seed,
           "train_seconds": round(time.time() - t_start, 1),
           "history": history, "selection": sel, "chosen_epoch": chosen,
           "chosen_val_db": float(val_curve[chosen]),
           "best_val_db": float(val_curve.min()),
           "best_path": str(out_dir / "best.pt")}
    if spec["loss"] in ("logdb", "balanced"):
        out["gradient_shares"] = gradient_shares(model, train_ld, rcfg,
                                                 spec["loss"])
    (out_dir / "result.json").write_text(json.dumps(out, indent=1) + "\n",
                                         encoding="utf-8")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, choices=sorted(RUNS) + ["list"])
    a = ap.parse_args(argv)
    if a.run == "list":
        for n in PRIORITY:
            done = (RESULTS / n / "result.json").exists()
            print(f"  {'DONE' if done else '    '}  {n:22s} {RUNS[n]}")
        return 0
    RESULTS.mkdir(parents=True, exist_ok=True)
    if (RESULTS / a.run / "result.json").exists():
        print(f"skip (done): {a.run}")
        return 0
    print(f"=== PROMPT 12 Part C: {a.run}  {RUNS[a.run]} ===", flush=True)
    r = train_one(a.run)
    print(f"[{a.run}] DONE chosen_epoch={r['chosen_epoch']} "
          f"val={r['chosen_val_db']:.3f} dB in {r['train_seconds']:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
