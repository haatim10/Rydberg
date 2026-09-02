"""Track D stage 5 - matched-training controls (PROMPT 9 Part C).

Four runs, all 80k / 13 epochs, N=32, K=3, L_k ~ U{3..7}, RSR 10 dB (ours),
spectral init, matched to stage 2/4 on seed, data order and schedule:

    C1  URformer, SNR-BALANCED loss, P=20   -- highest priority, runs first
    C2  URformer, matched training at P=10
    C3  URformer, matched training at P=35
    C4  G1 (scalar-gated Hankel), P=10 only

The SNR-balancing scheme, stated exactly
----------------------------------------
``nmse_loss`` is the mean of per-sample ``||dG||^2/||G||^2``. SNR is drawn
uniform on [-10,20], so bin COUNTS are already balanced; the imbalance is
entirely in per-sample MAGNITUDE (PROMPT 7 A4: 94.3% of loss and 85.9% of
gradient norm from below 5 dB).

So the weight is the inverse of the measured per-bin mean loss, normalised to
unit mean weight so the effective learning rate is unchanged:

    w(bin) = c / m(bin),   c = 1 / mean_bin(1 / m(bin))

with m(bin) taken from the U1 baseline measured in
reports/trackD_partA7_diagnostics.json (A4). This is a STATIC, precomputed
weighting -- not adaptive, not tuned, and reproducible from that file:

    bin      [-10,-5) [-5,0)  [0,5)   [5,10)  [10,15) [15,20)
    m         0.6852  0.3425  0.1228  0.0555  0.0232  0.0119
    w         0.0556  0.1113  0.3102  0.6871  1.6410  3.1948

Mean weight is exactly 1.0; the high/low ratio is 57.5x. The realized per-bin
gradient shares are measured after training and reported, to confirm the
balancing actually worked rather than assuming it.

Run:  PYTHONPATH=. python3 -m trackD_urformer.stage5 --i-have-approval
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
from .stage2 import SELECTION_RULE, select_epoch
from .stage3 import paired
from .stage4 import BINS, by_bin
from .train import make_initial_batch

RESULTS = Path("results") / "track_d" / "stage5"
REPORTS = Path("reports")
N_TRAIN, EPOCHS, HANKEL_RANK = 80_000, 13, 7
FULL_SNR = (-10.0, 20.0)

# Inverse of the measured per-bin mean loss, normalised to unit mean weight.
# Source: reports/trackD_partA7_diagnostics.json, A4, U1_urformer.
BIN_EDGES = np.array([-10.0, -5.0, 0.0, 5.0, 10.0, 15.0, 20.0])
BIN_WEIGHTS = np.array([0.055610, 0.111260, 0.310238,
                        0.687090, 1.640953, 3.194849])

RUNS: dict[str, dict] = {
    "C1_snr_balanced_P20": {"P": 20, "gate": "none", "hankel": False,
                            "balanced": True},
    "C2_urformer_P10":     {"P": 10, "gate": "none", "hankel": False,
                            "balanced": False},
    "C3_urformer_P35":     {"P": 35, "gate": "none", "hankel": False,
                            "balanced": False},
    "C4_g1_P10":           {"P": 10, "gate": "scalar", "hankel": True,
                            "balanced": False},
}


def snr_weights(snr_db: torch.Tensor) -> torch.Tensor:
    """Per-sample weight from the static bin table. Unit mean by construction."""
    idx = np.clip(np.digitize(snr_db.detach().cpu().numpy(), BIN_EDGES[1:-1]),
                  0, len(BIN_WEIGHTS) - 1)
    return torch.as_tensor(BIN_WEIGHTS[idx], dtype=snr_db.dtype,
                           device=snr_db.device)


def weighted_nmse_loss(G_hat, G, w=None):
    """Per-sample normalised NMSE, optionally weighted. w=None == nmse_loss."""
    num = torch.sum(torch.abs(G_hat - G) ** 2, dim=(1, 2))
    den = torch.sum(torch.abs(G) ** 2, dim=(1, 2))
    per = num / den
    if w is None:
        return torch.mean(per)
    return torch.mean(w.to(per.dtype) * per)


def run_cfg(cfg: TrackDConfig, spec: dict) -> TrackDConfig:
    return replace(
        cfg,
        system=replace(cfg.system, P=spec["P"]),
        data=replace(cfg.data, n_train=N_TRAIN, snr_range_db=FULL_SNR),
        model=replace(cfg.model, filter_init="random", use_transformer=True,
                      use_hankel=spec["hankel"], hankel_rank=HANKEL_RANK,
                      hankel_mode="fixed", hankel_gate=spec["gate"]))


def build_arm(cfg: TrackDConfig, name: str):
    m, meta = build_model(run_cfg(cfg, RUNS[name]), "arm1b_full_random")
    return m, {**meta, "arm": name, "spec": RUNS[name]}


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


def gradient_shares(model, loader, cfg, *, balanced: bool,
                    n_batches: int = 24) -> dict:
    """Realized share of gradient norm per SNR bin. Confirms the balancing."""
    shares = np.zeros(len(BIN_WEIGHTS))
    losses = np.zeros(len(BIN_WEIGHTS))
    for bi in range(len(BIN_WEIGHTS)):
        lo, hi = BIN_EDGES[bi], BIN_EDGES[bi + 1]
        tot_g, tot_l, seen = 0.0, 0.0, 0
        for b in loader:
            m = (b["snr_db"] >= lo) & (b["snr_db"] < hi)
            if not bool(m.any()):
                continue
            sub = {k: (v[m] if torch.is_tensor(v) else v) for k, v in b.items()}
            model.zero_grad(set_to_none=True)
            G0 = make_initial_batch(sub, cfg.train.init, cfg)
            est = model(G0, sub["Z"], sub["S"], sub["B"], sub["sigma2"])
            w = snr_weights(sub["snr_db"]) if balanced else None
            loss = weighted_nmse_loss(est, sub["G_true"], w)
            loss.backward()
            tot_g += float(torch.sqrt(sum((p.grad ** 2).sum()
                                          for p in model.parameters()
                                          if p.grad is not None)))
            tot_l += float(loss.detach())
            seen += 1
            if seen >= n_batches:
                break
        shares[bi] = tot_g / max(seen, 1)
        losses[bi] = tot_l / max(seen, 1)
    model.zero_grad(set_to_none=True)
    return {"bins": [[float(BIN_EDGES[i]), float(BIN_EDGES[i + 1])]
                     for i in range(len(BIN_WEIGHTS))],
            "grad_share": (shares / shares.sum()).tolist(),
            "loss_share": (losses / losses.sum()).tolist(),
            "grad_share_below_5dB": float(shares[:3].sum() / shares.sum()),
            "loss_share_below_5dB": float(losses[:3].sum() / losses.sum())}


def train_run(cfg: TrackDConfig, name: str, out_dir: Path) -> dict:
    spec = RUNS[name]
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(cfg.train.num_threads)
    torch.manual_seed(cfg.train.seed)
    np.random.seed(cfg.train.seed)
    rcfg = run_cfg(cfg, spec)

    train_ds = TrackDDataset("train", sysc=rcfg.system, datac=rcfg.data,
                             numeric=rcfg.numeric, P=spec["P"],
                             init=rcfg.train.init)
    val_ds = TrackDDataset("val", sysc=rcfg.system, datac=rcfg.data,
                           numeric=rcfg.numeric, P=spec["P"],
                           init=rcfg.train.init)
    train_ld = DataLoader(train_ds, batch_size=rcfg.train.batch_size,
                          shuffle=True, collate_fn=collate, num_workers=0)
    val_ld = DataLoader(val_ds, batch_size=rcfg.train.batch_size, shuffle=False,
                        collate_fn=collate, num_workers=0)

    model, meta = build_arm(cfg, name)
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
            csv.writer(fh).writerow([rec[k] for k in ("epoch", "train_loss_db",
                                                      "val_nmse_db", "lr",
                                                      "seconds")])
        torch.save({"model": model.state_dict()}, out_dir / f"ep{epoch:03d}.pt")
        torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(),
                    "scheduler": sched.state_dict(), "epoch": epoch,
                    "history": history, "config": rcfg.to_dict(),
                    "run": name}, ckpt)
        print(f"  [{name}] epoch {epoch:3d}  train {rec['train_loss_db']:7.2f} dB"
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
    model.load_state_dict(blob["model"])

    info = {**meta, "run": name, "n_train": N_TRAIN, "epochs": EPOCHS,
            "P": spec["P"], "balanced": spec["balanced"], "history": history,
            "selection": sel, "chosen_epoch": chosen,
            "chosen_val_db": float(val_curve[chosen]),
            "best_val_db": float(val_curve.min()),
            "best_path": str(out_dir / "best.pt")}
    if spec["balanced"]:
        info["bin_weights"] = BIN_WEIGHTS.tolist()
        info["realized_shares"] = gradient_shares(model, val_ld, rcfg,
                                                  balanced=True)
        info["unweighted_shares"] = gradient_shares(model, val_ld, rcfg,
                                                    balanced=False)
    return info


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Track D stage 5 (PROMPT 9 Part C)")
    ap.add_argument("--i-have-approval", action="store_true")
    ap.add_argument("--runs", type=str, default=",".join(RUNS))
    a = ap.parse_args(argv)
    if not a.i_have_approval:
        print("REFUSING: stage 5 requires --i-have-approval (PROMPT 9 Part C).")
        return 2

    cfg = replace(TrackDConfig(),
                  train=replace(TrackDConfig().train, init="spectral"))
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = REPORTS / "trackD_stage5_results.json"

    def merge_write(**up):
        cur = (json.loads(path.read_text()) if path.exists()
               else {"config": cfg.to_dict(), "runs": {}})
        for k, v in up.items():
            cur.setdefault("runs", {}).update(v) if k == "runs" else cur.update({k: v})
        path.write_text(json.dumps(cur, indent=2) + "\n", encoding="utf-8")
        return cur

    merge_write()
    for nm in [r.strip() for r in a.runs.split(",") if r.strip()]:
        print(f"\n=== {nm} ===", flush=True)
        t0 = time.time()
        info = train_run(cfg, nm, RESULTS / nm)
        info["train_seconds"] = round(time.time() - t0, 1)
        merge_write(runs={nm: info})
        print(f"  [{nm}] done in {info['train_seconds']}s, epoch "
              f"{info['chosen_epoch']}, val {info['chosen_val_db']:.3f} dB",
              flush=True)
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
