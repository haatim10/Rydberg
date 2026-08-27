"""Track D training loop.

Adam + cosine annealing, per paper Table I. Checkpoints carry model state,
optimizer state, scheduler state, epoch, best validation NMSE, the full config
and every random seed, so training can resume exactly.

Loss (PROMPT 2 sec. 7). Default is final-layer only::

    L_final = mean_batch( ||G - G_hat^(T)||_F^2 / ||G||_F^2 )

Deep supervision is available behind ``ModelConfig.deep_supervision`` but is
OFF in the reference run; it is a later ablation.

Because ``B`` is known there is no global phase ambiguity, so plain supervised
regression on complex ``G`` is valid. Phase-invariant losses from the generic
phase-retrieval literature would be wrong here and are deliberately not used.

Note on the loss vs the reported metric
---------------------------------------
The training loss is the **mean of per-sample normalized errors**. The reported
NMSE is the repository's **ratio-of-sums** (metrics.py:386). The two differ;
`evaluate.py` always reports the repository convention, and both are recorded.

This module is NOT executed in the build phase. Training starts only on
explicit approval.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import TrackDConfig
from .dataset import TrackDDataset, collate
from .urformer import URformer, count_parameters

__all__ = ["nmse_loss", "make_initial_batch", "evaluate_split", "train"]


def nmse_loss(G_hat: torch.Tensor, G: torch.Tensor) -> torch.Tensor:
    """Mean over the batch of ``||dG||_F^2 / ||G||_F^2`` (per-sample normalized)."""
    num = torch.sum(torch.abs(G_hat - G) ** 2, dim=(1, 2))
    den = torch.sum(torch.abs(G) ** 2, dim=(1, 2))
    return torch.mean(num / den)


def nmse_parts_torch(G_hat: torch.Tensor, G: torch.Tensor
                     ) -> tuple[torch.Tensor, torch.Tensor]:
    """Summed numerator and denominator, for ratio-of-sums aggregation."""
    return (torch.sum(torch.abs(G_hat - G) ** 2),
            torch.sum(torch.abs(G) ** 2))


def deep_supervision_loss(outs: list[torch.Tensor], G: torch.Tensor
                          ) -> torch.Tensor:
    """``sum_t w_t L_t`` with ``w_t = t / sum_t t`` (PROMPT 2 sec. 7)."""
    T = len(outs)
    ws = [(t + 1) / (T * (T + 1) / 2) for t in range(T)]
    return sum(w * nmse_loss(o, G) for w, o in zip(ws, outs))


def make_initial_batch(batch: dict, init: str, cfg: TrackDConfig) -> torch.Tensor:
    """Build ``G^(0)`` for a batch under the chosen initializer.

    ``random`` is generated in torch (fast, and matches the paper's
    "random complex Gaussian with normalized variance"). ``spectral`` and
    ``linearized_ls`` delegate to the repository's validated NumPy routines --
    they are not reimplemented here.
    """
    G_true = batch["G_true"]
    if init == "random":
        g = torch.randn_like(G_true.real), torch.randn_like(G_true.real)
        return torch.complex(g[0], g[1]).to(G_true.dtype) / np.sqrt(2.0)

    from .baselines import make_initial_G

    class _S:  # duck-typed view for the NumPy adapters
        pass

    outs = []
    for i in range(G_true.shape[0]):
        s = _S()
        s.S = batch["S"][i].detach().cpu().numpy().astype(np.complex128)
        s.Z = batch["Z"][i].detach().cpu().numpy().astype(np.float64)
        s.B = batch["B"][i].detach().cpu().numpy().astype(np.complex128)
        outs.append(make_initial_G(
            init, S=s.S, Z=s.Z, B=s.B,
            seed=int(batch["trial"][i]) if "trial" in batch else i,
        ))
    return torch.as_tensor(np.stack(outs), dtype=G_true.dtype)


@torch.no_grad()
def evaluate_split(model: URformer, loader: DataLoader, cfg: TrackDConfig
                   ) -> dict:
    """Validation pass. Reports BOTH the training-loss convention and the
    repository's ratio-of-sums NMSE."""
    model.eval()
    tot_num = tot_den = 0.0
    per_sample = []
    for batch in loader:
        G0 = make_initial_batch(batch, cfg.train.init, cfg)
        out = model(G0, batch["Z"], batch["S"], batch["B"], batch["sigma2"])
        num, den = nmse_parts_torch(out, batch["G_true"])
        tot_num += float(num)
        tot_den += float(den)
        n = torch.sum(torch.abs(out - batch["G_true"]) ** 2, dim=(1, 2))
        d = torch.sum(torch.abs(batch["G_true"]) ** 2, dim=(1, 2))
        per_sample.extend((n / d).tolist())
    model.train()
    ratio_of_sums = tot_num / tot_den
    mean_of_ratios = float(np.mean(per_sample))
    return {
        "nmse_ratio_of_sums": ratio_of_sums,
        "nmse_db": 10.0 * np.log10(max(ratio_of_sums, 1e-30)),
        "nmse_mean_of_ratios": mean_of_ratios,
        "nmse_mean_of_ratios_db": 10.0 * np.log10(max(mean_of_ratios, 1e-30)),
        "n_samples": len(per_sample),
    }


def train(cfg: TrackDConfig, out_dir: str | Path, *, resume: bool = True,
          max_epochs: int | None = None, log_every: int = 50) -> dict:
    """Full training loop. Not executed in the build phase."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
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

    model = URformer(cfg.system.N, cfg.system.K, cfg.model, cfg.numeric)
    if cfg.numeric.dtype == "float64":
        model = model.double()

    opt = torch.optim.Adam(model.parameters(), lr=cfg.train.lr,
                           weight_decay=cfg.train.weight_decay)
    epochs = int(max_epochs if max_epochs is not None else cfg.train.epochs)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    start_epoch, best = 0, float("inf")
    ckpt_path = out / "checkpoint.pt"
    if resume and ckpt_path.exists():
        blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(blob["model"])
        opt.load_state_dict(blob["optimizer"])
        sched.load_state_dict(blob["scheduler"])
        start_epoch = blob["epoch"] + 1
        best = blob["best_val_nmse"]
        print(f"resumed from epoch {start_epoch}, best val NMSE {best:.6e}")

    history = []
    for epoch in range(start_epoch, epochs):
        t0 = time.time()
        run_loss, n_batches = 0.0, 0
        for batch in train_ld:
            opt.zero_grad()
            G0 = make_initial_batch(batch, cfg.train.init, cfg)
            if cfg.model.deep_supervision:
                outs = model(G0, batch["Z"], batch["S"], batch["B"],
                             batch["sigma2"], return_all=True)
                loss = deep_supervision_loss(outs, batch["G_true"])
            else:
                o = model(G0, batch["Z"], batch["S"], batch["B"], batch["sigma2"])
                loss = nmse_loss(o, batch["G_true"])
            loss.backward()
            if cfg.train.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(),
                                               cfg.train.grad_clip)
            opt.step()
            run_loss += float(loss.detach())
            n_batches += 1
        sched.step()

        val = evaluate_split(model, val_ld, cfg)
        rec = {
            "epoch": epoch,
            "train_loss": run_loss / max(n_batches, 1),
            "train_loss_db": 10.0 * np.log10(max(run_loss / max(n_batches, 1), 1e-30)),
            "val_nmse_db": val["nmse_db"],
            "lr": sched.get_last_lr()[0],
            "seconds": round(time.time() - t0, 1),
            "alphas": model.initial_alphas(),
        }
        history.append(rec)
        print(f"epoch {epoch:3d}  train {rec['train_loss_db']:7.2f} dB  "
              f"val {rec['val_nmse_db']:7.2f} dB  {rec['seconds']:.1f}s")

        is_best = val["nmse_ratio_of_sums"] < best
        if is_best:
            best = val["nmse_ratio_of_sums"]
        torch.save({
            "model": model.state_dict(),
            "optimizer": opt.state_dict(),
            "scheduler": sched.state_dict(),
            "epoch": epoch,
            "best_val_nmse": best,
            "config": cfg.to_dict(),
            "seeds": {"torch": cfg.train.seed, "numpy": cfg.train.seed,
                      "master_seed": cfg.system.master_seed},
            "history": history,
        }, ckpt_path)
        if is_best:
            torch.save(torch.load(ckpt_path, map_location="cpu",
                                  weights_only=False), out / "best.pt")

        (out / "history.json").write_text(
            json.dumps({"config": cfg.to_dict(),
                        "params": count_parameters(model),
                        "history": history}, indent=2) + "\n",
            encoding="utf-8",
        )

    return {"best_val_nmse": best,
            "best_val_nmse_db": 10.0 * np.log10(max(best, 1e-30)),
            "history": history}
