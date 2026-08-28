"""FilterNet - the learnable replacement for the fixed Bessel filter.

The classical EM-GS update weights the reconstruction by
``R(kappa) = I1(kappa)/I0(kappa)`` (rydberg_sim/gs.py:426). The paper replaces
that fixed kernel with "a compact two-layer MLP that maps the scalar SNR proxy
kappa to a filtering coefficient in the range (0,1)". This module is that MLP,
applied elementwise to the ``(batch, N, P)`` array of kappa.

    input 1 -> Linear(1,h) -> activation -> Linear(h,1) -> sigmoid -> (0,1)

Input transform (config.model.filter_input)
-------------------------------------------
``kappa`` spans several decades, so feeding it raw saturates the first layer.
Default is ``log1p_kappa``. The third option additionally supplies
``log(sigma2)``; it is a clearly labeled experiment, not the reference
architecture.

Warm start (config.model.filter_init == "emgs_warmstart")
---------------------------------------------------------
Pretrains the MLP to regress the exact ``i1e/i0e`` until MSE < 1e-4, so the
network starts as classical EM-GS and learns a correction.

The audit measured ``kappa_max ~ 116`` at the probe configuration -- three
orders of magnitude below the 1e5 a naive grid would assume. Fitting over
``logspace(-3, 5)`` would therefore spend most of the MLP's capacity on kappa
values that never occur. The grid bounds are instead **measured** from the
actual training dataset by :func:`measure_kappa_range` and passed in.
"""
from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn as nn

from .torch_forward import bessel_ratio_torch, em_kappa, forward_field

__all__ = ["FilterNet", "measure_kappa_range", "warmstart_filternet"]


class FilterNet(nn.Module):
    """Elementwise learnable filter ``kappa -> (0,1)``."""

    def __init__(
        self,
        hidden: int = 32,
        filter_input: str = "log1p_kappa",
        activation: str = "relu",
        predict_one_minus_R: bool = False,
    ) -> None:
        super().__init__()
        if filter_input not in ("kappa", "log1p_kappa", "log1p_kappa_plus_logsigma2"):
            raise ValueError(f"unknown filter_input {filter_input!r}")
        self.filter_input = filter_input
        # Labeled variant (PROMPT 4 A3): predict 1-R instead of R. Better
        # conditioned when R ~ 1 across most of the kappa range, because the
        # sigmoid's resolution is then spent on the small-kappa knee -- exactly
        # the low-local-SNR entries the filter exists to handle. The output
        # remains in (0,1) either way; only the target changes.
        self.predict_one_minus_R = bool(predict_one_minus_R)
        self.in_dim = 2 if filter_input == "log1p_kappa_plus_logsigma2" else 1
        self.fc1 = nn.Linear(self.in_dim, hidden)
        self.act = {"relu": nn.ReLU(), "tanh": nn.Tanh(),
                    "gelu": nn.GELU()}[activation]
        self.fc2 = nn.Linear(hidden, 1)

    def features(self, kappa: torch.Tensor, sigma2: torch.Tensor | None = None
                 ) -> torch.Tensor:
        """Build the elementwise input feature of shape ``(..., in_dim)``."""
        if self.filter_input == "kappa":
            return kappa.unsqueeze(-1)
        lk = torch.log1p(kappa).unsqueeze(-1)
        if self.filter_input == "log1p_kappa":
            return lk
        if sigma2 is None:
            raise ValueError(
                "filter_input='log1p_kappa_plus_logsigma2' requires sigma2"
            )
        s2 = sigma2.view(-1, 1, 1) if sigma2.ndim == 1 else sigma2
        ls = torch.log(s2).expand_as(kappa).unsqueeze(-1)
        return torch.cat([lk, ls], dim=-1)

    def forward(self, kappa: torch.Tensor, sigma2: torch.Tensor | None = None
                ) -> torch.Tensor:
        """Return ``R_learned`` with the same shape as ``kappa``, in (0,1)."""
        x = self.features(kappa, sigma2)
        out = torch.sigmoid(self.fc2(self.act(self.fc1(x)))).squeeze(-1)
        return 1.0 - out if self.predict_one_minus_R else out


@torch.no_grad()
def measure_kappa_range(
    dataset,
    *,
    eps: float,
    n_samples: int = 512,
    percentiles: tuple[float, ...] = (0.1, 1.0, 50.0, 99.0, 99.9),
) -> dict:
    """Measure the kappa distribution over the ACTUAL training data.

    The warm-start grid is calibrated to this, not to an assumed range
    (PROMPT 2 sec. 5.2). Uses the classical first iteration ``G=0``, which is
    where kappa is largest -- ``|Y| = |B|`` is dominated by the reference at
    the RSR levels we run.
    """
    n = min(int(n_samples), len(dataset))
    vals: list[torch.Tensor] = []
    for i in range(n):
        s = dataset.sample(i)
        Z = torch.as_tensor(s.Z, dtype=torch.float64)[None]
        S = torch.as_tensor(s.S, dtype=torch.complex128)[None]
        B = torch.as_tensor(s.B, dtype=torch.complex128)[None]
        G0 = torch.zeros((1, s.N, s.K), dtype=torch.complex128)
        sig = torch.tensor([s.sigma2], dtype=torch.float64)
        Y = forward_field(G0, S, B)
        vals.append(em_kappa(Z, Y, sig, eps).flatten())
    allk = torch.cat(vals)
    out = {
        "n_samples": n,
        "min": float(allk.min()),
        "max": float(allk.max()),
        "mean": float(allk.mean()),
    }
    for p in percentiles:
        out[f"p{p}"] = float(torch.quantile(allk, p / 100.0))
    # Grid bounds: from the low percentile up to 4x the observed max, so the
    # fit has headroom above anything training will present.
    lo = max(out["p0.1"], 1e-6)
    hi = 4.0 * out["max"]
    out["grid_lo"] = float(lo)
    out["grid_hi"] = float(hi)
    return out


def warmstart_filternet(
    net: FilterNet,
    kappa_stats: dict,
    *,
    n_grid: int = 4096,
    max_steps: int = 20000,
    lr: float = 1e-2,
    target_mse: float = 1e-4,
    target_max_abs: float | None = None,
    cache_path: str | Path | None = None,
    device: str = "cpu",
) -> dict:
    """Pretrain ``net`` to regress ``i1e(kappa)/i0e(kappa)``.

    Fits over ``logspace(log10(grid_lo), log10(grid_hi))`` with bounds measured
    from the training set. Caches the fitted state dict so runs are
    deterministic and the fit is not repeated.

    Returns a dict with the achieved MSE and the grid actually used.
    """
    if cache_path is not None:
        cache_path = Path(cache_path)
        if cache_path.exists():
            blob = torch.load(cache_path, map_location=device, weights_only=False)
            net.load_state_dict(blob["state_dict"])
            return blob["info"] | {"loaded_from_cache": True}

    lo, hi = kappa_stats["grid_lo"], kappa_stats["grid_hi"]
    kappa = torch.logspace(math.log10(lo), math.log10(hi), n_grid,
                           dtype=torch.float64, device=device)
    target = bessel_ratio_torch(kappa)

    net = net.to(device).double()
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    mse = float("inf")
    max_abs = float("inf")
    step = 0
    for step in range(1, max_steps + 1):
        opt.zero_grad()
        pred = net(kappa.view(1, 1, -1)).view(-1)
        err = pred - target
        # Train on MSE but ALSO track the worst-case error: MSE alone hides a
        # large residual concentrated at the small-kappa knee, because R ~ 1
        # over most of the measured range (PROMPT 4 A3).
        loss = torch.mean(err ** 2)
        loss.backward()
        opt.step()
        mse = float(loss.detach())
        max_abs = float(err.detach().abs().max())
        if target_max_abs is not None:
            if max_abs < target_max_abs:
                break
        elif mse < target_mse:
            break

    converged = (max_abs < target_max_abs if target_max_abs is not None
                 else mse < target_mse)
    info = {
        "achieved_mse": mse,
        "achieved_max_abs": max_abs,
        "target_mse": target_mse,
        "target_max_abs": target_max_abs,
        "converged": converged,
        "steps": step,
        "n_grid": n_grid,
        "grid_lo": lo,
        "grid_hi": hi,
        "kappa_stats": kappa_stats,
        "loaded_from_cache": False,
    }
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": net.state_dict(), "info": info}, cache_path)
    return info
