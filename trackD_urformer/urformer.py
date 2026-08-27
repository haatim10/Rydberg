"""URformer - the unrolled EM-GS network.

``T_UR`` layers with **untied weights**. Layer ``t`` (PROMPT 2 sec. 5):

    Y        = G^(t-1) @ S + B
    Y_direct = Z * unit_phase(Y)                    eps-guarded, no torch.angle
    kappa    = 2 Z |Y| / sigma2                     eps-guarded
    R_learn  = FilterNet_t(kappa)                   in (0,1)
    alpha_t  = sigmoid(g_t)                         one trainable scalar / layer
    Y_rec    = alpha_t * (Y_direct * R_learn) + (1 - alpha_t) * Y_direct
    G_lin    = LS(Y_rec - B, S)                     NOT learned; repository M-step
    G^(t)    = G_lin + Former_t(G_lin)              user-token Transformer

Degeneration properties, all asserted in verify.py:

* ``alpha = 0`` and zero residual  => exactly one classical GS update  (gate D)
* FilterNet replaced by exact ``i1e/i0e``, ``alpha = 1``, zero residual
  => exactly one classical EM-GS iteration                              (gate E)
* Transformer output weights zeroed => exactly the preceding fixed update (gate F)

No Hankel projection, no Cadzow, no rank truncation, no ESPRIT anywhere in this
module (PROMPT 2 stop condition 2).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .config import ModelConfig, NumericConfig
from .filter_net import FilterNet
from .torch_forward import (
    assert_shapes,
    bessel_ratio_torch,
    em_kappa,
    forward_field,
    least_squares_G,
    unit_phase,
)
from .transformer import ChannelTransformer

__all__ = ["URformerLayer", "URformer", "count_parameters"]


class URformerLayer(nn.Module):
    """One unrolled layer: gated learned filtering -> LS -> Transformer residual."""

    def __init__(self, N: int, K: int, mcfg: ModelConfig, eps: float,
                 ridge: float = 0.0) -> None:
        super().__init__()
        self.eps = float(eps)
        self.ridge = float(ridge)
        self.filter_net = FilterNet(
            hidden=mcfg.filter_hidden, filter_input=mcfg.filter_input
        )
        self.gate = nn.Parameter(torch.tensor(float(mcfg.gate_init_value)))
        self.former = ChannelTransformer(
            N=N, K=K, d_model=mcfg.d_model, L_enc=mcfg.L_enc,
            n_heads=mcfg.n_heads, ffn_mult=mcfg.ffn_mult, dropout=mcfg.dropout,
            zero_init_out=True,
        )
        # Test hooks. Never set during training; used only by verify.py gates.
        self._override_filter: str | None = None   # None | "exact_bessel"
        self._override_alpha: float | None = None
        self._disable_residual: bool = False

    @property
    def alpha(self) -> torch.Tensor:
        return torch.sigmoid(self.gate)

    def forward(self, G: torch.Tensor, Z: torch.Tensor, S: torch.Tensor,
                B: torch.Tensor, sigma2: torch.Tensor) -> torch.Tensor:
        assert_shapes(G=G, S=S, B=B, Z=Z, where="URformerLayer")
        Y = forward_field(G, S, B)
        Zc = Z.to(Y.dtype)
        Y_direct = Zc * unit_phase(Y, self.eps)
        kappa = em_kappa(Z, Y, sigma2, self.eps)

        if self._override_filter == "exact_bessel":
            R = bessel_ratio_torch(kappa)
        else:
            R = self.filter_net(kappa, sigma2).to(kappa.dtype)

        alpha = (self.alpha if self._override_alpha is None
                 else torch.tensor(self._override_alpha, dtype=kappa.dtype,
                                   device=kappa.device))
        Y_filt = Y_direct * R.to(Y.dtype)
        Y_rec = alpha.to(Y.dtype) * Y_filt + (1.0 - alpha).to(Y.dtype) * Y_direct

        G_lin = least_squares_G(Y_rec - B, S, ridge=self.ridge)
        if self._disable_residual:
            return G_lin
        return G_lin + self.former(G_lin)


class URformer(nn.Module):
    """``T_UR``-layer unrolled network, untied weights by default."""

    def __init__(self, N: int, K: int, mcfg: ModelConfig, numeric: NumericConfig,
                 ridge: float = 0.0) -> None:
        super().__init__()
        self.N, self.K = int(N), int(K)
        self.mcfg = mcfg
        self.numeric = numeric
        self.T_UR = int(mcfg.T_UR)
        eps = numeric.eps
        if mcfg.tie_layers:
            shared = URformerLayer(N, K, mcfg, eps, ridge)
            self.layers = nn.ModuleList([shared] * self.T_UR)
        else:
            self.layers = nn.ModuleList(
                [URformerLayer(N, K, mcfg, eps, ridge) for _ in range(self.T_UR)]
            )

    def initial_alphas(self) -> list[float]:
        """Initial ``alpha_t`` for every layer. Printed into the report."""
        return [float(l.alpha.detach()) for l in self.layers]

    def forward(self, G0: torch.Tensor, Z: torch.Tensor, S: torch.Tensor,
                B: torch.Tensor, sigma2: torch.Tensor,
                return_all: bool = False):
        """Run the unrolled stack.

        Returns the final estimate, or every layer's estimate when
        ``return_all`` (used by deep supervision and by convergence plots).
        """
        G = G0
        outs = []
        for layer in self.layers:
            G = layer(G, Z, S, B, sigma2)
            if return_all:
                outs.append(G)
        return outs if return_all else G

    # -- test hooks, used only by verify.py -------------------------------
    def _set_test_mode(self, *, filter_override: str | None = None,
                       alpha: float | None = None,
                       disable_residual: bool = False) -> None:
        for l in self.layers:
            l._override_filter = filter_override
            l._override_alpha = alpha
            l._disable_residual = disable_residual

    def _clear_test_mode(self) -> None:
        self._set_test_mode(filter_override=None, alpha=None, disable_residual=False)


def count_parameters(model: URformer) -> dict:
    """Parameter count broken down by module, per layer and total."""
    def n(mod) -> int:
        return sum(p.numel() for p in mod.parameters())

    per_layer = []
    for i, l in enumerate(model.layers):
        per_layer.append({
            "layer": i,
            "filter_net": n(l.filter_net),
            "gate": l.gate.numel(),
            "transformer": n(l.former),
            "total": n(l),
        })
    tied = model.mcfg.tie_layers
    total = sum(p.numel() for p in model.parameters())
    return {
        "tie_layers": tied,
        "T_UR": model.T_UR,
        "per_layer": per_layer,
        "totals": {
            "filter_net": sum(d["filter_net"] for d in per_layer),
            "gate": sum(d["gate"] for d in per_layer),
            "transformer": sum(d["transformer"] for d in per_layer),
            "all_parameters": total,
        },
        "note": (
            "with tie_layers=True the same module object is repeated, so "
            "per-layer sums exceed the true distinct-parameter total"
            if tied else "untied: per-layer sums equal the total"
        ),
    }
