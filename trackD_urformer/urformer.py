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

Under PROMPT 2 this module contained no Hankel projection, no Cadzow, no rank
truncation and no ESPRIT (stop condition 2). PROMPT 6 lifts that for the
HS-URformer variant only: with ``ModelConfig.use_hankel=True`` a rank-``r``
Hankel projection is spliced between the LS step and the Transformer residual.
It is **off by default**, so the PROMPT 2 baseline is bit-identical to what it
was, and gate HK5 asserts the operator degenerates to the identity at full rank.
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

__all__ = ["URformerLayer", "URformer", "HankelGate", "count_parameters"]


class HankelGate(nn.Module):
    """``beta_t`` for the gated Hankel projection (PROMPT 7 B1).

    ``mode="scalar"`` (arm G1)
        one learned scalar per layer, exactly the ``alpha``-gate mechanism.
        Answers "does a gate help *without* SNR information?"
    ``mode="snr"`` (arm G2)
        ``sigmoid(MLP(standardized log sigma^2))``, a compact 2-layer MLP.
        ``sigma^2`` is already an estimator input -- ``kappa = 2 Z |Y| /
        sigma^2`` uses it -- so this adds no privileged information. It is
        perfectly anti-correlated with SNR (measured -1.000), so this is SNR
        conditioning by another name, and legitimately so.

    Both initialize at ``beta ~ sigmoid(gate_init)``, the alpha convention, so
    the model starts near plain URformer rather than near HS-URformer. For
    ``snr`` that is arranged by zeroing the MLP's output weight and setting its
    bias to ``gate_init``, which makes ``beta`` CONSTANT at initialization --
    identical to G1's starting point, so the two arms differ only in what they
    can learn, not where they start.
    """

    def __init__(self, mode: str, *, init: float = -2.0, hidden: int = 16,
                 mean: float = 0.0, std: float = 1.0) -> None:
        super().__init__()
        self.mode = str(mode)
        self.mean, self.std = float(mean), float(std)
        if self.mode == "scalar":
            self.g = nn.Parameter(torch.tensor(float(init)))
        elif self.mode == "snr":
            self.fc1 = nn.Linear(1, hidden)
            self.fc2 = nn.Linear(hidden, 1)
            nn.init.zeros_(self.fc2.weight)
            nn.init.constant_(self.fc2.bias, float(init))
        else:
            raise ValueError(f"unknown hankel gate mode {mode!r}")

    def forward(self, sigma2: torch.Tensor) -> torch.Tensor:
        """``(batch,)`` sigma^2 -> ``(batch, 1, 1)`` beta, broadcastable over G."""
        if self.mode == "scalar":
            b = torch.sigmoid(self.g).expand(sigma2.shape[0])
        else:
            x = ((torch.log(sigma2.reshape(-1, 1)) - self.mean) / self.std
                 ).to(self.fc1.weight.dtype)
            b = torch.sigmoid(self.fc2(torch.tanh(self.fc1(x)))).reshape(-1)
        return b.reshape(-1, 1, 1)


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
        # Arm 2 ("filteronly") does not CONSTRUCT the Transformer at all, so its
        # parameters do not exist rather than being zeroed or skipped.
        self.former = ChannelTransformer(
            N=N, K=K, d_model=mcfg.d_model, L_enc=mcfg.L_enc,
            n_heads=mcfg.n_heads, ffn_mult=mcfg.ffn_mult, dropout=mcfg.dropout,
            zero_init_out=True,
        ) if mcfg.use_transformer else None
        self.use_hankel = bool(mcfg.use_hankel)
        self.hankel_rank = int(mcfg.hankel_rank)
        self.hankel_mode = str(mcfg.hankel_mode)
        self.hankel_pencil = mcfg.hankel_pencil
        self.hankel_iters = int(mcfg.hankel_iters)
        self.hankel_gate_mode = str(getattr(mcfg, "hankel_gate", "none"))
        self.hankel_gate = (
            HankelGate(self.hankel_gate_mode, init=mcfg.hankel_gate_init,
                       hidden=mcfg.hankel_gate_hidden,
                       mean=mcfg.log_sigma2_mean, std=mcfg.log_sigma2_std)
            if self.use_hankel and self.hankel_gate_mode != "none" else None)
        # Gate overrides for the GK1/GK2 degeneration gates. None = learned.
        self._override_beta: float | None = None
        self._disable_hankel: bool = False
        # A3 diagnostic: use the exact autograd path through the SVD instead of
        # the straight-through estimator. NEVER set during training.
        self._exact_hankel_grad: bool = False

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

        # HS-URformer: the Hankel projection sits BETWEEN the LS step and the
        # Transformer residual, inside every unrolled layer.
        if self.use_hankel and not self._disable_hankel:
            if self._exact_hankel_grad:
                # A3 DIAGNOSTIC PATH ONLY, never training. Runs the projection
                # with autograd on, so the exact (ill-conditioned) SVD gradient
                # can be compared against the STE's.
                from .hankel import project_G_grad
                G_lin = project_G_grad(G_lin, rank=self.hankel_rank,
                                       pencil=self.hankel_pencil,
                                       n_iter=self.hankel_iters).to(G_lin.dtype)
                return (G_lin if self._disable_residual or self.former is None
                        else G_lin + self.former(G_lin))
            from .hankel import project_G
            proj = project_G(G_lin, rank=self.hankel_rank,
                             pencil=self.hankel_pencil,
                             n_iter=self.hankel_iters,
                             mode=self.hankel_mode).to(G_lin.dtype)
            # STRAIGHT-THROUGH ESTIMATOR. A fully detached projection severs
            # the unrolled gradient chain: gate HK6 measured EXACTLY ZERO
            # gradient in every layer except the last Transformer, because
            # @torch.no_grad() returns a leaf. PROMPT 6 justified detaching by
            # analogy to the LS/M-step, but that premise is wrong -- the LS
            # step is torch.linalg.solve and IS differentiable.
            #
            # forward:  G_lin -> proj            (exactly the projection)
            # backward: d/dG_lin = identity      (no path through the SVD)
            #
            # This satisfies both stated requirements at once: no gradient
            # through the ill-conditioned SVD, and gradients still flow around
            # the operator to every earlier layer.
            proj = G_lin + (proj - G_lin).detach()

            if self.hankel_gate is None and self._override_beta is None:
                G_lin = proj                      # stage-3 H1: unconditional
            else:
                # beta * Project(G_lin) + (1 - beta) * G_lin.
                # The (1 - beta) branch is an ordinary differentiable path, so
                # this form carries gradient to every earlier layer no matter
                # how the projection itself is treated -- the HK6 severing
                # failure is structurally impossible here.
                if self._override_beta is not None:
                    beta = torch.as_tensor(self._override_beta,
                                           dtype=G_lin.real.dtype,
                                           device=G_lin.device)
                else:
                    beta = self.hankel_gate(sigma2).to(G_lin.real.dtype)
                bc = beta.to(G_lin.dtype)
                G_lin = bc * proj + (1.0 - bc) * G_lin

        if self._disable_residual or self.former is None:
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

    def apply_filter_warmstart(self, cache_path: str | None = None) -> dict:
        """Load the Bessel warm start into every layer's FilterNet.

        Only meaningful when ``ModelConfig.filter_init == "emgs_warmstart"``;
        the caller is responsible for checking. Requires the cache produced by
        :func:`filter_net.warmstart_filternet`, which fits the MLP over a grid
        whose bounds were MEASURED from the training set -- never an assumed
        range. Returns the cached fit info so the achieved MSE is reportable.
        """
        from pathlib import Path

        path = Path(cache_path or self.mcfg.filter_warmstart_cache)
        if not path.exists():
            raise FileNotFoundError(
                f"warm-start cache {path} not found. Build it first with "
                "filter_net.warmstart_filternet(net, measure_kappa_range(ds))."
            )
        blob = torch.load(path, map_location="cpu", weights_only=False)
        for layer in self.layers:
            layer.filter_net.load_state_dict(blob["state_dict"])
            layer.filter_net.to(next(self.parameters()).dtype)
        return blob["info"]

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
                       disable_residual: bool = False,
                       disable_hankel: bool | None = None,
                       exact_hankel_grad: bool | None = None,
                       beta: float | None = None) -> None:
        for l in self.layers:
            l._override_filter = filter_override
            l._override_alpha = alpha
            l._disable_residual = disable_residual
            l._override_beta = beta
            if disable_hankel is not None:
                l._disable_hankel = bool(disable_hankel)
            if exact_hankel_grad is not None:
                l._exact_hankel_grad = bool(exact_hankel_grad)

    def _clear_test_mode(self) -> None:
        self._set_test_mode(filter_override=None, alpha=None,
                            disable_residual=False, disable_hankel=False,
                            exact_hankel_grad=False, beta=None)

    def initial_betas(self, sigma2: torch.Tensor) -> list[float]:
        """Each layer's ``beta_t`` at the given noise level. Printed into the report."""
        out = []
        for l in self.layers:
            if l.hankel_gate is None:
                out.append(1.0 if l.use_hankel else 0.0)
            else:
                with torch.no_grad():
                    out.append(float(l.hankel_gate(sigma2).reshape(-1)[0]))
        return out


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
            "transformer": n(l.former) if l.former is not None else 0,
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
