"""Channel Transformer residual-correction module - paper-faithful USER tokens.

Tokenization (paper sec. III-C, "Residual Correction Module")
-------------------------------------------------------------
The complex ``G_linear in C^{N x K}`` is split into real and imaginary parts and
reshaped into **K tokens of dimension 2N** -- one token per user::

    x_k   = [Re(G_linear[:,k]) ; Im(G_linear[:,k])]   in R^{2N}
    X     in R^{K x 2N}
    V0    = Linear(2N -> d_model)(X) + PosEmb          PosEmb in R^{K x d_model}
            L_enc x pre-LN blocks:
                A = MHSA(LN(V)) + V
                V = FFN(LN(A)) + A
    T_out = Linear(d_model -> 2N)(V)                   in R^{K x 2N}
    G_res = complexify(T_out)                          in C^{N x K}

Self-attention therefore runs over the ``K`` users, letting the module capture
the inter-user dependencies that the per-element LS step cannot see.

Shape-locking (PROMPT 2 sec. 6) - IMPORTANT FOR EXPERIMENT D3
-------------------------------------------------------------
This scheme is shape-locked to ``(N, K)``:

* the input/output projections are ``2N``-dimensional  -> **N-dependent**
* the positional embedding is ``K x d_model``          -> **K-dependent**

So a model trained at ``N=32`` cannot be evaluated at ``N=16``; that is a shape
error, not a performance question. D3 requires one trained model per ``N``.
``P`` and SNR are architecture-free and need no retraining.

Detokenization must not conjugate: ``G_res = T_out[:, :N] + 1j*T_out[:, N:]``,
transposed back to ``(N, K)``. Gate G asserts no conjugation flip survives the
round trip.

The antenna-token variant is deliberately NOT implemented in this phase
(PROMPT 2 stop condition 3).
"""
from __future__ import annotations

import torch
import torch.nn as nn

__all__ = ["ChannelTransformer", "tokenize", "detokenize"]


def tokenize(G: torch.Tensor) -> torch.Tensor:
    """``(batch, N, K)`` complex -> ``(batch, K, 2N)`` real user tokens.

    Token ``k`` is ``[Re(G[:,k]) ; Im(G[:,k])]``.
    """
    if G.ndim != 3 or not G.is_complex():
        raise ValueError(
            f"tokenize expects batched complex (b,N,K), got {tuple(G.shape)} "
            f"dtype {G.dtype}"
        )
    Gt = G.transpose(1, 2)                       # (b, K, N)
    return torch.cat([Gt.real, Gt.imag], dim=-1)  # (b, K, 2N)


def detokenize(T: torch.Tensor, N: int, complex_dtype: torch.dtype) -> torch.Tensor:
    """``(batch, K, 2N)`` real -> ``(batch, N, K)`` complex. No conjugation.

    Inverse of :func:`tokenize`: the first ``N`` entries are the real part and
    the last ``N`` the imaginary part, so the sign of the imaginary part is
    carried through unchanged.
    """
    if T.ndim != 3 or T.shape[-1] != 2 * N:
        raise ValueError(
            f"detokenize expects (b,K,2N) with N={N}, got {tuple(T.shape)}"
        )
    re, im = T[..., :N], T[..., N:]
    return torch.complex(re, im).transpose(1, 2).to(complex_dtype)


class _EncoderBlock(nn.Module):
    """Pre-LN Transformer encoder block: A = MHSA(LN(V))+V; V = FFN(LN(A))+A."""

    def __init__(self, d_model: int, n_heads: int, ffn_mult: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_mult * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_mult * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, V: torch.Tensor) -> torch.Tensor:
        h = self.ln1(V)
        a, _ = self.attn(h, h, h, need_weights=False)
        V = V + a
        return V + self.ffn(self.ln2(V))


class ChannelTransformer(nn.Module):
    """Residual correction over USER tokens.

    Parameters
    ----------
    N, K
        Array size and user count. The module is shape-locked to both.
    zero_init_out
        If True the output projection is zeroed, so the module returns exactly
        zero at initialization and the URformer layer reduces to its fixed
        classical update. This is both the gate-F condition and a sane training
        default -- the network starts as the classical algorithm.
    """

    def __init__(
        self,
        N: int,
        K: int,
        d_model: int = 64,
        L_enc: int = 3,
        n_heads: int = 4,
        ffn_mult: int = 4,
        dropout: float = 0.0,
        zero_init_out: bool = True,
    ) -> None:
        super().__init__()
        self.N, self.K, self.d_model = int(N), int(K), int(d_model)
        self.in_proj = nn.Linear(2 * N, d_model)
        self.pos = nn.Parameter(torch.zeros(K, d_model))
        nn.init.normal_(self.pos, std=0.02)
        self.blocks = nn.ModuleList(
            [_EncoderBlock(d_model, n_heads, ffn_mult, dropout) for _ in range(L_enc)]
        )
        self.ln_out = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, 2 * N)
        if zero_init_out:
            nn.init.zeros_(self.out_proj.weight)
            nn.init.zeros_(self.out_proj.bias)

    def forward(self, G_linear: torch.Tensor) -> torch.Tensor:
        """``(b,N,K)`` complex -> complex residual of the same shape."""
        if G_linear.shape[1] != self.N or G_linear.shape[2] != self.K:
            raise ValueError(
                f"ChannelTransformer is shape-locked to (N={self.N}, K={self.K}); "
                f"got {tuple(G_linear.shape[1:])}. Train a separate model per N "
                "(see module docstring, D3)."
            )
        cdtype = G_linear.dtype
        # The Transformer runs in real arithmetic at the module's own precision.
        X = tokenize(G_linear).to(self.in_proj.weight.dtype)   # (b, K, 2N)
        V = self.in_proj(X) + self.pos
        for blk in self.blocks:
            V = blk(V)
        T = self.out_proj(self.ln_out(V))                      # (b, K, 2N)
        return detokenize(T, self.N, cdtype)
