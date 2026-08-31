"""Hankel structure operator for HS-URformer (PROMPT 6 Part A).

PER-USER, PER-COLUMN. Each column ``g_k in C^N`` is a sum of ``L_k`` complex
exponentials in ``n``, so its Hankel matrix has rank exactly ``L_k`` (given
``N >= 2 L_k - 1``). The operator is built from *that length-N vector*, one per
user, and the results stacked. It is **never** built from ``G`` as a whole --
that is the easiest thing to get wrong here, and the shape assertions below
exist to make getting it wrong loud.

    H(g_k)    Hankel embedding of one column
    Pi_r      truncated SVD, keep top r triplets
    H^-1      anti-diagonal averaging back to length N

Shape convention -- INHERITED FROM TRACK B, not from the prompt
---------------------------------------------------------------
``rydberg_sim.track_b_structure.hankel_matrix`` uses ``p = N//2`` and builds a
``(N-p) x (p+1)`` matrix: 16x17 at N=32. PROMPT 6 specifies
``(N-p+1) x p`` = 17x16. These are transposes of one another, with identical
singular values, identical rank, and identical range of representable signals,
so the projection they induce is the same. The repository convention wins (the
standing rule), reuse is the stated priority, and gate HK4 asserts parity
against Track B's implementation on identical input.

Differentiability -- NO GRADIENT THROUGH THE SVD, BUT THE CHAIN SURVIVES
------------------------------------------------------------------------
``Pi_r`` is a truncated SVD, and SVD gradients carry ``1/(sigma_i^2 -
sigma_j^2)`` terms that blow up on near-degenerate singular values -- routine at
low SNR on a rank-7 truncation. So :func:`project_G` is ``@torch.no_grad()``:
the operator itself is fixed and non-differentiable, and the prior is *imposed*,
not learned.

That is NOT the same as detaching the projection from the network. A
``no_grad`` result is a leaf, and splicing a leaf into the unrolled stack
severs every layer below it -- gate HK6 measured exactly zero gradient in all
but the last Transformer when we did that. The call site in
:class:`urformer.URformerLayer` therefore wraps this function in a
straight-through estimator (``x + (f(x) - x).detach()``): forward is the exact
projection, backward is the identity. Both requirements hold at once -- no
gradient through the SVD, and gradients still reach every earlier layer.

One step is NOT a projection -- read Delta_H with this in mind
--------------------------------------------------------------
``Pi_r`` leaves the rank-r matrix set, but a truncated matrix is no longer
Hankel, and ``H^-1`` (anti-diagonal averaging) puts energy back above rank r.
So ``H^-1 . Pi_r . H`` is one ALTERNATING-PROJECTION sweep between the rank set
and the Hankel set, not a projection onto their intersection, and it is not
idempotent on generic input. That is why Cadzow iterates, and why Track B's
``hs_gs`` defaults to ``cadzow_iter=4``.

Measured here at ``N=32, r=7`` from a generic complex Gaussian vector, as the
fraction of Hankel spectral energy still outside rank 7:

    n_iter    1        2        4        8
    tail      3.7e-2   2.0e-2   8.4e-3   1.4e-3

HS-URformer runs at ``hankel_iters=1``, because that is the operator PROMPT 6
specifies. The structure it imposes is therefore APPROXIMATE -- about 4% of the
column's energy remains off-manifold -- and a small ``Delta_H`` has a reading
other than "the prior does not help": the prior may simply not have been
imposed very hard. ``stage3`` measures that directly by sweeping ``n_iter`` on
the training-free ``U1+post`` arm. Note also that H0 (Track B's HS-EM-GS) uses
four sweeps, so the classical and learned contrasts are NOT iteration-matched.

Rank rule -- three settings, never conflated
--------------------------------------------
``r = 7`` fixed (``L_max``)     PRIMARY. A system design assumption, the same one
                               the channel generator uses. NOT oracle information.
``adaptive``                   secondary; MDL on the Hankel spectrum (Track B's
                               ``estimate_order``), capped at ``L_max``.
``oracle`` (true ``L_k``)      DIAGNOSTIC UPPER BOUND ONLY. This *is* privileged
                               information URformer never receives. Never a headline.
"""
from __future__ import annotations

import numpy as np
import torch

__all__ = ["hankel_matrix_torch", "hankel_to_vector_torch", "hankel_project_torch",
           "project_G", "project_G_grad", "singular_gap_stats", "RANK_MODES",
           "max_representable_rank"]

RANK_MODES = ("fixed", "adaptive", "oracle")


def _pencil(N: int, pencil: int | None) -> int:
    """Track B's default: ``p = N // 2``."""
    p = int(pencil) if pencil is not None else N // 2
    if not (1 <= p <= N - 1):
        raise ValueError(f"pencil must satisfy 1 <= p <= N-1, got {p} for N={N}")
    return p


def max_representable_rank(N: int, pencil: int | None = None) -> int:
    """Largest rank the embedding can carry: ``min(N-p, p+1)``.

    At ``N=8, p=4`` this is 4, so ``L_k >= 5`` is NOT representable -- the
    documented HK7 degeneracy.
    """
    p = _pencil(N, pencil)
    return min(N - p, p + 1)


def hankel_matrix_torch(g: torch.Tensor, pencil: int | None = None) -> torch.Tensor:
    """``(..., N)`` -> ``(..., N-p, p+1)``. Matches Track B's layout exactly."""
    if g.shape[-1] < 2:
        raise ValueError(f"need N >= 2, got {g.shape[-1]}")
    N = g.shape[-1]
    p = _pencil(N, pencil)
    rows = N - p
    # unfold gives sliding windows of length p+1, stride 1 -> (..., rows+?, p+1)
    return g.unfold(-1, p + 1, 1)[..., :rows, :]


def hankel_to_vector_torch(H: torch.Tensor) -> torch.Tensor:
    """Anti-diagonal averaging, the inverse of :func:`hankel_matrix_torch`."""
    rows, cols = H.shape[-2], H.shape[-1]
    N = rows + cols - 1
    lead = H.shape[:-2]
    idx = (torch.arange(rows, device=H.device).view(-1, 1)
           + torch.arange(cols, device=H.device).view(1, -1)).reshape(-1)
    flat = H.reshape(*lead, rows * cols)
    out = torch.zeros(*lead, N, dtype=H.dtype, device=H.device)
    out.index_add_(-1, idx, flat)
    cnt = torch.zeros(N, dtype=H.real.dtype, device=H.device)
    cnt.index_add_(0, idx, torch.ones_like(idx, dtype=H.real.dtype))
    return out / cnt.to(out.dtype)


def hankel_project_torch(g: torch.Tensor, rank: int, *, pencil: int | None = None,
                         n_iter: int = 1) -> torch.Tensor:
    """Batched Cadzow projection of ``(..., N)`` onto rank-``rank`` structure.

    ``n_iter=1`` is exactly ``H^-1 . Pi_r . H`` as specified. More iterations
    are Track B's Cadzow, available but not the primary.
    """
    rank = max(1, int(rank))
    cur = g
    for _ in range(max(1, int(n_iter))):
        H = hankel_matrix_torch(cur, pencil)
        U, s, Vh = torch.linalg.svd(H, full_matrices=False)
        r = min(rank, s.shape[-1])
        Hr = (U[..., :, :r] * s[..., None, :r].to(U.dtype)) @ Vh[..., :r, :]
        cur = hankel_to_vector_torch(Hr)
    return cur


def project_G_grad(G: torch.Tensor, *, rank: int = 7, pencil: int | None = None,
                   n_iter: int = 1) -> torch.Tensor:
    """DIAGNOSTIC ONLY: the fixed-rank projection with autograd left ON.

    ``Pi_r`` is differentiable almost everywhere, so the exact gradient through
    the SVD is computable -- it is only *ill-conditioned*, carrying
    ``1/(sigma_i^2 - sigma_j^2)`` terms that blow up on near-degenerate
    singular values. This entry point exists so PROMPT 7's A3 can measure how
    faithful the straight-through approximation actually is, by comparing this
    gradient against the STE's.

    **Never call this from training.** The training path is the STE in
    :class:`urformer.URformerLayer`; this is a measurement instrument. Any
    caller must check the singular-value gaps before trusting the result.
    """
    if G.ndim != 3 or not G.is_complex():
        raise ValueError(
            f"project_G_grad expects batched complex (batch, N, K), got "
            f"{tuple(G.shape)} dtype {G.dtype}")
    b, N, K = G.shape
    cols = G.permute(0, 2, 1).reshape(b * K, N)
    out = hankel_project_torch(cols, rank, pencil=pencil, n_iter=n_iter)
    return out.reshape(b, K, N).permute(0, 2, 1)


def singular_gap_stats(G: torch.Tensor, *, rank: int = 7,
                       pencil: int | None = None) -> dict:
    """Smallest singular-value gaps in the Hankel spectra of ``G``'s columns.

    The exact SVD gradient carries ``1/(sigma_i - sigma_j)``-type terms, so a
    tiny gap makes it numerically meaningless. A3 reports these alongside any
    exact-gradient number, and discards the number if the gap is unusable.
    The gap that matters most is the one straddling the truncation, ``sigma_r
    - sigma_{r+1}``: that is the direction the projection actually cuts.
    """
    b, N, K = G.shape
    cols = G.permute(0, 2, 1).reshape(b * K, N)
    s = torch.linalg.svdvals(hankel_matrix_torch(cols, pencil))
    adjacent = (s[:, :-1] - s[:, 1:]).abs()
    r = min(int(rank), s.shape[-1] - 1)
    return {
        "min_adjacent_gap": float(adjacent.min()),
        "min_gap_at_truncation": float((s[:, r - 1] - s[:, r]).abs().min()),
        "median_gap_at_truncation": float((s[:, r - 1] - s[:, r]).abs().median()),
        "min_relative_gap_at_truncation":
            float(((s[:, r - 1] - s[:, r]).abs() / s[:, 0]).min()),
        "sigma_max_median": float(s[:, 0].median()),
    }


@torch.no_grad()
def project_G(G: torch.Tensor, *, rank: int = 7, pencil: int | None = None,
              n_iter: int = 1, mode: str = "fixed",
              L_k: torch.Tensor | None = None) -> torch.Tensor:
    """Project every USER COLUMN of a batched ``G`` onto Hankel rank structure.

    Parameters
    ----------
    G : (batch, N, K) complex
    mode :
        ``"fixed"``   - ``rank`` for every column (PRIMARY, ``rank = L_max = 7``)
        ``"oracle"``  - per-column true ``L_k``; DIAGNOSTIC ONLY, privileged info
        ``"adaptive"``- MDL order from the spectrum, capped at ``rank``

    Decorated ``@torch.no_grad()``: the projection is a fixed physics operator
    and no gradient path passes through the SVD (gate HK6).
    """
    if G.ndim != 3 or not G.is_complex():
        raise ValueError(
            f"project_G expects batched complex (batch, N, K), got "
            f"{tuple(G.shape)} dtype {G.dtype}")
    b, N, K = G.shape
    cols = G.permute(0, 2, 1).reshape(b * K, N)      # (b*K, N) -- PER USER COLUMN

    if mode == "fixed":
        out = hankel_project_torch(cols, rank, pencil=pencil, n_iter=n_iter)
    elif mode == "oracle":
        if L_k is None:
            raise ValueError("mode='oracle' requires L_k (privileged info)")
        Lk = torch.as_tensor(L_k).reshape(-1)
        if Lk.numel() != b * K:
            raise ValueError(f"L_k must have b*K={b*K} entries, got {Lk.numel()}")
        out = torch.empty_like(cols)
        for r in torch.unique(Lk):
            m = Lk == r
            out[m] = hankel_project_torch(cols[m], int(r), pencil=pencil,
                                          n_iter=n_iter)
    elif mode == "adaptive":
        from rydberg_sim.track_b_structure import estimate_order
        orders = np.array([
            min(int(estimate_order(c.detach().cpu().numpy(), max_order=rank)), rank)
            for c in cols])
        out = torch.empty_like(cols)
        for r in np.unique(orders):
            m = torch.as_tensor(orders == r, device=cols.device)
            out[m] = hankel_project_torch(cols[m], int(r), pencil=pencil,
                                          n_iter=n_iter)
    else:
        raise ValueError(f"unknown mode {mode!r}; expected one of {RANK_MODES}")

    return out.reshape(b, K, N).permute(0, 2, 1).contiguous()
