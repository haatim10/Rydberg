"""Differentiable Torch forward model and fixed-weight EM-GS layer.

This module is the bridge between the repository's validated NumPy physics and
the learnable network. It introduces **no new mathematical convention**: every
operation here is verified against `rydberg_sim` in `verify.py` gates B-E, J.

Conventions (audit report `reports/trackD_audit.json`, all confirmed empirically)
--------------------------------------------------------------------------------
    G in C^{N x K}      S in C^{K x P}      B in C^{N x P}
    Y = G @ S + B
    Z = |Y + W|                                (rydberg_sim/forward.py:206,226-227)

Least squares / M-step
----------------------
The repository solves, per receive element ``n``, the canonical problem
``z = |M^H u + b + w|`` with ``M = S``, ``u = conj(g_n)``, ``b = conj(B[n])``
(rydberg_sim/gs.py:397-407), updating

    gram = M M^H = S S^H
    rhs  = M r   = S (y - b)
    u    = solve(gram, rhs)                    (rydberg_sim/gs.py:326-331)

Unwinding the conjugations, that is algebraically the ordinary least squares

    G = R S^H (S S^H)^{-1},      R = Y_rec - B

which this module computes in batched form. The equivalence is not assumed --
gate C asserts it against `biased_gs_channel_rows` to < 1e-12 in float64.

Numerical guards (PROMPT 2 sec. 1)
----------------------------------
``torch.angle`` and ``|.|`` have singular gradients at zero, so the unit-phase
factor is formed as ``Y / (|Y| + eps)`` rather than ``exp(j*angle(Y))``, and
``kappa`` carries the same eps floor. Nothing here calls ``torch.angle``.

Bessel evaluation
-----------------
Exponentially scaled only: ``i1e(kappa)/i0e(kappa)``. The shared ``exp(-|x|)``
factor cancels, so this never overflows. Raw ``I1/I0`` is never formed.
"""
from __future__ import annotations

import torch

__all__ = [
    "assert_shapes",
    "forward_field",
    "observe",
    "unit_phase",
    "em_kappa",
    "bessel_ratio_torch",
    "least_squares_G",
    "em_gs_layer",
    "gs_layer",
]


# ---------------------------------------------------------------------------
# Shape discipline
# ---------------------------------------------------------------------------
def assert_shapes(
    G: torch.Tensor | None = None,
    S: torch.Tensor | None = None,
    B: torch.Tensor | None = None,
    Z: torch.Tensor | None = None,
    *,
    where: str = "",
) -> tuple[int, int, int, int]:
    """Validate the (batch, N, K, P) contract at a module boundary.

    Every tensor is batched: ``G (b,N,K)``, ``S (b,K,P)``, ``B (b,N,P)``,
    ``Z (b,N,P)``. Returns ``(batch, N, K, P)``. Raises on any mismatch --
    this is deliberately not a soft check (PROMPT 2 sec. 1).
    """
    tag = f" [{where}]" if where else ""
    batch = N = K = P = -1

    def _set(name, val, cur):
        if cur != -1 and cur != val:
            raise ValueError(f"{name} mismatch{tag}: {cur} vs {val}")
        return val

    for name, t, ndim in (("G", G, 3), ("S", S, 3), ("B", B, 3), ("Z", Z, 3)):
        if t is None:
            continue
        if t.ndim != ndim:
            raise ValueError(
                f"{name} must be {ndim}-D (batched){tag}, got shape {tuple(t.shape)}"
            )
        batch = _set("batch", t.shape[0], batch)

    if G is not None:
        N = _set("N", G.shape[1], N)
        K = _set("K", G.shape[2], K)
    if S is not None:
        K = _set("K", S.shape[1], K)
        P = _set("P", S.shape[2], P)
    if B is not None:
        N = _set("N", B.shape[1], N)
        P = _set("P", B.shape[2], P)
    if Z is not None:
        N = _set("N", Z.shape[1], N)
        P = _set("P", Z.shape[2], P)

    for name, t, want_complex in (
        ("G", G, True), ("S", S, True), ("B", B, True), ("Z", Z, False)
    ):
        if t is None:
            continue
        if want_complex and not t.is_complex():
            raise TypeError(f"{name} must be complex{tag}, got {t.dtype}")
        if not want_complex and t.is_complex():
            raise TypeError(f"{name} must be real{tag}, got {t.dtype}")

    return batch, N, K, P


# ---------------------------------------------------------------------------
# Forward model
# ---------------------------------------------------------------------------
def forward_field(G: torch.Tensor, S: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """``Y = G @ S + B``. Mirrors rydberg_sim/forward.py:206,226."""
    assert_shapes(G=G, S=S, B=B, where="forward_field")
    return torch.matmul(G, S) + B


def observe(
    G: torch.Tensor, S: torch.Tensor, B: torch.Tensor, W: torch.Tensor
) -> torch.Tensor:
    """``Z = |G @ S + B + W|``. Elementwise amplitude, never ``|.|^2``."""
    return torch.abs(forward_field(G, S, B) + W)


def unit_phase(Y: torch.Tensor, eps: float) -> torch.Tensor:
    """``Y / (|Y| + eps)`` -- the eps-guarded stand-in for ``exp(j*angle(Y))``.

    Using this instead of ``torch.angle`` keeps the gradient finite at ``Y=0``.
    The guard biases the modulus by ``|Y|/(|Y|+eps)``, which is within
    ``eps/|Y|`` of unity and is why the float64 gates use ``eps = 1e-12``.
    """
    return Y / (torch.abs(Y) + eps).to(Y.dtype)


def em_kappa(Z: torch.Tensor, Y: torch.Tensor, sigma2: torch.Tensor, eps: float
             ) -> torch.Tensor:
    """``kappa = (2 / sigma2) * Z * |Y|``  (rydberg_sim/gs.py:476-485).

    Factor 2, and ``sigma2`` not ``sigma``. ``sigma2`` broadcasts from
    ``(batch,)`` or ``(batch,1,1)``.
    """
    if sigma2.ndim == 1:
        sigma2 = sigma2.view(-1, 1, 1)
    return (2.0 / sigma2) * Z * (torch.abs(Y) + eps)


def bessel_ratio_torch(kappa: torch.Tensor) -> torch.Tensor:
    """``R(kappa) = I1/I0`` via exponentially scaled Bessels.

    ``i1e(x)/i0e(x)`` -- the shared ``exp(-|x|)`` cancels, so this is stable at
    every ``kappa`` reachable in this model. Raw ``I1/I0`` is never formed.
    Matches ``rydberg_sim.gs.bessel_ratio`` (which uses ``scipy.special.ive``)
    to machine precision over the observed range; gate E asserts it.
    """
    return torch.special.i1e(kappa) / torch.special.i0e(kappa)


# ---------------------------------------------------------------------------
# Least squares / M-step
# ---------------------------------------------------------------------------
def least_squares_G(R: torch.Tensor, S: torch.Tensor, ridge: float = 0.0
                    ) -> torch.Tensor:
    """``G = R S^H (S S^H + ridge I)^{-1}``, batched.

    This is the repository's M-step (rydberg_sim/gs.py:326-331) written
    directly on ``G`` instead of on the per-row canonical ``u = conj(g_n)``.
    Gate C asserts the two agree to < 1e-12 in float64.

    Solved as ``A G^H = S R^H`` with ``A = S S^H`` Hermitian, which avoids
    forming an explicit inverse -- the same reason gs.py uses ``np.linalg.solve``.

    Parameters
    ----------
    R : (batch, N, P) complex   -- the debiased reconstruction ``Y_rec - B``
    S : (batch, K, P) complex   -- pilots
    """
    if R.ndim != 3 or S.ndim != 3:
        raise ValueError(
            f"least_squares_G expects batched 3-D tensors, got R{tuple(R.shape)} "
            f"S{tuple(S.shape)}"
        )
    if R.shape[0] != S.shape[0] or R.shape[2] != S.shape[2]:
        raise ValueError(
            f"incompatible R{tuple(R.shape)} and S{tuple(S.shape)}: need matching "
            "batch and P"
        )
    Sh = S.conj().transpose(-1, -2)              # (b, P, K)
    gram = torch.matmul(S, Sh)                   # (b, K, K), Hermitian
    if ridge != 0.0:
        eye = torch.eye(gram.shape[-1], dtype=gram.dtype, device=gram.device)
        gram = gram + ridge * eye
    rhs = torch.matmul(S, R.conj().transpose(-1, -2))   # (b, K, N)
    Gh = torch.linalg.solve(gram, rhs)                  # (b, K, N)
    # resolve_conj(): torch tracks conjugation lazily as a tensor bit, which
    # leaks into downstream .numpy() calls and autograd. Materialize it here so
    # every caller sees an ordinary tensor.
    return Gh.conj().resolve_conj().transpose(-1, -2)   # (b, N, K)


# ---------------------------------------------------------------------------
# Fixed-weight classical layers (no learnable parameters)
# ---------------------------------------------------------------------------
def gs_layer(
    G: torch.Tensor, Z: torch.Tensor, S: torch.Tensor, B: torch.Tensor,
    *, eps: float, ridge: float = 0.0,
) -> torch.Tensor:
    """One classical biased-GS update. No Bessel weighting.

        Y     = G @ S + B
        Y_rec = Z * unit_phase(Y)
        G_new = LS(Y_rec - B, S)
    """
    assert_shapes(G=G, S=S, B=B, Z=Z, where="gs_layer")
    Y = forward_field(G, S, B)
    Y_rec = Z.to(Y.dtype) * unit_phase(Y, eps)
    return least_squares_G(Y_rec - B, S, ridge=ridge)


def em_gs_layer(
    G: torch.Tensor, Z: torch.Tensor, S: torch.Tensor, B: torch.Tensor,
    sigma2: torch.Tensor, *, eps: float, ridge: float = 0.0,
) -> torch.Tensor:
    """One classical EM-GS update with the exact Bessel filter.

        Y     = G @ S + B
        kappa = 2 Z |Y| / sigma2
        R     = i1e(kappa) / i0e(kappa)
        Y_rec = Z * unit_phase(Y) * R
        G_new = LS(Y_rec - B, S)

    Fixed weights throughout: this is the reference the learnable layer must
    reproduce when its FilterNet is replaced by ``R`` and its gate is 1
    (gate E).
    """
    assert_shapes(G=G, S=S, B=B, Z=Z, where="em_gs_layer")
    Y = forward_field(G, S, B)
    kappa = em_kappa(Z, Y, sigma2, eps)
    R = bessel_ratio_torch(kappa)
    Y_rec = Z.to(Y.dtype) * unit_phase(Y, eps) * R.to(Y.dtype)
    return least_squares_G(Y_rec - B, S, ridge=ridge)
