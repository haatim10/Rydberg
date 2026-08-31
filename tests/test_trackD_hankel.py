"""Track D HS-URformer unit tests (PROMPT 6).

Pins the Part A gate results as ordinary tests, so a regression in the Hankel
operator fails `pytest` and not only `scratch/trackD_hankel_gates.py`. Fast: no
training, small shapes, short iteration counts.

Two of these encode corrections that the gates forced:

* :func:`test_projection_gradient_reaches_every_layer` -- a fully detached
  projection severed the unrolled chain (gate HK6 measured exactly zero
  gradient below the last Transformer). The straight-through estimator is the
  fix, and this test fails if anyone reverts it.
* :func:`test_operator_is_vacuous_not_lossy_at_N8` -- at ``N=8`` a rank-7
  request cannot truncate anything, so ``Delta_H = 0`` there is an algebraic
  identity rather than a measurement.
"""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="Track D requires torch")

from trackD_urformer.config import (  # noqa: E402
    ModelConfig, NumericConfig, SystemConfig,
)
from trackD_urformer.hankel import (  # noqa: E402
    hankel_matrix_torch, hankel_project_torch, hankel_to_vector_torch,
    max_representable_rank, project_G,
)
from trackD_urformer.urformer import URformer  # noqa: E402

C128 = torch.complex128


def _exponential_sum(N: int, L: int, rng) -> np.ndarray:
    """A length-``N`` sum of ``L`` complex exponentials: Hankel rank exactly L."""
    freqs = rng.uniform(-0.5, 0.5, L)
    amps = rng.standard_normal(L) + 1j * rng.standard_normal(L)
    n = np.arange(N)
    return (amps[None, :] * np.exp(2j * np.pi * freqs[None, :] * n[:, None])
            ).sum(axis=1)


def _rel(a, b) -> float:
    return float(torch.linalg.norm(a - b) / torch.linalg.norm(b))


# --------------------------------------------------------------- HK1 / HK3
@pytest.mark.parametrize("N,L", [(32, 3), (32, 5), (32, 7), (16, 3), (16, 7)])
def test_projection_is_exact_on_structured_signals(N, L):
    """A true sum of L exponentials passes through rank-L projection unchanged."""
    rng = np.random.default_rng(20260831 + N * 100 + L)
    g = torch.as_tensor(_exponential_sum(N, L, rng), dtype=C128)
    out = hankel_project_torch(g, L)
    assert _rel(out, g) < 1e-12


@pytest.mark.parametrize("N,L", [(32, 3), (32, 7), (16, 5)])
def test_hankel_rank_of_structured_signal_is_exactly_L(N, L):
    rng = np.random.default_rng(4242 + L)
    g = torch.as_tensor(_exponential_sum(N, L, rng), dtype=C128)
    s = torch.linalg.svdvals(hankel_matrix_torch(g))
    assert int((s > 1e-9 * s[0]).sum()) == L


# --------------------------------------------------------------------- HK2
def test_single_step_is_NOT_idempotent_on_generic_input():
    """One step of ``H^-1 . Pi_r . H`` is not a projection. This is expected.

    ``Pi_r`` leaves the rank-r matrix set, but the truncated matrix is no
    longer Hankel, and anti-diagonal averaging back to a vector re-introduces
    energy above rank r. One step is a single ALTERNATING-projection sweep
    between the two sets, not a projection onto their intersection -- which is
    why Cadzow iterates and why Track B's ``hs_gs`` defaults to
    ``cadzow_iter=4``.

    Measured at ``N=32, r=7`` from a generic complex Gaussian vector: after one
    step 3.7% of the Hankel spectral energy still sits outside rank 7, falling
    to 0.84% after four steps. HS-URformer runs at ``hankel_iters=1`` because
    that is the operator PROMPT 6 specifies, so the structure it imposes is
    APPROXIMATE, and any reading of ``Delta_H`` has to say so.
    """
    rng = np.random.default_rng(99)
    g = torch.as_tensor(rng.standard_normal(32) + 1j * rng.standard_normal(32),
                        dtype=C128)
    p1 = hankel_project_torch(g, 7)
    assert _rel(hankel_project_torch(p1, 7), p1) > 1e-3

    def off_manifold(v) -> float:
        s = torch.linalg.svdvals(hankel_matrix_torch(v))
        return float((s[7:] ** 2).sum() / (s ** 2).sum())

    tails = [off_manifold(hankel_project_torch(g, 7, n_iter=k))
             for k in (1, 2, 4, 8)]
    assert tails == sorted(tails, reverse=True), (
        f"iterating must monotonically reduce off-manifold energy, got {tails}")
    assert tails[0] > 1e-2 and tails[-1] < tails[0] / 10


def test_structured_signals_are_exact_fixed_points():
    """The property HK2 actually pins: a true rank-L signal is a fixed point."""
    rng = np.random.default_rng(1010)
    g = torch.as_tensor(_exponential_sum(32, 7, rng), dtype=C128)
    p1 = hankel_project_torch(g, 7)
    assert _rel(p1, g) < 1e-12
    assert _rel(hankel_project_torch(p1, 7), p1) < 1e-12


def test_embedding_roundtrip_is_exact():
    rng = np.random.default_rng(7)
    g = torch.as_tensor(rng.standard_normal(32) + 1j * rng.standard_normal(32),
                        dtype=C128)
    assert _rel(hankel_to_vector_torch(hankel_matrix_torch(g)), g) < 1e-13


# --------------------------------------------------------------------- HK4
def test_matches_track_b_hankel_implementation():
    """Parity with ``rydberg_sim.track_b_structure`` on identical input.

    Track D inherited Track B's ``(N-p) x (p+1)`` layout rather than the
    prompt's transpose, so this must agree entry for entry, not just in
    singular values.
    """
    from rydberg_sim.track_b_structure import hankel_matrix, hankel_project

    rng = np.random.default_rng(1234)
    g = rng.standard_normal(32) + 1j * rng.standard_normal(32)
    gt = torch.as_tensor(g, dtype=C128)
    assert np.abs(hankel_matrix_torch(gt).numpy() - hankel_matrix(g)).max() == 0.0
    ours = hankel_project_torch(gt, 7, n_iter=1).numpy()
    theirs = hankel_project(g, 7, n_iter=1)
    assert np.abs(ours - theirs).max() < 1e-12


# --------------------------------------------------------------------- HK5
def test_degenerates_to_identity_at_full_rank():
    """At ``r = min(N-p, p+1)`` nothing is truncated, so P is the identity."""
    rng = np.random.default_rng(5)
    g = torch.as_tensor(rng.standard_normal(32) + 1j * rng.standard_normal(32),
                        dtype=C128)
    full = max_representable_rank(32)
    assert full == 16
    assert _rel(hankel_project_torch(g, full), g) < 1e-12


# --------------------------------------------------------------------- HK6
def test_projection_gradient_reaches_every_layer():
    """The straight-through estimator must keep the unrolled chain intact.

    A fully detached projection makes ``G_lin`` a leaf, and every parameter
    below the splice point receives exactly zero gradient. That is what gate
    HK6 measured before the fix, and it silently turns a ten-layer unrolled
    network into a one-layer one.
    """
    N, K, P, b = 16, 3, 20, 2
    mcfg = ModelConfig(T_UR=3, use_hankel=True, hankel_rank=5,
                       hankel_mode="fixed")
    # NumericConfig(dtype='float64') matches the complex128 inputs below;
    # stage1.build_model does the same .double() promotion.
    net = URformer(N, K, mcfg, NumericConfig(dtype="float64")).double()
    rng = np.random.default_rng(31337)
    cx = lambda *s: torch.as_tensor(
        rng.standard_normal(s) + 1j * rng.standard_normal(s), dtype=C128)
    G0, S, B = cx(b, N, K), cx(b, K, P), cx(b, N, P)
    Z = torch.as_tensor(np.abs(rng.standard_normal((b, N, P))),
                        dtype=torch.float64)
    sigma2 = torch.full((b,), 0.1, dtype=torch.float64)

    net(G0, Z, S, B, sigma2).abs().square().sum().backward()

    per_layer = [
        sum(float(p.grad.abs().sum()) for p in layer.parameters()
            if p.grad is not None)
        for layer in net.layers
    ]
    assert all(g > 0 for g in per_layer), (
        f"layer gradient sums {per_layer}: a zero means the projection severed "
        "the chain -- the straight-through estimator was removed")


def test_projection_forward_is_exactly_the_projection():
    """STE changes the backward pass only; the forward value is unchanged."""
    from trackD_urformer.hankel import project_G as pg

    rng = np.random.default_rng(808)
    G = torch.as_tensor(rng.standard_normal((2, 32, 3))
                        + 1j * rng.standard_normal((2, 32, 3)), dtype=C128)
    G = G.requires_grad_(True)
    proj = pg(G.detach(), rank=7)
    ste = G + (pg(G.detach(), rank=7) - G).detach()
    # G + (proj - G) reassociates in floating point, so this is exact to
    # rounding rather than bitwise; 2.2e-16 on a unit-scale input.
    assert float((ste - proj).detach().abs().max()) < 1e-13


# --------------------------------------------------------------------- HK7
def test_operator_is_vacuous_not_lossy_at_N8():
    """At ``N=8, p=4`` the embedding is 4x5, so rank 7 truncates to rank 4.

    Every length-8 vector already has Hankel rank <= 4, so the rank-7 request
    removes nothing at all. ``Delta_H = 0`` at ``N=8`` is therefore an
    algebraic identity, not an experimental prediction -- the same request is
    genuinely constraining at ``N=32``.
    """
    assert max_representable_rank(8) == 4
    rng = np.random.default_rng(2718)
    g8 = torch.as_tensor(rng.standard_normal(8) + 1j * rng.standard_normal(8),
                         dtype=C128)
    g32 = torch.as_tensor(rng.standard_normal(32) + 1j * rng.standard_normal(32),
                          dtype=C128)
    assert _rel(hankel_project_torch(g8, 7), g8) < 1e-12      # vacuous
    assert _rel(hankel_project_torch(g32, 7), g32) > 1e-2     # constraining


# ------------------------------------------------------- per-user semantics
def test_project_G_acts_per_user_column():
    """Each user's length-N column is projected independently.

    Building the embedding from ``G`` as a whole is the easy mistake here, and
    it would make the result depend on the other users' channels.
    """
    rng = np.random.default_rng(60613)
    cols = [_exponential_sum(32, L, rng) for L in (3, 5, 7)]
    G = torch.as_tensor(np.stack(cols, axis=1)[None], dtype=C128)
    out = project_G(G, rank=7)
    assert _rel(out, G) < 1e-12                     # all three are rank <= 7
    for k in range(3):
        alone = project_G(G[:, :, k:k + 1], rank=7)
        assert _rel(alone[0, :, 0], out[0, :, k]) < 1e-12


def test_project_G_rejects_unbatched_or_real_input():
    with pytest.raises(ValueError, match="batched complex"):
        project_G(torch.zeros(32, 3, dtype=C128))
    with pytest.raises(ValueError, match="batched complex"):
        project_G(torch.zeros(1, 32, 3))


def test_oracle_mode_requires_L_k():
    with pytest.raises(ValueError, match="privileged"):
        project_G(torch.zeros(1, 32, 3, dtype=C128), mode="oracle")


def test_unknown_mode_is_rejected():
    with pytest.raises(ValueError, match="unknown mode"):
        project_G(torch.zeros(1, 32, 3, dtype=C128), mode="cadzow")


# ------------------------------------------------------------ wiring / off
def test_hankel_is_off_by_default_and_changes_the_output_when_on():
    """The PROMPT 2 baseline must be untouched unless ``use_hankel`` is set."""
    assert ModelConfig().use_hankel is False

    N, K, P, b = 16, 3, 20, 2
    rng = np.random.default_rng(6060)
    cx = lambda *s: torch.as_tensor(
        rng.standard_normal(s) + 1j * rng.standard_normal(s), dtype=C128)
    G0, S, B = cx(b, N, K), cx(b, K, P), cx(b, N, P)
    Z = torch.as_tensor(np.abs(rng.standard_normal((b, N, P))),
                        dtype=torch.float64)
    sigma2 = torch.full((b,), 0.1, dtype=torch.float64)

    torch.manual_seed(0)
    off = URformer(N, K, ModelConfig(T_UR=2, use_hankel=False),
                   NumericConfig(dtype="float64")).double()
    torch.manual_seed(0)
    on = URformer(N, K, ModelConfig(T_UR=2, use_hankel=True, hankel_rank=3),
                  NumericConfig(dtype="float64")).double()
    with torch.no_grad():
        a = off(G0, Z, S, B, sigma2)
        c = on(G0, Z, S, B, sigma2)
        # The disable hook must restore the off behaviour exactly.
        on._set_test_mode(disable_hankel=True)
        d = on(G0, Z, S, B, sigma2)
    assert _rel(c, a) > 1e-6, "use_hankel=True did not change the output"
    assert float((d - a).abs().max()) == 0.0, (
        "the disable hook must bypass the operator entirely, not undo it")
