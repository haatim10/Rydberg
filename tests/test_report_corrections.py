"""Regression tests locking the claims the technical report makes.

Each test corresponds to a numbered fix in the report review; the point is
that a future edit which breaks the claim breaks a test rather than only the
prose.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from rydberg_sim.gs import em_gs_channel_rows
from rydberg_sim.spectral import spectral_initialize
from rydberg_sim.track_b_drivers import track_b_world
from rydberg_sim.track_b_proposed import cadzow_project, hankel_rank_cap, hs_gs


# --- FIX 1: the canonical mapping the report prints must be the real one ---
def test_canonical_mapping_convention_2():
    """The adapter uses M = S, u = conj(g_n), b = conj(B[n]), out = conj(u).

    The report's Sec. 1.3 table must state exactly this. The competing
    convention (M = conj(S), u = g_n, b = B[n]) is also valid; mixing the
    two -- conjugating both M and u -- is not, and that is what the first
    version of the table printed.

    Note the noise: Z = |GS + B + W|, so the identity only closes when W is
    carried into the canonical b alongside B. Checking against Z without it
    fails for the right conventions too.
    """
    w = track_b_world(3, 12, 8.0, N=8)
    for n in range(w.Z.shape[0]):
        M = w.S
        u = np.conjugate(w.G[n])
        b = np.conjugate(w.B[n] + w.W[n])
        assert np.allclose(np.abs(M.conj().T @ u + b), w.Z[n], atol=1e-12), n


def test_other_valid_convention_also_holds():
    """M = conj(S), u = g_n, b = B[n] -- the un-conjugated dual."""
    w = track_b_world(3, 12, 8.0, N=8)
    for n in range(w.Z.shape[0]):
        M = np.conjugate(w.S)
        u = w.G[n]
        b = w.B[n] + w.W[n]
        assert np.allclose(np.abs(M.conj().T @ u + b), w.Z[n], atol=1e-12), n


def test_the_mixed_convention_really_is_wrong():
    """Conjugating BOTH M and u -- what the report's first table printed --
    does not satisfy the model. Guards against reverting the fix."""
    w = track_b_world(3, 12, 8.0, N=8)
    M = np.conjugate(w.S)
    u = np.conjugate(w.G[0])
    b = w.B[0] + w.W[0]
    dev = float(np.max(np.abs(np.abs(M.conj().T @ u + b) - w.Z[0])))
    assert dev > 1e-3, f"mixed convention unexpectedly matched (dev {dev:.2e})"


def test_adapter_output_is_G_not_conj_G():
    """Noiseless recovery must return G, not its conjugate."""
    w = track_b_world(1, 40, 60.0, N=8)
    G_hat = em_gs_channel_rows(w.S, w.Z, w.B, w.sigma2, max_iter=200).G_hat
    err = np.linalg.norm(G_hat - w.G) / np.linalg.norm(w.G)
    err_conj = np.linalg.norm(G_hat - np.conjugate(w.G)) / np.linalg.norm(w.G)
    assert err < err_conj


# --- FIX 2: vacuity threshold, and the strictness of the active flag ------
@pytest.mark.parametrize("N,cap", [(8, 4), (16, 8), (32, 16)])
def test_rank_cap(N, cap):
    assert hankel_rank_cap(N) == cap


def test_hankel_rank_saturates_at_the_cap():
    """At N=8 a sum of L>=4 exponentials already has rank 4, so the
    constraint rank <= L_k is vacuous for L_k >= 4 (not >= 5)."""
    rng = np.random.default_rng(0)
    N = 8
    for L in (3, 4, 5, 6, 7):
        psi = rng.uniform(-np.pi, np.pi, L)
        a = rng.normal(size=L) + 1j * rng.normal(size=L)
        g = np.exp(-1j * np.outer(np.arange(N), psi)) @ a
        from rydberg_sim.track_b_structure import hankel_matrix
        r = np.linalg.matrix_rank(hankel_matrix(g, 4), tol=1e-8)
        assert r == min(L, hankel_rank_cap(N)), (L, r)


def test_active_flag_is_strict():
    """L_hat == cap must count as INACTIVE: the projection is a no-op there."""
    w = track_b_world(3, 10, 10.0, N=8)
    cap = hankel_rank_cap(8)
    assert not hs_gs(w.S, w.Z, w.B, w.sigma2, L_hat=cap, max_iter=5).constraint_active
    assert hs_gs(w.S, w.Z, w.B, w.sigma2, L_hat=cap - 1,
                 max_iter=5).constraint_active
    g = w.G[:, 0]
    assert np.array_equal(cadzow_project(g, cap), g)     # literally untouched


def test_inactive_constraint_reproduces_em_gs_bitwise():
    for N, P in ((8, 10), (16, 30)):
        w = track_b_world(3, P, 10.0, N=N)
        r = hs_gs(w.S, w.Z, w.B, w.sigma2, L_hat=hankel_rank_cap(N), max_iter=50)
        base = em_gs_channel_rows(w.S, w.Z, w.B, w.sigma2, max_iter=50).G_hat
        assert np.allclose(r.G_hat, base, rtol=0, atol=1e-12)


# --- FIX 8: the phase anchor in the spectral initializer is load-bearing --
def test_spectral_phase_anchor_matters():
    """Dropping the de-rotation must degrade the initializer measurably."""
    w = track_b_world(5, 20, 0.0, N=8)
    n = 0
    M, z, b = w.S, w.Z[n], np.conjugate(w.B[n])
    res = spectral_initialize(M, z, b)
    u_true = np.conjugate(w.G[n])
    anchored = np.linalg.norm(res.u0 - u_true)
    # same vector without the anchor: undo the rotation applied to ubar0
    unanchored = res.ubar0[:-1]
    assert anchored < np.linalg.norm(unanchored - u_true)


# --- FIX 13 (test I): the EM-GS baseline must not move with N -------------
def test_em_gs_baseline_is_flat_in_N():
    """Estimation is row-separable: each row has K unknowns and P
    measurements regardless of N, so the baseline must not move with N.
    Any N-dependence in the HS-GS gain is therefore attributable to the
    structural constraint, not to the baseline shifting underneath it."""
    import json
    rows = json.loads((REPO / "results/track_b/b3/summary.json").read_text())
    by: dict = {}
    for r in rows:
        by.setdefault((r["P"], r["snr_db"]), {})[r["N"]] = r["pooled_db"]["em_gs"]
    worst = max(max(v.values()) - min(v.values()) for v in by.values())
    assert worst <= 0.2, f"EM-GS baseline moves with N by {worst:.3f} dB"


# --- FIX 3: trial counts must reconcile -----------------------------------
def test_trial_counts_reconcile():
    import json
    p = REPO / "results/track_b/report_numbers.json"
    if not p.exists():
        pytest.skip("run scripts/report_numbers.py first")
    t = json.loads(p.read_text())["trials"]
    assert sum(t["b3_per_point"].values()) == t["b3_total"]
    assert t["grand_total_unique"] == t["b3_total"] + t["b4_new_only"]
    assert t["b4_total"] == t["b4_new_only"] + t["b4_copied_trials"]
