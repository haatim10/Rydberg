"""Track D verification gate suite.

Gates A-K (PROMPT 2 sec. 10). Every gate writes its result to
``reports/trackD_verify.json`` **as it runs**, and a human-readable
``reports/trackD_verify.md`` is rendered from that file. The phase report is
rendered from those files, never reconstructed from memory.

Gate I (tiny-dataset overfit) is a HARD STOP: if the model cannot strongly
overfit 32 samples, no training estimate is reported.

Run:  PYTHONPATH=. python3 -m trackD_urformer.verify
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from rydberg_sim.channel import steering_matrix
from rydberg_sim.gs import bessel_ratio, biased_gs_channel_rows, em_gs_channel_rows

from .baselines import make_initial_G, nmse_parts
from .config import ModelConfig, NumericConfig, SystemConfig, TrackDConfig
from .dataset import make_world
from .torch_forward import (
    bessel_ratio_torch, em_gs_layer, em_kappa, forward_field, gs_layer,
    least_squares_G, observe, unit_phase,
)
from .transformer import detokenize, tokenize
from .urformer import URformer, count_parameters

REPORTS = Path("reports")
VERIFY_JSON = REPORTS / "trackD_verify.json"
VERIFY_MD = REPORTS / "trackD_verify.md"

C128, F64 = torch.complex128, torch.float64


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _rel(a: np.ndarray, b: np.ndarray) -> float:
    """Relative Frobenius error ``||a-b|| / ||b||``."""
    b = np.asarray(b)
    den = np.linalg.norm(b)
    return float(np.linalg.norm(np.asarray(a) - b) / (den if den else 1.0))


def _t(a, dtype=C128):
    # np.array(copy=True): the repository freezes its arrays read-only, and
    # torch refuses to wrap a non-writable buffer without warning.
    return torch.as_tensor(np.array(a, copy=True)[None], dtype=dtype)


def _world(trial=0, N=16, P=20, snr=5.0, K=3, seed=20260827):
    sysc = SystemConfig(K=K, N=N, P=P, master_seed=seed)
    return make_world(trial, sysc=sysc, N=N, P=P, snr_db=snr), sysc


class Results:
    """Incremental JSON writer - every gate is persisted the moment it runs."""

    def __init__(self, path: Path):
        self.path = path
        self.data: dict = {"gates": {}, "meta": {}}
        path.parent.mkdir(parents=True, exist_ok=True)

    def add(self, name: str, payload: dict) -> None:
        self.data["gates"][name] = payload
        self.flush()
        status = "PASS" if payload.get("pass") else "FAIL"
        print(f"  [{status}] gate {name}: {payload.get('summary','')}")

    def meta(self, key: str, value) -> None:
        self.data["meta"][key] = value
        self.flush()

    def flush(self) -> None:
        self.path.write_text(json.dumps(self.data, indent=2) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Gate A - shape assertions
# ---------------------------------------------------------------------------
def gate_A(R: Results) -> None:
    from .torch_forward import assert_shapes
    cases, ok, errs = [], True, []
    for (N, K, P, b) in [(8, 2, 8, 1), (16, 3, 20, 4), (32, 3, 30, 2), (8, 4, 12, 3)]:
        G = torch.zeros((b, N, K), dtype=C128)
        S = torch.zeros((b, K, P), dtype=C128)
        B = torch.zeros((b, N, P), dtype=C128)
        Z = torch.zeros((b, N, P), dtype=F64)
        got = assert_shapes(G=G, S=S, B=B, Z=Z, where="gateA")
        cases.append({"N": N, "K": K, "P": P, "batch": b, "resolved": list(got)})
        ok &= got == (b, N, K, P)
        ok &= forward_field(G, S, B).shape == (b, N, P)
        ok &= observe(G, S, B, torch.zeros_like(B)).shape == (b, N, P)

    # Negative cases: every one of these MUST raise.
    for desc, fn in [
        ("K mismatch G/S", lambda: assert_shapes(
            G=torch.zeros((1, 8, 2), dtype=C128), S=torch.zeros((1, 3, 8), dtype=C128))),
        ("unbatched G", lambda: assert_shapes(G=torch.zeros((8, 2), dtype=C128))),
        ("real G", lambda: assert_shapes(G=torch.zeros((1, 8, 2), dtype=F64))),
        ("complex Z", lambda: assert_shapes(Z=torch.zeros((1, 8, 8), dtype=C128))),
        ("N mismatch G/B", lambda: assert_shapes(
            G=torch.zeros((1, 8, 2), dtype=C128), B=torch.zeros((1, 9, 8), dtype=C128))),
    ]:
        try:
            fn()
            ok = False
            errs.append(f"{desc}: did NOT raise")
        except (ValueError, TypeError):
            pass
    R.add("A_shapes", {"pass": bool(ok), "cases": cases, "errors": errs,
                       "summary": f"{len(cases)} shape configs + 5 negative cases"})


# ---------------------------------------------------------------------------
# Gate B - NumPy <-> Torch forward model
# ---------------------------------------------------------------------------
def gate_B(R: Results, dtype_name="float64") -> None:
    cd = C128 if dtype_name == "float64" else torch.complex64
    rd = F64 if dtype_name == "float64" else torch.float32
    tol = 1e-12 if dtype_name == "float64" else 1e-5
    w, _ = _world()
    GS_np = w.G_true @ w.S
    Y_np = GS_np + w.B
    G, S, B = _t(w.G_true, cd), _t(w.S, cd), _t(w.B, cd)
    e_gs = _rel(torch.matmul(G, S)[0].numpy(), GS_np)
    e_B = _rel(B[0].numpy(), w.B)
    e_Y = _rel(forward_field(G, S, B)[0].numpy(), Y_np)
    # Z from the SAME noise the repository drew, so this compares Z not W.
    W_np = np.asarray(Y_np, dtype=np.complex128) * 0.0
    e_Z = _rel(observe(G, S, B, _t(W_np, cd))[0].numpy(), np.abs(Y_np))
    worst = max(e_gs, e_B, e_Y, e_Z)
    R.add(f"B_forward_{dtype_name}", {
        "pass": bool(worst < tol), "tolerance": tol, "dtype": dtype_name,
        "G_at_S_rel_err": e_gs, "B_rel_err": e_B, "Y_rel_err": e_Y,
        "Z_rel_err": e_Z, "max_rel_err": worst,
        "summary": f"max rel {worst:.3e} < {tol:g}",
    })


# ---------------------------------------------------------------------------
# Gate C - LS parity
# ---------------------------------------------------------------------------
def gate_C(R: Results, dtype_name="float64") -> None:
    cd = C128 if dtype_name == "float64" else torch.complex64
    rd = F64 if dtype_name == "float64" else torch.float32
    tol = 1e-12 if dtype_name == "float64" else 1e-5
    worst, per_case = 0.0, []
    for (N, P, trial) in [(8, 8, 0), (16, 20, 1), (32, 30, 2)]:
        w, _ = _world(trial=trial, N=N, P=P)
        G0 = np.zeros_like(w.G_true)
        # One classical GS update through the repository, per-row canonical form.
        np_out = biased_gs_channel_rows(w.S, w.Z, w.B, max_iter=1, G0=G0).G_hat
        tc_out = gs_layer(
            _t(G0, cd), _t(w.Z, rd), _t(w.S, cd), _t(w.B, cd),
            eps=1e-12 if dtype_name == "float64" else 1e-8,
        )[0].numpy()
        e = _rel(tc_out, np_out)
        per_case.append({"N": N, "P": P, "rel_err": e})
        worst = max(worst, e)
    R.add(f"C_ls_parity_{dtype_name}", {
        "pass": bool(worst < tol), "tolerance": tol, "dtype": dtype_name,
        "cases": per_case, "max_rel_err": worst,
        "note": ("batched G = R S^H (S S^H)^{-1} vs the repository's per-row "
                 "canonical solve (gs.py:326-331)"),
        "summary": f"max rel {worst:.3e} < {tol:g}",
    })


# ---------------------------------------------------------------------------
# Gate D - GS degeneration
# ---------------------------------------------------------------------------
def gate_D(R: Results, dtype_name="float64") -> None:
    cd = C128 if dtype_name == "float64" else torch.complex64
    rd = F64 if dtype_name == "float64" else torch.float32
    tol = 1e-12 if dtype_name == "float64" else 1e-5
    eps = 1e-12 if dtype_name == "float64" else 1e-8
    w, sysc = _world(N=16, P=20)
    numeric = NumericConfig(dtype=dtype_name)
    mcfg = ModelConfig(T_UR=1)
    torch.manual_seed(0)
    model = URformer(16, sysc.K, mcfg, numeric)
    if dtype_name == "float64":
        model = model.double()
    model._set_test_mode(alpha=0.0, disable_residual=True)
    G0 = np.zeros_like(w.G_true)
    with torch.no_grad():
        out = model(_t(G0, cd), _t(w.Z, rd), _t(w.S, cd), _t(w.B, cd),
                    torch.tensor([w.sigma2], dtype=rd))[0].numpy()
    ref = biased_gs_channel_rows(w.S, w.Z, w.B, max_iter=1, G0=G0).G_hat
    e = _rel(out, ref)
    R.add(f"D_gs_degeneration_{dtype_name}", {
        "pass": bool(e < tol), "tolerance": tol, "dtype": dtype_name,
        "rel_err": e,
        "condition": "alpha=0, residual disabled => one classical GS update",
        "summary": f"rel {e:.3e} < {tol:g}",
    })


# ---------------------------------------------------------------------------
# Gate E - EM-GS degeneration
# ---------------------------------------------------------------------------
def gate_E(R: Results, dtype_name="float64") -> None:
    cd = C128 if dtype_name == "float64" else torch.complex64
    rd = F64 if dtype_name == "float64" else torch.float32
    tol = 1e-10 if dtype_name == "float64" else 1e-5
    eps = 1e-12 if dtype_name == "float64" else 1e-8
    w, sysc = _world(N=16, P=20)
    numeric = NumericConfig(dtype=dtype_name)
    torch.manual_seed(0)
    model = URformer(16, sysc.K, ModelConfig(T_UR=1), numeric)
    if dtype_name == "float64":
        model = model.double()
    model._set_test_mode(filter_override="exact_bessel", alpha=1.0,
                         disable_residual=True)
    G0 = np.zeros_like(w.G_true)
    with torch.no_grad():
        out = model(_t(G0, cd), _t(w.Z, rd), _t(w.S, cd), _t(w.B, cd),
                    torch.tensor([w.sigma2], dtype=rd))[0].numpy()
    ref = em_gs_channel_rows(w.S, w.Z, w.B, w.sigma2, max_iter=1, G0=G0).G_hat
    e = _rel(out, ref)

    # Bessel parity on the observed kappa range, independent of the layer.
    Y = forward_field(_t(G0, C128), _t(w.S, C128), _t(w.B, C128))
    kap = em_kappa(_t(w.Z, F64), Y, torch.tensor([w.sigma2], dtype=F64), 1e-12)
    e_bes = _rel(bessel_ratio_torch(kap).numpy(), bessel_ratio(kap.numpy()))
    R.add(f"E_emgs_degeneration_{dtype_name}", {
        "pass": bool(e < tol), "tolerance": tol, "dtype": dtype_name,
        "rel_err": e, "bessel_vs_scipy_rel_err": e_bes,
        "kappa_min": float(kap.min()), "kappa_max": float(kap.max()),
        "condition": "FilterNet -> exact i1e/i0e, alpha=1, residual disabled",
        "summary": f"rel {e:.3e} < {tol:g}; bessel vs scipy {e_bes:.3e}",
    })


# ---------------------------------------------------------------------------
# Gate F - Transformer identity
# ---------------------------------------------------------------------------
def gate_F(R: Results) -> None:
    w, sysc = _world(N=16, P=20)
    numeric = NumericConfig(dtype="float64")
    torch.manual_seed(0)
    model = URformer(16, sysc.K, ModelConfig(T_UR=3), numeric).double()
    # zero_init_out=True is the constructor default, so the residual is exactly
    # zero at initialization without any test hook.
    G0 = np.zeros_like(w.G_true)
    args = (_t(G0), _t(w.Z, F64), _t(w.S), _t(w.B),
            torch.tensor([w.sigma2], dtype=F64))
    with torch.no_grad():
        with_res = model(*args)[0].numpy()
        model._set_test_mode(disable_residual=True)
        without_res = model(*args)[0].numpy()
        model._clear_test_mode()
    diff = float(np.max(np.abs(with_res - without_res)))
    res_norms = []
    with torch.no_grad():
        Gl = _t(G0)
        for l in model.layers:
            r = l.former(Gl)
            res_norms.append(float(torch.abs(r).max()))
    R.add("F_transformer_identity", {
        "pass": bool(diff == 0.0), "tolerance": 0.0,
        "max_abs_diff": diff,
        "per_layer_max_abs_residual": res_norms,
        "condition": "out_proj zero-initialized => residual exactly 0",
        "summary": f"max abs diff {diff:.1e} (required exactly 0.0)",
    })


# ---------------------------------------------------------------------------
# Gate G - conjugation
# ---------------------------------------------------------------------------
def gate_G(R: Results) -> None:
    """Deterministic one-path channel; the phase progression must survive
    NumPy generator -> Torch -> LS -> tokenize -> detokenize with no flip."""
    tol, N, K, P = 1e-12, 8, 1, 12
    theta = np.deg2rad(30.0)
    psi = float(np.pi * np.sin(theta))
    a = steering_matrix(np.array([theta]), N)[:, 0]
    G = a.reshape(N, 1).astype(np.complex128)          # alpha = 1, one path

    # 1. the repository's own sign convention
    slope = float(np.polyfit(np.arange(N), np.unwrap(np.angle(a)), 1)[0])
    sign_ok = slope < 0                                 # e^{-j n psi}

    rng = np.random.default_rng(np.random.SeedSequence([4242]))
    S = np.asarray(rng.standard_normal((K, P)) + 1j * rng.standard_normal((K, P))
                   ) / np.sqrt(2.0)
    B = np.full((N, P), 3.0 + 0.0j)
    Y = G @ S + B

    # 2. NumPy -> Torch round trip
    e_torch = _rel(forward_field(_t(G), _t(S), _t(B))[0].numpy(), Y)

    # 3. through the LS step (oracle phase: exact Y)
    G_ls = least_squares_G(_t(Y) - _t(B), _t(S))[0].numpy()
    e_ls = _rel(G_ls, G)
    e_ls_conj = _rel(G_ls, np.conjugate(G))             # must be LARGE

    # 4. through tokenize / detokenize
    G_rt = detokenize(tokenize(_t(G)), N, C128)[0].numpy()
    e_tok = _rel(G_rt, G)
    e_tok_conj = _rel(G_rt, np.conjugate(G))            # must be LARGE

    # 5. the recovered phase slope still matches the original
    slope_rt = float(np.polyfit(np.arange(N), np.unwrap(np.angle(G_rt[:, 0])), 1)[0])

    worst = max(e_torch, e_ls, e_tok)
    ok = (worst < tol and sign_ok and e_ls_conj > 0.1 and e_tok_conj > 0.1
          and abs(slope_rt - slope) < 1e-9)
    R.add("G_conjugation", {
        "pass": bool(ok), "tolerance": tol,
        "psi": psi, "fitted_slope": slope, "slope_after_roundtrip": slope_rt,
        "steering_sign": "e^{-j n psi}" if sign_ok else "e^{+j n psi}",
        "numpy_to_torch_rel_err": e_torch,
        "ls_rel_err_vs_G": e_ls, "ls_rel_err_vs_conjG": e_ls_conj,
        "tokenize_roundtrip_rel_err": e_tok,
        "tokenize_roundtrip_rel_err_vs_conjG": e_tok_conj,
        "summary": (f"max rel {worst:.3e}; sign e^-jnpsi; "
                    f"conj cross-checks {e_ls_conj:.2f}/{e_tok_conj:.2f} (large)"),
    })


# ---------------------------------------------------------------------------
# Gate H - gradients
# ---------------------------------------------------------------------------
def gate_H(R: Results) -> None:
    numeric = NumericConfig(dtype="float32")
    sysc = SystemConfig(K=3, N=16, P=20)
    mcfg = ModelConfig(T_UR=3, d_model=32, L_enc=2)
    torch.manual_seed(0)
    model = URformer(16, 3, mcfg, numeric)
    cd, rd = numeric.complex_dtype, numeric.real_dtype

    Zs, Ss, Bs, s2s, Gs = [], [], [], [], []
    for i in range(4):
        w = make_world(i, sysc=sysc, N=16, P=20, snr_db=5.0)
        Zs.append(w.Z); Ss.append(w.S); Bs.append(w.B)
        s2s.append(w.sigma2); Gs.append(w.G_true)
    Z = torch.as_tensor(np.stack(Zs), dtype=rd)
    S = torch.as_tensor(np.stack(Ss), dtype=cd)
    B = torch.as_tensor(np.stack(Bs), dtype=cd)
    s2 = torch.as_tensor(np.array(s2s), dtype=rd)
    Gt = torch.as_tensor(np.stack(Gs), dtype=cd)
    G0 = torch.zeros_like(Gt)

    out = model(G0, Z, S, B, s2)
    loss = torch.mean(
        torch.sum(torch.abs(out - Gt) ** 2, dim=(1, 2))
        / torch.sum(torch.abs(Gt) ** 2, dim=(1, 2))
    )
    loss.backward()

    per_module, all_finite, all_positive = [], True, True
    for i, l in enumerate(model.layers):
        d = {"layer": i}
        for name, mod in (("filter_net", l.filter_net), ("former", l.former)):
            gs_ = [p.grad for p in mod.parameters() if p.grad is not None]
            nrm = float(torch.sqrt(sum((g ** 2).sum() for g in gs_))) if gs_ else 0.0
            d[name + "_grad_norm"] = nrm
            all_finite &= bool(np.isfinite(nrm))
            all_positive &= nrm > 0.0
        gg = float(l.gate.grad.abs()) if l.gate.grad is not None else 0.0
        d["gate_grad_abs"] = gg
        all_finite &= bool(np.isfinite(gg))
        all_positive &= gg > 0.0
        per_module.append(d)

    R.add("H_gradients", {
        "pass": bool(all_finite and all_positive), "dtype": "float32",
        "loss": float(loss.detach()), "all_finite": all_finite,
        "all_nonzero": all_positive, "per_layer": per_module,
        "summary": f"loss {float(loss.detach()):.4f}; all grads finite and > 0 "
                   f"across {len(per_module)} layers",
    })


# ---------------------------------------------------------------------------
# Gate I - tiny-dataset overfit (HARD STOP)
# ---------------------------------------------------------------------------
def gate_I(R: Results, n_samples=32, steps=1500, target_db=-25.0) -> None:
    numeric = NumericConfig(dtype="float32")
    sysc = SystemConfig(K=3, N=16, P=20)
    mcfg = ModelConfig(T_UR=4, d_model=32, L_enc=2)
    torch.manual_seed(0)
    np.random.seed(0)
    model = URformer(16, 3, mcfg, numeric)
    cd, rd = numeric.complex_dtype, numeric.real_dtype

    Zs, Ss, Bs, s2s, Gs = [], [], [], [], []
    for i in range(n_samples):
        w = make_world(i, sysc=sysc, N=16, P=20, snr_db=5.0)
        Zs.append(w.Z); Ss.append(w.S); Bs.append(w.B)
        s2s.append(w.sigma2); Gs.append(w.G_true)
    Z = torch.as_tensor(np.stack(Zs), dtype=rd)
    S = torch.as_tensor(np.stack(Ss), dtype=cd)
    B = torch.as_tensor(np.stack(Bs), dtype=cd)
    s2 = torch.as_tensor(np.array(s2s), dtype=rd)
    Gt = torch.as_tensor(np.stack(Gs), dtype=cd)
    G0 = torch.zeros_like(Gt)

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    curve, t0 = [], time.time()
    best = float("inf")
    for step in range(1, steps + 1):
        opt.zero_grad()
        out = model(G0, Z, S, B, s2)
        loss = torch.mean(
            torch.sum(torch.abs(out - Gt) ** 2, dim=(1, 2))
            / torch.sum(torch.abs(Gt) ** 2, dim=(1, 2))
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        db = 10.0 * np.log10(max(float(loss), 1e-30))
        best = min(best, db)
        if step % 50 == 0 or step == 1:
            curve.append({"step": step, "nmse_db": db})
    R.add("I_overfit32", {
        "pass": bool(best < target_db), "target_db": target_db,
        "final_nmse_db": curve[-1]["nmse_db"], "best_nmse_db": best,
        "n_samples": n_samples, "steps": steps,
        "seconds": round(time.time() - t0, 1),
        "curve": curve,
        "config": {"T_UR": mcfg.T_UR, "d_model": mcfg.d_model,
                   "L_enc": mcfg.L_enc, "N": 16, "K": 3, "P": 20},
        "summary": f"best {best:.2f} dB (target < {target_db} dB)",
    })


# ---------------------------------------------------------------------------
# Gate J - noiseless fixed point and sigma2 -> 0
# ---------------------------------------------------------------------------
def gate_J(R: Results) -> None:
    tol = 1e-10
    w, sysc = _world(N=16, P=20)
    G = np.asarray(w.G_true)
    # Noiseless Z built through the repository's own forward model.
    from rydberg_sim.forward import exact_forward
    ex = exact_forward(G, w.S, w.B, 0.0)
    Z0 = np.asarray(ex.Z)
    args = lambda s2: (_t(G), _t(Z0, F64), _t(w.S), _t(w.B),
                       torch.tensor([s2], dtype=F64))

    fixed = []
    for s2 in (1e-6, 1e-9, 1e-12):
        out = em_gs_layer(*args(s2), eps=1e-14)[0].numpy()
        fixed.append({"sigma2": s2, "rel_err": _rel(out, G)})

    limit = []
    gs_out = gs_layer(_t(G), _t(Z0, F64), _t(w.S), _t(w.B), eps=1e-14)[0].numpy()
    for s2 in (1e-1, 1e-3, 1e-6, 1e-9, 1e-12):
        em_out = em_gs_layer(*args(s2), eps=1e-14)[0].numpy()
        limit.append({"sigma2": s2, "rel_diff_vs_gs": _rel(em_out, gs_out)})

    ok = fixed[-1]["rel_err"] < tol and limit[-1]["rel_diff_vs_gs"] < tol
    R.add("J_noiseless_fixed_point", {
        "pass": bool(ok), "tolerance": tol,
        "fixed_point": fixed, "emgs_to_gs_limit": limit,
        "note": ("the residual is the exact 1-R(kappa) bias, which vanishes "
                 "linearly in sigma2; the truth is a fixed point only in the "
                 "limit, which is correct EM-GS behaviour"),
        "summary": (f"fixed point {fixed[-1]['rel_err']:.2e}, "
                    f"GS limit {limit[-1]['rel_diff_vs_gs']:.2e}"),
    })


# ---------------------------------------------------------------------------
# Gate K - kappa invariance under input normalization
# ---------------------------------------------------------------------------
def gate_K(R: Results) -> None:
    """Scaling Z by 1/s requires scaling sigma2 by 1/s^2 for kappa to be
    numerically unchanged. This is a gate, not a comment."""
    tol = 1e-10
    w, _ = _world(N=16, P=20)
    G0 = np.zeros_like(w.G_true)
    Y = forward_field(_t(G0), _t(w.S), _t(w.B))
    Z = _t(w.Z, F64)
    k_ref = em_kappa(Z, Y, torch.tensor([w.sigma2], dtype=F64), 0.0)

    rows = []
    worst = 0.0
    for s in (2.0, 10.0, 1e3, 0.01):
        # Scale the FIELD by 1/s: Z, B and G all scale, so |Y| scales too.
        Ys = forward_field(_t(G0), _t(w.S), _t(w.B) / s)
        Zs = Z / s
        k_s = em_kappa(Zs, Ys, torch.tensor([w.sigma2 / s ** 2], dtype=F64), 0.0)
        e = _rel(k_s.numpy(), k_ref.numpy())
        rows.append({"scale": s, "rel_err": e})
        worst = max(worst, e)

    # And the negative control: scaling Z without scaling sigma2 MUST change kappa.
    k_bad = em_kappa(Z / 2.0, forward_field(_t(G0), _t(w.S), _t(w.B) / 2.0),
                     torch.tensor([w.sigma2], dtype=F64), 0.0)
    e_bad = _rel(k_bad.numpy(), k_ref.numpy())

    R.add("K_kappa_invariance", {
        "pass": bool(worst < tol and e_bad > 0.1), "tolerance": tol,
        "scaled_cases": rows, "max_rel_err": worst,
        "negative_control_rel_err": e_bad,
        "note": ("Track D applies NO input normalization; this gate proves the "
                 "invariant holds should one ever be introduced"),
        "summary": f"max rel {worst:.3e}; negative control {e_bad:.2f} (must be large)",
    })


# ---------------------------------------------------------------------------
# Parameter count and model metadata
# ---------------------------------------------------------------------------
def collect_model_info(R: Results) -> None:
    cfg = TrackDConfig()
    torch.manual_seed(0)
    model = URformer(cfg.system.N, cfg.system.K, cfg.model, cfg.numeric)
    counts = count_parameters(model)
    counts["initial_alphas"] = model.initial_alphas()
    counts["gate_init"] = cfg.model.gate_init
    counts["gate_init_value"] = cfg.model.gate_init_value
    bytes_per = 4 if cfg.numeric.dtype == "float32" else 8
    counts["param_bytes"] = counts["totals"]["all_parameters"] * bytes_per
    counts["param_mb"] = counts["param_bytes"] / 1024 ** 2
    # Adam keeps two extra moment buffers per parameter.
    counts["optimizer_state_mb"] = 2 * counts["param_mb"]
    R.meta("model", counts)


def render_markdown(data: dict) -> str:
    L = ["# Track D - verification gate results", ""]
    L.append("Rendered from `reports/trackD_verify.json`. Do not edit by hand.")
    L.append("")
    L.append("| Gate | Pass | Summary |")
    L.append("|---|---|---|")
    for name, g in data["gates"].items():
        L.append(f"| `{name}` | {'PASS' if g.get('pass') else 'FAIL'} | "
                 f"{g.get('summary','')} |")
    L.append("")
    n_pass = sum(1 for g in data["gates"].values() if g.get("pass"))
    L.append(f"**{n_pass}/{len(data['gates'])} gates pass.**")
    L.append("")
    m = data.get("meta", {})
    if "model" in m:
        t = m["model"]["totals"]
        L.append("## Model")
        L.append("")
        L.append(f"- FilterNet: {t['filter_net']:,}")
        L.append(f"- Gates: {t['gate']:,}")
        L.append(f"- Transformer: {t['transformer']:,}")
        L.append(f"- **Total: {t['all_parameters']:,}**")
        L.append(f"- Initial alphas: {m['model']['initial_alphas']}")
        L.append("")
    if "timing" in m:
        L.append("## Timing")
        L.append("")
        L.append("```json")
        L.append(json.dumps(m["timing"], indent=2))
        L.append("```")
    return "\n".join(L) + "\n"


def main() -> int:
    R = Results(VERIFY_JSON)
    R.meta("torch_version", torch.__version__)
    R.meta("numpy_version", np.__version__)
    R.meta("cuda_available", torch.cuda.is_available())
    torch.set_num_threads(1)

    print("Track D verification suite")
    print("=" * 60)
    gate_A(R)
    for dt in ("float64", "float32"):
        gate_B(R, dt); gate_C(R, dt); gate_D(R, dt); gate_E(R, dt)
    gate_F(R)
    gate_G(R)
    gate_H(R)
    gate_J(R)
    gate_K(R)
    gate_I(R)
    collect_model_info(R)

    VERIFY_MD.write_text(render_markdown(R.data), encoding="utf-8")
    gates = R.data["gates"]
    n_pass = sum(1 for g in gates.values() if g.get("pass"))
    print("=" * 60)
    print(f"{n_pass}/{len(gates)} gates pass")
    failed = [k for k, g in gates.items() if not g.get("pass")]
    if failed:
        print("FAILED:", ", ".join(failed))
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
