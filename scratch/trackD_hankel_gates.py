"""PROMPT 6 Part A: Hankel operator gates HK1-HK7 and A4 dataset verification.

No training. All gates must pass before any HS-URformer run.
Writes reports/trackD_hankel_gates.json.

Run:  PYTHONPATH=. python3 scratch/trackD_hankel_gates.py
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from rydberg_sim.track_b_structure import hankel_matrix, hankel_project, hankel_rank
from trackD_urformer.config import ModelConfig, NumericConfig, TrackDConfig
from trackD_urformer.dataset import make_world
from trackD_urformer.hankel import (
    hankel_matrix_torch, hankel_project_torch, max_representable_rank, project_G,
)
from trackD_urformer.urformer import URformer

OUT = Path("reports/trackD_hankel_gates.json")
C128, F64 = torch.complex128, torch.float64
torch.set_num_threads(2)
res: dict = {"gates": {}}
rel = lambda a, b: float(np.linalg.norm(np.asarray(a) - np.asarray(b))
                         / max(np.linalg.norm(np.asarray(b)), 1e-300))


def add(name, payload):
    res["gates"][name] = payload
    print(f"  [{'PASS' if payload['pass'] else 'FAIL'}] {name}: {payload['summary']}")


def exact_signal(N, L, seed=0):
    """A length-N sum of exactly L complex exponentials on the ULA manifold."""
    rng = np.random.default_rng(seed)
    theta = rng.uniform(-np.pi / 2, np.pi / 2, L)
    psi = np.pi * np.sin(theta)
    a = rng.standard_normal(L) + 1j * rng.standard_normal(L)
    n = np.arange(N)[:, None]
    return (np.exp(-1j * n * psi[None, :]) @ a).astype(np.complex128)


# --- HK1 exactness ---------------------------------------------------------
worst, cases = 0.0, []
for N in (32, 16):
    for L in (1, 3, 5, 7):
        for sd in (0, 1, 2):
            g = exact_signal(N, L, sd)
            out = hankel_project_torch(torch.as_tensor(g)[None], 7,
                                       n_iter=1)[0].numpy()
            e = rel(out, g)
            worst = max(worst, e)
            cases.append({"N": N, "L": L, "seed": sd, "rel_err": e})
add("HK1_exactness", {"pass": bool(worst < 1e-12), "max_rel_err": worst,
                      "tolerance": 1e-12, "n_cases": len(cases), "cases": cases[:6],
                      "summary": f"max rel {worst:.3e} < 1e-12 over {len(cases)} cases"})

# --- HK2 fixed point -------------------------------------------------------
worst = 0.0
for N in (32, 16):
    for L in (3, 7):
        g = torch.as_tensor(exact_signal(N, L, 5))[None]
        one = hankel_project_torch(g, 7, n_iter=1)
        two = hankel_project_torch(one, 7, n_iter=1)
        worst = max(worst, rel(two[0].numpy(), one[0].numpy()))
add("HK2_fixed_point", {"pass": bool(worst < 1e-12), "max_rel_err": worst,
                        "tolerance": 1e-12,
                        "summary": f"P(P(g)) == P(g), max rel {worst:.3e}"})

# --- HK3 rank --------------------------------------------------------------
rows, ok = [], True
for L in (1, 3, 5, 7):
    g = exact_signal(32, L, 11)
    s = np.linalg.svd(hankel_matrix_torch(torch.as_tensor(g)).numpy(),
                      compute_uv=False)
    tail = float(s[L:].max() / s[0])
    rows.append({"L": L, "n_sv": len(s), "sv_L_over_sv0": float(s[L - 1] / s[0]),
                 "max_tail_over_sv0": tail,
                 "trackB_hankel_rank": int(hankel_rank(g))})
    ok &= tail < 1e-12 and int(hankel_rank(g)) == L
add("HK3_rank", {"pass": bool(ok), "rows": rows,
                 "summary": "exactly L nonzero singular values, tail < 1e-12"})

# --- HK4 Track B parity ----------------------------------------------------
worst_m, worst_p = 0.0, 0.0
for N in (32, 16, 8):
    for sd in (0, 3):
        g = exact_signal(N, 3, sd)
        worst_m = max(worst_m, rel(hankel_matrix_torch(torch.as_tensor(g)).numpy(),
                                   hankel_matrix(g)))
        worst_p = max(worst_p, rel(
            hankel_project_torch(torch.as_tensor(g)[None], 7, n_iter=4)[0].numpy(),
            hankel_project(g, 7, n_iter=4)))
add("HK4_trackB_parity", {"pass": bool(max(worst_m, worst_p) < 1e-12),
                          "hankel_matrix_rel_err": worst_m,
                          "hankel_project_rel_err": worst_p, "tolerance": 1e-12,
                          "note": "Track D torch operator vs Track B numpy, identical input",
                          "summary": f"matrix {worst_m:.3e}, project {worst_p:.3e}"})

# --- HK5 identity degeneration --------------------------------------------
cfg = TrackDConfig()
w = make_world(2_000_000, sysc=cfg.system, N=32, P=20, snr_db=5.0)
T = lambda a, d: torch.as_tensor(np.array(a, copy=True)[None], dtype=d)
args = (T(np.zeros_like(w.G_true), C128), T(w.Z, F64), T(w.S, C128), T(w.B, C128),
        torch.tensor([w.sigma2], dtype=F64))
full_rank = max_representable_rank(32)
torch.manual_seed(0)
m_ur = URformer(32, 3, ModelConfig(T_UR=3), NumericConfig("float64")).double()
torch.manual_seed(0)
m_hs = URformer(32, 3, ModelConfig(T_UR=3, use_hankel=True,
                                   hankel_rank=full_rank),
                NumericConfig("float64")).double()
with torch.no_grad():
    a_out, b_out = m_ur(*args)[0].numpy(), m_hs(*args)[0].numpy()
e5 = rel(b_out, a_out)
add("HK5_identity_degeneration", {
    "pass": bool(e5 < 1e-12), "rel_err": e5, "tolerance": 1e-12,
    "full_rank_used": full_rank,
    "condition": f"r = min(N-p, p+1) = {full_rank} => projection is identity => "
                 "HS-URformer == URformer layer for layer",
    "summary": f"rel {e5:.3e} < 1e-12 at full rank {full_rank}"})

# --- HK6 gradient safety ---------------------------------------------------
numeric = NumericConfig("float32")
torch.manual_seed(0)
mg = URformer(32, 3, ModelConfig(T_UR=3, d_model=32, L_enc=2, use_hankel=True),
              numeric)
Zs, Ss, Bs, s2s, Gs = [], [], [], [], []
for i in range(4):
    ww = make_world(2_000_000 + i, sysc=cfg.system, N=32, P=20, snr_db=5.0)
    Zs.append(ww.Z); Ss.append(ww.S); Bs.append(ww.B)
    s2s.append(ww.sigma2); Gs.append(ww.G_true)
Zb = torch.as_tensor(np.stack(Zs), dtype=torch.float32)
Sb = torch.as_tensor(np.stack(Ss), dtype=torch.complex64)
Bb = torch.as_tensor(np.stack(Bs), dtype=torch.complex64)
s2b = torch.as_tensor(np.array(s2s), dtype=torch.float32)
Gt = torch.as_tensor(np.stack(Gs), dtype=torch.complex64)
out = mg(torch.zeros_like(Gt), Zb, Sb, Bb, s2b)
loss = torch.mean(torch.sum(torch.abs(out - Gt) ** 2, dim=(1, 2))
                  / torch.sum(torch.abs(Gt) ** 2, dim=(1, 2)))
loss.backward()
per, fin, pos = [], True, True
for i, l in enumerate(mg.layers):
    d = {"layer": i}
    for nm, mod in (("filter_net", l.filter_net), ("former", l.former)):
        gs_ = [p.grad for p in mod.parameters() if p.grad is not None]
        nrm = float(torch.sqrt(sum((g ** 2).sum() for g in gs_))) if gs_ else 0.0
        d[nm + "_grad_norm"] = nrm
        fin &= bool(np.isfinite(nrm)); pos &= nrm > 0
    d["gate_grad_abs"] = float(l.gate.grad.abs()) if l.gate.grad is not None else 0.0
    fin &= bool(np.isfinite(d["gate_grad_abs"])); pos &= d["gate_grad_abs"] > 0
    per.append(d)
add("HK6_gradient_safety", {
    "pass": bool(fin and pos), "loss": float(loss.detach()),
    "all_finite": fin, "all_nonzero": pos, "per_layer": per,
    "note": "projection is @torch.no_grad(), so no gradient path through the SVD",
    "summary": f"loss {float(loss.detach()):.4f}; all grads finite and > 0"})

# --- HK7 N=8 degeneracy, documented ---------------------------------------
n8 = {"N": 8, "pencil": 8 // 2, "shape": list(hankel_matrix_torch(
    torch.zeros(8, dtype=C128)).shape), "max_representable_rank":
    max_representable_rank(8)}
errs = {}
for L in (3, 4, 5, 7):
    g = exact_signal(8, L, 17)
    out = hankel_project_torch(torch.as_tensor(g)[None], 7, n_iter=1)[0].numpy()
    errs[L] = rel(out, g)
# The decisive test: an UNSTRUCTURED random vector. If the projection returns
# it unchanged, the operator constrains nothing at all.
rngv = np.random.default_rng(0)
g_rand8 = rngv.standard_normal(8) + 1j * rngv.standard_normal(8)
g_rand32 = rngv.standard_normal(32) + 1j * rngv.standard_normal(32)
vac8 = rel(hankel_project_torch(torch.as_tensor(g_rand8)[None], 7,
                                n_iter=1)[0].numpy(), g_rand8)
vac32 = rel(hankel_project_torch(torch.as_tensor(g_rand32)[None], 7,
                                 n_iter=1)[0].numpy(), g_rand32)
n8["recovery_rel_err_by_L"] = errs
n8["identity_on_random_input_N8"] = vac8
n8["identity_on_random_input_N32"] = vac32
# CORRECTED characterization: VACUOUS, not lossy.
ok7 = (vac8 < 1e-12 and vac32 > 1e-2 and max(errs.values()) < 1e-12)
add("HK7_N8_degeneracy", {
    "pass": bool(ok7), **n8,
    "note": "CORRECTED characterization. PROMPT 6 predicted the operator "
            "'cannot represent L_k >= 5' at N=8, i.e. that it would be LOSSY. "
            "It is not lossy - it is VACUOUS. With p=4 the embedding is 4x5, so "
            "EVERY length-8 vector has Hankel rank <= 4; a rank-7 request "
            "truncates to min(7,4)=4, which is no truncation at all. Measured: "
            "the projection returns an UNSTRUCTURED random vector unchanged "
            f"(rel {vac8:.2e}) at N=8, while genuinely constraining one at N=32 "
            f"(rel {vac32:.2e}). So at N=8 HS-URformer is EXACTLY URformer, and "
            "Q3's prediction Delta_H <= 0 there is not merely likely but exact: "
            "Delta_H == 0 up to training noise.",
    "summary": f"VACUOUS not lossy: identity on random input at N=8 "
               f"({vac8:.2e}) vs constraining at N=32 ({vac32:.2e})"})

# --- A4 dataset identity vs stage 2 ---------------------------------------
s2cfg = json.load(open("reports/trackD_stage2_results.json"))["config"]
cur = cfg.to_dict()
fields = [("system", None), ("data", None), ("numeric", None), ("baseline", None)]
diffs = {}
for sec, _ in fields:
    for k in cur[sec]:
        if sec == "data" and k == "n_train":
            continue                      # intentionally varies per budget
        # JSON round-trips tuples to lists, so (0, 1e6) != [0, 1e6] in Python.
        # Normalize before comparing or every seed range reads as a difference.
        def _norm(v):
            return list(v) if isinstance(v, (tuple, list)) else v
        if _norm(cur[sec][k]) != _norm(s2cfg[sec][k]):
            diffs[f"{sec}.{k}"] = {"stage2": s2cfg[sec][k], "now": cur[sec][k]}


def h(o):
    return hashlib.sha256(json.dumps(o, sort_keys=True).encode()).hexdigest()[:16]


_nz = lambda d: {k: (list(v) if isinstance(v, (tuple, list)) else v)
                 for k, v in d.items()}
seed_hash_now = h(_nz({k: cur["data"][k] for k in
                   ("train_seed_range", "val_seed_range", "test_seed_range",
                    "n_val", "n_test", "pilot_mode", "fixed_S_seed",
                    "snr_mode", "snr_range_db", "rsr_train_mode")}))
seed_hash_s2 = h(_nz({k: s2cfg["data"][k] for k in
                  ("train_seed_range", "val_seed_range", "test_seed_range",
                   "n_val", "n_test", "pilot_mode", "fixed_S_seed",
                   "snr_mode", "snr_range_db", "rsr_train_mode")}))
same = seed_hash_now == seed_hash_s2 and not diffs
add("A4_dataset_identity", {
    "pass": bool(same), "seed_config_hash_stage2": seed_hash_s2,
    "seed_config_hash_now": seed_hash_now, "differences": diffs,
    "u1_checkpoint_reusable": bool(same),
    "summary": ("datasets identical, stage-2 U1 checkpoints reusable"
                if same else f"DIFFER: {list(diffs)} -> retrain U1")})

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
n_pass = sum(1 for g in res["gates"].values() if g["pass"])
print(f"\n{n_pass}/{len(res['gates'])} gates pass")
print(f"wrote {OUT}")
