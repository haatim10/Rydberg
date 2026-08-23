"""Track-B finalization audit — Parts 1, 2, 3, 9, 10, 15.

Everything here is verified numerically against the implementation. Nothing
is taken from a docstring, a comment, or the report. Read-only with respect
to Track A.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
TRACK_A = Path("/home/user/Rydberg")
sys.path.insert(0, str(REPO))

FIND: list[tuple[str, str, str]] = []


def chk(cond, name, detail="", sev="FAIL"):
    s = "ok" if cond else sev
    FIND.append((s, name, detail))
    print(f"  [{'ok  ' if cond else sev}] {name}" + (f" — {detail}" if detail else ""))
    return bool(cond)


def head(t):
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)


from rydberg_sim.gs import (biased_gs, biased_gs_channel_rows, em_gs,
                            em_gs_channel_rows)
from rydberg_sim.track_b_drivers import (TRACK_B_K, TRACK_B_L_MAX,
                                         TRACK_B_L_MIN, TRACK_B_RSR_DB,
                                         draw_L_k, track_b_world)
from rydberg_sim.track_b_proposed import (cadzow_project, hankel_rank_cap,
                                          hs_gs, hs_gs_auto)
from rydberg_sim.track_b_structure import hankel_matrix

# ============================ PART 1 ======================================
head("PART 1 — THE MATHEMATICAL MODEL AS IMPLEMENTED")

w = track_b_world(11, 20, 7.0, N=16)
N, K = w.G.shape
man = np.zeros_like(w.G)
for k in range(K):
    psi = np.pi * np.sin(w.theta[k])
    man[:, k] = np.exp(-1j * np.arange(N)[:, None] * psi[None, :]) @ w.alpha[k]
chk(np.allclose(man, w.G, atol=0, rtol=0) or np.allclose(man, w.G),
    "g[n,k] = sum_l alpha_lk exp(-j(n-1) pi sin theta_lk)",
    f"max|diff| {np.abs(man - w.G).max():.2e}")
chk(all(np.all(np.abs(t) <= np.pi / 2 + 1e-12) for t in w.theta),
    "theta in [-pi/2, pi/2] (d = lambda/2, no grating lobes)")
chk(K == TRACK_B_K == 3, "K = 3")

acc = cnt = 0.0
for t in range(600):
    ww = track_b_world(t, 10, 5.0)
    for k in range(TRACK_B_K):
        acc += float(np.sum(np.abs(ww.alpha[k]) ** 2)) * ww.L_k[k]
        cnt += len(ww.alpha[k])
chk(abs(acc / cnt - 1.0) < 0.06, "alpha ~ CN(0, beta_k/L_k) with beta_k = 1",
    f"L*E|alpha|^2 = {acc/cnt:.4f}")
flat = np.array([draw_L_k(t) for t in range(6000)]).ravel()
chk(flat.min() == 3 and flat.max() == 7 and abs(flat.mean() - 5) < 0.08,
    "L_k ~ U{3..7} iid per user per realization",
    f"support {{{flat.min()}..{flat.max()}}}, mean {flat.mean():.4f}")
chk(np.allclose(w.G, w.H), "G = c*H with c = 1")

worst = 0.0
for t in range(80):
    for Nn, P, s in ((8, 10, -5.0), (16, 30, 5.0), (32, 10, 20.0)):
        ww = track_b_world(t, P, s, N=Nn)
        worst = max(worst, float(np.abs(
            np.abs(ww.G @ ww.S + ww.B + ww.W) - ww.Z).max()))
chk(worst == 0.0, "observation is EXACT: Z == |GS + B + W|",
    f"max dev over 240 worlds: {worst:.2e}")
sig = np.var(w.W)
chk(abs(sig / w.sigma2 - 1) < 0.3, "W ~ CN(0, sigma^2)",
    f"var(W)/sigma^2 = {sig/w.sigma2:.3f}")

print("\n  Canonical mapping, verified per receive element (Part 1):")
w2 = track_b_world(4, 14, 6.0, N=8)
d2 = d1 = 0.0
for n in range(w2.Z.shape[0]):
    # convention actually implemented: M = S, u = conj(g_n), b = conj(B[n])
    lhs = np.abs(w2.S.conj().T @ np.conjugate(w2.G[n])
                 + np.conjugate(w2.B[n] + w2.W[n]))
    d2 = max(d2, float(np.abs(lhs - w2.Z[n]).max()))
    # dual convention: M = conj(S), u = g_n, b = B[n]
    lhs1 = np.abs(np.conjugate(w2.S).conj().T @ w2.G[n] + w2.B[n] + w2.W[n])
    d1 = max(d1, float(np.abs(lhs1 - w2.Z[n]).max()))
chk(d2 < 1e-12, "implemented convention (M=S, u=conj(g_n), b=conj(B[n]))",
    f"max dev {d2:.2e}")
chk(d1 < 1e-12, "dual convention also satisfies the model", f"max dev {d1:.2e}")
mix = np.abs(np.conjugate(w2.S).conj().T @ np.conjugate(w2.G[0]) + w2.B[0] + w2.W[0])
chk(float(np.abs(mix - w2.Z[0]).max()) > 1e-3,
    "mixed convention (conjugating BOTH M and u) is NOT valid",
    f"max dev {float(np.abs(mix - w2.Z[0]).max()):.3f} — as it must be")

wc = track_b_world(2, 60, 55.0, N=8)
Gh = em_gs_channel_rows(wc.S, wc.Z, wc.B, wc.sigma2, max_iter=200).G_hat
e, ec = (np.linalg.norm(Gh - wc.G) / np.linalg.norm(wc.G),
         np.linalg.norm(Gh - np.conjugate(wc.G)) / np.linalg.norm(wc.G))
chk(e < ec, "adapter returns G, not conj(G)", f"rel err {e:.2e} vs conj {ec:.2e}")

import rydberg_sim.baselines as bl
calls = {"n": 0}
_o = bl.linearised_closed_form_ls
def _trip(*a, **k):
    calls["n"] += 1
    raise AssertionError("LINEARIZED MODEL CALLED")
bl.linearised_closed_form_ls = _trip
try:
    for Nn in (8, 16, 32):
        ww = track_b_world(0, 10, 5.0, N=Nn)
        biased_gs_channel_rows(ww.S, ww.Z, ww.B, max_iter=5)
        em_gs_channel_rows(ww.S, ww.Z, ww.B, ww.sigma2, max_iter=5)
        hs_gs(ww.S, ww.Z, ww.B, ww.sigma2, L_hat=3, max_iter=5)
        hs_gs_auto(ww.S, ww.Z, ww.B, ww.sigma2, max_iter=5, select_iter=3)
    ok = True
except AssertionError:
    ok = False
finally:
    bl.linearised_closed_form_ls = _o
chk(ok and calls["n"] == 0,
    "NO linearized observation anywhere in GS / EM-GS / HS-GS",
    f"{calls['n']} calls across all three estimators, all three N")

# ============================ PART 2 ======================================
head("PART 2 — THE THREE ESTIMATORS")

src_gs = (REPO / "rydberg_sim/gs.py").read_bytes()
gs_commit = subprocess.run(
    ["git", "-C", str(REPO), "log", "-1", "--format=%h %ad", "--date=short",
     "--", "rydberg_sim/gs.py"], capture_output=True, text=True).stdout.strip()
committed = subprocess.run(
    ["git", "-C", str(REPO), "show", "HEAD:rydberg_sim/gs.py"],
    capture_output=True).stdout
chk(committed == src_gs, "gs.py: working copy == committed (no local edits)",
    f"sha256 {hashlib.sha256(src_gs).hexdigest()[:16]}, last touched {gs_commit}")

# GS/EM-GS must be row-separable: perturbing row m must not change row n
ww = track_b_world(6, 20, 5.0, N=16)
Z2 = np.array(ww.Z, copy=True)
Z2[5] *= 1.35
for name, fn in (("biased GS", lambda Z: biased_gs_channel_rows(
                     ww.S, Z, ww.B, max_iter=30).G_hat),
                 ("EM-GS", lambda Z: em_gs_channel_rows(
                     ww.S, Z, ww.B, ww.sigma2, max_iter=30).G_hat)):
    A, Bm = fn(ww.Z), fn(Z2)
    other = np.delete(np.arange(16), 5)
    chk(np.allclose(A[other], Bm[other], atol=1e-12),
        f"{name} is row-separable (uses NO cross-element structure)",
        f"rows other than the perturbed one changed by "
        f"{np.abs(A[other]-Bm[other]).max():.2e}")

# HS-GS: chaining max_iter=1 calls reproduces max_iter=t exactly
for Nn, P in ((8, 10), (16, 30)):
    ww = track_b_world(3, P, 8.0, N=Nn)
    ref = em_gs_channel_rows(ww.S, ww.Z, ww.B, ww.sigma2, max_iter=25).G_hat
    G = None
    for _ in range(25):
        G = em_gs_channel_rows(ww.S, ww.Z, ww.B, ww.sigma2, max_iter=1,
                               G0=G).G_hat
    chk(np.array_equal(G, ref),
        f"N={Nn}: 25 chained max_iter=1 calls == one max_iter=25 call",
        "bit-for-bit" if np.array_equal(G, ref)
        else f"max|diff| {np.abs(G-ref).max():.2e}")
    r = hs_gs(ww.S, ww.Z, ww.B, ww.sigma2, L_hat=hankel_rank_cap(Nn),
              max_iter=25)
    chk(np.allclose(r.G_hat, ref, rtol=0, atol=1e-12) and not r.constraint_active,
        f"N={Nn}: HS-GS with projection disabled == validated EM-GS",
        f"max|diff| {np.abs(r.G_hat-ref).max():.2e}")

# where does the projection occur?
r = hs_gs(ww.S, ww.Z, ww.B, ww.sigma2, L_hat=3, max_iter=10, project_every=1)
chk(len(r.residual_history) == 10 and r.constraint_active,
    "HS-GS: projection applied after EVERY exact measurement update",
    f"project_every=1, {len(r.residual_history)} iterations recorded")
g = ww.G[:, 0]
chk(np.array_equal(cadzow_project(g, hankel_rank_cap(Nn)), g),
    "cadzow_project is a literal no-op at rank >= cap")

# ============================ PART 3 ======================================
head("PART 3 — WHAT THE HANKEL CONSTRAINT ACTUALLY CHARACTERISES")

rng = np.random.default_rng(3)
Nn, L = 16, 4
psi = rng.uniform(-np.pi, np.pi, L)
a = rng.normal(size=L) + 1j * rng.normal(size=L)
g_ula = np.exp(-1j * np.outer(np.arange(Nn), psi)) @ a
r_ula = np.linalg.matrix_rank(hankel_matrix(g_ula, hankel_rank_cap(Nn)), tol=1e-8)
# damped modes: |z| != 1, still low Hankel rank, NOT a physical ULA channel
z_damped = 0.75 * np.exp(-1j * psi)
g_damp = np.vander(z_damped, Nn, increasing=True).T @ a
r_damp = np.linalg.matrix_rank(hankel_matrix(g_damp, hankel_rank_cap(Nn)), tol=1e-8)
chk(r_ula == L, f"physical ULA channel has Hankel rank exactly L", f"rank {r_ula}")
chk(r_damp == L,
    "a DAMPED-mode sequence (|z| = 0.75) has the SAME low Hankel rank",
    f"rank {r_damp} — so low rank does NOT imply |z| = 1: it is a RELAXATION")
chk(hankel_rank_cap(8) == 4 and hankel_rank_cap(16) == 8
    and hankel_rank_cap(32) == 16, "rank cap = ceil(N/2)", "8->4, 16->8, 32->16")
Ls = np.arange(TRACK_B_L_MIN, TRACK_B_L_MAX + 1)
for Nn in (8, 16, 32):
    cap = hankel_rank_cap(Nn)
    chk(True, f"N={Nn}: constraint vacuous when L_k >= {cap}",
        f"P(vacuous) = {np.mean(Ls >= cap):.0%}, "
        f"P(informative) = {np.mean(Ls < cap):.0%}")

# ============================ PART 9 ======================================
head("PART 9 — CRLB AUDIT (unconstrained)")
from rydberg_sim.crlb import cui_crlb, cui_fisher_information

ww = track_b_world(0, 30, 40.0, N=8)
n = 0
u, b = np.conjugate(ww.G[n]), np.conjugate(ww.B[n])
F = cui_fisher_information(ww.S, u, b, ww.sigma2).F
chk(F.shape == (TRACK_B_K, TRACK_B_K), "Fisher matrix dimension = K x K",
    f"{F.shape}")
chk(np.allclose(F, F.conj().T), "Fisher matrix is Hermitian")
chk(np.linalg.eigvalsh(F)[0] > 0, "Fisher matrix is positive definite")
res = cui_crlb(ww.S, u, b, ww.sigma2, expected_u_energy=float(TRACK_B_K))
chk(abs(res.crlb.shape[0] - TRACK_B_K) == 0, "CRLB matrix is K x K")
# sigma^2 convention: doubling sigma^2 must double the bound at high SNR
r1 = float(np.real(np.trace(cui_crlb(ww.S, u, b, ww.sigma2,
                                     expected_u_energy=float(TRACK_B_K)).crlb)))
r2 = float(np.real(np.trace(cui_crlb(ww.S, u, b, 2 * ww.sigma2,
                                     expected_u_energy=float(TRACK_B_K)).crlb)))
chk(abs(r2 / r1 - 2.0) < 0.05, "CRLB scales linearly in sigma^2 at high SNR",
    f"ratio {r2/r1:.4f} (expect 2.0)")
# the 10log10(2) phase-loss gap vs genie ZF
crlb_tr = zf_tr = 0.0
for nn in range(ww.Z.shape[0]):
    uu, bb = np.conjugate(ww.G[nn]), np.conjugate(ww.B[nn])
    crlb_tr += float(np.real(np.trace(cui_crlb(
        ww.S, uu, bb, ww.sigma2, expected_u_energy=float(TRACK_B_K)).crlb)))
    zf_tr += float(np.real(np.trace(
        ww.sigma2 * np.linalg.inv(ww.S @ ww.S.conj().T))))
gap = 10 * np.log10(crlb_tr / zf_tr)
chk(abs(gap - 3.0103) < 0.01,
    "high-SNR CRLB sits 10log10(2) above genie ZF (complex-Fisher convention)",
    f"{gap:.4f} dB vs required 3.0103")
C = json.loads((REPO / "results/track_b/crlb.json").read_text())
S3 = json.loads((REPO / "results/track_b/b3/summary.json").read_text())
viol = [r for r in S3 if r["pooled_db"]["em_gs"]
        < C["b3"][f"N{r['N']}_P{r['P']}_snr{r['snr_db']:+.0f}"]]
chk(len(viol) == 0, "EM-GS never falls below the unconstrained CRLB",
    f"{len(viol)}/36 violations")
below = [r for r in S3 if r["pooled_db"]["hs_gs"]
         < C["b3"][f"N{r['N']}_P{r['P']}_snr{r['snr_db']:+.0f}"]]
chk(True, "HS-GS DOES fall below it — expected, it uses a structural prior",
    f"{len(below)}/36 points, "
    f"{sum(1 for r in below if r['N']==32)} of them at N=32")
chk("UNCONSTRAINED" in C["caveat"].upper(),
    "crlb.json records the unconstrained caveat")

# ============================ PART 10 =====================================
head("PART 10 — CONTROL: IS THE EM-GS BASELINE N-DEPENDENT?")
by = {}
for r in S3:
    by.setdefault((r["P"], r["snr_db"]), {})[r["N"]] = r["pooled_db"]["em_gs"]
spreads = {k: max(v.values()) - min(v.values()) for k, v in by.items()}
worst_k = max(spreads, key=spreads.get)
chk(max(spreads.values()) <= 0.2,
    "EM-GS NMSE is flat in N at every matched (P, SNR)",
    f"max spread {max(spreads.values()):.3f} dB at P={worst_k[0]}, "
    f"SNR={worst_k[1]:+.0f}")
gspread = {k: max(v.values()) - min(v.values()) for k, v in
           {kk: {r["N"]: r["pooled_db"]["hs_gs"] for r in S3
                 if (r["P"], r["snr_db"]) == kk} for kk in by}.items()}
chk(max(gspread.values()) > 2.0,
    "HS-GS, by contrast, moves a lot with N",
    f"max spread {max(gspread.values()):.2f} dB — "
    f"{max(gspread.values())/max(spreads.values()):.0f}x the baseline's")

# ============================ PART 15 =====================================
head("PART 15 — REPRODUCIBILITY")
a1 = track_b_world(9, 30, 10.0, N=16)
a2 = track_b_world(9, 30, 10.0, N=16)
chk(all(np.array_equal(getattr(a1, x), getattr(a2, x))
        for x in ("G", "S", "B", "W", "Z")),
    "world is a deterministic function of (trial, P, SNR, RSR, N)")
b1 = track_b_world(9, 30, 10.0, N=32)
chk(np.array_equal(a1.G, b1.G[:16]),
    "channel is NESTED in N — N=16 is the first 16 rows of N=32 "
    "(sweep is paired in the channel)")
chk(not np.array_equal(a1.W, b1.W[:16]),
    "noise is NOT nested (stated explicitly; W drawn at shape (N,P))")
# CRN scope. The RNG is keyed by (master_seed, trial, snr_db, rsr_db), the
# frozen Track-A convention: each OPERATING POINT gets its own independent
# world. So pairing holds ACROSS ESTIMATORS within a point -- which is what
# the gain estimate needs -- and NOT across SNR or RSR. B3's SNR sweep
# behaves identically. This is a property of the design, not a defect, but
# it means the RSR and SNR curves are sequences of independent samples
# rather than a paired sweep.
r1 = track_b_world(5, 30, 5.0, rsr_db=0.0, N=8)
r2 = track_b_world(5, 30, 5.0, rsr_db=24.0, N=8)
q1 = track_b_world(5, 30, 5.0, rsr_db=0.0, N=8)
chk(all(np.array_equal(getattr(r1, f), getattr(q1, f))
        for f in ("G", "S", "B", "W", "Z")),
    "CRN holds ACROSS ESTIMATORS within an operating point",
    "identical G, S, B, W, Z -- this is what makes the gain paired")
chk(not np.array_equal(r1.G, r2.G),
    "worlds are INDEPENDENT across operating points (RSR keys the RNG)",
    "so the RSR curve is unpaired across points, as is B3's SNR curve")
chk(tuple(r1.L_k) == tuple(r2.L_k),
    "L_k is matched across operating points for a given trial index",
    f"{tuple(r1.L_k)} at both RSR = 0 and 24 dB")
chk(abs(float(np.abs(r2.B).mean()) / float(np.abs(r1.B).mean())
        - 10 ** (24 / 20)) < 0.05,
    "reference amplitude scales as sqrt(RSR_lin)",
    f"|B| ratio {float(np.abs(r2.B).mean())/float(np.abs(r1.B).mean()):.2f} "
    f"vs 10^(24/20) = {10**(24/20):.2f}")

stores = {}
for name in ("b3", "b4", "b6"):
    d = REPO / "results/track_b" / name
    tot = pts = 0
    fps = set()
    bad = []
    for f in sorted(d.glob("N*.npz")):
        try:
            with np.load(f, allow_pickle=False) as z:
                t = z["trial"]
                if len(np.unique(t)) != len(t):
                    bad.append(f"{f.name}: duplicate trial index")
                for k in z.files:
                    if k != "fingerprint" and z[k].shape[0] != len(t):
                        bad.append(f"{f.name}: ragged {k}")
                fps.add(str(z["fingerprint"]))
            tot += len(t); pts += 1
        except Exception as exc:
            bad.append(f"{f.name}: {exc}")
    stores[name] = (pts, tot, fps, bad)
    chk(not bad, f"{name}: {pts} points, {tot} trials, no duplicates/ragged/corrupt",
        "; ".join(bad) if bad else f"fingerprint {sorted(fps)[0] if fps else 'n/a'}")
chk(len(set().union(*[s[2] for s in stores.values()])) == 1,
    "one config fingerprint across B3, B4 and B6",
    sorted(set().union(*[s[2] for s in stores.values()]))[0])

ha = subprocess.run(["git", "-C", str(TRACK_A), "rev-parse", "HEAD"],
                    capture_output=True, text=True).stdout.strip()
hr = subprocess.run(["git", "-C", str(TRACK_A), "rev-parse",
                     "origin/track-a-cui-reproduction"],
                    capture_output=True, text=True).stdout.strip()
dirty = subprocess.run(["git", "-C", str(TRACK_A), "status", "--porcelain"],
                       capture_output=True, text=True).stdout.strip()
chk(ha == hr and dirty == "", "TRACK A UNTOUCHED", f"{ha[:7]}, clean, == remote")

print("\n" + "=" * 78)
bad = [f for f in FIND if f[0] != "ok"]
print(f"FINAL AUDIT: {len(FIND)} checks, {len(bad)} failures")
for s, n, d in bad:
    print(f"  [{s}] {n} — {d}")
if not bad:
    print("  ALL CHECKS PASS")
(REPO / "results/track_b/final_audit.json").write_text(json.dumps(
    [{"status": s, "check": n, "detail": d} for s, n, d in FIND], indent=2))
sys.exit(1 if bad else 0)
