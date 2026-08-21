"""Step-0 audit: everything that must hold before the B3/B4/B5 runs."""
from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

import numpy as np

FIND: list[tuple[str, str, str]] = []


def check(cond, name, detail="", sev="FAIL"):
    s = "ok" if cond else sev
    FIND.append((s, name, detail))
    print(f"[{'  ok  ' if cond else ' ' + sev + ' '}] {name}"
          + (f"  — {detail}" if detail else ""))
    return bool(cond)


REPO_A = Path("/home/user/Rydberg")
REPO_B = Path("/home/user/rydberg-trackb")


def git(repo, *args):
    return subprocess.run(["git", "-C", str(repo), *args],
                          capture_output=True, text=True).stdout.strip()


print("=" * 78)
print("1. TRACK A IS UNTOUCHED")
print("=" * 78)
head_a = git(REPO_A, "rev-parse", "HEAD")
remote_a = git(REPO_A, "rev-parse", "origin/track-a-cui-reproduction")
dirty_a = git(REPO_A, "status", "--porcelain")
check(head_a == remote_a, "Track-A HEAD == pushed remote", head_a[:12])
check(dirty_a == "", "Track-A worktree clean",
      f"{len(dirty_a.splitlines())} modified" if dirty_a else "clean")

print()
print("=" * 78)
print("2. ULA GENERATOR MATCHES THE FROZEN MODEL")
print("=" * 78)
from rydberg_sim.track_b_drivers import (
    TRACK_B_K, TRACK_B_L_MAX, TRACK_B_L_MIN, TRACK_B_N, TRACK_B_RSR_DB,
    B1_SNR_DB, B2_P, B2_SNR_DB, draw_L_k, track_b_world,
)

w = track_b_world(0, 10, 5.0)
N, K = w.G.shape
manual = np.zeros_like(w.G)
for k in range(K):
    psi = np.pi * np.sin(w.theta[k])                 # psi = pi sin(theta)
    n = np.arange(N)[:, None]
    manual[:, k] = np.exp(-1j * n * psi[None, :]) @ w.alpha[k]
check(np.allclose(manual, w.H), "g[n,k] = sum_l alpha exp(-j(n-1)psi)",
      f"max|diff| = {np.abs(manual - w.H).max():.2e}")
check(np.allclose(w.G, w.H), "G = c*H with c = 1")
check(all(np.all(np.abs(t) <= np.pi / 2 + 1e-12) for t in w.theta),
      "theta in [-pi/2, pi/2]")
# alpha ~ CN(0, beta/L): check second moment over many draws
acc, cnt = 0.0, 0
for t in range(400):
    ww = track_b_world(t, 10, 5.0)
    for k in range(K):
        acc += float(np.sum(np.abs(ww.alpha[k]) ** 2)) * ww.L_k[k]
        cnt += len(ww.alpha[k])
check(abs(acc / cnt - 1.0) < 0.08, "E|alpha|^2 = beta_k/L_k",
      f"L*E|alpha|^2 = {acc/cnt:.4f} (target 1.0)")

print()
print("=" * 78)
print("3. EXACT OBSERVATION Z = |GS + B + W|")
print("=" * 78)
worst = 0.0
for t in range(50):
    ww = track_b_world(t, 10, 5.0)
    worst = max(worst, float(np.abs(np.abs(ww.G @ ww.S + ww.B + ww.W) - ww.Z).max()))
check(worst == 0.0, "Z == |GS+B+W| exactly (50 worlds)", f"max dev {worst:.2e}")

print()
print("=" * 78)
print("4. LINEARIZED-MODEL TRIPWIRE")
print("=" * 78)
import rydberg_sim.baselines as bl
from rydberg_sim.track_b_proposed import hs_gs, hs_gs_auto

calls = {"n": 0}
_orig = bl.linearised_closed_form_ls


def _trip(*a, **k):
    calls["n"] += 1
    raise AssertionError("LINEARIZED MODEL CALLED during an exact-model run")


bl.linearised_closed_form_ls = _trip
try:
    hs_gs(w.S, w.Z, w.B, w.sigma2, L_hat=3, max_iter=10)
    hs_gs_auto(w.S, w.Z, w.B, w.sigma2, max_iter=10, select_iter=5)
    ok = True
except AssertionError as exc:                      # pragma: no cover
    ok = False
    print("   ", exc)
finally:
    bl.linearised_closed_form_ls = _orig
check(ok and calls["n"] == 0, "HS-GS never calls the linearised solver",
      f"{calls['n']} calls")

print()
print("=" * 78)
print("5. CRN — IDENTICAL WORLDS ACROSS ALL THREE ESTIMATORS")
print("=" * 78)
a = track_b_world(7, 30, 10.0)
b = track_b_world(7, 30, 10.0)
same = all(np.array_equal(getattr(a, x), getattr(b, x))
           for x in ("G", "S", "B", "W", "Z"))
check(same, "world is a deterministic function of (trial, P, SNR)")
check(tuple(a.L_k) == tuple(b.L_k), "L_k reproducible within a world")
# different trial -> different world
c = track_b_world(8, 30, 10.0)
check(not np.array_equal(a.G, c.G), "different trial gives a different world")

print()
print("=" * 78)
print("6. L_k DRAWN PER THE FROZEN SPECIFICATION")
print("=" * 78)
flat = np.array([draw_L_k(t) for t in range(6000)]).ravel()
check(flat.min() == TRACK_B_L_MIN and flat.max() == TRACK_B_L_MAX,
      f"support = {{{TRACK_B_L_MIN}..{TRACK_B_L_MAX}}}")
check(abs(flat.mean() - 5.0) < 0.08, "mean ~ 5", f"{flat.mean():.4f}")
counts = np.array([(flat == v).sum() for v in range(TRACK_B_L_MIN, TRACK_B_L_MAX + 1)])
check(counts.std() / counts.mean() < 0.04, "uniform over the support",
      f"cv = {counts.std()/counts.mean():.4f}")
check(tuple(w.L_k) == draw_L_k(0), "world L_k matches the independent draw")

print()
print("=" * 78)
print("7. HS-GS IS THE AUDITED VERSION")
print("=" * 78)
src = (REPO_B / "rydberg_sim/track_b_proposed.py").read_bytes()
digest = hashlib.sha256(src).hexdigest()
last_commit = git(REPO_B, "log", "-1", "--format=%h %s", "--",
                  "rydberg_sim/track_b_proposed.py")
# bytes, not git()'s stripped text: a trailing newline is a real difference
committed = subprocess.run(
    ["git", "-C", str(REPO_B), "show", "HEAD:rydberg_sim/track_b_proposed.py"],
    capture_output=True).stdout
check(committed == src, "working copy == committed version (no local edits)",
      f"{len(src)} bytes, byte-identical" if committed == src else "DIFFERS")
print(f"     sha256 {digest[:32]}…")
print(f"     last commit touching it: {last_commit}")
# the N=8/16/32 smoke run must have used exactly this file
smoke_commit = git(REPO_B, "log", "-1", "--format=%h", "--",
                   "results/track_b/structure_diagnostic.json")
src_commit = git(REPO_B, "log", "-1", "--format=%h", "--",
                 "rydberg_sim/track_b_proposed.py")
at_smoke = subprocess.run(
    ["git", "-C", str(REPO_B), "show",
     f"{smoke_commit}:rydberg_sim/track_b_proposed.py"],
    capture_output=True).stdout
check(at_smoke == src, "HS-GS unchanged since the N=8/16/32 smoke run",
      f"smoke commit {smoke_commit}, estimator last changed in {src_commit}")

print()
print("=" * 78)
print("8. INACTIVE CONSTRAINT => HS-GS REDUCES TO THE BASELINE")
print("=" * 78)
from rydberg_sim.gs import em_gs_channel_rows
from rydberg_sim.track_b_proposed import hankel_rank_cap

for Nt, P in ((8, 10), (16, 30)):
    ww = track_b_world(3, P, 10.0, N=Nt)
    cap = hankel_rank_cap(Nt)
    r = hs_gs(ww.S, ww.Z, ww.B, ww.sigma2, L_hat=cap, max_iter=50)
    base = em_gs_channel_rows(ww.S, ww.Z, ww.B, ww.sigma2, max_iter=50).G_hat
    check(not r.constraint_active, f"N={Nt}: L_hat=cap({cap}) marks constraint inactive")
    check(np.allclose(r.G_hat, base, rtol=0, atol=1e-12),
          f"N={Nt}: HS-GS == EM-GS exactly when inactive",
          f"max|diff| = {np.abs(r.G_hat - base).max():.2e}")

print()
print("=" * 78)
print("9. FROZEN PARAMETERS (recorded, not modified)")
print("=" * 78)
print(f"     N (baseline) = {TRACK_B_N}, K = {TRACK_B_K}, "
      f"L_k ~ U{{{TRACK_B_L_MIN}..{TRACK_B_L_MAX}}}")
print(f"     RSR = {TRACK_B_RSR_DB} dB, t0 = 50, c = 1, beta_k = 1, d = lambda/2")
print(f"     B1 SNR grid = {B1_SNR_DB}, P panels = (10, 30)")
print(f"     B2 P grid = {B2_P}, B2 SNR = {B2_SNR_DB} dB")

print()
print("=" * 78)
bad = [f for f in FIND if f[0] != "ok"]
print(f"SUMMARY: {len(FIND)} checks, {len(bad)} failures")
if bad:
    for s, n, d in bad:
        print(f"   [{s}] {n} — {d}")
else:
    print("   AUDIT PASSED — safe to launch B3/B4/B5")
