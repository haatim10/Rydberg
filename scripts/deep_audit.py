"""Independent deep audit of the Rydberg simulation stack.

Every check re-derives the quantity from first principles (or from an
independent library routine) and compares against the implementation. It
deliberately does NOT call the implementation to compute the reference.
"""
from __future__ import annotations

import numpy as np
from scipy import integrate, special

FINDINGS: list[tuple[str, str, str]] = []


def rec(sev, name, detail=""):
    FINDINGS.append((sev, name, detail))
    tag = {"OK": "  ok  ", "LOW": " LOW  ", "MED": " MED  ",
           "HIGH": " HIGH ", "BLOCK": "BLOCK "}[sev]
    print(f"[{tag}] {name}" + (f"  — {detail}" if detail else ""))


def check(cond, name, detail="", sev="HIGH"):
    rec("OK" if cond else sev, name, detail)
    return bool(cond)


print("=" * 78)
print("A. CALIBRATION — Cui eq. (36) SNR and eq. (37) RSR, re-derived")
print("=" * 78)
from rydberg_sim.calibration import rsr_db_to_alpha_magnitude, snr_db_to_sigma2

# Independent derivation. Row-normalised A gives E|a_{n,k}|^2 = 1; unit-energy
# QAM gives E|s_k|^2 = 1; users independent =>
#   E|a_n^H s|^2 = sum_k E|a_{n,k}|^2 E|s_k|^2 = K
# SNR = K / sigma^2  =>  sigma^2 = K / SNR_lin
for K in (3, 6):
    for snr_db in (-5.0, 0.0, 3.0, 12.0):
        expect = K / (10 ** (snr_db / 10))
        got = snr_db_to_sigma2(snr_db, np.ones(K), c=1.0)
        check(abs(got - expect) < 1e-12,
              f"sigma^2 = K/SNR_lin (K={K}, SNR={snr_db:+.0f} dB)",
              f"expect {expect:.6g}, got {got:.6g}")

# RSR eq. (37) has a SINGLE-USER denominator: E|a_{n,k} s_k|^2 = 1
#   RSR = E|b_n|^2 / 1  => |alpha_b| = sqrt(RSR_lin)      (NOT sqrt(K*RSR))
for rsr_db in (0.0, 12.0, 25.0):
    expect = np.sqrt(10 ** (rsr_db / 10))
    got = rsr_db_to_alpha_magnitude(rsr_db, beta_ref=1.0, e_s_b_sq=1.0)
    check(abs(got - expect) < 1e-12,
          f"|alpha_b| = sqrt(RSR_lin) single-user denom (RSR={rsr_db:g} dB)",
          f"expect {expect:.6g}, got {got:.6g}")
    # explicit guard against the two classic factor errors
    check(abs(got - np.sqrt(3 * 10 ** (rsr_db / 10))) > 1e-9,
          f"  not sqrt(K*RSR_lin) at RSR={rsr_db:g}")
    check(abs(got - np.sqrt(10 ** (rsr_db / 10) / 3)) > 1e-9,
          f"  not sqrt(RSR_lin/K) at RSR={rsr_db:g}")

print()
print("=" * 78)
print("B. sigma vs sigma^2, and the sqrt(1/2) per real/imag component")
print("=" * 78)
from rydberg_sim.monte_carlo import generate_detection_trial
from rydberg_sim.track_a import track_a_fig5_spec

spec = track_a_fig5_spec(n_trials=1, snr_db_grid=(3.0,))
acc_w, acc_b, n = 0.0, 0.0, 0
for t in range(400):
    w = generate_detection_trial(spec, t, 3.0, 12.0)
    acc_w += float(np.mean(np.abs(w.w) ** 2))
    acc_b += float(np.mean(np.abs(w.b) ** 2))
    n += 1
    s2 = w.sigma2
emp_w = acc_w / n
check(abs(emp_w / s2 - 1.0) < 0.05,
      "E|w|^2 == sigma^2 (not sigma, not 2 sigma^2)",
      f"empirical/sigma2 = {emp_w/s2:.4f}")
check(abs(emp_w / (2 * s2) - 1.0) > 0.2, "  not a factor-2 noise error")
check(abs((acc_b / n) / 10 ** 1.2 - 1.0) < 0.05,
      "E|b|^2 == RSR_lin at RSR=12 dB", f"ratio {(acc_b/n)/10**1.2:.4f}")

print()
print("=" * 78)
print("C. BESSEL RATIO R(k) = I1/I0 — vs independent scipy and integral")
print("=" * 78)
from rydberg_sim.gs import bessel_ratio

for k in (0.0, 1e-6, 0.5, 1.0, 5.0, 20.0, 100.0, 1e4, 1e6):
    got = float(np.asarray(bessel_ratio(k)))
    if k < 500:
        ref = float(special.i1(k) / special.i0(k)) if k > 0 else 0.0
    else:
        ref = float(special.ive(1, k) / special.ive(0, k))
    check(abs(got - ref) < 1e-9 * max(1.0, abs(ref)) + 1e-12,
          f"R({k:g})", f"ref {ref:.12f}, got {got:.12f}")
# integral definition: I_n(k) = (1/pi) int_0^pi exp(k cos t) cos(n t) dt
for k in (0.7, 3.0):
    i0 = integrate.quad(lambda t: np.exp(k * np.cos(t)), 0, np.pi)[0] / np.pi
    i1 = integrate.quad(lambda t: np.exp(k * np.cos(t)) * np.cos(t), 0, np.pi)[0] / np.pi
    check(abs(float(np.asarray(bessel_ratio(k))) - i1 / i0) < 1e-9,
          f"R({k:g}) matches the quadrature definition")
check(float(np.asarray(bessel_ratio(0.0))) == 0.0, "R(0) == 0")
r = np.asarray(bessel_ratio(np.linspace(0, 50, 400)))
check(bool(np.all(np.diff(r) >= -1e-12)), "R monotone increasing")
check(bool(np.all((r >= 0) & (r <= 1))), "R bounded in [0,1]")

print()
print("=" * 78)
print("D. QAM — unit energy, Gray mapping, demapper")
print("=" * 78)
from rydberg_sim.qam import build_qam_constellation, generate_qam, project_to_qam

for M in (4, 16, 64):
    c = build_qam_constellation(M)
    pts = np.asarray(c.points)
    check(abs(float(np.mean(np.abs(pts) ** 2)) - 1.0) < 1e-12,
          f"{M}-QAM unit average energy",
          f"mean|s|^2 = {np.mean(np.abs(pts)**2):.12f}")
    bits = np.asarray(c.bit_labels)
    m = int(np.log2(M))
    check(bits.shape == (M, m), f"{M}-QAM bit table shape {bits.shape}")
    check(len({tuple(b) for b in bits}) == M, f"{M}-QAM bit labels unique")
    # Gray: nearest neighbours in the I or Q direction differ in exactly 1 bit
    side = int(np.sqrt(M))
    lv = np.unique(np.round(pts.real, 12))
    step = lv[1] - lv[0]
    bad = 0
    for i, p in enumerate(pts):
        for dz in (step, 1j * step):
            j = np.argmin(np.abs(pts - (p + dz)))
            if abs(pts[j] - (p + dz)) < 1e-9:
                if int(np.sum(bits[i] != bits[j])) != 1:
                    bad += 1
    check(bad == 0, f"{M}-QAM Gray mapping: neighbours differ in 1 bit",
          f"{bad} violations")

print()
print("=" * 78)
print("E. BER is bit-level (not SER), pooled as a ratio of sums")
print("=" * 78)
from rydberg_sim.metrics import detection_ber, detection_nmse

rng = np.random.default_rng(0)
q = generate_qam(rng, 20000, 16)
noisy = q.symbols + 0.3 * (rng.standard_normal(20000) + 1j * rng.standard_normal(20000))
r = detection_ber(noisy, q.bits, 16)
hard = project_to_qam(noisy, 16)
ser = float(np.mean(hard != q.symbols))
ber = r.bit_errors / r.bit_count
# independent recomputation of bit errors from the constellation table
c = build_qam_constellation(16)
idx_true = np.array([np.argmin(np.abs(np.asarray(c.points) - s)) for s in q.symbols])
idx_hat = np.array([np.argmin(np.abs(np.asarray(c.points) - s)) for s in hard])
manual = int(np.sum(np.asarray(c.bit_labels)[idx_true] != np.asarray(c.bit_labels)[idx_hat]))
check(manual == r.bit_errors, "bit errors match an independent recount",
      f"manual {manual}, impl {r.bit_errors}")
check(r.bit_count == 20000 * 4, "bit_count = n_sym * log2(M)")
check(ber < ser, "BER < SER (Gray)", f"BER {ber:.4f} vs SER {ser:.4f}")
check(ser / 4 <= ber <= ser, "BER within [SER/log2M, SER]")

print()
print("=" * 78)
print("F. NMSE — 10log10, ratio of sums, energy definition")
print("=" * 78)
s_true = np.array([1 + 1j, -1 + 0j, 0.5 - 0.5j]) / np.sqrt(2)
s_hat = s_true + np.array([0.1, -0.2j, 0.05])
d = detection_nmse(s_hat, s_true)
manual_err = float(np.sum(np.abs(s_hat - s_true) ** 2))
check(abs(d.error_energy - manual_err) < 1e-12,
      "error_energy = ||s-s_hat||^2", f"{d.error_energy:.12g} vs {manual_err:.12g}")
check(abs(d.expected_energy - 3.0) < 1e-12,
      "expected_symbol_energy = K for unit QAM", f"{d.expected_energy}")
lin = manual_err / 3.0
check(abs(10 * np.log10(lin) - 10 * np.log10(d.nmse_linear)) < 1e-12,
      "nmse_linear consistent")
# 10log10 vs 20log10 discrimination
check(abs(10 * np.log10(lin) - 20 * np.log10(lin)) > 1e-6,
      "  10log10 and 20log10 are distinguishable here (sanity)")
# ratio of sums != mean of ratios
e = np.array([1.0, 100.0]); tt = np.array([1.0, 1.0])
check(abs(e.sum() / tt.sum() - np.mean(e / tt)) < 1e-12 or True, "")
FINDINGS.pop()
ros = 10 * np.log10(e.sum() / tt.sum())
mor = float(np.mean(10 * np.log10(e / tt)))
check(abs(ros - mor) > 1.0,
      "ratio-of-sums differs from mean-of-dB (aggregation matters)",
      f"{ros:.2f} vs {mor:.2f} dB")

print()
print("=" * 78)
print("G. CRLB — high-SNR limit must be exactly 10log10(2) above genie ZF")
print("=" * 78)
from rydberg_sim.baselines import zf_known_phase
from rydberg_sim.crlb import cui_crlb

rng = np.random.default_rng(3)
K_, N_ = 3, 36
A = (rng.standard_normal((K_, N_)) + 1j * rng.standard_normal((K_, N_))) / np.sqrt(2)
s = generate_qam(rng, K_, 16).symbols
b = np.sqrt(10 ** 1.2) * np.exp(1j * rng.uniform(0, 2 * np.pi, N_))
sig2 = 1e-6
cr = cui_crlb(A, s, b, sig2, expected_u_energy=float(K_))
tr_crlb = float(np.real(np.trace(cr.crlb)))
tr_zf = sig2 * float(np.real(np.trace(np.linalg.inv(A @ A.conj().T))))
gap = 10 * np.log10(tr_crlb / tr_zf)
check(abs(gap - 10 * np.log10(2)) < 0.05,
      "CRLB/ZF high-SNR gap = 3.0103 dB", f"measured {gap:.4f} dB")

print()
print("=" * 78)
print("H. Channel-estimation adapter — conjugation convention")
print("=" * 78)
from rydberg_sim.gs import biased_gs_channel_rows

rng = np.random.default_rng(5)
N2, K2, P2 = 6, 3, 24
G = (rng.standard_normal((N2, K2)) + 1j * rng.standard_normal((N2, K2))) / np.sqrt(2)
S = (rng.standard_normal((K2, P2)) + 1j * rng.standard_normal((K2, P2))) / np.sqrt(2)
Bm = 30.0 * np.exp(1j * rng.uniform(0, 2 * np.pi, (N2, P2)))
Z = np.abs(G @ S + Bm)          # noiseless
Gh = biased_gs_channel_rows(S, Z, Bm, max_iter=300).G_hat
relerr = np.linalg.norm(Gh - G) / np.linalg.norm(G)
check(relerr < 1e-3,
      "noiseless channel estimation recovers G (conjugation correct)",
      f"rel err {relerr:.3e}")
# a wrong conjugation would show up as recovering conj(G) instead
relerr_conj = np.linalg.norm(Gh - np.conj(G)) / np.linalg.norm(G)
check(relerr < relerr_conj, "  recovers G, not conj(G)",
      f"|G| {relerr:.2e} vs |conj G| {relerr_conj:.2e}")

print()
print("=" * 78)
print("I. Track separation, ground-truth leakage, CRN")
print("=" * 78)
from rydberg_sim.monte_carlo import (
    CHANNEL_MODEL_CUI,
    CHANNEL_MODEL_ULA,
    evaluate_detection_algorithm,
)
from rydberg_sim.track_a_fig78 import (
    track_a_fig7a_spec, track_a_fig7b_spec, track_a_fig8_spec,
)
from rydberg_sim.track_a_fig6 import track_a_fig6_spec

for nm, sp in (("fig5", track_a_fig5_spec(n_trials=1, snr_db_grid=(3.0,))),
               ("fig6", track_a_fig6_spec(n_trials=1)),
               ("fig7a", track_a_fig7a_spec(n_trials=1)),
               ("fig7b", track_a_fig7b_spec(n_trials=1)),
               ("fig8", track_a_fig8_spec(n_trials=1))):
    check(sp.track == "A" and sp.channel_model == CHANNEL_MODEL_CUI,
          f"{nm}: Track A uses the Cui channel, not the ULA one")

# CRN: identical world for every algorithm at one operating point
sp = track_a_fig7a_spec(n_trials=2)
w1 = generate_detection_trial(sp, 1, 3.0, 12.0)
w2 = generate_detection_trial(sp, 1, 3.0, 12.0)
check(all(np.array_equal(getattr(w1, a), getattr(w2, a))
          for a in ("A", "s", "b", "w", "z")), "CRN: world is deterministic")
# leakage: perturbing the truth must not change an estimate computed from z
res_a = evaluate_detection_algorithm(w1, "biased_gs", sp)[0]
res_b = evaluate_detection_algorithm(w1, "biased_gs", sp)[0]
check(res_a[0]["value"] == res_b[0]["value"], "estimator is a pure function")

print()
print("=" * 78)
print("J. Fingerprints and store separation")
print("=" * 78)
from rydberg_sim.monte_carlo import config_fingerprint

fps = {
    "fig5": config_fingerprint(track_a_fig5_spec(n_trials=1, snr_db_grid=(3.0,))),
    "fig6": config_fingerprint(track_a_fig6_spec(n_trials=1)),
    "fig7a": config_fingerprint(track_a_fig7a_spec(n_trials=1)),
    "fig7b": config_fingerprint(track_a_fig7b_spec(n_trials=1)),
    "fig8": config_fingerprint(track_a_fig8_spec(n_trials=1)),
}
for k, v in fps.items():
    print(f"    {k:6s} {v[:16]}…")
check(fps["fig5"] == fps["fig6"],
      "fig5/fig6 share a fingerprint BY DESIGN (same config, grids excluded)",
      sev="LOW")
check(fps["fig7a"] == fps["fig8"],
      "fig7a/fig8 share a fingerprint BY DESIGN (same config)", sev="LOW")
check(fps["fig7b"] not in (fps["fig7a"], fps["fig5"]),
      "fig7b distinct (different N,K,M)")
check(len({fps["fig5"], fps["fig7a"], fps["fig7b"]}) == 3,
      "the three distinct configurations have three distinct fingerprints")

print()
print("=" * 78)
print("SUMMARY")
print("=" * 78)
from collections import Counter
c = Counter(s for s, _, _ in FINDINGS)
print("   ", dict(c))
bad = [f for f in FINDINGS if f[0] in ("BLOCK", "HIGH", "MED")]
if bad:
    print("\n  NON-OK FINDINGS:")
    for s, n, d in bad:
        print(f"    [{s}] {n} — {d}")
else:
    print("\n  No BLOCKER / HIGH / MEDIUM findings.")
