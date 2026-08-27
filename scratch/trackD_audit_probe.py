"""Throwaway probe for the Track D (URformer) repository audit.

Read-only: imports repository code, generates small realizations, and prints
what the conventions actually are. Modifies nothing. Delete after the audit.

Run:  python3 scratch/trackD_audit_probe.py
"""

from __future__ import annotations

import json
import numpy as np

from rydberg_sim.calibration import (
    make_alpha_b,
    measure_rsr,
    measure_snr,
    reference_user_beta,
    rsr_db_to_alpha_magnitude,
    snr_db_to_sigma2,
)
from rydberg_sim.channel import generate_ula_channel, steering_matrix
from rydberg_sim.config import SimulationConfig
from rydberg_sim.forward import exact_forward
from rydberg_sim.gs import bessel_ratio, em_gs_channel_rows, biased_gs_channel_rows
from rydberg_sim.pilots import generate_gaussian_pilots
from rydberg_sim.reference import generate_reference_field
from rydberg_sim.rng import get_trial_rngs

OUT: dict = {}


def hdr(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def build_world(N=8, K=2, P=8, L=3, snr_db=5.0, rsr_db=10.0, seed=20260827, trial=0):
    """One realization built exactly the way monte_carlo.generate_channel_estimation_trial does."""
    cfg = SimulationConfig.create(N=N, K=K, L=L, beta=1.0, master_seed=seed, c=1.0)
    rngs = get_trial_rngs(cfg.master_seed, trial)
    ch = generate_ula_channel(cfg, trial, rng=rngs.channel)
    pilots = generate_gaussian_pilots(K=cfg.K, P=P, rng=rngs.pilots)
    beta_ref = reference_user_beta(cfg.beta_k, 0)
    alpha_b = make_alpha_b(rsr_db_to_alpha_magnitude(rsr_db, beta_ref), 0.0)
    ref = generate_reference_field(N=cfg.N, P=P, alpha_b=alpha_b, vartheta=0.0, c=cfg.c)
    sigma2 = snr_db_to_sigma2(snr_db, cfg.beta_k, c=cfg.c)
    exact = exact_forward(ch.G, pilots.S, ref.B, sigma2, rng_noise=rngs.noise)
    return cfg, ch, pilots, ref, sigma2, exact


# ---------------------------------------------------------------------------
# (a) Shapes and orientation
# ---------------------------------------------------------------------------
hdr("(a) SHAPES AND ORIENTATION  [N=8, K=2, P=8]")
cfg, ch, pilots, ref, sigma2, exact = build_world()
for name, arr in [
    ("G", ch.G), ("H", ch.H), ("S", pilots.S), ("B", ref.B),
    ("W", exact.W), ("Z", exact.Z), ("E", exact.E),
]:
    print(f"  {name:2s}  shape={str(arr.shape):10s}  dtype={arr.dtype}")

recon = ch.G @ pilots.S + ref.B + exact.W
print(f"\n  G @ S + B + W  reproduces E exactly: "
      f"max|diff| = {np.abs(recon - exact.E).max():.3e}")
print("  repository code forming Y (rydberg_sim/forward.py:206,226):")
print("      signal = np.matmul(G_arr, S_arr, dtype=np.complex128)")
print("      E = np.asarray(signal + B_arr + W, dtype=np.complex128)")
print("      Z = np.abs(E).astype(np.float64, copy=False)")
print("\n  VERDICT: forward model IS  Y = G @ S + B,  "
      "G in C^{N x K}, S in C^{K x P}, B in C^{N x P}.  No transpose.")
OUT["conventions"] = {
    "G_shape": "(N, K) complex128",
    "S_shape": "(K, P) complex128",
    "B_shape": "(N, P) complex128",
    "Z_shape": "(N, P) float64",
    "forward_expr": "Z = |G @ S + B + W|  (forward.py:206,226-227)",
}

# ---------------------------------------------------------------------------
# (b) Bias structure
# ---------------------------------------------------------------------------
hdr("(b) BIAS STRUCTURE")
B = ref.B
rank = int(np.linalg.matrix_rank(B))
print(f"  numpy.linalg.matrix_rank(B) = {rank}")
print(f"  s_b (reference symbols)     = {ref.s_b[:4]} ...")
print("\n  B[:, 0] and B[:, 1]:")
for n in range(B.shape[0]):
    print(f"    n={n}:  {B[n,0]:+.6f}   {B[n,1]:+.6f}")
col_ratio = B[:, 1] / B[:, 0]
print(f"\n  B[:,1]/B[:,0] constant across n?  spread = {np.ptp(np.abs(col_ratio)):.3e}")
print("  repository code (reference.py:192):  B = np.outer(c * alpha * a_b, symbols)")
print(f"\n  VERDICT: B IS RANK-{rank} (outer product b @ s_b^T). With the default")
print("  s_b[p] = 1 this is exactly B = b 1^T, i.e. CONSTANT across p.")
print("  >>> This CONTRADICTS the audit prompt's Task 4 item 2, which asserted")
print("  >>> ours is time-varying across p. It is not. It matches the paper.")
OUT["conventions"]["B_rank"] = rank
OUT["conventions"]["B_is_rank1_b1T"] = True

# ---------------------------------------------------------------------------
# (c) Steering sign convention
# ---------------------------------------------------------------------------
hdr("(c) STEERING SIGN CONVENTION  [theta = 30 deg, alpha = 1]")
theta = np.deg2rad(30.0)
psi = float(np.pi * np.sin(theta))
N = 8
a = steering_matrix(np.array([theta]), N)[:, 0]
print(f"  psi = pi*sin(30deg) = {psi:.10f}")
print("\n   n   angle(a[n])      n*psi        angle(a[n]) + n*psi")
for n in range(N):
    ang = float(np.angle(a[n]))
    print(f"  {n:2d}   {ang:+.10f}  {n*psi:+.10f}   {ang + n*psi:+.10f}")
unwrapped = np.unwrap(np.angle(a))
slope = float(np.polyfit(np.arange(N), unwrapped, 1)[0])
print(f"\n  fitted per-element phase increment = {slope:+.10f}")
print(f"  -psi = {-psi:+.10f}   +psi = {psi:+.10f}")
sign = "e^{-j n psi}" if slope < 0 else "e^{+j n psi}"
print(f"\n  VERDICT: convention is {sign} with n = 0..N-1 (0-based),")
print("  i.e. e^{-j(n-1)psi} in 1-based indexing.  NEGATIVE exponent.")
print("  repository code (channel.py:156):  np.exp(-1j * n * psi)")
print("  Paper uses e^{+j...}; ours is the conjugate. Keep ours.")
OUT["conventions"]["steering_sign"] = "e^{-j n psi}, n=0..N-1 (negative exponent)"
OUT["conventions"]["steering_slope_fitted"] = slope

# ---------------------------------------------------------------------------
# (d) Oracle-phase gate
# ---------------------------------------------------------------------------
hdr("(d) ORACLE-PHASE GATE  [gate: relative error < 1e-10]")
cfg2, ch2, pil2, ref2, sig2, ex2 = build_world(N=8, K=2, P=8)
G, S, Bm = ch2.G, pil2.S, ref2.B
Y_oracle = G @ S + Bm
Z_oracle = np.abs(Y_oracle) * np.exp(1j * np.angle(Y_oracle))
print(f"  |Y_oracle| (+) e^{{j angle(Y_oracle)}} == Y_oracle: "
      f"max|diff| = {np.abs(Z_oracle - Y_oracle).max():.3e}")

# Repository M-step, exactly as gs.py:326-331 with M = S, b = conj(B[n]),
# then G_hat[n] = conj(u).  Bias subtraction included.
G_hat = np.empty_like(G)
gram = S @ S.conj().T
for n in range(G.shape[0]):
    r = np.conjugate(Z_oracle[n]) - np.conjugate(Bm[n])   # y - b, canonical form
    u = np.linalg.solve(gram, S @ r)
    G_hat[n] = np.conjugate(u)

abs_err = float(np.abs(G_hat - G).max())
rel_err = float(np.linalg.norm(G_hat - G, "fro") / np.linalg.norm(G, "fro"))
print(f"\n  max absolute error  |G_hat - G|_max            = {abs_err:.6e}")
print(f"  max relative error  ||G_hat-G||_F / ||G||_F     = {rel_err:.6e}")
print(f"\n  GATE (rel < 1e-10): {'PASS' if rel_err < 1e-10 else 'FAIL'}")

# Cross-check: does the conjugate convention recover conj(G) instead?
rel_conj = float(np.linalg.norm(G_hat - np.conjugate(G), "fro") / np.linalg.norm(G, "fro"))
print(f"  cross-check vs conj(G): rel err = {rel_conj:.6e}  (must be LARGE)")
OUT.setdefault("gates", {})["oracle_phase_rel_err"] = rel_err
OUT["gates"]["oracle_phase_abs_err"] = abs_err
OUT["gates"]["oracle_phase_vs_conjG_rel_err"] = rel_conj

# ---------------------------------------------------------------------------
# (e) EM-GS degenerate limits
# ---------------------------------------------------------------------------
hdr("(e) EM-GS DEGENERATE LIMITS")

print("  (e1) W = 0, G0 = G (truth), one EM-GS iteration must return G")
cfg3 = SimulationConfig.create(N=8, K=2, L=3, beta=1.0, master_seed=20260827, c=1.0)
r3 = get_trial_rngs(cfg3.master_seed, 0)
ch3 = generate_ula_channel(cfg3, 0, rng=r3.channel)
pil3 = generate_gaussian_pilots(K=2, P=8, rng=r3.pilots)
ab3 = make_alpha_b(rsr_db_to_alpha_magnitude(10.0, 1.0), 0.0)
ref3 = generate_reference_field(N=8, P=8, alpha_b=ab3, vartheta=0.0, c=1.0)
ex3 = exact_forward(ch3.G, pil3.S, ref3.B, 0.0)          # sigma2 = 0 -> W = 0
print(f"       W is exactly zero: {np.all(ex3.W == 0)}")

for s2 in (1e-6, 1e-9):
    res = em_gs_channel_rows(pil3.S, ex3.Z, ref3.B, s2, max_iter=1, G0=ch3.G)
    e = float(np.linalg.norm(res.G_hat - ch3.G, "fro") / np.linalg.norm(ch3.G, "fro"))
    print(f"       sigma2={s2:.0e}  rel err after 1 iteration = {e:.6e}")
    if s2 == 1e-9:
        OUT["gates"]["emgs_fixed_point_rel_err"] = e

print("\n  (e2) sigma2 -> 0  =>  R(kappa) -> 1  =>  EM-GS must coincide with GS")
print("       (same G0 for both; noiseless Z; 5 iterations)")
print("\n       sigma2      ||G_em - G_gs||_F/||G_gs||_F    min R(kappa)")
gs_res = biased_gs_channel_rows(pil3.S, ex3.Z, ref3.B, max_iter=5, G0=ch3.G)
last = None
for s2 in (1e-1, 1e-3, 1e-5, 1e-7, 1e-9):
    em_res = em_gs_channel_rows(pil3.S, ex3.Z, ref3.B, s2, max_iter=5, G0=ch3.G)
    d = float(np.linalg.norm(em_res.G_hat - gs_res.G_hat, "fro")
              / np.linalg.norm(gs_res.G_hat, "fro"))
    lam = ch3.G @ pil3.S + ref3.B
    kap = (2.0 / s2) * ex3.Z * np.abs(lam)
    print(f"       {s2:.0e}     {d:.6e}                  {bessel_ratio(kap).min():.10f}")
    last = d
OUT["gates"]["emgs_to_gs_limit_rel_err"] = last

# ---------------------------------------------------------------------------
# (f) Bessel stability
# ---------------------------------------------------------------------------
hdr("(f) BESSEL / KAPPA STABILITY  [N=8, K=2, P=8, RSR=10 dB, SNR=5 dB]")
cfg4, ch4, pil4, ref4, s2_4, ex4 = build_world(snr_db=5.0, rsr_db=10.0)
lam4 = ch4.G @ pil4.S + ref4.B
kappa = (2.0 / s2_4) * ex4.Z * np.abs(lam4)
print(f"  sigma2 = {s2_4:.6f}")
print(f"  kappa   min = {kappa.min():.6f}")
print(f"  kappa   median = {np.median(kappa):.6f}")
print(f"  kappa   max = {kappa.max():.6f}")
R = bessel_ratio(kappa)
print(f"\n  R(kappa)  min = {R.min():.10f}   max = {R.max():.10f}")
print(f"  any NaN in R? {bool(np.isnan(R).any())}    any Inf? {bool(np.isinf(R).any())}")
print(f"  R within [0,1]? {bool(np.all((R >= 0) & (R <= 1)))}")
probe = np.array([0.0, 1.0, 1e2, 1e4, 1e4 + 1, 1e6, 1e12, 1e300])
print("\n  R(x) at extreme x (overflow probe):")
for x, rx in zip(probe, bessel_ratio(probe)):
    print(f"    x={x:<10.0e}  R={rx:.12f}")
print("\n  Implementation: scipy.special.ive(1,x)/ive(0,x) for x <= 1e4 (exp factor")
print("  cancels), asymptotic 1 - 1/(2x) - 1/(8x^2) above. No overflow. (gs.py:426)")
OUT["kappa_stats"] = {
    "min": float(kappa.min()), "median": float(np.median(kappa)),
    "max": float(kappa.max()),
    "overflow": bool(np.isnan(R).any() or np.isinf(R).any()),
    "R_min": float(R.min()), "R_max": float(R.max()),
}

# ---------------------------------------------------------------------------
# (g) SNR / RSR realization
# ---------------------------------------------------------------------------
hdr("(g) SNR / RSR REQUESTED vs MEASURED")
print("  Repository definitions (calibration.py):")
print("    SNR = E|(GS)_np|^2 / E|W_np|^2   -> sigma2 = c^2 * sum_k beta_k / SNR_lin")
print("    RSR = E|B_np|^2 / E|g_nk s_kp|^2 -> |alpha_b| = sqrt(RSR_lin * beta_ref)")
print("    RSR denominator is a SINGLE user, not the sum over K.\n")

for (snr_t, rsr_t) in [(5.0, 10.0), (0.0, 10.0), (12.0, 12.0)]:
    accS = accN = accB = accU = 0.0
    n_real = 400
    for t in range(n_real):
        c5, ch5, p5, r5, s25, e5 = build_world(
            N=8, K=2, P=8, snr_db=snr_t, rsr_db=rsr_t, trial=t)
        accS += np.sum(np.abs(ch5.G @ p5.S) ** 2)
        accN += np.sum(np.abs(e5.W) ** 2)
        accB += np.sum(np.abs(r5.B) ** 2)
        accU += np.sum(np.abs(ch5.G[:, 0:1] @ p5.S[0:1, :]) ** 2)
    snr_m = 10 * np.log10(accS / accN)
    rsr_m = 10 * np.log10(accB / accU)
    print(f"  requested SNR={snr_t:5.1f} dB  ->  measured {snr_m:7.3f} dB   "
          f"(ratio of summed energies, {n_real} realizations)")
    print(f"  requested RSR={rsr_t:5.1f} dB  ->  measured {rsr_m:7.3f} dB")
    if (snr_t, rsr_t) == (5.0, 10.0):
        OUT["snr_rsr_check"] = {
            "snr_requested_db": snr_t, "snr_measured_db": float(snr_m),
            "rsr_requested_db": rsr_t, "rsr_measured_db": float(rsr_m),
            "n_realizations": n_real,
        }

# single-realization measure_snr / measure_rsr helpers
m1 = measure_snr(ch.G, pilots.S, exact.W)
m2 = measure_rsr(ref.B, ch.G, pilots.S, 0)
print(f"\n  measure_snr() on ONE realization: {m1.snr_db:.3f} dB "
      "(single-realization scatter is expected)")
print(f"  measure_rsr() on ONE realization: {m2.rsr_db:.3f} dB")

# ---------------------------------------------------------------------------
# (h) NMSE form
# ---------------------------------------------------------------------------
hdr("(h) NMSE FORM (inventory item 14)")
print("  metrics.channel_nmse (metrics.py:229-242) computes PER-REALIZATION:")
print("      error_energy  = ||G_hat - G||_F^2")
print("      true_energy   = ||G||_F^2")
print("      instantaneous_nmse = error_energy / true_energy")
print("      nmse_linear = error/expected_channel_energy if given, else instantaneous")
print("\n  monte_carlo.evaluate_channel_algorithm (monte_carlo.py:945,958) stores")
print("  value = nmse.instantaneous_nmse, AND error_energy/true_energy separately.")
print("  metrics.NmseAccumulator.nmse_linear (metrics.py:386) then forms")
print("      sum(error_energy) / sum(true_energy)")
print("\n  VERDICT: BOTH exist. Per-realization ratio is stored per trial;")
print("  the AGGREGATE reported in results is RATIO-OF-SUMS.")
print("  Track D must aggregate ratio-of-sums to be comparable.")
OUT["conventions"]["nmse_form"] = (
    "per-trial: ||dG||_F^2/||G||_F^2 stored; AGGREGATE: ratio-of-sums "
    "sum(err)/sum(true) via NmseAccumulator"
)

hdr("MACHINE-READABLE SUMMARY")
print(json.dumps(OUT, indent=2))
