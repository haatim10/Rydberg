"""Diagnostic: per-element vs per-ray polarization -> channel conditioning.

Tr((A A^H)^-1) is exactly proportional to genie-ZF NMSE (sigma^2 * Tr / K),
so it predicts the NMSE offset without running any solver.
"""
import numpy as np
from rydberg_sim.channel_cui import CuiChannelParams, _mu_hat, _polarization_basis, _array_phase, _cn01

P = CuiChannelParams()
N, K = 36, 3


def gen(rng, per_ray_pol: bool):
    mu = _mu_hat(P)
    n_idx = np.arange(N, dtype=np.float64)
    two_pi_f = 2.0 * np.pi * P.carrier_hz
    A = np.zeros((K, N), dtype=np.complex128)
    for k in range(K):
        ds = float(rng.uniform(0.0, P.delay_spread_max_s))
        for _c in range(P.n_clusters):
            th_c = float(rng.uniform(P.angle_min_deg, P.angle_max_deg))
            tau_c = float(rng.uniform(0.0, ds)) if ds > 0 else 0.0
            for _r in range(P.n_rays_per_cluster):
                off = float(rng.uniform(-P.cluster_ray_offset_deg, P.cluster_ray_offset_deg))
                th = np.deg2rad(np.clip(th_c + off, P.angle_min_deg, P.angle_max_deg))
                alpha = _cn01(rng)
                e1, e2 = _polarization_basis(th)
                if per_ray_pol:
                    # 38.901 eq. (10): Phi indexed by (cluster, ray) only.
                    psi = rng.uniform(0.0, 2.0 * np.pi)
                else:
                    # current code: independent psi per antenna element
                    psi = rng.uniform(0.0, 2.0 * np.pi, size=N)
                coup = np.dot(mu, e1) * np.cos(psi) + np.dot(mu, e2) * np.sin(psi)
                A[k, :] += alpha * np.exp(-1j * two_pi_f * tau_c) * coup * _array_phase(n_idx, th)
    # production row normalization
    A = A / np.sqrt(np.mean(np.abs(A) ** 2, axis=1, keepdims=True))
    return A


def stats(per_ray_pol, n=300, seed=7):
    tr, cond = [], []
    for t in range(n):
        A = gen(np.random.default_rng([seed, t]), per_ray_pol)
        G = A @ A.conj().T
        tr.append(np.real(np.trace(np.linalg.inv(G))))
        cond.append(np.linalg.cond(G))
    return np.array(tr), np.array(cond)


tr_cur, c_cur = stats(False)
tr_new, c_new = stats(True)
iid = K / (N - K)

print(f"i.i.d. Wishart K/(N-K)          = {iid:.5f}")
print(f"current (per-element pol) mean  = {tr_cur.mean():.5f}   median={np.median(tr_cur):.5f}  cond={np.median(c_cur):.1f}")
print(f"38.901  (per-ray pol)     mean  = {tr_new.mean():.5f}   median={np.median(tr_new):.5f}  cond={np.median(c_new):.1f}")
print()
print(f"ratio new/current (mean)        = {tr_new.mean()/tr_cur.mean():.3f}")
print(f"predicted ZF NMSE shift         = {10*np.log10(tr_new.mean()/tr_cur.mean()):+.2f} dB")
print(f"(audit target from Cui pixels: ~1.57x  = +1.96 dB)")
