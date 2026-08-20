"""Faithful TR 38.901 chain (eqs 12,13,15,16,18,19) vs current Track-A model.

Tr((A A^H)^-1) is exactly proportional to genie-ZF NMSE, so it predicts the
NMSE offset without running a solver.
"""
import numpy as np

N, K = 36, 3
N_CL, M_RAYS = 23, 20            # Cui Table I
R_TAU = 2.3                      # spec Table VI, UMa NLOS
ZETA_DB = 3.0                    # spec Table VI, per-cluster shadowing STD
C_PHI = 1.289                    # spec Table VIII, NLOS
FC_GHZ = 5.0
# spec Table VIII: mu_lgASA = 2.08 - 0.27 log10(fc[GHz]); sigma = 0.11
MU_LG_ASA = 2.08 - 0.27 * np.log10(FC_GHZ)
SIG_LG_ASA = 0.11
# spec Table VII: fixed ray offset basis vectors
ALPHA_M = np.array([0.0447, 0.1413, 0.2492, 0.3715, 0.5129,
                    0.6797, 0.8844, 1.1481, 1.5195, 2.1551])
ALPHA_M = np.concatenate([ALPHA_M, -ALPHA_M])
C_ASA_DEG = 15.0                 # spec Table VIII, NLOS cluster ASA

n_idx = np.arange(N, dtype=np.float64)


def steer(theta_rad):
    return np.exp(-1j * n_idx * np.pi * np.sin(theta_rad))


def gen(rng, *, mode, asa_deg=None, ray_offset="cui"):
    """mode: 'current' (uniform angles, equal power) or 'spec' (full chain)."""
    A = np.zeros((K, N), dtype=np.complex128)
    for k in range(K):
        ds = float(rng.uniform(0.0, 30e-9))
        if mode == "current":
            P = np.full(N_CL, 1.0 / N_CL)
            phi_c = rng.uniform(-90.0, 90.0, size=N_CL)
        else:
            # eq (12)-(13): delays
            X = rng.uniform(1e-12, 1.0, size=N_CL)
            tau = np.sort(-R_TAU * ds * np.log(X))
            tau = tau - tau.min()
            # eq (15)-(16): exponential PDP + per-cluster shadowing, normalized
            Z = rng.normal(0.0, ZETA_DB, size=N_CL)
            Pp = np.exp(-tau * (R_TAU - 1.0) / (R_TAU * ds)) * 10.0 ** (-Z / 10.0)
            P = Pp / Pp.sum()
            # eq (18): cluster angles DERIVED from cluster powers
            asa = asa_deg if asa_deg is not None else \
                10.0 ** rng.normal(MU_LG_ASA, SIG_LG_ASA)
            phi_p = 2.0 * (asa / 1.4) * np.sqrt(-np.log(P / P.max())) / C_PHI
            sign = rng.choice([-1.0, 1.0], size=N_CL)
            Y = rng.normal(0.0, asa / 7.0, size=N_CL)
            phi_user = rng.uniform(-90.0, 90.0)      # Table I incident angle
            phi_c = sign * phi_p + Y + phi_user
        for c in range(N_CL):
            if ray_offset == "cui":
                offs = rng.uniform(-5.0, 5.0, size=M_RAYS)
            else:                                     # eq (19) + Table VII
                offs = C_ASA_DEG * ALPHA_M
            for m in range(M_RAYS):
                th = np.deg2rad(np.clip(phi_c[c] + offs[m], -90.0, 90.0))
                # CN(0,1) ray gain scaled by sqrt(P_n/M)  (spec eq. 10)
                a = (rng.standard_normal() + 1j * rng.standard_normal()) / np.sqrt(2)
                psi = rng.uniform(0.0, 2 * np.pi)     # per-RAY polarization
                A[k] += np.sqrt(P[c] / M_RAYS) * a * np.cos(psi) * steer(th)
    A = A / np.sqrt(np.mean(np.abs(A) ** 2, axis=1, keepdims=True))
    return A


def stats(n=300, seed=11, **kw):
    tr = []
    for t in range(n):
        A = gen(np.random.default_rng([seed, t]), **kw)
        tr.append(np.real(np.trace(np.linalg.inv(A @ A.conj().T))))
    return np.array(tr)


base = stats(mode="current")
print(f"i.i.d. Wishart K/(N-K)        = {K/(N-K):.5f}")
print(f"current model                 = {base.mean():.5f}")
print()
spec_auto = stats(mode="spec")
print(f"full 38.901 chain (ASA~LN)    = {spec_auto.mean():.5f}   "
      f"shift = {10*np.log10(spec_auto.mean()/base.mean()):+.2f} dB")
spec_tab7 = stats(mode="spec", ray_offset="table7")
print(f"  + Table VII ray offsets     = {spec_tab7.mean():.5f}   "
      f"shift = {10*np.log10(spec_tab7.mean()/base.mean()):+.2f} dB")
print()
print("ASA sensitivity (deg -> Tr, shift vs current):")
for asa in (10.0, 20.0, 30.0, 45.0, 60.0, 78.0):
    s = stats(mode="spec", asa_deg=asa)
    print(f"   ASA={asa:5.1f}  Tr={s.mean():.5f}  shift={10*np.log10(s.mean()/base.mean()):+.2f} dB")
print(f"\n(Cui pixel-extraction target: ~1.57x = +1.96 dB)")
print(f"ASA implied by spec Table VIII at {FC_GHZ} GHz = {10**MU_LG_ASA:.1f} deg")
