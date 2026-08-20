"""Robustness of the +1.57 dB conditioning shift: clip vs wrap, isolate mechanisms."""
import numpy as np

N, K, N_CL, M = 36, 3, 23, 20
R_TAU, ZETA, C_PHI = 2.3, 3.0, 1.289
MU_LG_ASA, SIG_LG_ASA = 2.08 - 0.27 * np.log10(5.0), 0.11
n_idx = np.arange(N, dtype=np.float64)


def gen(rng, *, pdp=True, derived_ang=True, angle_mode="clip", asa=None):
    A = np.zeros((K, N), dtype=np.complex128)
    for k in range(K):
        ds = float(rng.uniform(0.0, 30e-9))
        if pdp:
            X = rng.uniform(1e-12, 1.0, size=N_CL)
            tau = np.sort(-R_TAU * ds * np.log(X)); tau -= tau.min()
            Z = rng.normal(0.0, ZETA, size=N_CL)
            Pp = np.exp(-tau * (R_TAU - 1) / (R_TAU * ds)) * 10.0 ** (-Z / 10)
            P = Pp / Pp.sum()
        else:
            P = np.full(N_CL, 1.0 / N_CL)
        if derived_ang:
            a_ = asa if asa is not None else 10.0 ** rng.normal(MU_LG_ASA, SIG_LG_ASA)
            phi_p = 2.0 * (a_ / 1.4) * np.sqrt(-np.log(P / P.max())) / C_PHI
            phi = (rng.choice([-1.0, 1.0], size=N_CL) * phi_p
                   + rng.normal(0.0, a_ / 7.0, size=N_CL) + rng.uniform(-90, 90))
        else:
            phi = rng.uniform(-90.0, 90.0, size=N_CL)
        for c in range(N_CL):
            for m in range(M):
                raw = phi[c] + rng.uniform(-5.0, 5.0)
                if angle_mode == "clip":
                    th = np.clip(raw, -90.0, 90.0)
                elif angle_mode == "wrap":            # wrap into [-180,180), keep sin
                    th = (raw + 180.0) % 360.0 - 180.0
                else:                                  # reject outside FoV
                    if not (-90.0 <= raw <= 90.0):
                        continue
                    th = raw
                a = (rng.standard_normal() + 1j * rng.standard_normal()) / np.sqrt(2)
                psi = rng.uniform(0.0, 2 * np.pi)
                A[k] += np.sqrt(P[c] / M) * a * np.cos(psi) * np.exp(
                    -1j * n_idx * np.pi * np.sin(np.deg2rad(th)))
    return A / np.sqrt(np.mean(np.abs(A) ** 2, axis=1, keepdims=True))


def stats(n=400, seed=23, **kw):
    tr = [np.real(np.trace(np.linalg.inv((lambda A: A @ A.conj().T)(
        gen(np.random.default_rng([seed, t]), **kw))))) for t in range(n)]
    tr = np.array(tr)
    return tr.mean(), tr.std(ddof=1) / np.sqrt(n)


base, base_se = stats(pdp=False, derived_ang=False)
print(f"baseline (current model)              Tr={base:.5f} +-{base_se:.5f}\n")

print("mechanism isolation (each vs baseline):")
for lbl, kw in [
    ("PDP only (uniform angles)",      dict(pdp=True,  derived_ang=False)),
    ("derived angles only (flat PDP)", dict(pdp=False, derived_ang=True)),
    ("both (full 38.901)",             dict(pdp=True,  derived_ang=True)),
]:
    m, se = stats(**kw)
    print(f"  {lbl:32s} Tr={m:.5f} +-{se:.5f}  shift={10*np.log10(m/base):+.2f} dB")

print("\nangle handling robustness (full chain):")
for am in ("clip", "wrap", "reject"):
    m, se = stats(pdp=True, derived_ang=True, angle_mode=am)
    print(f"  {am:8s} Tr={m:.5f} +-{se:.5f}  shift={10*np.log10(m/base):+.2f} dB")

print("\nseed stability (full chain, clip):")
for s in (23, 101, 999):
    m, se = stats(seed=s, pdp=True, derived_ang=True)
    print(f"  seed={s:4d} Tr={m:.5f} +-{se:.5f}  shift={10*np.log10(m/base):+.2f} dB")
