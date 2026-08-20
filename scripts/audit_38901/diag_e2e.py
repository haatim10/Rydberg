"""End-to-end NMSE: current channel vs spec-faithful 38.901, real solvers, CRN."""
import numpy as np
from rydberg_sim.gs import biased_gs, em_gs
from rydberg_sim.baselines import zf_known_phase
from rydberg_sim.metrics import detection_nmse
from rydberg_sim.qam import generate_qam
from rydberg_sim.calibration import snr_db_to_sigma2

N, K, N_CL, M_RAYS, QAM_M, MAX_ITER = 36, 3, 23, 20, 16, 50
R_TAU, ZETA, C_PHI = 2.3, 3.0, 1.289
MU_LG_ASA, SIG_LG_ASA = 2.08 - 0.27 * np.log10(5.0), 0.11
n_idx = np.arange(N, dtype=np.float64)


def channel(rng, spec_faithful: bool):
    A = np.zeros((K, N), dtype=np.complex128)
    for k in range(K):
        ds = float(rng.uniform(0.0, 30e-9))
        if spec_faithful:
            X = rng.uniform(1e-12, 1.0, size=N_CL)
            tau = np.sort(-R_TAU * ds * np.log(X)); tau -= tau.min()
            Pp = (np.exp(-tau * (R_TAU - 1) / (R_TAU * ds))
                  * 10.0 ** (-rng.normal(0.0, ZETA, size=N_CL) / 10))
            P = Pp / Pp.sum()
            asa = 10.0 ** rng.normal(MU_LG_ASA, SIG_LG_ASA)
            phi_p = 2.0 * (asa / 1.4) * np.sqrt(-np.log(P / P.max())) / C_PHI
            phi = (rng.choice([-1.0, 1.0], size=N_CL) * phi_p
                   + rng.normal(0.0, asa / 7.0, size=N_CL) + rng.uniform(-90, 90))
            phi = (phi + 180.0) % 360.0 - 180.0            # wrap, do NOT clip
        else:
            P = np.full(N_CL, 1.0 / N_CL)
            phi = rng.uniform(-90.0, 90.0, size=N_CL)
        for c in range(N_CL):
            for m in range(M_RAYS):
                raw = phi[c] + rng.uniform(-5.0, 5.0)
                th = np.deg2rad((raw + 180.0) % 360.0 - 180.0)
                a = (rng.standard_normal() + 1j * rng.standard_normal()) / np.sqrt(2)
                psi = rng.uniform(0.0, 2 * np.pi)          # per-RAY (spec eq. 10)
                A[k] += np.sqrt(P[c] / M_RAYS) * a * np.cos(psi) * np.exp(
                    -1j * n_idx * np.pi * np.sin(th))
    return A / np.sqrt(np.mean(np.abs(A) ** 2, axis=1, keepdims=True))


def run(spec_faithful, snr_db, rsr_db, n_trials, seed=5150):
    acc = {a: [0.0, 0.0] for a in ("biased_gs", "em_gs", "genie_zf")}
    for t in range(n_trials):
        r = np.random.default_rng([seed, t, int(snr_db) + 100, int(rsr_db)])
        A = channel(r, spec_faithful)
        qam = generate_qam(r, K, QAM_M)
        s = qam.symbols
        b = np.sqrt(10 ** (rsr_db / 10)) * np.exp(1j * r.uniform(0, 2 * np.pi, N))
        sigma2 = snr_db_to_sigma2(snr_db, np.ones(K), c=1.0)
        w = np.sqrt(sigma2 / 2) * (r.standard_normal(N) + 1j * r.standard_normal(N))
        field = A.conj().T @ s + b + w
        z, theta = np.abs(field), np.angle(field)
        Mx = A
        out = {
            "biased_gs": biased_gs(Mx, z, b, max_iter=MAX_ITER).u_hat,
            "em_gs": em_gs(Mx, z, b, sigma2, max_iter=MAX_ITER).u_hat,
            "genie_zf": zf_known_phase(Mx, z, theta, b),
        }
        for a, sh in out.items():
            d = detection_nmse(sh, s)
            acc[a][0] += d.error_energy
            acc[a][1] += d.expected_energy
    return {a: 10 * np.log10(e / x) for a, (e, x) in acc.items()}


NT = 300
print(f"End-to-end detection NMSE (dB), {NT} trials, CRN, real solvers")
print(f"{'SNR':>5} {'RSR':>5} | {'algorithm':<10} {'current':>9} {'spec-38901':>11} {'delta':>8}")
print("-" * 62)
for snr, rsr in [(-2.0, 12.0), (3.0, 12.0), (8.0, 12.0), (3.0, 0.0), (3.0, 25.0)]:
    cur = run(False, snr, rsr, NT)
    new = run(True, snr, rsr, NT)
    for a in ("biased_gs", "em_gs", "genie_zf"):
        print(f"{snr:5.1f} {rsr:5.1f} | {a:<10} {cur[a]:9.3f} {new[a]:11.3f} "
              f"{new[a]-cur[a]:+8.3f}")
