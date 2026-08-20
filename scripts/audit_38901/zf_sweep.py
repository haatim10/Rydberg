"""ZF-only scenario/geometry sweep for exact-Cui reproduction.

Genie ZF error = (A A^H)^-1 A w, so E||err||^2 = sigma^2 Tr((A A^H)^-1) and,
with sigma^2 = K/SNR_lin and expected symbol energy K,

    NMSE_zf(linear) = Tr((A A^H)^-1) / SNR_lin

which is exact (verified: Tr=0.0909 -> -22.41 dB at SNR=12, matching the
measured production value). No solver run is needed for the sweep.
"""
import numpy as np

N, K = 36, 3
SNR_DB = 12.0
SNR_LIN = 10 ** (SNR_DB / 10)
CUI_ZF_TARGET_DB = -20.5          # our -22.41 dB + ~1.9 dB (pixel extraction)

# ---- Cui Table I overrides (paper, explicit) ----
CUI_NCL, CUI_M = 23, 20
CUI_DS_MAX = 30e-9
CUI_RAY_OFF = 5.0
CUI_ANG = (-90.0, 90.0)

# ---- 38.901 scenario params at fc = 5 GHz ----
# UMa rows are from the UPLOADED chapter (Tables VI/VIII).
# Other scenarios are from TR 38.901 Table 7.5-6 proper (not in the upload) --
# flagged as such in the report.
def lg(mu):
    return 10 ** mu


SCEN = {
    # name: (r_tau, zeta_dB, ASA_deg, ZSA_deg, c_ASA, c_ZSA, n_cl, source)
    "UMa-NLOS": (2.3, 3.0, lg(2.08 - 0.27 * np.log10(5)), lg(-0.3236 * np.log10(5) + 1.512), 15, 7, 20, "upload"),
    "UMa-LOS":  (2.5, 3.0, lg(1.81), lg(0.95), 11, 7, 12, "upload"),
    "UMi-NLOS": (2.1, 3.0, lg(-0.08 * np.log10(1 + 5) + 1.81), lg(-0.04 * np.log10(1 + 5) + 1.21), 10, 7, 19, "TR 7.5-6"),
    "UMi-LOS":  (3.0, 3.0, lg(-0.05 * np.log10(1 + 5) + 1.62), lg(-0.1 * np.log10(1 + 5) + 0.73), 3, 7, 12, "TR 7.5-6"),
    "RMa-NLOS": (1.7, 3.0, lg(1.52), lg(0.88), 2, 3, 10, "TR 7.5-6"),
    "RMa-LOS":  (3.8, 3.0, lg(1.52), lg(0.47), 2, 3, 11, "TR 7.5-6"),
    "InH-NLOS": (3.0, 3.0, lg(1.863 - 0.11 * np.log10(1 + 5)), lg(1.387 - 0.15 * np.log10(1 + 5)), 11, 9, 19, "TR 7.5-6"),
    "InH-LOS":  (3.6, 6.0, lg(1.781 - 0.19 * np.log10(1 + 5)), lg(1.44 - 0.26 * np.log10(1 + 5)), 8, 9, 15, "TR 7.5-6"),
}
C_PHI, C_THETA = 1.289, 1.178


def positions(geom):
    """Element positions in units of lambda/2."""
    if geom.startswith("ULA"):
        axis = geom.split("-")[1]           # x or y
        u = np.arange(N, dtype=float)
        z = np.zeros(N)
        return {"x": (u, z, z), "y": (z, u, z)}[axis]
    r, c = (int(v) for v in geom.split("-")[1].split("x"))
    assert r * c == N
    i, j = np.meshgrid(np.arange(r, dtype=float), np.arange(c, dtype=float), indexing="ij")
    return i.ravel(), np.zeros(N), j.ravel()      # x-z plane (horiz x, vert z)


def gen(rng, *, geom, per_elem_pol, cui_ang, scen, three_d, row_norm, mu_axis):
    px, py, pz = positions(geom)
    r_tau, zeta, ASA, ZSA, cASA, cZSA, ncl_s, _ = SCEN[scen]
    ncl, M = CUI_NCL, CUI_M
    mu = {"y": np.array([0.0, 1.0, 0.0]), "x": np.array([1.0, 0.0, 0.0])}[mu_axis]
    A = np.zeros((K, N), dtype=np.complex128)
    for k in range(K):
        ds = float(rng.uniform(0.0, CUI_DS_MAX))
        X = rng.uniform(1e-12, 1.0, size=ncl)
        tau = np.sort(-r_tau * ds * np.log(X)); tau -= tau.min()
        Pp = np.exp(-tau * (r_tau - 1) / (r_tau * ds)) * 10.0 ** (-rng.normal(0, zeta, ncl) / 10)
        P = Pp / Pp.sum()
        if cui_ang:                      # Table I: incident angles U(-90,90)
            phi_c = rng.uniform(*CUI_ANG, size=ncl)
        else:                            # 38.901 eq. (18), power-derived, wrapped
            pp = 2 * (ASA / 1.4) * np.sqrt(-np.log(P / P.max())) / C_PHI
            phi_c = (rng.choice([-1.0, 1.0], ncl) * pp + rng.normal(0, ASA / 7, ncl)
                     + rng.uniform(-90, 90))
            phi_c = (phi_c + 180) % 360 - 180
        if three_d:
            tp = -ZSA * np.log(P / P.max()) / C_THETA
            th_c = 90.0 + rng.choice([-1.0, 1.0], ncl) * tp + rng.normal(0, ZSA / 7, ncl)
        else:
            th_c = np.full(ncl, 90.0)
        for c in range(ncl):
            off = rng.uniform(-CUI_RAY_OFF, CUI_RAY_OFF, size=M)
            phi = np.deg2rad((phi_c[c] + off + 180) % 360 - 180)
            th_off = rng.uniform(-CUI_RAY_OFF, CUI_RAY_OFF, M) if three_d else np.zeros(M)
            th = np.deg2rad(np.clip(th_c[c] + th_off, 0.1, 179.9))
            # arrival unit vectors
            kx, ky, kz = np.sin(th) * np.cos(phi), np.sin(th) * np.sin(phi), np.cos(th)
            ph = np.pi * (np.outer(kx, px) + np.outer(ky, py) + np.outer(kz, pz))  # (M,N)
            a = (rng.standard_normal(M) + 1j * rng.standard_normal(M)) / np.sqrt(2)
            kv = np.stack([kx, ky, kz], 1)                       # (M,3)
            mperp = mu - kv * (kv @ mu)[:, None]                 # project mu onto plane _|_ k
            nrm = np.linalg.norm(mperp, axis=1, keepdims=True)
            e1 = np.where(nrm > 1e-12, mperp / np.maximum(nrm, 1e-12), 0.0)
            e2 = np.cross(kv, e1)
            g1, g2 = e1 @ mu, e2 @ mu                            # (M,)
            if per_elem_pol:                                     # Cui: eps per (n,k,l)
                psi = rng.uniform(0, 2 * np.pi, size=(M, N))
                coup = g1[:, None] * np.cos(psi) + g2[:, None] * np.sin(psi)
            else:                                                # 38.901 eq.(10): per ray
                psi = rng.uniform(0, 2 * np.pi, size=M)
                coup = (g1 * np.cos(psi) + g2 * np.sin(psi))[:, None]
            A[k] += np.sqrt(P[c] / M) * np.sum(
                a[:, None] * coup * np.exp(-1j * ph), axis=0)
    if row_norm:
        A = A / np.sqrt(np.mean(np.abs(A) ** 2, axis=1, keepdims=True))
    return A


def evaluate(n=250, seed=31, **kw):
    tr, cond, corr, bad = [], [], [], 0
    for t in range(n):
        A = gen(np.random.default_rng([seed, t]), **kw)
        if not np.all(np.isfinite(A)) or np.linalg.matrix_rank(A) < K:
            bad += 1
            continue
        G = A @ A.conj().T
        tr.append(np.real(np.trace(np.linalg.inv(G))))
        cond.append(np.linalg.cond(G))
        num = np.mean(A[:, :-1] * np.conj(A[:, 1:]))
        corr.append(abs(num) / np.mean(np.abs(A) ** 2))
    tr = np.array(tr)
    return dict(tr=tr.mean(), tr_se=tr.std(ddof=1) / np.sqrt(len(tr)),
                zf_db=10 * np.log10(tr.mean() / SNR_LIN),
                cond=float(np.median(cond)), corr=float(np.mean(corr)), bad=bad)
