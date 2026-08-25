"""Re-derive every numerical claim in paper/hsgs.tex from the stored data.

Fails loudly if any claim in the manuscript disagrees with the .npz stores,
the audit, or the CRLB JSON. Run before submitting.
"""
from __future__ import annotations
import csv, glob, json, re, sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parent.parent
R = REPO / "results" / "track_b"
TEX_RAW = (REPO / "paper" / "hsgs.tex").read_text()
# LaTeX wraps lines, so match on whitespace-collapsed text
TEX = " ".join(TEX_RAW.split())
ok, bad = [], []


def claim(desc, present_in_tex, computed, expected, tol=0.005):
    hit = present_in_tex in TEX
    agree = abs(computed - expected) <= tol
    (ok if (hit and agree) else bad).append(
        f"{'PASS' if hit and agree else 'FAIL'}  {desc}: tex={present_in_tex!r} "
        f"computed={computed:.4f} expected={expected} in_tex={hit}")


rows = list(csv.DictReader(open(R / "artifact/table_B3_nmse_vs_snr.csv")))
f = lambda k, N=None, P=None: [float(x[k]) for x in rows
                               if (N is None or x["N"] == N) and (P is None or x["P"] == P)]

# --- headline gains (mean over the 12 operating points at each N)
for N, exp in (("8", -0.19), ("16", 0.78), ("32", 2.85)):
    claim(f"mean gain N={N}", f"${'-' if exp<0 else '+'}{abs(exp):.2f}$", np.mean(f("gain_dB", N)), exp)
claim("max single-point gain", "$4.33$~dB", max(f("gain_dB", "32")), 4.33)

# --- EM-GS flatness (same pooling)
em = [np.mean(f("EMGS_dB", N)) for N in ("8", "16", "32")]
claim("EM-GS flatness", "$0.012$~dB", max(em) - min(em), 0.012)

# --- win rates and active fractions
for N, w, a in (("8", 33.5, 57.5), ("16", 74.0, 92.7), ("32", 95.3, 99.6)):
    claim(f"win rate N={N}", f"${w}\\%$", 100 * np.mean(f("win_rate", N)), w, 0.05)
    claim(f"active frac N={N}", f"${a}\\%$", 100 * np.mean(f("active_frac", N)), a, 0.05)

# --- N=32,P=30 sweep
g32 = f("gain_dB", "32", "30")
claim("N=32,P=30 gain at 5 dB", "$2.27$~dB at $5$~dB", g32[2], 2.27)
claim("N=32,P=30 gain range lo", "$2.27$--$3.04$", min(g32), 2.27)
claim("N=32,P=30 gain range hi", "$2.27$--$3.04$", max(g32), 3.04)

# --- EM-GS vs unconstrained CRLB, SNR>=5, N=32 P=30
d = [abs(a - b) for a, b, s in zip(f("EMGS_dB", "32", "30"), f("uncon_CRLB_dB", "32", "30"),
                                   f("SNR_dB", "32", "30")) if s >= 5]
claim("EM-GS within of CRLB", "$0.05$~dB for SNR", max(d), 0.0513, 0.002)

# --- CCRB below unconstrained
gap = [a - b for a, b in zip(f("uncon_CRLB_dB"), f("con_CRLB_dB"))]
claim("CCRB below CRLB lo", "$0.87$--$9.98$", min(gap), 0.87)
claim("CCRB below CRLB hi", "$0.87$--$9.98$", max(gap), 9.98)

# --- crossings
u = f("uncon_CRLB_dB")
n_gs = sum(1 for a, b in zip(f("GS_dB"), u) if a < b)
n_em = sum(1 for a, b in zip(f("EMGS_dB"), u) if a < b)
n_hs = sum(1 for a, b in zip(f("HSGS_dB"), u) if a < b)
claim("GS crossings", "at $18$ and $19$ of $36$ points", n_gs, 18, 0)
claim("EM-GS crossings", "at $18$ and $19$ of $36$ points", n_em, 19, 0)
claim("HS-GS below uncon CRLB", "$26$ of $36$", n_hs, 26, 0)
claim("HS-GS max below uncon", "$6.94$~dB", max(b - a for a, b in zip(f("HSGS_dB"), u) if a < b), 6.94)
ccgap = [a - b for a, b in zip(f("HSGS_dB"), f("con_CRLB_dB"))]
claim("HS-GS above CCRB count", "$34$ of $36$", sum(1 for v in ccgap if v > 0), 34, 0)
claim("CCRB gap lo (positive)", "$0.32$--$8.85$", min(v for v in ccgap if v > 0), 0.32)
claim("CCRB gap hi", "$0.32$--$8.85$", max(ccgap), 8.85)

# --- RSR sweep
r6 = list(csv.DictReader(open(R / "artifact/table_B6_nmse_vs_rsr.csv")))
g = lambda rsr: float(next(x["EMGS_advantage_over_GS_dB"] for x in r6
                           if x["N"] == "32" and float(x["RSR_dB"]) == rsr))
claim("EM over GS RSR=0", "$1.31$~dB", g(0.0), 1.31)
claim("EM over GS RSR=12", "$0.113$~dB", g(12.0), 0.113)
claim("EM over GS RSR=24", "$0.002$~dB", g(24.0), 0.002)

# --- B7 controlled path count
b7 = []
for p in sorted(glob.glob(str(R / "b7/L*.npz"))):
    d_ = np.load(p)
    b7.append((int(Path(p).stem[1:]),
               10 * np.log10(d_["num_em_gs"].sum() / d_["num_hs_gs"].sum()),
               10 * np.log10(d_["num_em_gs"].sum() / d_["denom"].sum()),
               (d_["num_hs_gs"] < d_["num_em_gs"]).mean(), d_["L_hat"].mean(),
               d_["active"].mean(), d_["denom"].size))
seq = ", ".join(f"{v:.2f}".rstrip("0").rstrip(".") if False else f"{v:.2f}" for _, v, *_ in b7)
for L, v, *_ in b7:
    lit = f"{v:.2f}".replace("-", "$-$") if v < 0 else f"{v:.2f}"
    ok.append(f"INFO  B7 L={L}: gain {v:+.3f} dB")
claim("B7 monotone", "7.13,\\;3.58,\\;1.84,\\;1.03,\\;0.59,\\;0.27,\\;0.04,\\;-0.12",
      1.0 if all(b7[i][1] > b7[i + 1][1] for i in range(len(b7) - 1)) else 0.0, 1.0, 0)
claim("B7 gain at L=2", "$7.13$~dB at $L=2$", b7[0][1], 7.13)
claim("B7 gain at L=14", "$+0.04$~dB", b7[6][1], 0.04)
claim("B7 gain at L=16", "$-0.12$~dB", b7[7][1], -0.12)
claim("B7 win rate L=2", "$100\\%$", 100 * b7[0][3], 100.0, 0.05)
claim("B7 win rate L=16", "$45.3\\%$", 100 * b7[7][3], 45.3, 0.05)
emb7 = [t[2] for t in b7]
claim("B7 EM-GS flat", "$0.140$~dB", max(emb7) - min(emb7), 0.140)
claim("B7 mean Lhat at L=16", "$11.55$", b7[7][4], 11.55)
claim("B7 active at L=16", "$90.2\\%$", 100 * b7[7][5], 90.2, 0.05)

# --- trial counts
tot = sum(np.load(p)["denom"].size
          for sub in ("b3", "b4", "b6", "b7") for p in glob.glob(str(R / sub / "*.npz")))
claim("total trials", "$2.8\\times10^{4}$", tot, 28000, 0)
claim("B7 trials/point", "$400$ trials per point", b7[0][6], 400, 0)
claim("extended points", "five points", sum(1 for x in rows if int(x["trials"]) > 400), 5, 0)

# --- CRLB internals
CC = json.loads((R / "constrained_crlb.json").read_text())
old = json.loads((R / "crlb.json").read_text())["b3"]
dd = [CC["unconstrained_rank1"]["b3"][k] - v for k, v in old.items()
      if k in CC["unconstrained_rank1"]["b3"]]
claim("rank1 vs rank2 lo", "$0.17$--$4.64$~dB", min(dd), 0.17)
claim("rank1 vs rank2 hi", "$0.17$--$4.64$~dB", max(dd), 4.64)
jr = CC["jacobian_rank"]["b3"]
claim("Jacobian rank N=8", "$40.4$",
      np.mean([jr[k] for k in jr if k.startswith("N8_")]), 40.4, 0.05)
allc = list(CC["jacobian_cond"]["b3"].values()) + list(CC["jacobian_cond"]["b4"].values())
claim("worst conditioning", "$80.1$", max(allc), 80.1, 0.05)
claim("CRLB trials", "$400$ channel realisations", CC["n_trials"], 400, 0)

# --- structural / audit facts asserted in the text
A = json.loads((R / "artifact_audit.json").read_text())
txt = [("exact observation identity", "never linearised", float(A["Z_equals_abs_GS_B_W"]) == 0.0),
       ("no oracle rank", "No oracle rank is used anywhere in this paper", A["hs_gs_auto_uses_true_L"] is False),
       ("shared scalar Lhat", "single scalar $\\hat L$", "SHARED" in A["per_user_or_shared_order"]),
       ("cadzow sweeps = 4", "$n_{\\mathrm{cz}}=4$", A["cadzow_default_n_iter"] == 4),
       ("project every 1", "project every", A["project_every_default"] == 1),
       ("val_frac 0.3", "\\nu=0.3", abs(A["val_frac_default"] - 0.3) < 1e-9),
       ("select_iter 20", "T_{\\mathrm{sel}}=20", A["select_iter_used_in_runs"] == 20),
       ("pilot non-orthogonality", "0.233", "0.2328" in A["S_orthogonality"]),
       ("RSR single-user", "single-user", "SINGLE USER" in A["RSR_denominator_convention"]),
       ("10log10 not 20", "$10$, not $20$", "NOT 20" in A["db_factor"]),
       ("ratio of sums", "ratio of sums", "ratio of sums" in A["pooling"])]
for desc, lit, cond in txt:
    hit = lit in TEX
    (ok if (hit and cond) else bad).append(
        f"{'PASS' if hit and cond else 'FAIL'}  {desc}: code_ok={cond} in_tex={hit}")

print("\n".join(x for x in ok if x.startswith("INFO")))
print()
print("\n".join(x for x in ok if not x.startswith("INFO")))
if bad:
    print("\n".join(bad))
print(f"\n{len([x for x in ok if x.startswith('PASS')])} passed, {len(bad)} failed")
sys.exit(1 if bad else 0)
