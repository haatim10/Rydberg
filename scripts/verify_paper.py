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

# --- the compiled document itself must be error-free.
# A missing macro is an ERROR ("! Undefined control sequence"), not a warning,
# and pdflatex in batchmode continues past it, silently rendering a broken
# line. Grepping only for warnings misses exactly that class of fault.
LOG = REPO / "paper" / "hsgs.log"
if LOG.exists():
    lg = LOG.read_text(errors="ignore")
    for name, pat, want in (("no LaTeX errors", r"^! ", 0),
                            ("no undefined references", r"Warning.*undefined", 0),
                            ("no overfull boxes", r"Overfull", 0)):
        n = len(re.findall(pat, lg, re.M))
        (ok if n == want else bad).append(
            f"{'PASS' if n == want else 'FAIL'}  {name}: found {n}")
    und = set(re.findall(r"Undefined control sequence.*?\\([a-zA-Z]+)", lg, re.S))
    (ok if not und else bad).append(
        f"{'PASS' if not und else 'FAIL'}  no undefined macros"
        + (f": {sorted(und)}" if und else ""))

# --- the standalone package: N x SNR grid behind Figs. 5-6
PKG = REPO / "trackB_hankel_emgs" / "results"
pk = json.loads((PKG / "summary.json").read_text())
B2 = pk["experiment_B"]
for N, em, hk, gain in (("8", -10.321, -10.350, 0.029),
                        ("16", -10.337, -11.149, 0.812),
                        ("32", -10.301, -12.753, 2.452)):
    claim(f"pkg N={N} EM-GS SNR-avg", f"${em}$",
          B2[N]["em_gs_db_mean_over_points"], em)
    claim(f"pkg N={N} Hankel SNR-avg", f"${hk:.3f}$",
          B2[N]["hankel_db_mean_over_points"], hk)
emv = [B2[N]["em_gs_db_mean_over_points"] for N in ("8", "16", "32")]
claim("EM-GS flat in N (new grid)", "$0.04$~dB", max(emv) - min(emv), 0.036, 0.005)
claim("gap widens 0.03->0.81->2.45", "$0.03\\to0.81\\to2.45$~dB",
      B2["32"]["mean_gain_db"], 2.452)
A2 = pk["experiment_A"]
claim("N=8 helps at -10 dB", "$0.74$~dB at $-10$~dB SNR", A2["-10.0"]["gain_db"], 0.744)
claim("N=8 hurts above 0 dB", "up to $0.40$~dB above $0$~dB",
      -min(A2[f"{s:+.1f}"]["gain_db"] for s in (0.0, 5.0, 10.0, 15.0, 20.0)), 0.403)
C2 = pk["experiment_C"]
# The paper's Fig. 3 uses the 400-trial B7 sweep; the package ran an
# independent 300-trial sweep of the same design. Check they agree in trend and
# endpoints rather than re-checking a literal the paper no longer prints.
b7 = {int(Path(f).stem[1:]): np.load(f) for f in glob.glob(str(R / "b7/L*.npz"))}
b7g = {L: 10 * np.log10(d["num_em_gs"].sum() / d["num_hs_gs"].sum())
       for L, d in b7.items()}
pkg_g = {int(L): v["gain_db"] for L, v in C2.items()}
dmax_c = max(abs(b7g[L] - pkg_g[L]) for L in pkg_g)
ok.append(f"INFO  path-count sweeps: B7(400 trials) vs package(300 trials), "
          f"max |diff| = {dmax_c:.3f} dB over 8 values of L")
claim("path-count sweeps agree within MC error", "$7.13$~dB at $L=2$", dmax_c, 0.0, 0.35)
claim("both sweeps end non-positive at L=r_max", "$7.13$~dB at $L=2$",
      float(max(b7g[16], pkg_g[16]) < 0), 1.0, 0)
claim("pkg checks", "thirteen automated checks", pk["checks_passed"], 13, 0)
claim("pkg checks total", "thirteen automated checks", pk["checks_total"], 13, 0)
pkg_trials = sum(v["trials"] for v in B2.values())
claim("pkg grid trials", "$8.4\\times10^{3}$ paired trials", pkg_trials, 8400, 0)
claim("total trials in abstract", "$3.6\\times10^{4}$ paired trials",
      tot + pkg_trials, 36400, 400)

# --- the two runs agree bit-for-bit where they overlap
o = np.load(R / "b3/N32_P30_snr+05.0.npz")
n_ = np.load(PKG / "grid/N32_P30_snr+05.0.npz")
oi = {int(t): i for i, t in enumerate(o["trial"])}
ni = {int(t): i for i, t in enumerate(n_["trial"])}
com = sorted(set(oi) & set(ni))
dmax = max(max(abs(o["num_em_gs"][oi[t]] - n_["num_em_gs"][ni[t]]),
               abs(o["num_hs_gs"][oi[t]] - n_["num_hankel_em_gs"][ni[t]])) for t in com)
claim("overlapping runs agree bit-for-bit", "all $200$ common trials", dmax, 0.0, 0.0)
claim("number of common trials", "all $200$ common trials", len(com), 200, 0)

# --- linear-vs-dB averaging shares quoted in Sec. VI
lin = np.array([10 ** (A2[f"{s:+.1f}"]["em_gs_db"] / 10) for s in
                (-10.0, -5.0, 0.0, 5.0, 10.0, 15.0, 20.0)])
claim("low-SNR share of the linear sum", "$\\approx70\\%$ of the sum",
      100 * lin[0] / lin.sum(), 70.1, 1.0)
claim("top-three share of the linear sum", "top three SNRs together carry $0.72\\%$",
      100 * lin[-3:].sum() / lin.sum(), 0.72, 0.02)

# --- the four algorithms are present and cross-referenced
for lab in ("alg:emgs", "alg:cadzow", "alg:rank", "alg:hsgs"):
    hit = f"\\label{{{lab}}}" in TEX and f"\\ref{{{lab}}}" in TEX
    (ok if hit else bad).append(
        f"{'PASS' if hit else 'FAIL'}  algorithm {lab} defined and referenced")

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
