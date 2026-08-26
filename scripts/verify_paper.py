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


def fact(desc, computed, expected, tol=0.005):
    """A data-integrity check with no printed counterpart in the manuscript."""
    agree = abs(computed - expected) <= tol
    (ok if agree else bad).append(
        f"{'PASS' if agree else 'FAIL'}  {desc}: computed={computed:.4f} "
        f"expected={expected}")


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
WIN_LIT = "win rate $33.5$, $74.0$, $95.3\\%$"
ACT_LIT = "constraint active in $57.5$, $92.7$, $99.6\\%$ of trials"
for N, w, a in (("8", 33.5, 57.5), ("16", 74.0, 92.7), ("32", 95.3, 99.6)):
    claim(f"win rate N={N}", WIN_LIT, 100 * np.mean(f("win_rate", N)), w, 0.05)
    claim(f"active frac N={N}", ACT_LIT, 100 * np.mean(f("active_frac", N)), a, 0.05)

# --- N=32,P=30 sweep
g32 = f("gain_dB", "32", "30")
claim("N=32,P=30 gain at 5 dB", "$2.27$--$3.04$~dB", g32[2], 2.27)
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
fact("GS crossings", n_gs, 18, 0)
fact("EM-GS crossings", n_em, 19, 0)
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

# --- controlled path count.  Fig. 3 and Sec. VII-B are sourced from the
# package's 300-trial sweep (trackB_hankel_emgs/results/pathcount), not the
# older 400-trial B7 sweep; B7 is retained below as an independent replication.
PC = REPO / "trackB_hankel_emgs" / "results" / "pathcount"
pc = []
for p_ in sorted(glob.glob(str(PC / "L*.npz"))):
    d_ = np.load(p_)
    e_, h_, dn_ = d_["num_em_gs"], d_["num_hankel_em_gs"], d_["denom"]
    pc.append((int(Path(p_).stem[1:]),
               10 * np.log10(e_.sum() / h_.sum()),
               10 * np.log10(e_.sum() / dn_.sum()),
               (h_ < e_).mean(), d_["L_hat"].mean(), d_["active"].mean(), dn_.size))


def boot_ci(f):
    d_ = np.load(f)
    e_, h_ = d_["num_em_gs"], d_["num_hankel_em_gs"]
    rng = np.random.default_rng(987654321)
    n_ = e_.size
    v = []
    for _ in range(2000):
        i = rng.integers(0, n_, n_)
        v.append(10 * np.log10(e_[i].sum() / h_[i].sum()))
    return np.percentile(v, [2.5, 97.5])


for L, v, *_ in pc:
    ok.append(f"INFO  path count L={L}: gain {v:+.3f} dB")
claim("path-count sequence",
      "7.04,\\;3.56,\\;1.79,\\;1.04,\\;0.58,\\;0.27,\\;0.05,\\;-0.12",
      1.0 if all(pc[i][1] > pc[i + 1][1] for i in range(len(pc) - 1)) else 0.0, 1.0, 0)
for i, want in enumerate((7.04, 3.56, 1.79, 1.04, 0.58, 0.27, 0.05, -0.12)):
    fact(f"path-count gain at L={pc[i][0]}", pc[i][1], want)
claim("path-count win rate L=2", "from $100\\%$", 100 * pc[0][3], 100.0, 0.05)
claim("path-count win rate L=16", "to $45.0\\%$", 100 * pc[7][3], 45.0, 0.05)
empc = [t[2] for t in pc]
claim("path-count EM-GS flat", "EM-GS flat to $0.19$~dB", max(empc) - min(empc), 0.191)
claim("path-count mean Lhat at L=16", "only $11.63$ at $L=16$", pc[7][4], 11.63)
claim("path-count active at L=16", "active in $89.3\\%$", 100 * pc[7][5], 89.3, 0.05)
lo14, hi14 = boot_ci(str(PC / "L14.npz"))
lo16, hi16 = boot_ci(str(PC / "L16.npz"))
claim("CI at L=14", "CI $[-0.05,+0.14]$", lo14, -0.05, 0.005)
fact("CI at L=14 upper", hi14, 0.14)
claim("CI at L=16", "CI $[-0.21,-0.04]$", lo16, -0.21, 0.005)
fact("CI at L=16 upper", hi16, -0.04)
fact("null reached one grid point before the ceiling", float(lo14 < 0 < hi14), 1.0, 0)
fact("L=16 significantly negative", float(hi16 < 0), 1.0, 0)
claim("path-count trials/point", "$300$ trials per point", pc[0][6], 300, 0)

# --- trial counts
sweeps = sum(np.load(p)["denom"].size
             for sub in ("b3", "b4", "b6") for p in glob.glob(str(R / sub / "*.npz")))
pc_trials = sum(t[6] for t in pc)
claim("SNR + pilot sweep trials", "$2.5\\times10^{4}$ trials for the SNR and",
      sweeps, 24800, 200)
claim("path-count trials in budget", "$2.4\\times10^{3}$ on the", pc_trials, 2400, 0)
claim("tiered trial counts", "$200$--$1200$, tiered",
      float(min(int(x["trials"]) for x in rows) >= 200
            and max(int(x["trials"]) for x in rows) == 1200), 1.0, 0)

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
fact("CRLB trials", CC["n_trials"], 400, 0)

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
# The paper's Fig. 3 uses the package's 300-trial sweep; B7 ran an independent
# 400-trial sweep of the same design. Neither is printed as a literal here --
# this is a replication check between two runs, so it uses fact(), not claim().
b7 = {int(Path(f).stem[1:]): np.load(f) for f in glob.glob(str(R / "b7/L*.npz"))}
b7g = {L: 10 * np.log10(d["num_em_gs"].sum() / d["num_hs_gs"].sum())
       for L, d in b7.items()}
pkg_g = {int(L): v["gain_db"] for L, v in C2.items()}
dmax_c = max(abs(b7g[L] - pkg_g[L]) for L in pkg_g)
ok.append(f"INFO  path-count sweeps: B7(400 trials) vs package(300 trials), "
          f"max |diff| = {dmax_c:.3f} dB over 8 values of L")
fact("path-count sweeps agree within MC error", dmax_c, 0.0, 0.35)
fact("both sweeps end non-positive at L=r_max",
     float(max(b7g[16], pkg_g[16]) < 0), 1.0, 0)
claim("pkg checks", "thirteen automated checks", pk["checks_passed"], 13, 0)
claim("pkg checks total", "thirteen automated checks", pk["checks_total"], 13, 0)
pkg_trials = sum(v["trials"] for v in B2.values())
claim("pkg grid trials", "$8.4\\times10^{3}$ on the", pkg_trials, 8400, 0)
claim("total trials in abstract", "$3.6\\times10^{4}$ paired trials",
      sweeps + pkg_trials + pc_trials, 35600, 500)

# --- the two runs agree bit-for-bit where they overlap
o = np.load(R / "b3/N32_P30_snr+05.0.npz")
n_ = np.load(PKG / "grid/N32_P30_snr+05.0.npz")
oi = {int(t): i for i, t in enumerate(o["trial"])}
ni = {int(t): i for i, t in enumerate(n_["trial"])}
com = sorted(set(oi) & set(ni))
dmax = max(max(abs(o["num_em_gs"][oi[t]] - n_["num_em_gs"][ni[t]]),
               abs(o["num_hs_gs"][oi[t]] - n_["num_hankel_em_gs"][ni[t]])) for t in com)
fact("overlapping runs agree bit-for-bit", dmax, 0.0, 0.0)
fact("number of common trials", len(com), 200, 0)

# --- linear-vs-dB averaging shares quoted in Sec. VI
lin = np.array([10 ** (A2[f"{s:+.1f}"]["em_gs_db"] / 10) for s in
                (-10.0, -5.0, 0.0, 5.0, 10.0, 15.0, 20.0)])
claim("low-SNR share of the linear sum", "$\\approx70\\%$ of the sum",
      100 * lin[0] / lin.sum(), 70.1, 1.0)
claim("top-three share of the linear sum", "against $0.72\\%$",
      100 * lin[-3:].sum() / lin.sum(), 0.72, 0.02)

# --- the four algorithms are present and cross-referenced
for lab in ("alg:emgs", "alg:cadzow", "alg:hsgs"):
    hit = f"\\label{{{lab}}}" in TEX and f"\\ref{{{lab}}}" in TEX
    (ok if hit else bad).append(
        f"{'PASS' if hit else 'FAIL'}  algorithm {lab} defined and referenced")

# The rank-selection box was merged into the text when the paper was cut to six
# pages; the procedure itself must still be fully specified there.
for lit in ("held-out", "\\nu=0.3", "T_{\\mathrm{sel}}=20", "1,\\dots,r_{\\max}"):
    hit = lit in TEX
    (ok if hit else bad).append(
        f"{'PASS' if hit else 'FAIL'}  rank selection described in text: {lit!r}")

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
