"""Track D stage-4 figures (PROMPT 7).

Palette: validated dataviz categorical slots. Repo conventions kept.
Primary presentation everywhere: Delta(SNR) per bin, paired per-trial median.

Run:  PYTHONPATH=. python3 scratch/trackD_stage4_plots.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = Path("results/track_d/stage4")
D = json.load(open("reports/trackD_stage4_results.json"))
A = json.load(open("reports/trackD_partA7_diagnostics.json"))
B, C = D["part_b"], D["part_c"]

plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
    "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "figure.dpi": 200, "savefig.bbox": "tight", "savefig.facecolor": "white",
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#52514e", "axes.labelcolor": "#0b0b0b",
    "text.color": "#0b0b0b", "xtick.color": "#52514e", "ytick.color": "#52514e",
})
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, RED = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#e34948")
INK, MUTED = "#0b0b0b", "#52514e"
mid = np.array([-7.5, -2.5, 2.5, 7.5, 12.5, 17.5])


def _save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}")
    plt.close(fig)
    print(f"  wrote {OUT/name}.png")


def series(key):
    b = B["contrasts"][key]["bins"]
    m = np.array([r["median_diff_db"] for r in b])
    er = np.array([[r["median_diff_db"] - r["boot_ci95_median"][0],
                    r["boot_ci95_median"][1] - r["median_diff_db"]] for r in b]).T
    return m, er


# ------------------------------------------------------- 1. the main result
fig, (ax, ax2) = plt.subplots(1, 2, figsize=(7.4, 3.1),
                              gridspec_kw={"width_ratios": [1.25, 1]})
for key, c, mk, lab in [("H1_vs_U1_recheck", MAGENTA, "o", "H1  ungated (stage 3)"),
                        ("G1_gate_scalar_vs_U1", YELLOW, "s", "G1  scalar gate"),
                        ("G2_gate_snr_vs_U1", AQUA, "D", "G2  SNR-conditioned gate")]:
    m, er = series(key)
    ax.errorbar(mid, m, yerr=er, color=c, lw=2, marker=mk, ms=5, capsize=3,
                label=lab)
ax.axhline(0, color=INK, lw=1.2)
ax.axvspan(-10, 5, color=RED, alpha=0.06)
ax.set_xlabel("SNR (dB)")
ax.set_ylabel("$\\Delta$ vs U1, paired median (dB)")
ax.legend(frameon=False, loc="upper left", fontsize=6.5)
ax.set_title("a. gating removes most of the low-SNR damage\nand keeps the "
             "high-SNR gain", loc="left", fontweight="bold", fontsize=7.5)

# The two-part criterion, each condition on its own axis.
arms = ["H1_vs_U1_recheck", "G1_gate_scalar_vs_U1", "G2_gate_snr_vs_U1"]
names = ["H1", "G1", "G2"]
cols = [MAGENTA, YELLOW, AQUA]
hi = [B["contrasts"][a]["high_snr_ge5"] for a in arms]
lo = [B["contrasts"][a]["low_snr_lt5"] for a in arms]
x = np.arange(3)
ax2.axhline(0.85, color=AQUA, ls="--", lw=1.3)
ax2.axhline(-0.05, color=RED, ls="--", lw=1.3)
ax2.axhline(0, color=INK, lw=1.0)
for i, (h, l, c) in enumerate(zip(hi, lo, cols)):
    for off, v, mk in ((-0.16, h, "o"), (0.16, l, "s")):
        ci = v["boot_ci95_median"]
        ax2.plot([x[i] + off] * 2, ci, color=c, lw=2.4, solid_capstyle="round")
        ax2.plot(x[i] + off, v["median_diff_db"], mk, color=c, ms=7,
                 markeredgecolor="white", markeredgewidth=0.8)
        ax2.annotate(f"{v['median_diff_db']:+.2f}", (x[i] + off, v["median_diff_db"]),
                     textcoords="offset points", xytext=(0, 9), ha="center",
                     fontsize=6.2, fontweight="bold", color=c)
ax2.annotate("(i) SNR$\\geq$5 bar +0.85", (2.45, 0.90), color=AQUA, fontsize=6,
             ha="right", fontweight="bold")
ax2.annotate("(ii) SNR<5 bar $-$0.05", (2.45, -0.30), color=RED, fontsize=6,
             ha="right", fontweight="bold")
ax2.set_xticks(x)
ax2.set_xticklabels([f"{n}\n○ SNR≥5   □ SNR<5" for n in names], fontsize=6.3)
ax2.set_ylabel("$\\Delta$ vs U1, paired median (dB)")
ax2.set_title("b. both gated arms PASS (i), FAIL (ii)", loc="left",
              fontweight="bold", fontsize=7.5)
_save(fig, "fig1_gated_delta_by_snr")


# ------------------------------------------- 2. what the gate actually learned
fig, (ax, ax2) = plt.subplots(1, 2, figsize=(7.4, 3.0))
bc = D["runs"]["G2_gate_snr"]["beta_curve"]
snr = np.array(bc["snr_db"])
cm = plt.cm.viridis(np.linspace(0.05, 0.95, 10))
for r in bc["per_layer"]:
    ax.plot(snr, r["beta"], color=cm[r["layer"]], lw=1.6,
            label=f"layer {r['layer']}" if r["layer"] in (0, 4, 6, 9) else None)
allb = np.array([r["beta"] for r in bc["per_layer"]])
ax.plot(snr, allb.mean(axis=0), color=INK, lw=2.6, ls="--", label="mean")
ax.axvline(5, color=RED, ls=":", lw=1.2)
ax.set_xlabel("SNR (dB)")
ax.set_ylabel("learned $\\beta_t$")
ax.set_ylim(-0.03, 1.05)
ax.legend(frameon=False, fontsize=6.2, ncol=2)
ax.set_title("a. G2: $\\beta$ rises with SNR in every layer\n(mean 0.33 → 0.98)",
             loc="left", fontweight="bold", fontsize=7.5)

g1b = np.array([r["beta"][0] for r in D["runs"]["G1_gate_scalar"]["beta_curve"]["per_layer"]])
g2m = allb.mean(axis=1)
ax2.plot(range(10), g1b, color=YELLOW, lw=2, marker="s", ms=6,
         label="G1 (constant in SNR)")
ax2.plot(range(10), g2m, color=AQUA, lw=2, marker="D", ms=6,
         label="G2 (mean over SNR)")
ax2.axhline(0.119, color=MUTED, ls=":", lw=1.3)
ax2.annotate("initialization 0.119", (9, 0.119), textcoords="offset points",
             xytext=(0, 6), ha="right", color=MUTED, fontsize=6.2,
             fontweight="bold")
ax2.set_xlabel("unrolled layer")
ax2.set_ylabel("$\\beta_t$")
ax2.set_ylim(0, 1.05)
ax2.legend(frameon=False, fontsize=6.5)
ax2.set_title("b. EARLY layers project hardest —\nI predicted the opposite",
              loc="left", fontweight="bold", fontsize=7.5)
_save(fig, "fig2_learned_beta")


# ------------------------------------------------ 3. Part C: the reframing
fig, (ax, ax2) = plt.subplots(1, 2, figsize=(7.4, 3.0))
cb = C["contrasts"]["C_H1_vs_C_U1"]["bins"]
cm_ = np.array([r["median_diff_db"] for r in cb])
cer = np.array([[r["median_diff_db"] - r["boot_ci95_median"][0],
                 r["boot_ci95_median"][1] - r["median_diff_db"]] for r in cb]).T
cmid = np.array([7.5, 12.5, 17.5])
hm, her = series("H1_vs_U1_recheck")
ax.errorbar(cmid, hm[3:], yerr=her[:, 3:], color=MAGENTA, lw=2, marker="o",
            ms=5, capsize=3, label="H1 vs U1, trained on [-10,20]")
ax.errorbar(cmid, cm_, yerr=cer, color=ORANGE, lw=2, marker="s", ms=5,
            capsize=3, label="H1 vs U1, trained on [5,20] only")
ax.axhline(0, color=INK, lw=1.2)
ax.set_xlabel("SNR (dB)")
ax.set_ylabel("Hankel advantage, paired median (dB)")
ax.legend(frameon=False, fontsize=6.3, loc="upper left")
ax.set_title("a. focus training on [5,20] and the\nHankel advantage COLLAPSES",
             loc="left", fontweight="bold", fontsize=7.5)

vals = [1.209, C["contrasts"]["C_H1_vs_C_U1"]["pooled_SAMPLING_DESIGN_DEPENDENT"]["median_diff_db"]]
ax2b = ax2
ax2b.bar([0, 1], [2.653, 1.629], color=[BLUE, MAGENTA], width=0.55)
for i, v in enumerate([2.653, 1.629]):
    ax2b.annotate(f"+{v:.2f} dB", (i, v), textcoords="offset points",
                  xytext=(0, 4), ha="center", fontsize=7.5, fontweight="bold")
ax2b.set_xticks([0, 1])
ax2b.set_xticklabels(["U1 (no prior)", "H1 (Hankel prior)"], fontsize=7)
ax2b.set_ylabel("high-SNR gain from training on [5,20] (dB)")
ax2b.set_ylim(0, 3.1)
ax2b.set_title("b. U1 gains far more from focused training.\nThe prior was "
               "substituting for it.", loc="left", fontweight="bold",
               fontsize=7.5)
_save(fig, "fig3_partC_reframing")


# -------------------------------------------- 4. A3 STE fidelity by depth
fig, ax = plt.subplots(figsize=(3.9, 2.9))
for tag, c, lab in (("low_snr_-10_0", RED, "low SNR (-10..0)"),
                    ("high_snr_10_20", BLUE, "high SNR (+10..+20)")):
    rows = [r for r in A["A3_ste_fidelity"][tag]["per_layer_group"]
            if r["group"] == "transformer"]
    ax.plot([r["layer"] for r in rows], [r["cosine"] for r in rows],
            color=c, lw=2, marker="o", ms=5, label=lab)
ax.axhline(0, color=INK, lw=1.0)
ax.axhline(1.0, color=MUTED, ls=":", lw=1.0)
ax.set_xlabel("unrolled layer  (9 = last, nearest the loss)")
ax.set_ylabel("cosine(exact grad, STE grad)")
ax.legend(frameon=False, fontsize=6.5, loc="lower right")
ax.set_title("A3: STE fidelity decays with DEPTH,\nnot with SNR", loc="left",
             fontweight="bold", fontsize=7.5)
_save(fig, "fig4_ste_fidelity")

print("\nstage-4 figures written to", OUT)
