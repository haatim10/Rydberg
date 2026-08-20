"""Publication figures for the frozen Track-B 400-trial baseline.

Reads results/track_b/baseline_preliminary.json only; computes nothing.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "track_b"
DATA = json.loads((OUT / "baseline_preliminary.json").read_text())

STYLE = {
    "biased_gs": dict(label="Biased GS (Cui)", color="#EDB120", ls="-", marker="o"),
    "em_gs": dict(label="EM-GS (Cui)", color="#D95319", ls="-", marker="s"),
}
ORDER = ["biased_gs", "em_gs"]


def rows_for(sweep):
    return [r for r in DATA["rows"] if r["sweep"] == sweep]


def draw(ax, sweep, xlabel, title):
    for alg in ORDER:
        rs = sorted((r for r in rows_for(sweep) if r["algorithm"] == alg),
                    key=lambda r: r["x"])
        x = [r["x"] for r in rs]
        y = [r["nmse_db"] for r in rs]
        lo = [r["nmse_db"] - r["ci_low"] for r in rs]
        hi = [r["ci_high"] - r["nmse_db"] for r in rs]
        st = STYLE[alg]
        ax.errorbar(x, y, yerr=[lo, hi], color=st["color"], linestyle=st["ls"],
                    marker=st["marker"], markersize=6.5, markerfacecolor="none",
                    markeredgewidth=1.4, linewidth=1.6, label=st["label"],
                    capsize=3.0, elinewidth=1.0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"NMSE$_G$ (dB)")
    ax.grid(True, which="both", linestyle=":", linewidth=0.6, alpha=0.65)
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=9, framealpha=0.95)


sub = (f"$N$={DATA['N']}, $K$={DATA['K']}, $L_k\\sim\\mathcal{{U}}\\{{3..7\\}}$, "
       f"RSR={DATA['rsr_db']:g} dB, $t_0$={DATA['max_iter']}, "
       f"{DATA['n_trials']} trials/point")

# --- B1: both panels side by side ---
fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.3), sharey=True)
for ax, P in zip(axes, (10, 30)):
    draw(ax, f"B1 (P={P})", "SNR (dB)", f"$P$ = {P}")
axes[1].set_ylabel("")
fig.suptitle("B1 — channel NMSE vs SNR   (" + sub + ")", fontsize=10.5)
fig.tight_layout(rect=(0, 0, 1, 0.95))
for ext in ("png", "pdf"):
    fig.savefig(OUT / f"b1_nmse_vs_snr.{ext}", dpi=300)
plt.close(fig)

# --- B1 overlay, single axes ---
fig, ax = plt.subplots(figsize=(6.4, 4.8))
for P, dash in ((10, (4, 2)), (30, None)):
    for alg in ORDER:
        rs = sorted((r for r in rows_for(f"B1 (P={P})") if r["algorithm"] == alg),
                    key=lambda r: r["x"])
        st = STYLE[alg]
        ax.errorbar([r["x"] for r in rs], [r["nmse_db"] for r in rs],
                    yerr=[[r["nmse_db"] - r["ci_low"] for r in rs],
                          [r["ci_high"] - r["nmse_db"] for r in rs]],
                    color=st["color"], marker=st["marker"], markersize=6,
                    markerfacecolor="none", markeredgewidth=1.3, linewidth=1.6,
                    dashes=dash, capsize=2.5, elinewidth=0.9,
                    label=f"{st['label']}, $P$={P}")
ax.set_xlabel("SNR (dB)")
ax.set_ylabel(r"NMSE$_G$ (dB)")
ax.grid(True, which="both", linestyle=":", linewidth=0.6, alpha=0.65)
ax.set_title("B1 — channel NMSE vs SNR\n" + sub, fontsize=9.5)
ax.legend(fontsize=8.5, framealpha=0.95)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(OUT / f"b1_nmse_vs_snr_overlay.{ext}", dpi=300)
plt.close(fig)

# --- B2 ---
sweep_b2 = next(r["sweep"] for r in DATA["rows"] if r["sweep"].startswith("B2"))
fig, ax = plt.subplots(figsize=(6.4, 4.8))
draw(ax, sweep_b2, "pilot length $P$",
     "B2 — channel NMSE vs pilot length\n" + sub.replace(
         f"$t_0$={DATA['max_iter']}, ", f"$t_0$={DATA['max_iter']}, SNR=5 dB, "))
ax.axvline(2 * DATA["K"], color="0.45", linestyle="--", linewidth=1.0)
ax.annotate(r"$P=2K$", xy=(2 * DATA["K"], ax.get_ylim()[1]),
            xytext=(3, -12), textcoords="offset points", fontsize=8.5,
            color="0.35", va="top")
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(OUT / f"b2_nmse_vs_pilots.{ext}", dpi=300)
plt.close(fig)

print("wrote:")
for f in sorted(OUT.glob("b[12]_*.p*")):
    print("  ", f)
