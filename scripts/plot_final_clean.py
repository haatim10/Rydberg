"""Publication-style regeneration of every final figure.

Reads only the saved aggregate stores; runs no Monte Carlo. Style follows
IEEE-paper conventions: thin lines, small open markers, subtle grid, thin
spines, compact legend, no title (the caption carries it).

Uncertainty is deliberately NOT drawn. Confidence intervals remain in the
aggregate CSV/JSON and in each figure's README.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
TA = REPO / "results/track_a"
TB = Path("/home/user/rydberg-trackb/results/track_b")
FINAL = REPO / "results/final_figures"

# ---------------------------------------------------------------- style ----
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "font.size": 9,
    "axes.labelsize": 9.5,
    "axes.titlesize": 9.5,
    "legend.fontsize": 8,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.minor.width": 0.4,
    "ytick.minor.width": 0.4,
    "lines.linewidth": 1.3,
    "lines.markersize": 4.5,
    "lines.markeredgewidth": 0.9,
    "grid.linewidth": 0.4,
    "grid.alpha": 0.35,
    "legend.frameon": True,
    "legend.framealpha": 0.9,
    "legend.edgecolor": "0.7",
    "legend.borderpad": 0.4,
    "legend.labelspacing": 0.3,
    "legend.handlelength": 2.0,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03,
})

FIGSIZE = (3.6, 2.9)          # single-column IEEE
FIGSIZE_W = (5.4, 3.4)        # wider, for busier panels

# Cui's colour/marker convention, kept for cross-comparability.
STYLE = {
    "biased_gs":     dict(label="Biased GS",     c="#D9A404", ls="-",  m="o"),
    "em_gs":         dict(label="EM-GS",         c="#C4451C", ls="-",  m="s"),
    "exhaustive_ls": dict(label="Exh. search (LS)", c="#D9A404", ls="--", m="o"),
    "exhaustive_ml": dict(label="Exh. search (ML)", c="#C4451C", ls="--", m="s"),
    "genie_zf":      dict(label="ZF w/ known phase", c="#6A3D9A", ls="--", m="*"),
    "cui_crlb":      dict(label="CRLB",          c="#2E7D32", ls="-.", m=None),
}
ORDER = ["biased_gs", "em_gs", "exhaustive_ls", "exhaustive_ml",
         "genie_zf", "cui_crlb"]


from matplotlib.ticker import MultipleLocator


def style_axes(ax, logy=False, xstep=None):
    ax.grid(True, which="major", linestyle="-", linewidth=0.4, alpha=0.3)
    if logy:
        ax.grid(True, which="minor", linestyle="-", linewidth=0.25, alpha=0.15)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction="in", length=3)
    if xstep:
        ax.xaxis.set_major_locator(MultipleLocator(xstep))


def draw(ax, rows, xkey, ykey, order=ORDER, drop=(), ms=4.5):
    by = {}
    for r in rows:
        by.setdefault(r["algorithm"], []).append(r)
    for alg in order:
        if alg in drop or alg not in by:
            continue
        pts = sorted(by[alg], key=lambda r: r[xkey])
        xs = [p[xkey] for p in pts]
        ys = [p[ykey] for p in pts]
        if ykey == "ber":                      # log axis: drop zero-error pts
            xy = [(x, y) for x, y in zip(xs, ys) if y > 0]
            if not xy:
                continue
            xs, ys = zip(*xy)
        st = STYLE[alg]
        ax.plot(xs, ys, color=st["c"], linestyle=st["ls"], marker=st["m"],
                markersize=ms, markerfacecolor="none",
                markeredgewidth=0.9, label=st["label"])


def save(fig, stem, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    FINAL.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        p = out_dir / f"{stem}.{ext}"
        fig.savefig(p)
        shutil.copy2(p, FINAL / f"{stem}.{ext}")
    plt.close(fig)
    print(f"  wrote {stem}.png/.pdf")


def agg_of(path):
    d = json.loads(Path(path).read_text())
    return d["aggregate"] if isinstance(d, dict) and "aggregate" in d else d


# --------------------------------------------------------------- Track A ---
def fig5():
    rows = agg_of(TA / "fig5_final/aggregate.json")
    fig, ax = plt.subplots(figsize=FIGSIZE)
    draw(ax, rows, "snr_db", "nmse_db")
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Detection NMSE (dB)")
    ax.set_xlim(-5, 12)
    ax.set_ylim(-25, 10)
    style_axes(ax, xstep=2)
    ax.legend(loc="upper right")
    save(fig, "fig5_clean", TA / "fig5_final")


def fig6():
    rows = agg_of(TA / "fig6/aggregate.json")
    fig, ax = plt.subplots(figsize=FIGSIZE)
    draw(ax, rows, "rsr_db", "nmse_db")
    ax.set_xlabel("RSR (dB)")
    ax.set_ylabel("Detection NMSE (dB)")
    ax.set_xlim(0, 25)
    ax.set_ylim(-16, 0)
    style_axes(ax, xstep=5)
    ax.legend(loc="upper right")
    save(fig, "fig6_clean", TA / "fig6")


def ber_fig(store, stem, xkey, xlabel, xlim, ylim, drop=(), figsize=FIGSIZE,
            loc="lower left", xstep=None):
    rows = agg_of(TA / store / "aggregate.json")
    fig, ax = plt.subplots(figsize=figsize)
    draw(ax, rows, xkey, "ber", drop=drop)
    ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("BER")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    style_axes(ax, logy=True, xstep=xstep)
    ax.legend(loc=loc)
    save(fig, stem, TA / store)


# --------------------------------------------------------------- Track B ---
TB_STYLE = {
    "biased_gs": dict(label="Biased GS (exact model)", c="#D9A404", ls="-", m="o"),
    "em_gs": dict(label="EM-GS (exact model)", c="#C4451C", ls="-", m="s"),
}


def tb_draw(ax, rows, xkey):
    for alg, st in TB_STYLE.items():
        pts = sorted((r for r in rows if r["algorithm"] == alg),
                     key=lambda r: r["x"])
        if not pts:
            continue
        ax.plot([p["x"] for p in pts], [p["nmse_db"] for p in pts],
                color=st["c"], linestyle=st["ls"], marker=st["m"],
                markersize=4.5, markerfacecolor="none", markeredgewidth=0.9,
                label=st["label"])


def track_b():
    f = TB / "baseline_preliminary.json"
    if not f.exists():
        print("  (Track-B baseline not found; skipped)")
        return
    d = json.loads(f.read_text())
    rows = d["rows"]

    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.9), sharey=True)
    for ax, P in zip(axes, (10, 30)):
        tb_draw(ax, [r for r in rows if r["sweep"] == f"B1 (P={P})"], "x")
        ax.set_xlabel("SNR (dB)")
        style_axes(ax, xstep=5)
        ax.set_title(f"$P$ = {P}", fontsize=9)
    axes[0].set_ylabel(r"Channel NMSE$_G$ (dB)")
    axes[0].legend(loc="lower left")
    fig.tight_layout()
    save(fig, "b1_clean", TB)

    sweep = next(r["sweep"] for r in rows if r["sweep"].startswith("B2"))
    fig, ax = plt.subplots(figsize=FIGSIZE)
    tb_draw(ax, [r for r in rows if r["sweep"] == sweep], "x")
    ax.set_xlabel("Pilot length $P$")
    ax.set_ylabel(r"Channel NMSE$_G$ (dB)")
    ax.axvline(6, color="0.6", linestyle=":", linewidth=0.7)
    ax.annotate("$P=2K$", xy=(6, ax.get_ylim()[1]), xytext=(3, -10),
                textcoords="offset points", fontsize=7.5, color="0.4", va="top")
    style_axes(ax, xstep=10)
    ax.legend(loc="upper right")
    save(fig, "b2_clean", TB)


if __name__ == "__main__":
    print("Track A:")
    fig5()
    fig6()
    ber_fig("fig7a", "fig7a_clean", "snr_db", "SNR (dB)", (-5, 12), (1e-5, 1e0),
            figsize=FIGSIZE_W, loc="upper right", xstep=2)
    ber_fig("fig7b", "fig7b_clean", "snr_db", "SNR (dB)", (-5, 12), (1e-5, 1e0),
            loc="lower left", xstep=5)
    ber_fig("fig8", "fig8_clean", "rsr_db", "RSR (dB)", (0, 25), (1e-4, 1e0),
            drop=("genie_zf",), figsize=FIGSIZE_W, xstep=5)
    ber_fig("fig8_16qam", "fig8_16qam_diagnostic", "rsr_db", "RSR (dB)",
            (0, 25), (1e-2, 1e0), drop=("genie_zf",), xstep=5)
    print("Track B:")
    track_b()
    print(f"\nall copies in {FINAL}")
