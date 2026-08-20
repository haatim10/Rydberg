"""Regenerate Fig. 5 and Fig. 6 in Cui's published legend/marker style.

Reads the already-computed aggregate stores; computes nothing new.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path("/home/user/Rydberg")
FIG5 = REPO / "results/track_a/fig5_final"
FIG6 = REPO / "results/track_a/fig6"

# Cui's MATLAB default colour order, matched to the published legend:
#   CM-ZF  blue diamond | GS yellow circle | EM-GS orange square
#   CRLB   green dashed | ZF w/ known phase purple dashed asterisk
STYLE = {
    "cm_zf":     dict(label="CM-ZF",            color="#0072BD", ls="-",  marker="D"),
    "biased_gs": dict(label="GS",               color="#EDB120", ls="-",  marker="o"),
    "em_gs":     dict(label="EM-GS",            color="#D95319", ls="-",  marker="s"),
    "cui_crlb":  dict(label="CRLB",             color="#77AC30", ls="--", marker=None),
    "genie_zf":  dict(label="ZF w/ known phase", color="#7E2F8E", ls="--", marker="*"),
}
# Cui's legend order; cm_zf is intentionally absent (not implemented -- see
# baselines.py: "Cui's channel-magnitude ZF is not specified in the available
# source; do not invent an approximation").
ORDER = ["biased_gs", "em_gs", "cui_crlb", "genie_zf"]


def load(path: Path) -> list[dict]:
    d = json.loads((path / "aggregate.json").read_text())
    return d["aggregate"] if isinstance(d, dict) else d


def series(agg: list[dict], alg: str, xkey: str):
    rows = sorted((r for r in agg if r["algorithm"] == alg), key=lambda r: r[xkey])
    return ([r[xkey] for r in rows],
            [r["nmse_db"] for r in rows],
            [r.get("se_db") or 0.0 for r in rows])


def draw(agg, xkey, xlabel, xlim, ylim, title, out_stem, *, errorbars=False):
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    offscale = []
    for alg in ORDER:
        x, y, se = series(agg, alg, xkey)
        if not x:
            continue
        st = STYLE[alg]
        kw = dict(color=st["color"], linestyle=st["ls"], linewidth=1.6,
                  marker=st["marker"], markersize=6.5, markerfacecolor="none",
                  markeredgewidth=1.4, label=st["label"], clip_on=True)
        if errorbars:
            ax.errorbar(x, y, yerr=se, capsize=2.5, elinewidth=0.9, **kw)
        else:
            ax.plot(x, y, **kw)
        if max(y) < ylim[0] or min(y) > ylim[1]:
            offscale.append((st["label"], sum(y) / len(y)))

    ax.set_xlabel(xlabel)
    ax.set_ylabel("NMSE (dB)")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.grid(True, which="both", linestyle=":", linewidth=0.6, alpha=0.65)
    ax.set_title(title)
    ax.legend(loc="best", framealpha=0.95, fontsize=9)

    for label, mean_db in offscale:
        ax.annotate(
            f"{label} off-scale at ~{mean_db:.1f} dB",
            xy=(0.5, 0.02), xycoords="axes fraction", ha="center", va="bottom",
            fontsize=8.5, color=STYLE["genie_zf"]["color"],
        )

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{out_stem}.{ext}", dpi=300)
    plt.close(fig)
    return offscale


def main() -> None:
    a5, a6 = load(FIG5), load(FIG6)

    o5 = draw(a5, "snr_db", "SNR (dB)", (-5, 12), (-25, 10),
              "Detection NMSE vs SNR (RSR = 12 dB)", FIG5 / "fig5_cui_style")
    draw(a5, "snr_db", "SNR (dB)", (-5, 12), (-25, 10),
         "Detection NMSE vs SNR (RSR = 12 dB)", FIG5 / "fig5_cui_style_errorbars",
         errorbars=True)

    o6 = draw(a6, "rsr_db", "RSR (dB)", (0, 25), (-12, 4),
              "Detection NMSE vs RSR (SNR = 3 dB)", FIG6 / "fig6_cui_style")
    draw(a6, "rsr_db", "RSR (dB)", (0, 25), (-12, 4),
         "Detection NMSE vs RSR (SNR = 3 dB)", FIG6 / "fig6_cui_style_errorbars",
         errorbars=True)

    # Full-range companion so no curve is hidden by the requested Fig. 6 floor.
    draw(a6, "rsr_db", "RSR (dB)", (0, 25), (-16, 4),
         "Detection NMSE vs RSR (SNR = 3 dB) -- full range",
         FIG6 / "fig6_cui_style_fullrange")

    print("fig5 off-scale:", o5)
    print("fig6 off-scale:", o6)


if __name__ == "__main__":
    main()
