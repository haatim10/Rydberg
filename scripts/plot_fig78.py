"""Publication figures for Cui Fig. 7(a)/7(b)/8, in the paper's own style.

Usage: python3 scripts/plot_fig78.py <fig7a|fig7b|fig8>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent

# Cui's MATLAB colour/marker convention (Fig. 7(a) legend).
STYLE = {
    "cm_zf":         dict(label="CM-ZF", color="#0072BD", ls="-", marker="D"),
    "biased_gs":     dict(label="Biased GS", color="#EDB120", ls="-", marker="o"),
    "em_gs":         dict(label="EM-GS", color="#D95319", ls="-", marker="s"),
    "exhaustive_ls": dict(label="Exhaustive search (LS)", color="#EDB120",
                          ls="--", marker="o"),
    "exhaustive_ml": dict(label="Exhaustive search (ML)", color="#D95319",
                          ls="--", marker="s"),
    "genie_zf":      dict(label="ZF w/ known phase", color="#7E2F8E",
                          ls="--", marker="*"),
}
# Cui's plotting order. CM-ZF is absent everywhere (unspecified in the paper).
ORDER = ["biased_gs", "em_gs", "exhaustive_ls", "exhaustive_ml", "genie_zf"]

SPEC = {
    "fig7a": dict(sweep="snr_db", xlabel="SNR [dB]", xlim=(-5, 12),
                  ylim=(1e-5, 1e0),
                  title="BER vs SNR — 4-QAM, $N\\times K=36\\times3$, RSR = 12 dB",
                  # Cui Fig. 8 omits ZF; Fig. 7 shows it
                  drop=()),
    "fig7b": dict(sweep="snr_db", xlabel="SNR [dB]", xlim=(-5, 12),
                  ylim=(1e-5, 1e0),
                  title="BER vs SNR — 16-QAM, $N\\times K=100\\times6$, RSR = 12 dB",
                  drop=()),
    "fig8": dict(sweep="rsr_db", xlabel="RSR [dB]", xlim=(0, 25),
                 ylim=(1e-4, 1e0),
                 title="BER vs RSR — 4-QAM, $N\\times K=36\\times3$, SNR = 3 dB",
                 # Cui's Fig. 8 plots no ZF-known-phase curve
                 drop=("genie_zf",)),
}


def main() -> None:
    name = sys.argv[1]
    cfg = SPEC[name]
    out = REPO / "results" / "track_a" / name
    agg = json.loads((out / "aggregate.json").read_text())["aggregate"]

    by = {}
    for r in agg:
        by.setdefault(r["algorithm"], []).append(r)

    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    for alg in ORDER:
        if alg in cfg["drop"] or alg not in by:
            continue
        rows = sorted(by[alg], key=lambda r: r[cfg["sweep"]])
        # log axis: a zero-error point has no finite position, so drop it and
        # note it in the README rather than pretending it is on the floor
        xs = [r[cfg["sweep"]] for r in rows if r["ber"] > 0]
        ys = [r["ber"] for r in rows if r["ber"] > 0]
        lo = [r["ber"] - r["ber_ci95_low"] for r in rows if r["ber"] > 0]
        hi = [r["ber_ci95_high"] - r["ber"] for r in rows if r["ber"] > 0]
        st = STYLE[alg]
        ax.errorbar(xs, ys, yerr=[lo, hi], color=st["color"], linestyle=st["ls"],
                    marker=st["marker"], markersize=6.5, markerfacecolor="none",
                    markeredgewidth=1.4, linewidth=1.6, label=st["label"],
                    capsize=2.0, elinewidth=0.8)

    ax.set_yscale("log")
    ax.set_xlabel(cfg["xlabel"])
    ax.set_ylabel("BER")
    ax.set_xlim(*cfg["xlim"])
    ax.set_ylim(*cfg["ylim"])
    ax.grid(True, which="both", linestyle=":", linewidth=0.6, alpha=0.65)
    ax.set_title(cfg["title"], fontsize=10)
    ax.legend(loc="lower left", fontsize=8.5, framealpha=0.95)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{name}_ber.{ext}", dpi=300)
    print(f"wrote {out/(name+'_ber.png')}")


if __name__ == "__main__":
    main()
