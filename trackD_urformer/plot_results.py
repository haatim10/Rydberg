"""Track D figures - SCAFFOLD ONLY, NOT RUN in the build phase.

Follows the repository's plotting conventions exactly (as recorded in the audit,
item 19): matplotlib Agg, rcParams set once, figsize (3.6,2.7) single /
(7.0,2.7) two-panel, BOTH .png and .pdf emitted, outputs under
results/track_d/ with a co-located README.md.

Every legend entry names the initializer. Never plot two estimators with
different initializers without labeling both (PROMPT 2 sec. 8).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

FIG = Path("results") / "track_d"

plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
    "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "figure.dpi": 200, "savefig.bbox": "tight", "savefig.facecolor": "white",
    "axes.grid": True, "grid.alpha": 0.3,
})

# Colour-blind-safe, consistent across every Track D figure.
STYLE = {
    "gs":            {"color": "#4C72B0", "marker": "o", "ls": "-"},
    "em_gs":         {"color": "#DD8452", "marker": "s", "ls": "-"},
    "linearised_ls": {"color": "#8172B3", "marker": "^", "ls": "--"},
    "urformer":      {"color": "#C44E52", "marker": "D", "ls": "-"},
}
INIT_LS = {"random": "-", "spectral": "--", "linearized_ls": ":"}


def _save(fig, name: str) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"{name}.{ext}")
    plt.close(fig)


def _label(est: str, init: str) -> str:
    """Initializer is ALWAYS in the label - never an unlabeled comparison."""
    pretty = {"gs": "GS", "em_gs": "EM-GS", "urformer": "URformer",
              "linearised_ls": "Linearised LS"}[est]
    return pretty if init == "closed_form" else f"{pretty} ({init})"


def plot_vs_x(agg: list[dict], xkey: str, xlabel: str, name: str, title: str
              ) -> None:
    """Generic NMSE-vs-x figure over aggregated rows."""
    fig, ax = plt.subplots(figsize=(3.6, 2.7))
    seen = {}
    for r in agg:
        seen.setdefault((r["estimator"], r["initializer"]), []).append(r)
    for (est, init), rs in sorted(seen.items()):
        rs = sorted(rs, key=lambda r: r[xkey])
        st = dict(STYLE[est])
        st["ls"] = INIT_LS.get(init, st["ls"])
        ax.plot([r[xkey] for r in rs], [r["nmse_db"] for r in rs],
                label=_label(est, init), lw=1.4, ms=3.5, **st)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("channel NMSE (dB)")
    ax.set_title(title)
    ax.legend(loc="best", framealpha=0.9)
    _save(fig, name)


def plot_d1(agg): plot_vs_x(agg, "snr_db", "SNR (dB)", "D1_nmse_vs_snr",
                            "D1 - NMSE vs SNR")


def plot_d2(agg): plot_vs_x(agg, "P", "pilots $P$", "D2_nmse_vs_pilots",
                            "D2 - NMSE vs pilot count")


def plot_d3(agg): plot_vs_x(agg, "N", "array size $N$", "D3_nmse_vs_array_size",
                            "D3 - NMSE vs array size")


if __name__ == "__main__":
    raise SystemExit(
        "plot_results.py is scaffolding: it is not run in the build phase "
        "(PROMPT 2 sec. 11). Call plot_d1/d2/d3 on aggregated rows after the "
        "D-experiments have been approved and run."
    )
