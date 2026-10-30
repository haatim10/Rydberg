"""Presentation-only figure: distinguish measured BER from zero-error markers."""
import json

import numpy as np

from .ber import ROOT, REGIMES, SNRS


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    rows = json.loads((ROOT / "summary_with_sparse.json").read_text())["cells"]
    styles = {"perfect_csi": ("Perfect CSI", "#555555", "--"),
              "em100": ("EM100", "#d55e00", "-"),
              "optshrink": ("Frozen OptShrink", "#0072b2", "-"),
              "sparse_angle": ("Sparse angle EM", "#cc79a7", "-"),
              "learned_mean": ("Learned (3-fit mean)", "#009e73", "-")}
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 4.9), sharey=True)
    floor = 1e-6
    for ax, (regime, _) in zip(axes, REGIMES):
        for name, (label, color, style) in styles.items():
            selected = [r for r in rows if r["regime"] == regime and r["method"] == name]
            values = np.array([r["ber"] for r in selected])
            intervals = np.array([r["bootstrap95"] for r in selected])
            nonzero = values > 0
            ax.plot(SNRS, np.where(nonzero, values, np.nan), marker="o", markersize=4.5,
                    color=color, linestyle=style, label=label)
            ax.fill_between(SNRS, np.maximum(intervals[:, 0], floor), np.maximum(intervals[:, 1], floor),
                            where=nonzero, color=color, alpha=.1)
            ax.scatter(np.array(SNRS)[~nonzero], np.full((~nonzero).sum(), floor),
                       marker="v", color=color, s=34, zorder=4)
        ax.set_title({"L2": "2 paths per user", "L4": "4 paths per user", "random3to7": "3–7 paths per user"}[regime])
        ax.set_xlabel("SNR (dB)"); ax.set_xticks(SNRS)
        ax.set_yscale("log"); ax.set_ylim(.6e-6, .16)
        ax.grid(alpha=.22, which="both")
    axes[0].set_ylabel("Uncoded bit error rate")
    handles, labels = axes[-1].get_legend_handles_labels()
    handles.append(Line2D([0], [0], color="#555555", marker="v", linestyle="none"))
    labels.append("Zero observed errors (marker only)")
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=8.8, bbox_to_anchor=(.5, .055))
    fig.suptitle("Same exact magnitude-domain detector and data observations for every estimator", fontsize=11)
    fig.text(.5, .02, "Shading: 95% channel-block bootstrap intervals. Zero-error markers at 10⁻⁶ are a display choice, not a BER estimate or bound.",
             ha="center", fontsize=8.4)
    fig.tight_layout(rect=[0, .16, 1, .98])
    fig.savefig(ROOT / "ber_with_sparse_final.png", dpi=180, bbox_inches="tight")
    fig.savefig(ROOT / "ber_with_sparse_final.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
