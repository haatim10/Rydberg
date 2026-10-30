"""Reporting-only redraw from saved scores; no scoring or estimator execution."""
import hashlib
import json
import os
from pathlib import Path

os.environ["MPLCONFIGDIR"] = "/private/tmp/tcom_start_budget_reporting_matplotlib"
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parents[1] / "results/tcom_start_budget_20261020"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    summary_path = OUT / "raw_start_summary.json"
    data = json.loads(summary_path.read_text())
    source_hash = sha(summary_path)
    old_paths = [OUT / f"raw_start_nmse.{ext}" for ext in ("png", "svg")]
    originals = {p.name: sha(p) for p in old_paths}
    styles = (
        ("adaptive_mean", "#0969da", "Adaptive learned", "-"),
        ("global_mean", "#bf3989", "Global learned", "--"),
        ("classical", "#26965c", "Classical structured", "-."),
        ("spectral", "#777777", "Spectral start", ":"),
        ("em10", "#d99322", "EM10 start", "--"),
    )
    fig, axes = plt.subplots(3, 3, figsize=(13.8, 10.6), sharex=True, sharey=True)
    for i, regime in enumerate(("L2", "L4", "random3to7")):
        for j, ref in enumerate((5, 12, 20)):
            ax = axes[i, j]
            rows = sorted(
                (r for r in data["cells"] if r["regime"] == regime and r["rsr"] == ref),
                key=lambda r: r["snr"],
            )
            assert len(rows) == 17
            for method, color, label, linestyle in styles:
                ax.plot(
                    [r["snr"] for r in rows],
                    [r["methods"][method]["nmse_db"] for r in rows],
                    color=color, label=label, linestyle=linestyle,
                    marker=".", linewidth=1.5,
                )
            ax.set_title(f"{regime}, reference {ref} dB", fontsize=11)
            ax.set_xticks((0, 10, 20, 30, 40, 50, 60))
            ax.grid(alpha=0.2)
            if i == 2:
                ax.set_xlabel("Nominal SNR (dB)")
            if j == 0:
                ax.set_ylabel("Channel NMSE (dB; lower is better)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.015), ncol=3, fontsize=9)
    fig.suptitle(
        "Starting estimates before common continuation\n"
        "Saved fresh paired scores; all five start families at every condition",
        fontsize=13, y=0.97,
    )
    fig.subplots_adjust(left=0.08, right=0.985, bottom=0.13, top=0.895, hspace=0.23, wspace=0.13)
    new_paths = [OUT / f"raw_start_nmse_readable.{ext}" for ext in ("png", "svg")]
    for path in new_paths:
        fig.savefig(path, dpi=180)
    plt.close(fig)
    assert sha(summary_path) == source_hash
    assert all(sha(OUT / name) == digest for name, digest in originals.items())
    record = {
        "reporting_only": True,
        "purpose": "Correct figure margins and visible axis labels; reuse every saved score unchanged.",
        "script_sha256": sha(__file__),
        "raw_start_summary_sha256": source_hash,
        "selection_sha256": data["selection_sha256"],
        "preserved_original_figures": originals,
        "new_figures": {p.name: sha(p) for p in new_paths},
        "no_scoring_or_estimator_execution": True,
    }
    (OUT / "raw_start_nmse_readable_provenance.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record))


if __name__ == "__main__":
    main()
