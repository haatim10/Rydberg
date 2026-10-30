"""Read frozen results and generate manuscript tables, figure and small evidence pack.

This does not run estimators, train models, tune settings or modify old results.
"""
from pathlib import Path
import hashlib
import json
import shutil

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
OUT = HERE / "evidence" / "training_updates"
STUDIES = ("rician_full_comparison", "rician_mixed_screen", "rician_selector_screen", "rician_pilot_screen")
LABEL = {"holdout_best": "Original (selected)", "allpilot_best": "All-pilot (selected)",
         "holdout_final": "Original (final)", "allpilot_final": "All-pilot (final)",
         "mixed_best": "Mixed (selected)", "mixed_final": "Mixed (final)",
         "rank_loop": "OptShrink, block rank", "opt_selected": "OptShrink, offline",
         "selector": "Selected rule"}
METHODS = ("holdout_best", "allpilot_best", "opt_selected", "rank_loop")


def read(path):
    return json.loads(path.read_text())


def row(*items):
    return " & ".join(map(str, items)) + r"\\" + "\n"


def table(caption, label, columns, header, rows, wide=False):
    env = "table*" if wide else "table"
    return (f"\\begin{{{env}}}[!t]\n\\caption{{{caption}}}\n"
            f"\\label{{{label}}}\\centering\\footnotesize\n"
            r"\setlength{\tabcolsep}{3pt}" + "\n"
            f"\\begin{{tabular}}{{@{{}}{columns}@{{}}}}\\toprule\n" + row(*header)
            + "\\midrule\n" + "".join(rows) + "\\bottomrule\n\\end{tabular}\n"
            f"\\end{{{env}}}\n")


def verify_arrays(summary, name):
    a = np.load(REPO / "results" / name / "all_errors.npz", allow_pickle=False)
    error, power, names = a["squared_error"], a["channel_power"], a["names"].tolist()
    if name == "rician_selector_screen":
        for oi, offset in enumerate(a["noise_offsets_db"]):
            for mi, method in enumerate(names):
                e = error[:, :126, oi, mi]
                p = power[:, :126]
                macro = np.mean(10 * np.log10(e.sum(axis=(0, 2)) / p.sum(axis=(0, 2))))
                severe = np.mean(e / p > 1)
                target = summary["offsets"][str(float(offset))]["primary"][method]
                np.testing.assert_allclose([macro, severe], [target["macro_nmse_db"], target["severe_rate"]], atol=1e-12)
        assert not a["allpilot_mask"].any()
        return
    for key, value in summary["primary"].items():
        ii = [names.index(key)] if key in names else [names.index(key + "_" + str(s)) for s in (9101, 9102, 9103)]
        macro, severe = [], []
        for mi in ii:
            e, p = error[:, :126, mi], power[:, :126]
            macro.append(np.mean(10 * np.log10(e.sum(axis=(0, 2)) / p.sum(axis=(0, 2)))))
            severe.append(np.mean(e / p > 1))
        expected = value if isinstance(value, (float, int)) else value["macro_nmse_db"]
        expected_severe = summary["severe_rate"][key] if isinstance(value, (float, int)) else value["severe_rate"]
        np.testing.assert_allclose([np.mean(macro), np.mean(severe)], [expected, expected_severe], atol=1e-12)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    summaries = {name: read(REPO / "results" / name / "summary.json") for name in STUDIES}
    full, mixed, selector, small = (summaries[x] for x in STUDIES)
    phase = read(REPO / "results/rician_pilot_screen/phase/summary.json")
    hashes = {}
    for name, summary in summaries.items():
        base = REPO / "results" / name
        assert read(base / "status.json")["state"] == "complete", name
        if name != "rician_pilot_screen":
            assert summary["audit"]["passed"]
            assert summary["independent_headline_recalculation"] == "passed"
            verify_arrays(summary, name)
        # Summaries retain per-cell values, not raw predictions or channel arrays.
        compact = json.loads(json.dumps(summary))
        if "audit" in compact:
            compact["audit"].pop("hashes", None)
        (OUT / (name + "_summary.json")).write_text(json.dumps(compact, indent=2) + "\n")
        shutil.copyfile(REPO / name / "PROTOCOL.txt", OUT / (name + "_PROTOCOL.txt"))
        for p in (base / "summary.json", base / "manifest.json", REPO / name / "PROTOCOL.txt"):
            hashes[str(p.relative_to(REPO))] = hashlib.sha256(p.read_bytes()).hexdigest()
    phase.pop("artifact_hashes", None)
    (OUT / "phase_summary.json").write_text(json.dumps(phase, indent=2) + "\n")
    shutil.copyfile(REPO / "results/rician_pilot_screen/phase/protocol.json", OUT / "phase_protocol.json")
    selection = read(REPO / "results/rician_selector_screen/frozen_selection.json")
    selection["development"]["candidate_scores"] = [
        {k: v for k, v in d.items() if k != "per_condition_nll"}
        for d in selection["development"]["candidate_scores"]]
    (OUT / "selector_selection.json").write_text(json.dumps(selection, indent=2) + "\n")
    (OUT / "provenance.json").write_text(json.dumps({"input_sha256": hashes,
        "independent_table_recalculation": "passed for all three large/new screens",
        "scope": "Saved primary errors and severe frequencies recomputed; no new training or test tuning."}, indent=2) + "\n")

    rows = [row(LABEL[m].replace(" (selected)", ""), f'{full["primary"][m]:.3f}',
                f'{100 * full["severe_rate"][m]:.2f}', f'{full["timing"][m]["mean_ms"]:.2f}') for m in METHODS]
    (HERE / "training_main_table.tex").write_text(table(
        "Matched training-recipe comparison: 8,064 fresh primary worlds. Neural results average three separate fits, not an ensemble. Error is the equal-cell dB NMSE; severe means per-case NMSE $>1$. One-thread CPU time includes initialization and any rank search/refit.",
        "tab:training", "lrrr", ["Method", "NMSE (dB)", r"Severe (\%)", "Time (ms)"], rows))

    parts = []
    rows = []
    for m in full["primary"]:
        rows.append(row(LABEL[m], f'{full["primary"][m]:.3f}', f'{full["pooled_db"][m]:.3f}',
                        f'{100 * full["severe_rate"][m]:.2f}',
                        f'{full["cells"]["weak_reference"][m]["nmse_db"]:.3f}',
                        f'{full["cells"]["nearly_noiseless"][m]["nmse_db"]:.3f}'))
    parts.append(table("Full-budget study: selected and final-epoch sensitivity, plus separate stress cells. Primary scores use 8,064 worlds; each stress cell has 64 worlds. Weak reference: SNR 5/RSR 0 dB. Nearly noiseless: SNR 60/RSR 12 dB. Stress scores do not enter the primary score.",
        "tab:training-full", "lrrrrr", ["Method", "Macro (dB)", "Pooled (dB)", r"Severe (\%)", "Weak reference", "Nearly noiseless"], rows, wide=True))
    rows = []
    for m in ("holdout_best", "rank_loop"):
        c = full["contrasts"][m]
        lo, hi = c["simultaneous95_macro_ci"]
        rows.append(row(LABEL[m], f'{c["gain_db"]:+.3f}', f'[{lo:+.3f}, {hi:+.3f}]',
                        f'{c["improved_cells"]}/126', f'{c["worst_cell_deterioration_db"]:.3f}'))
    parts.append(table(r"All-pilot gains: positive favors all-pilot training. Two predeclared contrasts, Bonferroni-adjusted conditional 95\% intervals. Worst observed harm is not a confidence bound.",
        "tab:training-contrasts", "lrrrr", ["Comparator", "Gain (dB)", r"95\% interval", "Cell wins", "Worst harm"], rows, wide=True))
    rows = []
    for seed in (9101, 9102, 9103):
        h = full["audit"]["histories"][str(seed)]
        rows.append(row(seed, f'{full["seed_primary"]["holdout_best"][str(seed)]:.3f}',
                        f'{full["seed_primary"]["allpilot_best"][str(seed)]:.3f}', h["best_epoch"],
                        f'{h["training_seconds"] / 60:.2f}'))
    parts.append(table("Full-budget training records. Seeds share one training dataset. All-pilot training times include validation and concurrent-worker contention; they are not isolated-device benchmarks.",
        "tab:training-seeds", "rrrrr", ["Seed", "Original", "All-pilot", "Best epoch", "Minutes"], rows))
    rows = []
    for m in ("holdout_best", "allpilot_best", "mixed_best", "holdout_final", "allpilot_final", "mixed_final", "opt_selected", "rank_loop"):
        v = mixed["primary"][m]
        rows.append(row(LABEL[m], f'{v["macro_nmse_db"]:.3f}', f'{v["mean_case_nmse"]:.4f}',
                        f'{100*v["severe_rate"]:.2f}', f'{mixed["timing"][m]["mean_ms"]:.2f}'))
    parts.append(table("Separate one-seed mixed-training screen: 4,032 primary worlds, not the full-budget test bank. Mean-case NMSE is linear. Selected-checkpoint comparisons are primary; final checkpoints are sensitivity checks.",
        "tab:training-mixed", "lrrrr", ["Method", "Macro (dB)", "Case mean", r"Severe (\%)", "Time (ms)"], rows, wide=True))
    rows = []
    for offset in (0., -3., 3.):
        v = selector["offsets"][str(offset)]
        for m in ("holdout_best", "allpilot_best", "selector"):
            p = v["primary"][m]
            rows.append(row(f'{offset:+.0f}', LABEL[m], f'{p["macro_nmse_db"]:.3f}',
                            f'{p["mean_case_nmse"]:.4f}', f'{100*p["severe_rate"]:.2f}',
                            f'{v["timing"][m]["mean_ms"]:.2f}'))
    parts.append(table("Separate one-seed selector/noise-input screen: the same 4,032 primary worlds at all three supplied-variance offsets. Physical observations and actual noise are unchanged. The selected rule always runs the original model.",
        "tab:training-noise", "rlrrrr", ["Offset (dB)", "Method", "Macro (dB)", "Case mean", r"Severe (\%)", "Time (ms)"], rows, wide=True))
    rows = []
    for c in phase["cells"]:
        v = c["methods"]["adaptive_seed_average"]
        rows.append(row(f'{c["spec"]["snr"]:.0f}', f'{c["spec"]["rsr"]:.0f}',
                        *[f'{x:.3f}' for x in v["nmse_ratio_db"]],
                        f'{100*v["row_phase_removable_squared_error_fraction"]:.1f}'))
    parts.append(table("Separate phase diagnostic: 64 worlds per cell, averaged across three original fitted models. Both aligned scores use the true channel and are oracle-only, not receiver performance. Last column is the fraction of squared error removed by row-wise alignment.",
        "tab:training-phase", "rrrrrr", ["SNR", "RSR", "Raw (dB)", "Global oracle", "Row oracle", r"Removed (\%)"], rows, wide=True))
    (HERE / "training_supplement_tables.tex").write_text("\n".join(parts))

    specs = read(REPO / "results/rician_full_comparison/manifest.json")["config"]["specs"][:126]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "pdf.fonttype": 42})
    fig, ax = plt.subplots(1, 2, figsize=(6.8, 2.65), layout="constrained")
    colors = ("#156082", "#D55E00", "#868686", "#009E73")
    for m, color, marker in zip(METHODS, colors, ("o", "s", "v", "^")):
        snrs = list(range(-10, 21, 5))
        nmse = [np.mean([full["cells"][c["id"]][m]["nmse_db"] for c in specs if c["snr"] == s]) for s in snrs]
        rates = [100 * np.mean([full["cells"][c["id"]][m]["severe_rate"] for c in specs if c["snr"] == s]) for s in snrs]
        label = LABEL[m].replace(" (selected)", "")
        ax[0].plot(snrs, nmse, color=color, marker=marker, ms=3, lw=1.2, label=label)
        ax[1].plot(snrs, rates, color=color, marker=marker, ms=3, lw=1.2)
    for a in ax:
        a.set_xlabel("SNR (dB)")
        a.set_xticks(range(-10, 21, 5))
        a.grid(alpha=.25)
        a.spines[["top", "right"]].set_visible(False)
    ax[0].set_ylabel("Mean cell NMSE (dB)")
    ax[0].legend(frameon=False, fontsize=7, loc="lower left")
    ax[1].set_ylabel("Cases with NMSE > 1 (%)")
    ax[1].set_ylim(bottom=-1)
    (HERE / "figures").mkdir(exist_ok=True)
    fig.savefig(HERE / "figures/training_tradeoff.pdf", bbox_inches="tight")
    fig.savefig(HERE / "figures/training_tradeoff.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("Training materials regenerated; independent headline and severe-rate checks passed.")


if __name__ == "__main__":
    main()
