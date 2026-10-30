"""Add the prespecified sparse estimator to the same saved BER experiment.

Reuses the pilot worlds, transmitted bits, and data magnitudes byte for byte.
No channel estimator is rerun or selected using detection results.
"""
from pathlib import Path
import argparse
import json
import time

import numpy as np
import torch

from . import ber
from .ber_detector import detect
from .common import sha, write_json

SPARSE_ROOT = Path("results/tcom_followups_20261017/sparse")
NEW_NAME = "sparse_angle"


def freeze():
    selection = SPARSE_ROOT / "frozen_selection.json"
    protocol = dict(study="prespecified sparse addition to paired BER study", version=1,
        base_protocol_sha256=sha(ber.ROOT / "protocol.json"), sparse_selection_sha256=sha(selection),
        sources={str(Path(__file__)): sha(__file__), "detector": sha(Path(ber.__file__).with_name("ber_detector.py"))},
        method=NEW_NAME, scope="all 15 original cells; no outcome-dependent omissions",
        matching="require exact same pilot observations, reference, variance, and channel truth",
        data="reuse saved magnitude observations and transmitted bits for each channel world",
        detector="same exhaustive exact Rician QPSK detector as every other estimator")
    path = ber.ROOT / "sparse_protocol.json"
    if path.exists():
        if json.loads(path.read_text()) != protocol:
            raise RuntimeError("sparse BER protocol changed after freeze")
    else:
        write_json(path, protocol)
    return sha(path)


def evaluate(digest, available_only=False):
    complete = True
    for regime, paths, snr in ber.cells():
        label = f"{regime}_snr{snr}"
        out = ber.ROOT / "sparse_cells" / f"{label}.npz"
        if out.exists():
            with np.load(out) as a:
                if str(a["protocol_sha256"]) != digest:
                    raise RuntimeError("cached sparse BER protocol mismatch")
            print(f"Sparse BER cached {label}", flush=True)
            continue
        sparse_regime = "random3-7" if paths is None else regime
        source = SPARSE_ROOT / f"test_snr{snr}_{sparse_regime}.npz"
        if not source.exists() and available_only:
            complete = False
            continue
        with np.load(source) as s, \
             np.load(ber.ROOT / "observations" / f"{label}_pilots.npz") as p, \
             np.load(ber.ROOT / "observations" / f"{label}_data.npz") as d, \
             np.load(ber.ROOT / "truth" / f"{label}.npz") as t:
            if int(s["master_seed"]) != 2026101712 or float(s["rsr_db"]) != 12 or float(s["snr_db"]) != snr:
                raise ValueError("sparse metadata mismatch")
            for key in ("z", "s", "b", "sigma2"):
                np.testing.assert_array_equal(s[key], p[key])
            np.testing.assert_array_equal(s["truth_g"], t["g"])
            estimate = s["sparse_g"]
            errors = np.zeros((ber.CASES, 3), dtype=np.int64)
            seconds = np.zeros(ber.CASES)
            for case in range(ber.CASES):
                start = time.perf_counter()
                decoded = detect(d["z"][case], estimate[case], d["b"][case], float(d["sigma2"][case]))
                seconds[case] = time.perf_counter() - start
                errors[case] = np.count_nonzero(decoded != t["data_bits"][case], axis=(0, 2))
            squared_error = (np.abs(estimate - t["g"]) ** 2).sum((-2, -1))
            energy = (np.abs(t["g"]) ** 2).sum((-2, -1))
        ber.save_arrays(out, method=NEW_NAME, errors_per_user=errors, bits_per_user=np.full((ber.CASES, 3), 2 * ber.VECTORS),
            squared_channel_error=squared_error, true_channel_energy=energy, detector_seconds=seconds,
            source_prediction_sha256=sha(source), protocol_sha256=digest)
        print(json.dumps(dict(study="sparse BER addition", cell=label,
            ber=float(errors.sum() / (ber.CASES * 3 * 2 * ber.VECTORS)))), flush=True)
    return complete


def summarize(digest):
    rng = np.random.default_rng(2026101793)
    names = ber.NAMES + (NEW_NAME, "learned_mean")
    rows, rates, samples, raw_errors = [], [], [], []
    bits_per_world = 3 * 2 * ber.VECTORS
    for regime, paths, snr in ber.cells():
        label = f"{regime}_snr{snr}"
        with np.load(ber.ROOT / "cells" / f"{label}.npz") as base, \
             np.load(ber.ROOT / "sparse_cells" / f"{label}.npz") as sparse:
            errors = np.concatenate((base["errors_per_user"].sum(-1), sparse["errors_per_user"].sum(-1)[None]), 0)
            energies = base["true_channel_energy"].sum()
            nmse = np.concatenate((base["squared_channel_error"].sum(-1), [sparse["squared_channel_error"].sum()])) / energies
        perworld = errors / bits_per_world
        neural = perworld[[names.index(n) for n in ber.SEEDS]].mean(0)
        perworld = np.concatenate((perworld, neural[None]), 0)
        indices = rng.integers(0, ber.CASES, size=(ber.BOOTSTRAPS, ber.CASES))
        bootstrap = perworld[:, indices].mean(-1)
        rates.append(perworld.mean(-1)); samples.append(bootstrap); raw_errors.append(errors)
        for mi, name in enumerate(names):
            row = dict(regime=regime, snr_db=snr, method=name, ber=float(perworld[mi].mean()),
                bootstrap95=ber.ci(bootstrap[mi]), evaluated_bits=ber.CASES * bits_per_world,
                independent_channel_worlds=ber.CASES)
            if name != "learned_mean":
                row.update(bit_errors=int(errors[mi].sum()), nmse_db=float(10 * np.log10(nmse[mi])) if nmse[mi] else None)
            else:
                row.update(errors_per_training_seed={n: int(errors[names.index(n)].sum()) for n in ber.SEEDS},
                    ber_training_seed_range=[float(perworld[[names.index(n) for n in ber.SEEDS]].mean(-1).min()),
                                             float(perworld[[names.index(n) for n in ber.SEEDS]].mean(-1).max())])
            rows.append(row)
    rates, samples, raw_errors = map(np.stack, (rates, samples, raw_errors))
    def group(indices):
        overall = {}
        for mi, name in enumerate(names):
            row = dict(ber=float(rates[indices, mi].mean()), bootstrap95=ber.ci(samples[indices, mi].mean(0)),
                evaluated_bits=len(indices) * ber.CASES * bits_per_world,
                independent_channel_worlds=len(indices) * ber.CASES)
            if name != "learned_mean":
                row["bit_errors"] = int(raw_errors[indices, mi].sum())
            else:
                row["errors_per_training_seed"] = {n: int(raw_errors[indices, names.index(n)].sum()) for n in ber.SEEDS}
            overall[name] = row
        paired = {}
        neural = names.index("learned_mean")
        for comparator in ("em100", "optshrink", NEW_NAME):
            j = names.index(comparator)
            delta = rates[indices, neural] - rates[indices, j]
            delta_boot = samples[indices, neural] - samples[indices, j]
            paired[comparator] = dict(learned_minus_comparator_ber=float(delta.mean()),
                bootstrap95=ber.ci(delta_boot.mean(0)), learned_better_cells=int((delta < 0).sum()), total_cells=len(indices))
        return dict(methods=overall, paired=paired)
    summary = dict(protocol_sha256=digest, overall=group(np.arange(len(ber.cells()))),
        by_snr={str(snr): group(np.array([i for i, c in enumerate(ber.cells()) if c[2] == snr])) for snr in ber.SNRS},
        by_regime={regime: group(np.array([i for i, c in enumerate(ber.cells()) if c[0] == regime])) for regime, _ in ber.REGIMES},
        cells=rows,
        uncertainty="95% percentile bootstrap resampling independent channel worlds within each condition. Neural fits are fixed; three fits are not tripled independent data.",
        zero_error_policy="One error per cell corresponds to 1/98304 bits. Zero observed errors are not zero BER; degenerate zero bootstrap intervals do not bound rare failures.",
        scope="Uncoded exact-ML magnitude-domain detector, N32 K3 P30 RSR12, QPSK, synthetic ULA channels. No throughput or hardware claim.")
    write_json(ber.ROOT / "summary_with_sparse.json", summary)
    draw_plot(rows)
    print(json.dumps(summary["overall"]), flush=True)


def draw_plot(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 4.3), sharey=True)
    styles = {"perfect_csi": ("Perfect CSI", "#555555", "--"),
              "em100": ("EM100", "#d55e00", "-"),
              "optshrink": ("Frozen OptShrink", "#0072b2", "-"),
              NEW_NAME: ("Sparse angle EM", "#cc79a7", "-"),
              "learned_mean": ("Learned (3-fit mean)", "#009e73", "-")}
    floor = 1 / (ber.CASES * 3 * 2 * ber.VECTORS)
    zero_present = False
    for ax, (regime, _) in zip(axes, ber.REGIMES):
        for name, (label, color, style) in styles.items():
            selected = [r for r in rows if r["regime"] == regime and r["method"] == name]
            values = np.array([r["ber"] for r in selected])
            intervals = np.array([r["bootstrap95"] for r in selected])
            nonzero = values > 0
            # Lines end at the last observed positive BER; zero-error points
            # use a separate marker at the empirical one-bit resolution.
            ax.plot(ber.SNRS, np.where(nonzero, values, np.nan), marker="o", color=color, linestyle=style, label=label)
            ax.fill_between(ber.SNRS, np.maximum(intervals[:, 0], floor), np.maximum(intervals[:, 1], floor),
                            where=nonzero, color=color, alpha=.1)
            if (~nonzero).any():
                zero_present = True
                ax.scatter(np.array(ber.SNRS)[~nonzero], np.full((~nonzero).sum(), floor),
                           marker="v", color=color, s=34, zorder=4)
        ax.set_title({"L2": "2 paths per user", "L4": "4 paths per user", "random3to7": "3–7 paths per user"}[regime])
        ax.set_xlabel("SNR (dB)"); ax.set_xticks(ber.SNRS); ax.set_yscale("log"); ax.grid(alpha=.22, which="both")
    axes[0].set_ylabel("Uncoded bit error rate")
    handles, labels = axes[-1].get_legend_handles_labels()
    if zero_present:
        handles.append(Line2D([0], [0], color="#555555", marker="v", linestyle="none"))
        labels.append("Zero observed errors (display floor)")
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=8, bbox_to_anchor=(.5, -.015))
    fig.suptitle("Same exact magnitude-domain detector and data observations for every estimator", fontsize=11)
    fig.tight_layout(rect=[0, .09, 1, 1])
    fig.savefig(ber.ROOT / "ber_with_sparse.png", dpi=180, bbox_inches="tight")
    fig.savefig(ber.ROOT / "ber_with_sparse.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--analyze-only", action="store_true")
    parser.add_argument("--available-only", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    digest = freeze()
    if not args.analyze_only and not evaluate(digest, args.available_only):
        print("Sparse BER: processed all currently available cells; remaining predictions are still being generated.", flush=True)
        return
    summarize(digest)


if __name__ == "__main__":
    main()
