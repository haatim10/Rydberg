"""Paired BER for runtime-selected estimators, using only frozen predictions.

No estimator is trained, timed, or selected by this module. Every saved method
receives precisely the same new data magnitudes and exact Rician ML detector.
"""
from pathlib import Path
import argparse
import csv
import json
import os
import time

for variable in ("VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[variable] = "1"

import numpy as np
import torch

from tcom_followups_20261017.ber_detector import detect, draw_data, verify_detector
from tcom_followups_20261017.common import sha, write_json, effective_trial_id


REPO = Path(__file__).resolve().parents[1]
SOURCE = REPO / "results/tcom_runtime_bypass_20261017/runtime"
ROOT = REPO / "results/tcom_runtime_bypass_20261017/runtime_ber"
SELECTION = SOURCE / "validation_selection.json"
SNRS = (0, 5, 10, 15, 20)
REGIMES = ("L2", "L4", "random3to7")
CASES = 64
VECTORS = 256
BOOTSTRAPS = 4000
NEURAL = ("learned_9101", "learned_9102", "learned_9103")
EXPECTED = NEURAL + ("sparse_matched", "sparse_10ms", "sparse_20ms", "sparse_50ms",
                     "em100", "optshrink", "fast_full")


def cells():
    return [(regime, snr) for regime in REGIMES for snr in SNRS]


def save_arrays(path, **arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def freeze():
    detector_path = REPO / "tcom_followups_20261017/ber_detector.py"
    selection = json.loads(SELECTION.read_text())
    protocol = dict(study="runtime-matched sparse versus learned estimate-then-detect BER", version=1,
        N=32, K=3, P=30, RSR_dB=12, SNR_dB=list(SNRS), path_regimes=list(REGIMES),
        independent_worlds_per_cell=CASES, pilot_world_seed=2026101742,
        data_seed=2026101743, data_seed_rule="SeedSequence([2026101743, cell_index, local_world_index])",
        independent_QPSK_vectors_per_world=VECTORS,
        modulation="unit-power Gray QPSK; two bits per user symbol",
        expected_source_methods=list(EXPECTED), additional_method="perfect_csi",
        nominal_caps_ms=selection["caps_ms"],
        selection_sha256=sha(SELECTION), source_hashes={str(Path(__file__).relative_to(REPO)): sha(__file__),
                                                     str(detector_path.relative_to(REPO)): sha(detector_path)},
        observation_policy="All saved channel estimates use identical pilot data. All detectors use identical new data magnitudes, known reference, variance, and 64 joint QPSK hypotheses.",
        inference="No unknown data symbol or true channel is used to fit or select any channel estimate; true CSI is an explicitly labeled reference detector only.",
        aggregation="Three frozen neural fits evaluated separately; learned_mean averages BER, never channels or predictions.",
        confidence="95% paired coupled-block percentile bootstrap; 4000 resamples with seed2026101794. Within each path regime the same local-world indices are resampled across every SNR and method, preserving possible dependence from shared random path-count draws. Regimes are resampled independently. Training fits remain fixed.",
        timing="This module must remain idle during isolated latency calibration; no speed claim is based on its observed wall time.",
        sparse_budget_note="Nominal budget labels refer to the separately frozen runtime study; sparse_matched uses its learned-latency matching rule. These are not hard real-time guarantees.")
    path = ROOT / "protocol.json"
    if path.exists():
        if json.loads(path.read_text()) != protocol:
            raise RuntimeError("runtime BER protocol changed after freeze")
    else:
        write_json(path, protocol)
    return sha(path)


def evaluate(digest, available_only=False):
    complete = True
    for cell_index, (regime, snr) in enumerate(cells()):
        label = f"{regime}_snr{snr}"
        source = SOURCE / "test" / f"{label}.npz"
        out = ROOT / "cells" / f"{label}.npz"
        if out.exists():
            with np.load(out) as previous:
                if str(previous["protocol_sha256"]) != digest or str(previous["source_sha256"]) != sha(source):
                    raise RuntimeError("cached prediction/protocol mismatch")
            print(f"Runtime BER cached {label}", flush=True)
            continue
        if not source.exists() and available_only:
            complete = False
            continue
        with np.load(source) as a:
            if int(a["seed"]) != 2026101742 or float(a["snr"]) != snr or float(a["rsr"]) != 12 or str(a["regime"]) != regime:
                raise ValueError("runtime source metadata mismatch")
            if str(a["selection_sha256"]) != sha(SELECTION):
                raise ValueError("source predictions do not match the frozen runtime selection")
            source_names = tuple(str(n) for n in a["method_names"])
            if set(source_names) != set(EXPECTED) or len(source_names) != len(EXPECTED):
                raise ValueError(f"unexpected saved source methods: {source_names}")
            names = ("perfect_csi",) + source_names
            truth = a["truth_g"]
            if truth.shape != (CASES, 32, 3):
                raise ValueError("unexpected channel dimensions")
            estimates = np.concatenate((truth[None], a["estimates"]), axis=0)
            if estimates.shape != (len(names), CASES, 32, 3) or not np.isfinite(estimates).all():
                raise ValueError("invalid saved channel estimates")
            references = a["b"][:, :, :1]
            variances = a["sigma2"]
            identifiers = a["effective_trial_ids"]
            paths_per_user = {"L2": 2, "L4": 4, "random3to7": None}[regime]
            np.testing.assert_array_equal(identifiers, [effective_trial_id(i, paths_per_user) for i in range(CASES)])
            if a["z"].shape != (CASES, 32, 30) or a["s"].shape != (CASES, 3, 30):
                raise ValueError("source pilot observations are not the planned N32/K3/P30 experiment")
            data_z, data_bits, symbol_indices = [], [], []
            for case in range(CASES):
                rng = np.random.default_rng(np.random.SeedSequence([2026101743, cell_index, case]))
                z, bits, indices = draw_data(truth[case], references[case], float(variances[case]), rng, VECTORS)
                data_z.append(z); data_bits.append(bits); symbol_indices.append(indices)
            data_z, data_bits, symbol_indices = map(np.stack, (data_z, data_bits, symbol_indices))
            save_arrays(ROOT / "data" / f"{label}.npz", z=data_z, b=references,
                sigma2=variances, effective_trial_ids=identifiers, protocol_sha256=digest)
            save_arrays(ROOT / "truth" / f"{label}.npz", data_bits=data_bits,
                symbol_indices=symbol_indices, effective_trial_ids=identifiers, protocol_sha256=digest)
        errors = np.zeros((len(names), CASES, 3), dtype=np.int64)
        seconds = np.zeros((len(names), CASES))
        for mi, name in enumerate(names):
            for case in range(CASES):
                start = time.perf_counter()
                decoded = detect(data_z[case], estimates[mi, case], references[case], float(variances[case]))
                seconds[mi, case] = time.perf_counter() - start
                errors[mi, case] = np.count_nonzero(decoded != data_bits[case], axis=(0, 2))
        error_energy = (np.abs(estimates - truth[None]) ** 2).sum((-2, -1))
        truth_energy = (np.abs(truth) ** 2).sum((-2, -1))
        save_arrays(out, method_names=np.array(names), errors_per_user=errors,
            bits_per_user=np.full((CASES, 3), 2 * VECTORS, dtype=np.int64),
            detector_wall_seconds=seconds, squared_channel_error=error_energy,
            true_channel_energy=truth_energy, effective_trial_ids=identifiers,
            source_sha256=sha(source), protocol_sha256=digest)
        print(json.dumps(dict(study="runtime BER", cell=label,
            ber={name: float(errors[mi].sum() / (CASES * 3 * 2 * VECTORS)) for mi, name in enumerate(names)})), flush=True)
    return complete


def ci(replicates):
    return np.quantile(replicates, [.025, .975]).tolist()


def analyze(digest):
    rng = np.random.default_rng(2026101794)
    # The generator shares random path counts for a local world across SNRs,
    # although its channel/pilot/noise streams use distinct SNR keys. Keep that
    # dependence in pooled intervals by resampling the complete SNR cluster.
    coupled_indices = {regime: rng.integers(0, CASES, size=(BOOTSTRAPS, CASES)) for regime in REGIMES}
    rows, rates, samples, raw_errors = [], [], [], []
    bits_per_world = 3 * 2 * VECTORS
    method_order = None
    for regime, snr in cells():
        with np.load(ROOT / "cells" / f"{regime}_snr{snr}.npz") as a:
            names = tuple(str(n) for n in a["method_names"])
            if method_order is None:
                method_order = names
            if method_order != names:
                raise ValueError("inconsistent saved method order across conditions")
            errors = a["errors_per_user"].sum(-1)
            nmse = a["squared_channel_error"].sum(-1) / a["true_channel_energy"].sum()
        perworld = errors / bits_per_world
        neural_average = perworld[[names.index(n) for n in NEURAL]].mean(0)
        perworld = np.concatenate((perworld, neural_average[None]), axis=0)
        names = names + ("learned_mean",)
        indices = coupled_indices[regime]
        boot = perworld[:, indices].mean(-1)
        rates.append(perworld.mean(-1)); samples.append(boot); raw_errors.append(errors)
        for mi, name in enumerate(names):
            row = dict(regime=regime, snr_db=snr, method=name,
                ber=float(perworld[mi].mean()), bootstrap95=ci(boot[mi]),
                evaluated_bits=CASES * bits_per_world, independent_channel_worlds=CASES)
            if name != "learned_mean":
                row.update(bit_errors=int(errors[mi].sum()),
                    nmse_db=float(10 * np.log10(nmse[mi])) if nmse[mi] else None)
            else:
                seed_rates = perworld[[names.index(n) for n in NEURAL]].mean(-1)
                row.update(errors_per_training_seed={n: int(errors[names.index(n)].sum()) for n in NEURAL},
                    ber_training_seed_range=[float(seed_rates.min()), float(seed_rates.max())])
            rows.append(row)
    rates, samples, raw_errors = map(np.stack, (rates, samples, raw_errors))
    def aggregate(indices):
        methods = {}
        clusters = len({cells()[i][0] for i in indices}) * CASES
        for mi, name in enumerate(names):
            row = dict(ber=float(rates[indices, mi].mean()), bootstrap95=ci(samples[indices, mi].mean(0)),
                channel_realizations=len(indices) * CASES, independent_channel_clusters=clusters,
                evaluated_bits=len(indices) * CASES * bits_per_world)
            if name != "learned_mean":
                row["bit_errors"] = int(raw_errors[indices, mi].sum())
            else:
                row["errors_per_training_seed"] = {n: int(raw_errors[indices, names.index(n)].sum()) for n in NEURAL}
            methods[name] = row
        paired = {}
        neural = names.index("learned_mean")
        for comparator in EXPECTED:
            if comparator in NEURAL:
                continue
            mi = names.index(comparator)
            differences = rates[indices, neural] - rates[indices, mi]
            replicates = samples[indices, neural] - samples[indices, mi]
            paired[comparator] = dict(learned_minus_comparator_ber=float(differences.mean()),
                bootstrap95=ci(replicates.mean(0)), learned_better_cells=int((differences < 0).sum()),
                tied_cells=int((differences == 0).sum()), total_cells=len(indices))
        return dict(methods=methods, paired=paired)
    summary = dict(protocol_sha256=digest, nominal_caps_ms=json.loads(SELECTION.read_text())["caps_ms"],
        overall=aggregate(np.arange(len(cells()))),
        by_snr={str(snr): aggregate(np.array([i for i, c in enumerate(cells()) if c[1] == snr])) for snr in SNRS},
        by_regime={regime: aggregate(np.array([i for i, c in enumerate(cells()) if c[0] == regime])) for regime in REGIMES},
        cells=rows,
        uncertainty="Paired coupled-block bootstrap conditional on fixed models and selections. The same local-world indices are resampled across SNRs within each regime to preserve possible shared path-count dependence; regimes are independent. Overall there are 960 channel realizations grouped into 192 resampling clusters. Three neural fits are not three independent test banks.",
        zero_error_policy="Zero observed errors do not establish zero BER or a rare-event bound. One error per cell is 1/98304 bits, and 64 independent channels cannot characterize very rare channel failures.",
        scope="Uncoded vector-ML QPSK detection, N32/K3/P30, RSR12, synthetic ULA channels. No coded-throughput or hardware claim.")
    write_json(ROOT / "summary.json", summary)
    with (ROOT / "ber.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["regime", "snr_db", "method", "ber", "ci_low", "ci_high", "bit_errors", "nmse_db"])
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in ("regime", "snr_db", "method", "ber", "bit_errors", "nmse_db")} |
                dict(ci_low=row["bootstrap95"][0], ci_high=row["bootstrap95"][1]))
    plot(rows)
    print(json.dumps(summary["overall"]), flush=True)


def plot(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    matched_cap = json.loads(SELECTION.read_text())["caps_ms"]["sparse_matched"]
    styles = {"perfect_csi": ("Perfect CSI", "#555555", "--"),
              "learned_mean": ("Learned (3-fit mean)", "#009e73", "-"),
              "sparse_matched": (f"Sparse, matched ({matched_cap:.2f} ms)", "#d55e00", "-"),
              "sparse_10ms": ("Sparse, 10 ms budget", "#e69f00", "-"),
              "sparse_20ms": ("Sparse, 20 ms budget", "#0072b2", "-"),
              "sparse_50ms": ("Sparse, 50 ms budget", "#cc79a7", "-")}
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 5.1), sharey=True)
    floor = 1e-6
    ceiling = max(.16, min(.55, 1.3 * max(row["ber"] for row in rows)))
    for ax, regime in zip(axes, REGIMES):
        for name, (label, color, style) in styles.items():
            selected = [row for row in rows if row["regime"] == regime and row["method"] == name]
            values = np.array([row["ber"] for row in selected])
            intervals = np.array([row["bootstrap95"] for row in selected])
            positive = values > 0
            ax.plot(SNRS, np.where(positive, values, np.nan), marker="o", markersize=4.2,
                linestyle=style, color=color, label=label)
            if name in ("learned_mean", "sparse_matched"):
                ax.fill_between(SNRS, np.maximum(intervals[:, 0], floor), np.maximum(intervals[:, 1], floor),
                    where=positive, color=color, alpha=.11)
            ax.scatter(np.array(SNRS)[~positive], np.full((~positive).sum(), floor), marker="v", color=color, s=30)
        ax.set_title({"L2": "2 paths per user", "L4": "4 paths per user", "random3to7": "3–7 paths per user"}[regime])
        ax.set_xlabel("SNR (dB)"); ax.set_xticks(SNRS); ax.set_yscale("log")
        ax.set_ylim(.6e-6, ceiling); ax.grid(alpha=.22, which="both")
    axes[0].set_ylabel("Uncoded bit error rate")
    handles, labels = axes[-1].get_legend_handles_labels()
    handles.append(Line2D([0], [0], color="#555555", marker="v", linestyle="none"))
    labels.append("Zero observed errors (marker only)")
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=8.2, bbox_to_anchor=(.5, .035))
    fig.suptitle("BER after channel estimation with matched and increasing sparse-computation budgets", fontsize=11)
    fig.text(.5, .012, "Shading: 95% coupled-block intervals for learned/matched. Same data and detector. Zero-error markers at 10⁻⁶ are not estimates or bounds.", ha="center", fontsize=8.2)
    fig.tight_layout(rect=[0, .19, 1, .98])
    fig.savefig(ROOT / "runtime_ber.png", dpi=180, bbox_inches="tight")
    fig.savefig(ROOT / "runtime_ber.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--available-only", action="store_true")
    parser.add_argument("--analyze-only", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    checks = verify_detector()
    digest = freeze()
    write_json(ROOT / "detector_checks.json", checks)
    if not args.analyze_only and not evaluate(digest, args.available_only):
        print("Processed available runtime prediction cells; remaining cells have not been generated.", flush=True)
        return
    analyze(digest)


if __name__ == "__main__":
    main()
