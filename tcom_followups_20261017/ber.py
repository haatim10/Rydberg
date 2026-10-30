"""Frozen-channel, paired estimate-then-detect study with common observations.

Run from the repository root: python -m tcom_followups_20261017.ber
"""
from pathlib import Path
import argparse
import csv
import json
import time

import numpy as np
import torch

from .ber_detector import detect, draw_data, verify_detector
from .common import world_batch, load_estimators, frozen_inputs, sha, write_json


ROOT = Path("results/tcom_followups_20261017/ber")
SNRS = (0, 5, 10, 15, 20)
REGIMES = (("L2", 2), ("L4", 4), ("random3to7", None))
NAMES = ("perfect_csi", "learned_9101", "learned_9102", "learned_9103", "em100", "optshrink")
SEEDS = ("learned_9101", "learned_9102", "learned_9103")
CASES = 64
VECTORS = 256
BOOTSTRAPS = 4000


def cells():
    return [(name, paths, snr) for name, paths in REGIMES for snr in SNRS]


def freeze():
    ROOT.mkdir(parents=True, exist_ok=True)
    protocol = dict(study="paired estimate-then-detect BER", version=1,
        N=32, K=3, P=30, RSR_dB=12, SNR_dB=list(SNRS),
        path_regimes={name: paths for name, paths in REGIMES}, cases_per_cell=CASES,
        world_seed=2026101712, local_world_indices=list(range(CASES)),
        data_seed=2026101703, data_seed_rule="SeedSequence([data_seed, cell_index, local_world_index])",
        independent_data_vectors_per_world=VECTORS,
        modulation="unit-power Gray QPSK; each real/imaginary sign carries one bit",
        detector="exhaustive exact Rician maximum likelihood over 4^3 transmitted vectors",
        observations="same data magnitudes, known reference, and total complex noise variance for all estimators",
        reference="same known constant reference as the pilots, at RSR=12 dB",
        fitting="pilot observations only; no data symbols or data magnitudes used to fit channels",
        names=list(NAMES), planned_additional_method="sparse_angle on all 15 cells, from separately frozen sparse selection; evaluate on these identical saved data observations",
        neural_aggregation="mean of three separately evaluated fits; no ensemble",
        confidence="95% percentile bootstrap of independent channel worlds, stratified by condition; 4000 resamples",
        bootstrap_seed=2026101793, selection="all channel estimators frozen before test generation",
        inputs=frozen_inputs(), source_hashes={str(p): sha(p) for p in (
            Path(__file__), Path(__file__).with_name("ber_detector.py"), Path(__file__).with_name("common.py"))})
    path = ROOT / "protocol.json"
    if path.exists():
        if json.loads(path.read_text()) != protocol:
            raise RuntimeError("BER protocol changed after freeze; use a new output directory")
    else:
        write_json(path, protocol)
    return sha(path)


def save_arrays(path, **arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


@torch.no_grad()
def evaluate(digest):
    estimators = load_estimators()
    if set(estimators) != set(NAMES) - {"perfect_csi"}:
        raise ValueError("unexpected estimator set")
    for cell_index, (regime, paths, snr) in enumerate(cells()):
        label = f"{regime}_snr{snr}"
        result_path = ROOT / "cells" / f"{label}.npz"
        if result_path.exists():
            with np.load(result_path) as old:
                if str(old["protocol_sha256"]) != digest:
                    raise RuntimeError("cached BER result protocol mismatch")
            print(f"BER cached {label}", flush=True)
            continue
        obs, truth, path_counts = world_batch(2026101712, range(CASES), snr=snr, rsr=12., L=paths)
        save_arrays(ROOT / "observations" / f"{label}_pilots.npz",
            z=obs.z.numpy(), s=obs.s.numpy(), b=obs.b.numpy(), sigma2=obs.sigma2.numpy(),
            protocol_sha256=digest)
        data_z, data_bits, data_indices = [], [], []
        for case in range(CASES):
            rng = np.random.default_rng(np.random.SeedSequence([2026101703, cell_index, case]))
            z, bits, indices = draw_data(truth[case], obs.b[case, :, 0].numpy(),
                float(obs.sigma2[case]), rng, VECTORS)
            data_z.append(z); data_bits.append(bits); data_indices.append(indices)
        data_z, data_bits, data_indices = map(np.stack, (data_z, data_bits, data_indices))
        save_arrays(ROOT / "observations" / f"{label}_data.npz",
            z=data_z, b=obs.b[:, :, :1].numpy(), sigma2=obs.sigma2.numpy(), protocol_sha256=digest)
        save_arrays(ROOT / "truth" / f"{label}.npz", g=truth, paths=path_counts,
            data_bits=data_bits, data_symbol_indices=data_indices, protocol_sha256=digest)
        estimates = {"perfect_csi": truth}
        channel_seconds = np.zeros(len(NAMES))
        for name, estimator in estimators.items():
            start = time.perf_counter()
            estimates[name] = estimator(obs).numpy()
            channel_seconds[NAMES.index(name)] = time.perf_counter() - start
        errors = np.zeros((len(NAMES), CASES, 3), dtype=np.int64)
        squared_error = np.zeros((len(NAMES), CASES))
        truth_energy = (np.abs(truth) ** 2).sum((-2, -1))
        detect_seconds = np.zeros((len(NAMES), CASES))
        for mi, name in enumerate(NAMES):
            squared_error[mi] = (np.abs(estimates[name] - truth) ** 2).sum((-2, -1))
            for case in range(CASES):
                start = time.perf_counter()
                decoded = detect(data_z[case], estimates[name][case], obs.b[case, :, 0].numpy(),
                    float(obs.sigma2[case]))
                detect_seconds[mi, case] = time.perf_counter() - start
                errors[mi, case] = np.count_nonzero(decoded != data_bits[case], axis=(0, 2))
        save_arrays(result_path, names=np.array(NAMES), estimates=np.stack([estimates[n] for n in NAMES]),
            errors_per_user=errors, bits_per_user=np.full((CASES, 3), 2 * VECTORS, dtype=np.int64),
            squared_channel_error=squared_error, true_channel_energy=truth_energy,
            channel_batch_seconds=channel_seconds, detector_seconds=detect_seconds,
            snr=snr, regime=regime, protocol_sha256=digest)
        print(json.dumps(dict(study="BER", cell=label, completed=cell_index + 1, total=len(cells()),
            ber={name: float(errors[i].sum() / (CASES * 3 * 2 * VECTORS)) for i, name in enumerate(NAMES)})), flush=True)


def ci(samples):
    return np.quantile(samples, [.025, .975]).tolist()


def analyze(digest):
    rng = np.random.default_rng(2026101793)
    rows, all_ber, all_boot, all_errors = [], [], [], []
    bit_count = CASES * 3 * 2 * VECTORS
    for regime, paths, snr in cells():
        label = f"{regime}_snr{snr}"
        with np.load(ROOT / "cells" / f"{label}.npz") as a:
            errors = a["errors_per_user"].sum(-1)
            nmse = a["squared_channel_error"].sum(-1) / a["true_channel_energy"].sum()
            times = a["detector_seconds"].sum(-1)
            channel_times = a["channel_batch_seconds"]
        perworld = errors / (3 * 2 * VECTORS)
        average = perworld[[NAMES.index(name) for name in SEEDS]].mean(0)
        perworld = np.concatenate((perworld, average[None]), axis=0)
        indices = rng.integers(0, CASES, size=(BOOTSTRAPS, CASES))
        boot = perworld[:, indices].mean(-1)
        names = NAMES + ("learned_mean",)
        for mi, name in enumerate(names):
            row = dict(regime=regime, snr_db=snr, method=name, ber=float(perworld[mi].mean()),
                bootstrap95=ci(boot[mi]), independent_channel_worlds=CASES, evaluated_bits=bit_count)
            if name == "learned_mean":
                row.update(errors_per_training_seed={n: int(errors[NAMES.index(n)].sum()) for n in SEEDS},
                    ber_training_seed_range=[float(perworld[[NAMES.index(n) for n in SEEDS]].mean(-1).min()),
                                             float(perworld[[NAMES.index(n) for n in SEEDS]].mean(-1).max())])
            else:
                row.update(bit_errors=int(errors[mi].sum()), nmse_db=float(10 * np.log10(nmse[mi])) if nmse[mi] else None,
                    detector_wall_seconds=float(times[mi]), channel_batch_wall_seconds=float(channel_times[mi]))
            rows.append(row)
        all_ber.append(perworld.mean(-1)); all_boot.append(boot); all_errors.append(errors)
    all_ber, all_boot, all_errors = map(np.stack, (all_ber, all_boot, all_errors))
    overall = {}
    for mi, name in enumerate(names):
        item = dict(ber=float(all_ber[:, mi].mean()), bootstrap95=ci(all_boot[:, mi].mean(0)),
            evaluated_bits=bit_count * len(cells()), independent_channel_worlds=CASES * len(cells()))
        if name == "learned_mean":
            item["errors_per_training_seed"] = {n: int(all_errors[:, NAMES.index(n)].sum()) for n in SEEDS}
        else:
            item["bit_errors"] = int(all_errors[:, mi].sum())
        overall[name] = item
    learned = names.index("learned_mean")
    paired = {}
    for other in ("em100", "optshrink"):
        j = names.index(other)
        difference = all_ber[:, learned] - all_ber[:, j]
        replicates = all_boot[:, learned] - all_boot[:, j]
        paired[other] = dict(learned_minus_comparator_ber=float(difference.mean()), bootstrap95=ci(replicates.mean(0)),
            learned_better_cells=int((difference < 0).sum()), total_cells=len(cells()))
    output = dict(protocol_sha256=digest, overall=overall, paired_overall=paired, cells=rows,
        uncertainty="Channel-block bootstrap, conditional on the fixed trained models. Three neural seeds are not counted as independent test worlds.",
        zero_error_policy="Zero observed errors are not zero BER. One error per cell corresponds to 1/98304 bits; a zero bootstrap interval is uninformative about unseen rare channel failures.",
        timing_note="Observed wall time during potentially concurrent studies; not a hardware speed comparison. Same exhaustive detector for all methods.",
        scope="Uncoded, K=3, unit-power QPSK, fixed RSR12, N32/P30, synthetic ULA channels; no claim of coded throughput or hardware validation.")
    write_json(ROOT / "summary.json", output)
    with (ROOT / "ber.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["regime", "snr_db", "method", "ber", "ci_low", "ci_high", "bit_errors", "nmse_db"])
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in ("regime", "snr_db", "method", "ber", "bit_errors", "nmse_db")} |
                            {"ci_low": row["bootstrap95"][0], "ci_high": row["bootstrap95"][1]})
    plot(rows)
    print(json.dumps(dict(overall=overall, paired=paired)), flush=True)


def plot(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 3.8), sharey=True)
    styles = {"perfect_csi": ("Perfect CSI", "#555555", "--"),
              "em100": ("EM100", "#d55e00", "-"),
              "optshrink": ("Frozen OptShrink", "#0072b2", "-"),
              "learned_mean": ("Learned (3-fit mean)", "#009e73", "-")}
    for ax, (regime, _) in zip(axes, REGIMES):
        for name, (label, color, style) in styles.items():
            selected = [r for r in rows if r["regime"] == regime and r["method"] == name]
            values = np.array([r["ber"] for r in selected])
            intervals = np.array([r["bootstrap95"] for r in selected])
            ax.plot(SNRS, np.maximum(values, 1 / 98304), marker="o", color=color, linestyle=style, label=label)
            ax.fill_between(SNRS, np.maximum(intervals[:, 0], 1 / 98304), np.maximum(intervals[:, 1], 1 / 98304), color=color, alpha=.12)
        ax.set_title({"L2": "2 paths per user", "L4": "4 paths per user", "random3to7": "3–7 paths per user"}[regime])
        ax.set_xlabel("SNR (dB)"); ax.set_xticks(SNRS); ax.set_yscale("log"); ax.grid(alpha=.25, which="both")
    axes[0].set_ylabel("Uncoded bit error rate")
    axes[-1].legend(fontsize=8, loc="lower left")
    fig.suptitle("Same exact magnitude-domain detector and data observations for every estimator", fontsize=11)
    fig.tight_layout()
    fig.savefig(ROOT / "ber_vs_snr.png", dpi=180)
    fig.savefig(ROOT / "ber_vs_snr.pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--analyze-only", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    verification = verify_detector()
    if args.verify_only:
        print(json.dumps(verification)); return
    digest = freeze()
    write_json(ROOT / "detector_checks.json", verification)
    if not args.analyze_only:
        evaluate(digest)
    analyze(digest)


if __name__ == "__main__":
    main()
