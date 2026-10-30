"""Prespecified supplied-noise sensitivity; fixed observations and policies.

Actual SNR is held fixed while only the variance reported to the receiver is
changed. Correct-variance predictions are reused from the primary experiment.
"""
import os
for variable in ("VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[variable] = "1"

from pathlib import Path
import argparse
import json
import numpy as np
import torch

from tcom_followups_20261017.common import sha, write_json
from . import bypass_study as study

ROOT = study.OUT.parent / "bypass_noise_sensitivity"
SNRS = (20., 30., 60.)
OFFSETS = (-3., 0., 3.)
GROUPS = dict(original_mean=[0, 1, 2], off_mean=[3, 4, 5],
              fast_bypass_mean=[6, 7, 8], em100=[9], em100_bypass_mean=[10, 11, 12])


def freeze():
    policy, policy_sha = study.check_selection()
    declared = json.loads((study.OUT / "protocol.json").read_text())["sensitivity"]
    expected = dict(actual_snrs=list(SNRS), rsr=12., regime="random3to7", supplied_variance_offsets_db=list(OFFSETS))
    for name, value in expected.items():
        if declared[name] != value:
            raise ValueError("sensitivity conditions differ from parent protocol")
    record = dict(version=1, name="Fixed-policy supplied-variance sensitivity", **expected,
        test_seed=2026101762, test_anchors=64,
        selection_sha256=policy_sha, parent_protocol_sha256=policy["protocol_sha256"],
        policies=policy["policies"], source_sha256=sha(__file__),
        receiver_variance="actual variance multiplied by10^(offset_dB/10); measured magnitudes and actual channel are unchanged",
        zero_offset="Reuse all predictions from the primary experiment without fitting again",
        methods=list(study.NAMES), group_indices=GROUPS,
        uncertainty="3000 paired bootstrap replicates of64 whole anchors, with identical indices across SNR, supplied-variance offsets, methods, and fits; seed2026101773. Pointwise95 intervals; conditional on frozen policies and weights.",
        scope="Diagnostic only, with no policy adaptation or new deployment claim")
    path = ROOT / "protocol.json"
    if path.exists():
        if json.loads(path.read_text()) != record:
            raise RuntimeError("sensitivity protocol changed after freeze")
    else:
        write_json(path, record)
    return policy, sha(path)


@torch.no_grad()
def evaluate(policy, digest):
    source = study.OUT / "test/anchors/random3to7_rsr12.npz"
    anchor = study.load(source)
    policy_sha = sha(study.OUT / "policy.json")
    if int(anchor["seed"]) != 2026101762 or str(anchor["selection_sha256"]) != policy_sha:
        raise ValueError("wrong fresh sensitivity anchor or selection")
    if len(anchor["g"]) != 64:
        raise ValueError("expected64 independent anchors")
    np.testing.assert_array_equal(anchor["indices"], np.arange(64))
    estimator = study.engine()
    for snr in SNRS:
        baseline_file = study.OUT / "test/cells" / f"random3to7_rsr12_snr{snr:g}.npz"
        base = study.load(baseline_file)
        if str(base["selection_sha256"]) != policy_sha:
            raise ValueError("wrong zero-offset prediction selection")
        np.testing.assert_array_equal(base["method_names"], study.NAMES)
        correct_obs = study.observed(anchor, snr)
        for offset in OFFSETS:
            target = ROOT / "cells" / f"snr{snr:g}_offset{offset:+g}.npz"
            if target.exists():
                cached = study.load(target)
                if str(cached["protocol_sha256"]) != digest:
                    raise RuntimeError("cached sensitivity protocol mismatch")
                continue
            obs = study.observed(anchor, snr, variance_offset_db=offset)
            for unchanged, current in zip(correct_obs.args()[:3], obs.args()[:3]):
                if not torch.equal(unchanged, current):
                    raise AssertionError("variance sensitivity changed measurements or known pilots/reference")
            torch.testing.assert_close(obs.sigma2, correct_obs.sigma2 * 10 ** (offset / 10), rtol=0, atol=0)
            if offset == 0:
                predictions = torch.from_numpy(base["estimates"])
                fast_mask = torch.from_numpy(base["bypass_fast"])
                em_mask = torch.from_numpy(base["bypass_em100"])
            else:
                original, off, em = study.base_predictions(estimator, obs)
                predictions, fast_mask, em_mask = study.assemble(estimator, obs, policy["policies"], original, off, em)
            if not torch.isfinite(predictions).all():
                raise FloatingPointError("nonfinite sensitivity output")
            dispatch_error = 0.
            for case in (0, 63):
                for seed_index, seed in enumerate(study.SEEDS):
                    for kind, group_start in (("same_steps", 6), ("em100", 10)):
                        actual = estimator.estimate(obs.batch(slice(case, case + 1)), f"learned_{seed}",
                            policy["policies"][kind], fallback=kind)
                        difference = float((actual - predictions[group_start + seed_index, case:case + 1]).abs().max())
                        if difference > 1e-10:
                            raise AssertionError("sensitivity cached branch differs from real dispatch")
                        dispatch_error = max(dispatch_error, difference)
            values = predictions.numpy()
            error = (np.abs(values - anchor["g"][None]) ** 2).sum((-2, -1))
            energy = (np.abs(anchor["g"]) ** 2).sum((-2, -1))
            study.save(target, method_names=np.array(study.NAMES), estimates=values,
                errors=error, truth_energy=energy, nmse=error / energy[None], actual_snr=snr,
                supplied_variance_offset_db=offset, supplied_snr_proxy_db=snr - offset,
                bypass_fast=fast_mask.numpy(), bypass_em100=em_mask.numpy(),
                indices=anchor["indices"], anchor_sha256=sha(source), baseline_sha256=sha(baseline_file),
                actual_dispatch_max_abs_error=dispatch_error, protocol_sha256=digest)
            print(json.dumps(dict(study="bypass noise sensitivity", actual_snr=snr,
                variance_offset_db=offset, fast_bypass_rate=float(fast_mask.double().mean()),
                em100_bypass_rate=float(em_mask.double().mean()))), flush=True)


def interval(values):
    return np.quantile(values, [.025, .975]).tolist()


def analyze(digest):
    rng = np.random.default_rng(2026101773)
    indices = rng.integers(0, 64, size=(3000, 64))
    rows, records = [], {}
    for snr in SNRS:
        for offset in OFFSETS:
            result = study.load(ROOT / "cells" / f"snr{snr:g}_offset{offset:+g}.npz")
            errors, energy = result["errors"], result["truth_energy"]
            per_fit_db = 10 * np.log10(errors.sum(-1) / energy.sum())
            bootstrap_db = 10 * np.log10(errors[:, indices].sum(-1) / energy[indices].sum(-1)[None])
            methods = {}
            for name, group in GROUPS.items():
                methods[name] = dict(nmse_db=float(per_fit_db[group].mean()),
                    bootstrap95=interval(bootstrap_db[group].mean(0)),
                    per_fit_nmse_db=per_fit_db[group].tolist(),
                    severe_nmse_above1_rate=float((result["nmse"][group] > 1).mean()))
            fast_gain = per_fit_db[:3].mean() - per_fit_db[6:9].mean()
            fast_boot = bootstrap_db[:3].mean(0) - bootstrap_db[6:9].mean(0)
            em_gain = per_fit_db[:3].mean() - per_fit_db[10:13].mean()
            em_boot = bootstrap_db[:3].mean(0) - bootstrap_db[10:13].mean(0)
            row = dict(actual_snr_db=snr, variance_offset_db=offset, supplied_snr_proxy_db=snr - offset,
                methods=methods, fast_bypass_rate=float(result["bypass_fast"].mean()),
                em100_bypass_rate=float(result["bypass_em100"].mean()),
                fast_gain_vs_original_same_supplied_noise_db=float(fast_gain), fast_gain_bootstrap95=interval(fast_boot),
                em100_bypass_gain_vs_original_same_supplied_noise_db=float(em_gain), em100_gain_bootstrap95=interval(em_boot))
            rows.append(row)
            records[(snr, offset)] = dict(row=row, boot=bootstrap_db)
    for snr in SNRS:
        zero = records[(snr, 0.)]
        for offset in OFFSETS:
            current = records[(snr, offset)]
            for name, group in GROUPS.items():
                current["row"]["methods"][name]["degradation_vs_correct_variance_db"] = (
                    current["row"]["methods"][name]["nmse_db"] - zero["row"]["methods"][name]["nmse_db"])
                current["row"]["methods"][name]["degradation_bootstrap95"] = interval(
                    current["boot"][group].mean(0) - zero["boot"][group].mean(0))
    output = dict(protocol_sha256=digest, cells=rows, independent_test_anchors=64,
        policies=json.loads((study.OUT / "policy.json").read_text())["policies"],
        metric="Each fit uses ratio of summed squared channel error to summed truth energy in dB; group reports average fit score, never an ensemble.",
        interpretation="Positive gain means bypass helps relative to the original estimator with the same supplied variance. Positive degradation means a variance error hurts the same method relative to correctly supplied noise. Primary calibration is unchanged.",
        caveat="Ordinary-range exact preservation is conditional on the receiver SNR proxy, not on unknown actual SNR under variance miscalibration. These nine coupled diagnostic cells do not establish robustness to arbitrary mismatch.")
    write_json(ROOT / "summary.json", output)
    plot(rows)
    print(json.dumps({f"snr{row['actual_snr_db']:g}_offset{row['variance_offset_db']:+g}": {
        "fast_gain_db": row["fast_gain_vs_original_same_supplied_noise_db"],
        "fast_bypass_rate": row["fast_bypass_rate"],
        "methods": {name: values["nmse_db"] for name, values in row["methods"].items()}} for row in rows}), flush=True)


def plot(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(12.7, 4.2))
    styles = dict(original_mean=("Original learned", "#d55e00", "-"),
        off_mean=("Same steps, filtering off", "#0072b2", "--"),
        fast_bypass_mean=("Primary bypass", "#009e73", "-"),
        em100=("EM100", "#555555", "--"),
        em100_bypass_mean=("EM100 bypass", "#cc79a7", "-"))
    for ax, snr in zip(axes, SNRS):
        selected = [row for row in rows if row["actual_snr_db"] == snr]
        for name, (label, color, style) in styles.items():
            ax.plot(OFFSETS, [row["methods"][name]["nmse_db"] for row in selected],
                color=color, linestyle=style, marker="o", label=label)
        ax.set_title(f"Actual SNR {snr:g} dB")
        ax.set_xlabel("Supplied variance error (dB)")
        ax.set_xticks(OFFSETS); ax.grid(alpha=.25)
    axes[0].set_ylabel("Channel NMSE (dB; lower is better)")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=3, loc="lower center", bbox_to_anchor=(.5, -.005), fontsize=8.4)
    fig.suptitle("Fixed-policy sensitivity to supplied noise variance: same channels and observations", fontsize=11)
    fig.tight_layout(rect=[0, .12, 1, .99])
    fig.savefig(ROOT / "noise_sensitivity.png", dpi=180, bbox_inches="tight")
    fig.savefig(ROOT / "noise_sensitivity.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--analyze-only", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    policy, digest = freeze()
    if not args.analyze_only:
        evaluate(policy, digest)
    analyze(digest)


if __name__ == "__main__":
    main()
