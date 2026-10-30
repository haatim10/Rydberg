"""Independent read-only audit of frozen cutoff selection and all test scores."""
from pathlib import Path
import json
import numpy as np

from tcom_followups_20261017.common import sha, write_json
from . import bypass_study as study

ROOT = study.OUT.parent / "bypass_audit"


def run():
    policy, policy_sha = study.check_selection()
    protocol = json.loads((study.OUT / "protocol.json").read_text())
    loss_files = sorted((study.OUT / "development/losses").glob("*.npz"))
    records = [study.load(path) for path in loss_files]
    if len(records) != 117:
        raise AssertionError("incomplete development grid")
    selections = {}
    for rsr in study.RSRS:
        cells = [item for item in records if float(item["rsr"]) == rsr]
        if len(cells) != 39:
            raise AssertionError("incomplete reference development grid")
        for kind in ("same_steps", "em100"):
            candidates = []
            for cutoff_index, cutoff in enumerate(study.CUTOFFS):
                scores = []
                for cell in cells:
                    expected = float(cell["snr"]) > cutoff + 1e-10 and float(cell["snr"]) > 20 + 1e-10
                    mask = np.full(len(cell["indices"]), expected, dtype=bool)
                    np.testing.assert_array_equal(cell["bypass_masks"][cutoff_index], mask)
                    original = cell["losses"][:3]
                    fallback = cell["losses"][3:6] if kind == "same_steps" else np.repeat(cell["losses"][6:7], 3, axis=0)
                    value = float(np.where(mask[None], fallback, original).mean())
                    scores.append(dict(snr=float(cell["snr"]), regime=str(cell["regime"]), mean_nll=value))
                mean = float(np.mean([item["mean_nll"] for item in scores]))
                recorded = policy["validation_tables"][kind][f"{rsr:g}"][cutoff_index]
                np.testing.assert_allclose(mean, recorded["mean_nll"], atol=1e-12, rtol=1e-12)
                candidates.append(dict(cutoff_db=cutoff, mean_nll=mean, cells=scores))
            ordered = sorted(candidates, key=lambda item: (item["mean_nll"], -item["cutoff_db"]))
            winner, runner_up = ordered[:2]
            if winner["cutoff_db"] != policy["policies"][kind][f"{rsr:g}"]:
                raise AssertionError("selected cutoff does not minimize the declared criterion")
            # Show which SNRs distinguish the winner from the next-best policy.
            contributions = {f"{snr:g}": float(sum(
                other["mean_nll"] - best["mean_nll"]
                for best, other in zip(winner["cells"], runner_up["cells"])
                if best["snr"] == snr) / len(cells)) for snr in study.SNRS}
            selections[f"{kind}_rsr{rsr:g}"] = dict(selected_cutoff_db=winner["cutoff_db"],
                selected_mean_nll=winner["mean_nll"], runner_up_cutoff_db=runner_up["cutoff_db"],
                winning_mean_nll_margin=runner_up["mean_nll"] - winner["mean_nll"],
                runner_up_minus_winner_contribution_by_snr=contributions)
    anchors = sorted((study.OUT / "test/anchors").glob("*.npz"))
    files = sorted((study.OUT / "test/cells").glob("*.npz"))
    if len(anchors) != 9 or len(files) != 117:
        raise AssertionError("test grid not complete")
    all_test_files = anchors + files
    earliest = min(path.stat().st_mtime for path in all_test_files)
    if earliest < policy["frozen_unix"]:
        raise AssertionError("a test file predates the recorded policy freeze")
    max_error_difference = 0.
    max_nmse_difference = 0.
    ordinary_checked = 0
    anchor_hashes, test_hashes = {}, {}
    for path in anchors:
        anchor = study.load(path)
        if int(anchor["seed"]) != 2026101762 or str(anchor["selection_sha256"]) != policy_sha:
            raise AssertionError("wrong test anchor seed or policy")
        if str(anchor["protocol_sha256"]) != policy["protocol_sha256"]:
            raise AssertionError("wrong test anchor source protocol")
        reconstructed = np.abs(anchor["g"] @ anchor["s"] + anchor["b"] +
                               anchor["unit_noise"] * np.sqrt(anchor["anchor_variance"])[:, None, None])
        np.testing.assert_allclose(reconstructed, anchor["anchor_z"], atol=1e-13, rtol=0)
        anchor_hashes[path.name] = sha(path)
    for path in files:
        result = study.load(path)
        regime, rsr, snr = str(result["regime"]), float(result["rsr"]), float(result["snr"])
        anchor = study.load(study.OUT / "test/anchors" / f"{regime}_rsr{rsr:g}.npz")
        if str(result["selection_sha256"]) != policy_sha or str(result["protocol_sha256"]) != policy["protocol_sha256"]:
            raise AssertionError("test predictions have incorrect frozen hashes")
        np.testing.assert_array_equal(result["indices"], anchor["indices"])
        np.testing.assert_array_equal(result["paths"], anchor["paths"])
        np.testing.assert_array_equal(result["method_names"], study.NAMES)
        errors = (np.abs(result["estimates"] - anchor["g"][None]) ** 2).sum((-2, -1))
        energy = (np.abs(anchor["g"]) ** 2).sum((-2, -1))
        nmse = errors / energy[None]
        max_error_difference = max(max_error_difference, float(np.max(np.abs(errors - result["errors"]))))
        max_nmse_difference = max(max_nmse_difference, float(np.max(np.abs(nmse - result["nmse"]))))
        np.testing.assert_array_equal(errors, result["errors"])
        np.testing.assert_array_equal(energy, result["truth_energy"])
        np.testing.assert_array_equal(nmse, result["nmse"])
        for kind, mask_key, start, fallback in (("same_steps", "bypass_fast", 6, result["estimates"][3:6]),
                ("em100", "bypass_em100", 10, np.repeat(result["estimates"][9:10], 3, axis=0))):
            decision = snr > policy["policies"][kind][f"{rsr:g}"] + 1e-10 and snr > 20 + 1e-10
            np.testing.assert_array_equal(result[mask_key], np.full(64, decision, dtype=bool))
            expected = fallback if decision else result["estimates"][:3]
            np.testing.assert_array_equal(result["estimates"][start:start + 3], expected)
        if snr <= 20:
            np.testing.assert_array_equal(result["estimates"][6:9], result["estimates"][:3])
            np.testing.assert_array_equal(result["estimates"][10:13], result["estimates"][:3])
            ordinary_checked += 1
        test_hashes[path.name] = sha(path)
    return dict(passed=True, selection_sha256=policy_sha, parent_protocol_sha256=policy["protocol_sha256"],
        all_frozen_source_checkpoint_development_hashes_match=True,
        cutoff_recomputation=selections, test_anchors_checked=9, test_cells_checked=117,
        ordinary_cells_with_exact_identity=ordinary_checked,
        saved_error_recomputation_max_abs_difference=max_error_difference,
        saved_nmse_recomputation_max_abs_difference=max_nmse_difference,
        test_files_after_policy_freeze=True, earliest_test_file_minus_freeze_seconds=earliest - policy["frozen_unix"],
        timestamp_caveat="Local file times support the recorded sequence; frozen hashes and runner assertions provide the complementary reproducibility record.",
        anchor_hashes=anchor_hashes, test_prediction_hashes=test_hashes, audit_source_sha256=sha(__file__),
        score_caveat="Mean held-out NLL is the declared selection criterion. Primary claims must still examine every test condition and all negative-gain cells; a favorable pooled score is not a uniform guarantee.")


if __name__ == "__main__":
    output = run()
    write_json(ROOT / "study_audit.json", output)
    print(json.dumps({key: value for key, value in output.items()
                      if key not in ("anchor_hashes", "test_prediction_hashes")}, indent=2))
