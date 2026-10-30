"""Independent bypass audit: branch execution, scalar grouping, and state.

Synthetic policies and a separate smoke seed test implementation only. No
validation or test outcome is used to choose the deployment cutoff here.
"""
import os
for variable in ("VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[variable] = "1"

from pathlib import Path
import json
import numpy as np
import torch

from sahure.data import Observations
from tcom_followups_20261017.common import world_batch, sha, write_json
from .bypass import BypassEstimator


ROOT = Path(__file__).resolve().parents[1] / "results/tcom_runtime_bypass_20261017/bypass_audit"


def combine(items):
    return Observations(*(torch.cat([obs.args()[j] for obs in items]) for j in range(4)))


def tagged_observations(snrs, references):
    count = len(snrs)
    z = torch.zeros((count, 32, 30), dtype=torch.float64)
    z[:, 0, 0] = torch.arange(count)
    s = torch.zeros((count, 3, 30), dtype=torch.complex128)
    b = (10 ** (torch.tensor(references, dtype=torch.float64) / 20))[:, None, None]
    b = b.expand(count, 32, 30).to(torch.complex128)
    variance = 3 * 10 ** (-torch.tensor(snrs, dtype=torch.float64) / 10)
    return Observations(z, s, b, variance)


def call_count_checks(wrapper, policy):
    """Replace branch computations with tagged sentinels, retaining real dispatch."""
    saved = {name: getattr(wrapper, name) for name in ("original", "unfiltered", "em100")}
    cases = {
        "ordinary": tagged_observations([0, 20, 25], [5, 12, 12]),
        "high": tagged_observations([30, 50, 60], [5, 20, 12]),
        "mixed": tagged_observations([10, 20, 25, 26, 35, 40, 60], [5, 12, 20, 12, 20, 20, 5]),
    }
    results = {}
    model = next(iter(wrapper.models))
    try:
        for case_name, obs in cases.items():
            expected_bypass = wrapper.masks(obs, policy)["bypass"]
            for fallback in ("same_steps", "em100"):
                calls = []
                def make_spy(branch, value):
                    def spy(subset, *_):
                        identifiers = subset.z[:, 0, 0].to(torch.int64).tolist()
                        calls.append(dict(branch=branch, local_ids=identifiers))
                        return torch.full((len(subset), 32, 3), value, dtype=torch.complex128)
                    return spy
                wrapper.original = make_spy("original", 1)
                wrapper.unfiltered = make_spy("unfiltered", 2)
                wrapper.em100 = make_spy("em100", 3)
                output = wrapper.estimate(obs, model, policy, fallback=fallback)
                fallback_branch = "unfiltered" if fallback == "same_steps" else "em100"
                expected_calls = []
                for branch, mask in (("original", ~expected_bypass), (fallback_branch, expected_bypass)):
                    if mask.any():
                        expected_calls.append(dict(branch=branch, local_ids=mask.nonzero().flatten().tolist()))
                if calls != expected_calls:
                    raise AssertionError(dict(case=case_name, fallback=fallback, observed=calls, expected=expected_calls))
                all_ids = [index for entry in calls for index in entry["local_ids"]]
                if sorted(all_ids) != list(range(len(obs))):
                    raise AssertionError("a world was skipped or evaluated twice")
                expected_values = torch.where(expected_bypass, 2 if fallback == "same_steps" else 3, 1)
                if not torch.equal(output[:, 0, 0].real.to(torch.int64), expected_values):
                    raise AssertionError("mixed outputs were scattered incorrectly")
                results[f"{case_name}/{fallback}"] = calls
    finally:
        for name, function in saved.items():
            setattr(wrapper, name, function)
    return results


@torch.no_grad()
def run():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    wrapper = BypassEstimator()
    policy = {"5": 20., "12": 25., "20": 40.}
    call_checks = call_count_checks(wrapper, policy)
    patterns = tagged_observations([20, 21, 25, 26, 40, 41], [5, 5, 12, 12, 20, 20])
    masks = wrapper.masks(patterns, policy)
    expected = [False, True, False, True, False, True]
    if masks["bypass"].tolist() != expected:
        raise AssertionError("strict per-reference cutoff comparison is incorrect")
    changed = Observations(patterns.z + 700, patterns.s + 100 + 50j, patterns.b, patterns.sigma2)
    if not torch.equal(masks["bypass"], wrapper.masks(changed, policy)["bypass"]):
        raise AssertionError("dispatch depends on pilot values or measured magnitudes")
    ordinary = combine([world_batch(2026101770, [i], snr, rsr=ref, L=None)[0]
        for i, (snr, ref) in enumerate([(0, 5), (10, 12), (20, 20), (20, 5), (20, 12)])])
    mixed = combine([world_batch(2026101770, [20 + i], snr, rsr=ref, L=None)[0]
        for i, (snr, ref) in enumerate([(20, 5), (30, 5), (25, 12), (30, 12), (40, 20), (60, 20)])])
    numerical = {}
    for name in wrapper.models:
        original = wrapper.original(ordinary, name)
        row = {}
        for fallback in ("same_steps", "em100"):
            ordinary_result = wrapper.estimate(ordinary, name, policy, fallback=fallback)
            if not torch.equal(original, ordinary_result):
                raise AssertionError("ordinary result changed")
            grouped = wrapper.estimate(mixed, name, policy, fallback=fallback)
            scalar = torch.cat([wrapper.estimate(mixed.batch(slice(i, i + 1)), name, policy, fallback=fallback)
                                for i in range(len(mixed))])
            max_error = float((grouped - scalar).abs().max())
            torch.testing.assert_close(grouped, scalar, atol=1e-11, rtol=1e-11)
            row[f"{fallback}_ordinary_bitwise_equal"] = True
            row[f"{fallback}_grouped_vs_scalar_max_abs_error"] = max_error
        state_a, state_b = wrapper.models[name].state_dict(), wrapper.unfiltered_models[name].state_dict()
        if state_a.keys() != state_b.keys() or not all(torch.equal(state_a[k], state_b[k]) for k in state_a):
            raise AssertionError("filter-off clone changed checkpoint weights")
        row["complete_checkpoint_state_unchanged"] = True
        numerical[name] = row
    return dict(passed=True, audit_seed=2026101770, synthetic_policy=policy,
        branch_execution=call_checks, strict_boundary_pattern=expected,
        dispatch_invariant_to_measurement_and_pilot_values=True,
        numerical=numerical, source_sha256=sha(Path(__file__).with_name("bypass.py")),
        audit_source_sha256=sha(__file__),
        interpretation="Checks implementation using synthetic cutoffs and a separate smoke seed. They neither select cutoffs nor establish robustness to noise calibration errors.")


if __name__ == "__main__":
    result = run()
    write_json(ROOT / "audit.json", result)
    print(json.dumps(result, indent=2))
