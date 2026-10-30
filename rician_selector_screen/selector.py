"""A fixed noise-based choice between two already trained estimators.

Only the supplied, normalized noise variance decides the route.  The value
3 is calibrated expected signal power in this K=3 benchmark, not power from
the unknown channel.  This is not an SNR estimator for an uncalibrated link.
The two extreme threshold values are explicit constant-policy sentinels.
"""
import math
from pathlib import Path

import numpy as np
import torch

from sahure.data import Observations, load_observations, split_observations
from sahure.math import nll


ALWAYS_ALLPILOT = -100.0
ALWAYS_ORIGINAL = 100.0
THRESHOLDS = (ALWAYS_ALLPILOT, -7.5, -2.5, 2.5, 7.5, 12.5,
              17.5, ALWAYS_ORIGINAL)
MODEL_NAMES = ("holdout_best", "allpilot_best")
BOUNDARY_ATOL_DB = 1e-10  # Fixed float64 roundoff tolerance, not a tuned gate.


def nominal_snr(sigma2, nominal_power=3.0):
    """Nominal dB SNR from calibrated expected power and supplied variance."""
    variance = torch.as_tensor(sigma2)
    power = float(nominal_power)
    if variance.ndim != 1 or not variance.is_floating_point():
        raise ValueError("variance must be a one-dimensional floating-point tensor")
    if not bool(torch.isfinite(variance).all()) or not bool((variance > 0).all()):
        raise ValueError("supplied variance must be finite and strictly positive")
    if not math.isfinite(power) or power <= 0:
        raise ValueError("calibrated nominal power must be finite and positive")
    # Difference of logs avoids overflow for very small positive variances.
    return 10 * (math.log10(power) - torch.log10(variance))


def route_mask(sigma2, threshold_db, nominal_power=3.0):
    """True selects all-pilot; false selects original, without channel truth.

    Ordinary thresholds use all-pilot at equality (within 1e-10 dB roundoff).
    +100 and -100 are explicit
    always-original and always-all-pilot policies, including outside the grid.
    """
    snr = nominal_snr(sigma2, nominal_power)
    threshold = float(threshold_db)
    if not math.isfinite(threshold):
        raise ValueError("threshold must be finite")
    if threshold == ALWAYS_ORIGINAL:
        return torch.zeros_like(snr, dtype=torch.bool)
    if threshold == ALWAYS_ALLPILOT:
        return torch.ones_like(snr, dtype=torch.bool)
    return snr >= threshold - BOUNDARY_ATOL_DB


def supplied_variance(obs, bias_db):
    """Change only the receiver's variance input; never regenerate observations.

    The multiplier is 10**(bias_db/10), because sigma2 is a variance, not an
    amplitude standard deviation. Both routing and the selected model receive
    this same altered input. Actual measurement noise remains unchanged.
    """
    bias = float(bias_db)
    if not math.isfinite(bias):
        raise ValueError("variance bias must be finite")
    try:
        factor = 10.0 ** (bias / 10.0)
    except OverflowError as exc:
        raise ValueError("variance bias is outside the numerical range") from exc
    variance = obs.sigma2 * factor
    nominal_snr(variance)
    return Observations(obs.z, obs.s, obs.b, variance)


@torch.no_grad()
def estimate(obs, models, threshold_db, nominal_power=3.0, diagnostics=False):
    """Run one selected estimator per case, grouping equal routes in a batch.

    A mixed batch makes two group calls, never two predictions for one case.
    A single-case batch therefore always makes exactly one model call.
    Loading and storing two frozen models is required and must be disclosed.
    """
    if (obs.z.ndim != 3 or obs.s.ndim != 3 or obs.b.shape != obs.z.shape
            or obs.s.shape[0] != len(obs) or obs.s.shape[-1] != obs.z.shape[-1]
            or obs.sigma2.shape != (len(obs),) or len(obs) == 0):
        raise ValueError("expected nonempty batched observations with one variance per case")
    if not all(name in models for name in MODEL_NAMES):
        raise ValueError("both frozen estimator checkpoints are required")
    mask = route_mask(obs.sigma2, threshold_db, nominal_power)
    output = torch.empty((len(obs), obs.z.shape[1], obs.s.shape[1]),
                         dtype=obs.s.dtype, device=obs.z.device)
    counts, calls = {}, 0
    for use_allpilot, name in enumerate(MODEL_NAMES):
        ids = torch.nonzero(mask == bool(use_allpilot), as_tuple=False).flatten()
        counts[name] = len(ids)
        if len(ids) == 0:
            continue
        model = models[name]
        if hasattr(model, "training") and model.training:
            raise ValueError("selector requires frozen models in evaluation mode")
        selected = model(*obs.batch(ids).args())
        expected = (len(ids), obs.z.shape[1], obs.s.shape[1])
        if selected.shape != expected or not selected.is_complex():
            raise ValueError("estimator returned an incorrect channel shape or dtype")
        if selected.dtype != output.dtype or selected.device != output.device:
            raise ValueError("estimator output dtype/device must match observations")
        if not bool(torch.isfinite(selected).all()):
            raise FloatingPointError("nonfinite selected channel estimate")
        output[ids] = selected
        calls += 1
    if diagnostics:
        return output, dict(allpilot_mask=mask, nominal_snr_db=nominal_snr(obs.sigma2, nominal_power),
                            forward_calls=calls, cases_per_model=counts)
    return output


def select_threshold(nominal_snr_db, old_nll, allpilot_nll, conditions):
    """Choose among eight fixed rules using observation-only validation losses.

    Condition IDs balance the objective; they never determine an individual
    route. No channel truth, channel errors or severe-error labels are inputs.
    Every candidate is scored by equal-weight mean of within-condition means.
    Ties within 1e-12 prefer a larger threshold (more original-model use).
    """
    snr, old, new = (np.asarray(value, dtype=np.float64)
                     for value in (nominal_snr_db, old_nll, allpilot_nll))
    groups = np.asarray(conditions)
    if (snr.ndim != 1 or len(snr) == 0 or old.shape != snr.shape
            or new.shape != snr.shape or groups.shape != snr.shape):
        raise ValueError("one SNR, two losses and one condition ID are required per validation case")
    if not all(np.isfinite(value).all() for value in (snr, old, new)):
        raise ValueError("validation SNR and losses must be finite")
    if not np.issubdtype(groups.dtype, np.integer):
        raise ValueError("condition IDs must be integers")
    ids, counts = np.unique(groups, return_counts=True)
    if not np.array_equal(ids, np.arange(len(ids))):
        raise ValueError("all contiguous condition IDs must be represented")
    candidates = []
    for threshold in THRESHOLDS:
        if threshold == ALWAYS_ORIGINAL:
            mask = np.zeros(len(snr), dtype=bool)
        elif threshold == ALWAYS_ALLPILOT:
            mask = np.ones(len(snr), dtype=bool)
        else:
            mask = snr >= threshold - BOUNDARY_ATOL_DB
        case_scores = np.where(mask, new, old)
        strata = np.bincount(groups, weights=case_scores, minlength=len(ids)) / counts
        routing = np.bincount(groups, weights=mask.astype(float), minlength=len(ids)) / counts
        candidates.append(dict(threshold_db=threshold, score=float(strata.mean()),
            per_condition_nll=strata.tolist(), allpilot_fraction=float(mask.mean()),
            per_condition_allpilot_fraction=routing.tolist()))
    lowest = min(row['score'] for row in candidates)
    selected = max((row for row in candidates if row['score'] <= lowest + 1e-12),
                   key=lambda row: row['threshold_db'])
    return dict(threshold_db=selected['threshold_db'], score=selected['score'],
        allpilot_fraction=selected['allpilot_fraction'], candidate_scores=candidates,
        validation_cases=len(snr), condition_count=len(ids), condition_counts=counts.tolist(),
        selection_metric='equal-condition mean held-out Rician negative log-likelihood',
        tie_rule='within 1e-12 of minimum NLL, prefer larger threshold',
        boundary_tolerance_db=BOUNDARY_ATOL_DB,
        fitting_uses_channel_truth=False)


@torch.no_grad()
def fit_rule(models):
    """Fit a fixed rule on the original validation observations, without writes.

    Both already selected checkpoints are evaluated on the same 21/9 split
    and all 126 strata. They are not retrained. Selection is now in float64;
    historical training checkpoint selection was in float32. This validation
    reuse is development, not independent evidence for the chosen router.
    """
    for name in MODEL_NAMES:
        if name not in models or models[name].training:
            raise ValueError("both frozen estimators must be in evaluation mode")
        if any(parameter.dtype != torch.float64 or parameter.device.type != 'cpu'
               for parameter in models[name].parameters()):
            raise ValueError("router validation requires CPU float64 models")
    directory = Path('results/sahure_expansion/data')
    observed = load_observations(directory / 'val_observations.npz', dtype=torch.float64)
    conditions = np.load(directory / 'val_conditions.npy', allow_pickle=False)
    if len(observed) != 2048 or observed.s.shape[-1] != 30 or len(np.unique(conditions)) != 126:
        raise RuntimeError("expected the frozen 2048-case, 30-pilot, 126-stratum validation set")
    context, target = split_observations(observed, 2026092203)
    losses = {}
    for name in MODEL_NAMES:
        rows = []
        for start in range(0, len(observed), 32):
            a, b = context.batch(slice(start, start + 32)), target.batch(slice(start, start + 32))
            channel = models[name](*a.args())
            values = nll(b.z, channel @ b.s + b.b, b.sigma2).mean((-2, -1))
            if not bool(torch.isfinite(channel).all()) or not bool(torch.isfinite(values).all()):
                raise FloatingPointError("nonfinite held-out router validation result")
            rows.append(values.cpu())
        losses[name] = torch.cat(rows).numpy()
    snr = nominal_snr(observed.sigma2).numpy()
    result = select_threshold(snr, losses['holdout_best'], losses['allpilot_best'], conditions)
    result.update(validation_mask_seed=2026092203, context_pilots=21, target_pilots=9,
        nominal_power=3., dtype='CPU float64',
        validation_sources=[str(directory / 'val_observations.npz'), str(directory / 'val_conditions.npy')],
        per_case_nll={name: values.tolist() for name, values in losses.items()},
        nominal_snr_db=snr.tolist(), conditions=conditions.tolist(),
        note='Observation-only development selection on reused validation data; condition IDs balance scores but never route cases.')
    return result
