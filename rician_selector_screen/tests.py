"""Small isolated tests of observable routing; no training or source freezing."""
import math
import unittest

import numpy as np
import torch

from sahure.data import Observations
from sahure.model import AdaptiveHankelEstimator, ModelConfig
from .selector import (ALWAYS_ALLPILOT, ALWAYS_ORIGINAL, estimate, nominal_snr,
                       route_mask, supplied_variance, select_threshold, THRESHOLDS)


class Counter(torch.nn.Module):
    def __init__(self, offset):
        super().__init__()
        self.offset = offset
        self.calls = []
        self.eval()

    def forward(self, z, s, b, variance):
        self.calls.append((z.detach().clone(), variance.detach().clone()))
        values = z[..., :1] + self.offset
        return values.expand(-1, -1, s.shape[1]).to(s.dtype)


def example(snr=(-10., 0., 10.), seed=107):
    gen = torch.Generator().manual_seed(seed)
    count, n, k, p = len(snr), 6, 2, 8
    real = lambda shape: torch.randn(shape, generator=gen, dtype=torch.float64)
    complex_values = lambda shape: torch.complex(real(shape), real(shape))
    s, b, g = complex_values((count, k, p)), complex_values((count, n, p)), complex_values((count, n, k))
    z = (g @ s + b).abs()
    variance = 3 * torch.pow(torch.tensor(10., dtype=torch.float64), -torch.tensor(snr, dtype=torch.float64) / 10)
    return Observations(z, s, b, variance)


class SelectorTests(unittest.TestCase):
    def test_boundary_and_constant_policies(self):
        obs = example()
        torch.testing.assert_close(nominal_snr(obs.sigma2), torch.tensor([-10., 0., 10.], dtype=torch.float64))
        self.assertEqual(route_mask(obs.sigma2, 0.).tolist(), [False, True, True])
        extreme = torch.tensor([1e-300, 1e300], dtype=torch.float64)
        self.assertEqual(route_mask(extreme, ALWAYS_ORIGINAL).tolist(), [False, False])
        self.assertEqual(route_mask(extreme, ALWAYS_ALLPILOT).tolist(), [True, True])
        self.assertTrue(torch.isfinite(nominal_snr(extreme)).all())
        for threshold in THRESHOLDS[1:-1]:
            values = torch.tensor([3 / 10**((threshold - 1e-6) / 10),
                                   3 / 10**(threshold / 10),
                                   3 / 10**((threshold + 1e-6) / 10)], dtype=torch.float64)
            self.assertEqual(route_mask(values, threshold).tolist(), [False, True, True])

    def test_variance_bias_and_no_observation_mutation(self):
        obs = example()
        saved = tuple(value.clone() for value in obs.args())
        for bias in (-3., 3.):
            altered = supplied_variance(obs, bias)
            self.assertIs(altered.z, obs.z)
            self.assertIs(altered.s, obs.s)
            self.assertIs(altered.b, obs.b)
            torch.testing.assert_close(altered.sigma2, obs.sigma2 * 10 ** (bias / 10))
            torch.testing.assert_close(nominal_snr(altered.sigma2), nominal_snr(obs.sigma2) - bias)
        for old, current in zip(saved, obs.args()):
            torch.testing.assert_close(old, current, rtol=0, atol=0)

    def test_grouping_selects_each_case_once(self):
        obs = example()
        models = dict(holdout_best=Counter(0), allpilot_best=Counter(100))
        got, detail = estimate(obs, models, 0., diagnostics=True)
        expected = obs.z[..., :1].expand(-1, -1, obs.s.shape[1]).to(obs.s.dtype).clone()
        expected[1:] += 100
        torch.testing.assert_close(got, expected, rtol=0, atol=0)
        self.assertEqual(detail['forward_calls'], 2)
        self.assertEqual(detail['cases_per_model'], dict(holdout_best=1, allpilot_best=2))
        self.assertEqual([len(models[n].calls) for n in ('holdout_best', 'allpilot_best')], [1, 1])
        self.assertEqual(sum(len(v[0]) for m in models.values() for v in m.calls), len(obs))

    def test_single_case_exactly_one_forward_with_supplied_variance(self):
        obs = supplied_variance(example((0.,)), 3.)
        models = dict(holdout_best=Counter(0), allpilot_best=Counter(100))
        _, detail = estimate(obs, models, 0., diagnostics=True)
        self.assertEqual(detail['forward_calls'], 1)
        self.assertEqual(len(models['holdout_best'].calls), 1)
        self.assertEqual(len(models['allpilot_best'].calls), 0)
        torch.testing.assert_close(models['holdout_best'].calls[0][1], obs.sigma2, rtol=0, atol=0)

    def test_routes_ignore_channel_observations_and_pilots(self):
        obs = example()
        changed = Observations(obs.z * 9, obs.s * 4j, obs.b * 17, obs.sigma2)
        models = dict(holdout_best=Counter(0), allpilot_best=Counter(100))
        _, first = estimate(obs, models, 0., diagnostics=True)
        _, second = estimate(changed, models, 0., diagnostics=True)
        torch.testing.assert_close(first['allpilot_mask'], second['allpilot_mask'])

    def test_grouped_output_matches_unchanged_estimator(self):
        torch.set_num_threads(1)
        obs = example()
        torch.manual_seed(71)
        first = AdaptiveHankelEstimator(ModelConfig(stages=2)).double().eval()
        torch.manual_seed(72)
        second = AdaptiveHankelEstimator(ModelConfig(stages=2)).double().eval()
        # Initial last layers are identical; use distinct frozen controls.
        with torch.no_grad():
            second.steps.add_(.4)
        models = dict(holdout_best=first, allpilot_best=second)
        got = estimate(obs, models, 0.)
        with torch.no_grad():
            expected = torch.cat([models['holdout_best' if i == 0 else 'allpilot_best'](*obs.batch(slice(i, i+1)).args())
                                  for i in range(len(obs))])
        torch.testing.assert_close(got, expected, rtol=1e-10, atol=1e-11)

    def test_invalid_inputs_rejected(self):
        for variance in (torch.tensor([0.]), torch.tensor([-1.]), torch.tensor([math.nan]),
                         torch.tensor([math.inf]), torch.tensor([1]), torch.ones(2, 2)):
            with self.assertRaises(ValueError):
                nominal_snr(variance)
        with self.assertRaises(ValueError):
            route_mask(torch.tensor([1.]), math.inf)
        with self.assertRaises(ValueError):
            nominal_snr(torch.tensor([1.]), 0)
        with self.assertRaises(ValueError):
            supplied_variance(example(), math.nan)

    def test_selection_finds_observation_only_switch(self):
        result = select_threshold([-10., -5., 0., 5.], [1., 1., 4., 4.],
                                  [4., 4., 1., 1.], [0, 1, 2, 3])
        self.assertEqual(result['threshold_db'], -2.5)
        self.assertEqual(result['score'], 1.)
        self.assertEqual(result['allpilot_fraction'], .5)
        self.assertFalse(result['fitting_uses_channel_truth'])
        self.assertEqual([row['threshold_db'] for row in result['candidate_scores']], list(THRESHOLDS))

    def test_selection_equal_condition_not_equal_case(self):
        # Nine cases prefer old by one, but the other condition prefers new by
        # two: equal-condition averaging selects new, equal-case averaging old.
        result = select_threshold(np.zeros(10), [1.] * 9 + [3.],
                                  [2.] * 9 + [1.], [0] * 9 + [1])
        self.assertLess(result['threshold_db'], 0.)
        self.assertEqual(result['score'], 1.5)
        self.assertEqual(result['condition_counts'], [9, 1])

    def test_selection_tie_and_constant_fallback(self):
        tie = select_threshold([-10., 10.], [1., 1.], [1., 1.], [0, 1])
        self.assertEqual(tie['threshold_db'], ALWAYS_ORIGINAL)
        old = select_threshold([-10., 10.], [0., 0.], [1., 1.], [0, 1])
        self.assertEqual(old['threshold_db'], ALWAYS_ORIGINAL)
        new = select_threshold([-1000., -1000.], [1., 1.], [0., 0.], [0, 1])
        self.assertEqual(new['threshold_db'], ALWAYS_ALLPILOT)

    def test_selection_rejects_nonfinite_or_missing_groups(self):
        with self.assertRaises(ValueError):
            select_threshold([0.], [math.nan], [1.], [0])
        with self.assertRaises(ValueError):
            select_threshold([0., 1.], [1., 1.], [2., 2.], [0, 2])
        with self.assertRaises(ValueError):
            select_threshold([0.], [1.], [2.], [0.5])


if __name__ == '__main__':
    unittest.main()
