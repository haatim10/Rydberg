import unittest
import numpy as np
import torch
from sahure.data import Observations, split_observations
from sahure_expansion.data import world
from sahure.math import nll
from paper2_optshrink_loop.estimator import estimate as independent_opt
from codex_verification.solver import fit
from merged_review_checks.run import one_sweep
from .estimator import tensors, estimate, hard_one, development_scores, lookup_key, PARAMS, BUDGETS
from .run import models, choose, select_settings, CONFIG, SPECS, validation_data


class Checks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.w = world(0, 202610123991, snr=5, rsr=12, L=7)[0]
        w = cls.w
        cls.obs = tensors(dict(z=w.Z[None], s=w.S[None], b=w.B[None], sigma2=np.array([w.sigma2])))

    def test_classical_numpy_parity(self):
        w = self.w
        for p in ('loop', 'final'):
            for r, a, t in ((1, .25, 5), (7, 1., 20), (15, .75, 10), (0, 0., 100)):
                cfg = dict(rank=r, strength=a, iterations=t, placement=p)
                got = estimate(self.obs, cfg).numpy()[0]
                expected = independent_opt(w.Z, w.S, w.B, w.sigma2, cfg)
                np.testing.assert_allclose(got, expected, rtol=3e-8, atol=3e-9)

    def test_zero_filter_em_and_batch(self):
        cfg = dict(rank=0, strength=0., iterations=10, placement='loop')
        got = estimate(self.obs, cfg).numpy()[0]
        w = self.w
        np.testing.assert_allclose(got, fit(w.S, w.Z, w.B, w.sigma2, iterations=10)[0], rtol=3e-9, atol=3e-10)
        obs = Observations(*(x.expand(3, *x.shape[1:]) for x in self.obs.args()))
        actual = estimate(obs, cfg).numpy()
        np.testing.assert_allclose(actual, np.repeat(got[None], 3, axis=0), rtol=3e-9, atol=3e-10)

    def test_hard_one_sweep_parity(self):
        w = self.w
        got, ranks = hard_one(self.obs)
        expected, rank, _ = one_sweep(w.Z, w.S, w.B, w.sigma2)
        self.assertEqual(ranks, [rank])
        np.testing.assert_allclose(got.numpy()[0], expected, rtol=3e-8, atol=3e-9)

    def test_candidate_scores_and_no_heldout_initialization(self):
        a, b = split_observations(self.obs, 72)
        scores = development_scores(a, b)
        self.assertEqual(scores.shape, (2, 5, 61))
        for pi, ti, ri in ((0, 1, 0), (0, 2, 27), (1, 3, 59)):
            cfg = dict(**PARAMS[ri], placement=('loop', 'final')[pi], iterations=BUDGETS[ti])
            g = estimate(a, cfg)
            loss = float(nll(b.z, g @ b.s + b.b, b.sigma2).mean())
            np.testing.assert_allclose(scores[pi, ti, ri], loss, rtol=3e-9, atol=3e-9)
            altered = Observations(b.z * 2, b.s, b.b, b.sigma2)
            self.assertNotEqual(float(nll(altered.z, g @ b.s + b.b, b.sigma2).mean()), loss)
            torch.testing.assert_close(estimate(a, cfg), g, rtol=0, atol=0)

    def test_lookup_uses_only_known_reference_noise(self):
        self.assertEqual(lookup_key(self.obs), ['5:12'])
        altered = Observations(self.obs.z * 100, self.obs.s * 3, self.obs.b, self.obs.sigma2)
        self.assertEqual(lookup_key(altered), lookup_key(self.obs))

    def test_equal_stratum_weight_not_case_count(self):
        # Unequal case counts deliberately favour a different candidate under pooling.
        ids = np.repeat(np.arange(126), [50 if j % 6 == 0 else 1 for j in range(126)])
        scores = np.full((len(ids), 2, 5, 61), 100.)
        scores[:, 0, 0, 0] = np.where(ids % 6 == 0, 0., 10.)
        scores[:, 0, 0, 1] = 4.
        chosen, strata = select_settings(scores, ids)
        self.assertEqual(strata.shape, (126, 2, 5, 61))
        for entry in chosen.values():
            self.assertEqual(entry['selected']['rank'], 1)
        self.assertEqual(choose(scores.mean(0))['rank'], 0)

    def test_original_models_finite_and_unchanged(self):
        for model in models().values():
            self.assertEqual(sum(p.numel() for p in model.parameters()), 188)
            before = {k: v.clone() for k, v in model.state_dict().items()}
            with torch.no_grad():
                self.assertTrue(torch.isfinite(model(*self.obs.args())).all())
            for k, v in before.items():
                torch.testing.assert_close(v, model.state_dict()[k], rtol=0, atol=0)

    def test_test_bank_seed_separation(self):
        seeds = [s + 100 * j for s in CONFIG['test_banks'] for j in range(len(SPECS))]
        self.assertEqual(len(seeds), len(set(seeds)))
        self.assertNotIn(CONFIG['timing_seed'], seeds)
        self.assertNotIn(2026092202, seeds)

    def test_original_validation_split(self):
        a, b, ids = validation_data()
        self.assertEqual(len(a), 2048)
        self.assertEqual(a.z.shape[-1], 21)
        self.assertEqual(b.z.shape[-1], 9)
        a2, b2, ids2 = validation_data()
        torch.testing.assert_close(a.z, a2.z, rtol=0, atol=0)
        torch.testing.assert_close(b.s, b2.s, rtol=0, atol=0)
        np.testing.assert_array_equal(ids, ids2)


if __name__ == '__main__':
    unittest.main()
