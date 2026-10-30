"""Small routing tests independent of all scientific result files."""
import unittest
import sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from paper2_snr_switch import choose_threshold, route_mask, routed_forward


class FakeLayer:
    _disable_hankel = False


class FakeModel:
    def __init__(self, value):
        self.value, self.layers = value, [FakeLayer()]

    def __call__(self, G0, Z, S, B, sigma2):
        value = self.value + (10 if self.layers[0]._disable_hankel else 0)
        return torch.full_like(G0, value)


class SwitchingTests(unittest.TestCase):
    def test_threshold_from_validation(self):
        snr = np.array([-5, 1, 4, 6, 15])
        u = np.array([[1., 1., 1., 4., 4.]])
        h = np.array([[4., 4., 4., 1., 1.]])
        threshold, _ = choose_threshold(u, h, snr)
        self.assertEqual(threshold, 5)

    def test_tie_prefers_no_hankel(self):
        x = np.ones((3, 5))
        threshold, _ = choose_threshold(x, x, np.arange(5.))
        self.assertEqual(threshold, 21)

    def test_boundary_and_endpoints(self):
        np.testing.assert_array_equal(route_mask([4.9, 5., 5.1], 5), [False, True, True])
        self.assertTrue(route_mask([-10, 20], -11).all())
        self.assertFalse(route_mask([-10, 20], 21).any())

    def test_routing_and_toggle_restore(self):
        batch = {key: torch.zeros(2, 1, 1, dtype=torch.complex64)
                 for key in ('G0', 'Z', 'S', 'B')}
        batch['sigma2'] = torch.tensor([30., .03])
        u, h = FakeModel(1), FakeModel(2)
        np.testing.assert_array_equal(routed_forward(u, h, batch, 5).numpy().ravel(), [1, 2])
        np.testing.assert_array_equal(routed_forward(u, h, batch, 5, True).numpy().ravel(), [12, 2])
        self.assertFalse(h.layers[0]._disable_hankel)
        for threshold, value in ((-11, 2), (21, 1)):
            np.testing.assert_array_equal(routed_forward(u, h, batch, threshold).numpy().ravel(), [value, value])


if __name__ == '__main__':
    unittest.main()
