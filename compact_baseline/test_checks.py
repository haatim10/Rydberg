"""Numerical and fairness checks for the residual-driven recurrent control.

Run with ``python -m unittest compact_baseline.test_checks``. These tests use
small simulator observations; channel truth is never passed to a controller.
"""
import copy
import inspect
import io
import unittest

import numpy as np
import torch
from torch import nn

from compact_baseline.model import RecurrentHankelEstimator
from sahure.data import Observations, split_observations
from sahure.math import nll
from sahure.model import AdaptiveHankelEstimator, ModelConfig
from sahure_expansion.data import world


def observation_batch(dtype=torch.float64):
    """Two independent worlds, with no channel labels in the returned object."""
    worlds = [world(i, 202610129100, N=8, K=2, P=12, snr=5., rsr=12., L=2)[0]
              for i in range(2)]
    cdtype = torch.complex128 if dtype == torch.float64 else torch.complex64
    return Observations(
        torch.as_tensor(np.stack([w.Z for w in worlds]), dtype=dtype),
        torch.as_tensor(np.stack([w.S for w in worlds]), dtype=cdtype),
        torch.as_tensor(np.stack([w.B for w in worlds]), dtype=cdtype),
        torch.as_tensor(np.asarray([w.sigma2 for w in worlds]), dtype=dtype),
    )


def output_layer(model):
    layers = [m for m in model.modules() if isinstance(m, nn.Linear)]
    if len(layers) != 1:
        raise AssertionError('The recurrent comparator must have one output layer')
    return layers[0]


def update(model, optimizer, context, heldout):
    optimizer.zero_grad(set_to_none=True)
    estimate = model(*context.args())
    loss = nll(heldout.z, estimate @ heldout.s + heldout.b,
               heldout.sigma2).mean()
    loss.backward()
    optimizer.step()
    return loss.detach()


class CompactChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.data = observation_batch()

    def setUp(self):
        torch.manual_seed(410)

    def test_parameter_budget_and_truth_free_interface(self):
        model = RecurrentHankelEstimator(stages=10, hidden_size=4)
        self.assertEqual(sum(p.numel() for p in model.parameters()), 148)
        cells = [m for m in model.modules() if isinstance(m, nn.LSTMCell)]
        self.assertEqual(len(cells), 1)
        self.assertEqual((cells[0].input_size, cells[0].hidden_size), (2, 4))
        parameters = inspect.signature(model.forward).parameters
        self.assertTrue({'z', 's', 'b', 'sigma2', 'diagnostics'} <= set(parameters))
        self.assertFalse({'g', 'G', 'truth', 'rank', 'paths', 'channel'} & set(parameters))

    def test_constant_output_matches_global_control(self):
        """Independent existing forward path checks EM, shrinkage and mixing."""
        model = RecurrentHankelEstimator(stages=10, hidden_size=4).double()
        constant = AdaptiveHankelEstimator(
            ModelConfig(core='em', structure='global', stages=10)).double()
        with torch.no_grad():
            steps = torch.linspace(-.2, .2, 10, dtype=torch.float64)
            model.steps.copy_(steps)
            constant.steps.copy_(steps)
            output_layer(model).weight.zero_()
            constant.structural.copy_(output_layer(model).bias.expand(10, 2))
        actual, trace = model(*self.data.args(), diagnostics=True)
        expected, constant_trace = constant(*self.data.args(), diagnostics=True)
        torch.testing.assert_close(actual, expected, rtol=2e-10, atol=2e-10)
        self.assertEqual(len(trace), 10)
        for left, right in zip(trace, constant_trace):
            self.assertTrue({'gamma', 'beta', 'tau', 'correction', 'nll'} <= left.keys())
            for key in ('gamma', 'beta', 'tau', 'correction', 'nll'):
                self.assertTrue(torch.isfinite(left[key]).all())
                torch.testing.assert_close(left[key], right[key], rtol=2e-10, atol=2e-10)

    def test_finite_forward_backward_and_dtype_agreement(self):
        model64 = RecurrentHankelEstimator(stages=10, hidden_size=4).double()
        with torch.no_grad():
            output_layer(model64).weight.normal_(mean=0., std=.05)
        model32 = copy.deepcopy(model64).float()
        data32 = observation_batch(torch.float32)
        estimates = []
        for model, data in ((model64, self.data), (model32, data32)):
            context, heldout = split_observations(data, 8221)
            estimate = model(*context.args())
            loss = nll(heldout.z, estimate @ heldout.s + heldout.b,
                       heldout.sigma2).mean()
            self.assertEqual(estimate.shape, (2, 8, 2))
            self.assertTrue(torch.isfinite(estimate).all())
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            for name, parameter in model.named_parameters():
                self.assertIsNotNone(parameter.grad, name)
                self.assertTrue(torch.isfinite(parameter.grad).all(), name)
            estimates.append(estimate.detach().to(torch.complex128))
        torch.testing.assert_close(estimates[0], estimates[1], rtol=1e-4, atol=1e-4)

    def test_recurrent_state_resets_and_does_not_mix_worlds(self):
        model = RecurrentHankelEstimator(stages=10, hidden_size=4).double()
        with torch.no_grad():
            output_layer(model).weight.normal_(mean=0., std=.1)
        with torch.no_grad():
            first = model(*self.data.args())
            model(*self.data.batch(slice(1, 2)).args())
            repeated = model(*self.data.args())
            singles = torch.cat([model(*self.data.batch(slice(i, i + 1)).args())
                                 for i in range(2)])
        torch.testing.assert_close(first, repeated, rtol=0., atol=0.)
        torch.testing.assert_close(first, singles, rtol=2e-10, atol=2e-10)

    def test_user_permutation_equivariance(self):
        """User states share weights, but must not leak through a user sequence."""
        model = RecurrentHankelEstimator(stages=10, hidden_size=4).double()
        with torch.no_grad():
            output_layer(model).weight.normal_(mean=0., std=.1)
            expected = model(*self.data.args())
            actual = model(self.data.z, self.data.s[:, [1, 0]],
                           self.data.b, self.data.sigma2)
        torch.testing.assert_close(actual[:, :, [1, 0]], expected,
                                   rtol=2e-10, atol=2e-10)

    def test_checkpoint_and_optimizer_resume_is_deterministic(self):
        reference = RecurrentHankelEstimator(stages=3, hidden_size=4).double()
        resumed = copy.deepcopy(reference)
        optimizer = torch.optim.Adam(reference.parameters(), lr=.001)
        resume_optimizer = torch.optim.Adam(resumed.parameters(), lr=.001)
        context, heldout = split_observations(self.data, 5151)
        first = slice(0, 1)
        second = slice(1, 2)
        update(reference, optimizer, context.batch(first), heldout.batch(first))
        update(resumed, resume_optimizer, context.batch(first), heldout.batch(first))
        stream = io.BytesIO()
        torch.save(dict(model=resumed.state_dict(), optimizer=resume_optimizer.state_dict()), stream)
        stream.seek(0)
        saved = torch.load(stream, map_location='cpu', weights_only=False)
        restored = RecurrentHankelEstimator(stages=3, hidden_size=4).double()
        restored.load_state_dict(saved['model'])
        restored_optimizer = torch.optim.Adam(restored.parameters(), lr=.001)
        restored_optimizer.load_state_dict(saved['optimizer'])
        left = update(reference, optimizer, context.batch(second), heldout.batch(second))
        right = update(restored, restored_optimizer, context.batch(second), heldout.batch(second))
        torch.testing.assert_close(left, right, rtol=0., atol=0.)
        for (name, parameter), (other, value) in zip(reference.named_parameters(),
                                                   restored.named_parameters()):
            self.assertEqual(name, other)
            torch.testing.assert_close(parameter, value, rtol=0., atol=0.)

    def test_output_weight_gradient_matches_central_difference(self):
        model = RecurrentHankelEstimator(stages=3, hidden_size=4).double()
        with torch.no_grad():
            output_layer(model).weight.normal_(mean=0., std=.05)
        context, heldout = split_observations(self.data, 7211)

        def loss_value():
            estimate = model(*context.args())
            return nll(heldout.z, estimate @ heldout.s + heldout.b,
                       heldout.sigma2).mean()

        loss_value().backward()
        parameter = output_layer(model).weight
        analytic = float(parameter.grad[0, 0])
        initial = float(parameter[0, 0].detach())
        epsilon = 1e-5
        with torch.no_grad():
            parameter[0, 0] = initial + epsilon
            upper = float(loss_value())
            parameter[0, 0] = initial - epsilon
            lower = float(loss_value())
            parameter[0, 0] = initial
        numeric = (upper - lower) / (2 * epsilon)
        self.assertGreater(abs(analytic), 1e-10)
        self.assertAlmostEqual(analytic, numeric, delta=max(1e-7, 3e-4 * abs(numeric)))


if __name__ == '__main__':
    unittest.main()
