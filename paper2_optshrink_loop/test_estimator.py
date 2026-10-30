import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[key] = '1'
import unittest
import numpy as np
from codex_verification.solver import fit
from paper2_review_checks.optshrink_control import opt_weights, unit_checks
from .estimator import weights, shrink, trajectory, estimate, noise_index, CANDIDATES


class Checks(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(403)

    def complex(self, shape):
        return self.rng.normal(size=shape)+1j*self.rng.normal(size=shape)

    def inputs(self):
        g, s, b = self.complex((32, 3)), self.complex((3, 30)), self.complex((32, 30))
        z = abs(g@s+b+self.complex((32, 30))*.3)
        return z, s, b, .18

    def test_independent_prior_primitive(self):
        self.assertTrue(unit_checks()['all_passed'])
        for shape in [(17,16), (16,17), (8,8)]:
            n,m = shape
            sv = np.linalg.svd(self.complex((5, n, m)), compute_uv=False)
            for rank in range(1,min(shape)):
                actual = weights(sv,n,m,rank)
                expected = np.stack([opt_weights(row,n,m,rank) for row in sv])
                np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=2e-12)

    def test_repeated_and_zero(self):
        sv = np.array([[0.,0.,0.,0.], [2.,1.,1.,.2], [7.,3.,0.,0.]])
        for rank in (1,2,3):
            out = weights(sv,4,7,rank)
            expected = np.stack([opt_weights(row,4,7,rank) for row in sv])
            np.testing.assert_allclose(out,expected,atol=1e-12)

    def test_scaling(self):
        sv = np.sort(self.rng.uniform(.1,10,(3,16)),axis=-1)[:,::-1]
        base = weights(sv,17,16,np.array([2,7,15]))
        for scale in (1e-8,1e8):
            np.testing.assert_allclose(weights(sv*scale,17,16,np.array([2,7,15]))/scale,
                                       base,rtol=1e-11,atol=1e-11)

    def test_no_filter_matches_em(self):
        z,s,b,v = self.inputs()
        gs = trajectory(z,s,b,v,[CANDIDATES[0]],(1,10))
        for t in (1,10):
            np.testing.assert_allclose(gs[t][0],fit(s,z,b,v,iterations=t)[0],rtol=1e-12,atol=1e-12)

    def test_prefix_and_placement(self):
        z,s,b,v = self.inputs()
        cfg = dict(rank=7,strength=.5)
        both = trajectory(z,s,b,v,[cfg],(3,10))
        short = trajectory(z,s,b,v,[cfg],(3,))[3]
        np.testing.assert_array_equal(short,both[3])
        g = fit(s,z,b,v,iterations=10)[0]
        direct = shrink(g[None],[cfg])[0]
        np.testing.assert_allclose(direct,estimate(z,s,b,v,dict(**cfg,iterations=10,placement='final')))

    def test_mixing_and_noop(self):
        g = self.complex((32,3))
        configs = [dict(rank=7,strength=a) for a in (0.,.25,1.)]
        out = shrink(np.repeat(g[None],3,axis=0),configs)
        np.testing.assert_array_equal(out[0],g)
        np.testing.assert_allclose(out[1],.75*g+.25*out[2],atol=1e-12)
        np.testing.assert_array_equal(shrink(np.zeros((3,32,3),complex),configs),0)

    def test_receiver_noise_mapping(self):
        for i,snr in enumerate(range(-10,21,5)):
            self.assertEqual(noise_index(3/10**(snr/10)),i)

if __name__ == '__main__':
    unittest.main()
