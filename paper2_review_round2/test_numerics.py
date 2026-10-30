import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key]='1'
import unittest
import numpy as np
import torch
from scipy.stats import rice
from codex_verification.solver import fit as old_fit
from sahure.model import AdaptiveHankelEstimator,ModelConfig
from .models import ReviewEstimator,ReviewConfig,ARMS
from .calibration import fit,nll,amplitude_jacobian,conditioning


class Checks(unittest.TestCase):
    def setUp(self):
        self.rng=np.random.default_rng(912)
        self.s=self.c((3,30));self.g=self.c((32,3));self.b=np.ones((32,30),complex)*2
        self.z=abs(self.g@self.s+self.b+self.c((32,30))*.4);self.v=.32
    def c(self,shape):return self.rng.normal(size=shape)+1j*self.rng.normal(size=shape)
    def test_full_likelihood(self):
        mu=self.g@self.s+self.b
        for v in (.1,1.,10.):
            exact=-rice.logpdf(self.z,abs(mu)/np.sqrt(v/2),scale=np.sqrt(v/2))
            np.testing.assert_allclose(nll(self.z,mu,v)-np.log(2*self.z),exact,rtol=1e-10,atol=1e-10)
    def test_no_calibration_matches_old(self):
        bank,_=fit(self.z,self.s,self.b,self.v,mode='none',budgets=(10,))
        np.testing.assert_allclose(bank[10]['g'],old_fit(self.s,self.z,self.b,self.v,iterations=10)[0],rtol=1e-12,atol=1e-12)
    def test_em_likelihood_decreases(self):
        for mode in ('none','noise','gain','joint'):
            _,hist=fit(self.z,self.s,self.b,self.v*4,mode=mode,budgets=(100,))
            self.assertLessEqual(float(np.diff(hist).max()),1e-9)
    def test_jacobian_finite_differences(self):
        a=amplitude_jacobian(self.g,self.s,self.b);eps=1e-6
        for j in range(6):
            delta=np.zeros_like(self.g);delta[:,j%3]=eps*(1 if j<3 else 1j)
            numeric=(abs((self.g+delta)@self.s+self.b)-abs((self.g-delta)@self.s+self.b))/(2*eps)
            np.testing.assert_allclose(a[:,:,j],numeric,rtol=1e-6,atol=1e-7)
    def test_phase_ambiguity_and_zero_reference_rank(self):
        phase=np.exp(.3j)
        np.testing.assert_allclose(abs(phase*self.g@self.s+phase*self.b),abs(self.g@self.s+self.b),atol=1e-12)
        self.assertTrue(np.all(conditioning(self.g,self.s,np.zeros_like(self.b))['rank']<=5))
    def test_baseline_exact_architecture(self):
        torch.manual_seed(81);a=AdaptiveHankelEstimator(ModelConfig())
        torch.manual_seed(81);b=ReviewEstimator()
        self.assertEqual(a.state_dict().keys(),b.state_dict().keys())
        for key in a.state_dict():torch.testing.assert_close(a.state_dict()[key],b.state_dict()[key],rtol=0,atol=0)
    def test_model_gradients_and_counts(self):
        counts={'baseline':188,'depth5':183,'depth20':198,'width8':100,'width32':364,'experts':374,'curriculum':188}
        for name,cfg in ARMS.items():
            model=ReviewEstimator(cfg)
            self.assertEqual(sum(p.numel() for p in model.parameters()),counts[name])
            g=model.estimate(self.z,self.s,self.b,self.v)
            g.abs().square().mean().backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))

if __name__=='__main__':unittest.main()
