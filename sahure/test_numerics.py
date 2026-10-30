"""Run: python -m unittest sahure.test_numerics -v"""
import unittest
import tempfile
from pathlib import Path
import numpy as np
from scipy.integrate import quad
from scipy.stats import rice
import torch
from .math import nll, posterior_field, wirtinger, right_solve, spectral_initial, hankel, unhankel, smooth_shrink
from .model import AdaptiveHankelEstimator,ModelConfig
from .data import Observations,split_observations,generate,load_observations


torch.set_num_threads(1)

class NumericalTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(101)
        self.s=torch.randn(2,3,12,dtype=torch.complex128)
        self.g=torch.randn(2,8,3,dtype=torch.complex128)
        self.b=torch.randn(2,8,12,dtype=torch.complex128)*2
        self.v=torch.tensor([.2,2.],dtype=torch.float64)
        self.z=(self.g@self.s+self.b+torch.randn(2,8,12,dtype=torch.complex128)*self.v.sqrt()[:,None,None]).abs()
        self.data=Observations(self.z,self.s,self.b,self.v)

    def test_density(self):
        for a,v in [(0.,.7),(1e-10,2.),(3.,.2),(20.,.01)]:
            z=np.linspace(max(1e-5,a-8*np.sqrt(v)),a+8*np.sqrt(v),100)
            t=nll(torch.tensor(z),torch.full((100,),complex(a),dtype=torch.complex128),v,full=True)
            np.testing.assert_allclose(t.numpy(),-rice.logpdf(z,a/np.sqrt(v/2),scale=np.sqrt(v/2)),rtol=1e-9,atol=1e-9)
            def density(z):
                return float(torch.exp(-nll(torch.tensor(z,dtype=torch.float64),torch.tensor(complex(a)),v,full=True)))
            integral=quad(density,max(0,a-10*np.sqrt(v)),a+10*np.sqrt(v),epsabs=1e-7)[0]
            self.assertAlmostEqual(integral,1.,places=6)

    def test_complex_noise_variance(self):
        w=torch.randn(400000,dtype=torch.complex128)*np.sqrt(2.3)
        self.assertAlmostEqual(float(w.abs().square().mean()),2.3,delta=.02)
        self.assertAlmostEqual(float(w.real.square().mean()),1.15,delta=.01)

    def test_analytic_autograd_and_finite_difference(self):
        g=self.g.clone().requires_grad_()
        loss=nll(self.z,g@self.s+self.b,self.v).sum()
        grad=torch.autograd.grad(loss,g)[0]
        torch.testing.assert_close(grad,2*wirtinger(g,self.z,self.s,self.b,self.v),rtol=1e-9,atol=1e-9)
        direction=torch.randn_like(g); eps=1e-6
        fd=(nll(self.z,(g+eps*direction)@self.s+self.b,self.v).sum()-nll(self.z,(g-eps*direction)@self.s+self.b,self.v).sum())/(2*eps)
        torch.testing.assert_close(fd,(grad.conj()*direction).real.sum(),rtol=1e-7,atol=1e-7)

    def test_zero_weak_and_high_fields(self):
        for dtype in (torch.complex64,torch.complex128):
            mu=torch.tensor([0,1e-20+1e-20j,1e-6j,1+2j,1e4+1e4j],dtype=dtype,requires_grad=True)
            z=torch.tensor([1.,1.,1.,3.,1e4],dtype=mu.real.dtype)
            v=.3
            grad=torch.autograd.grad(nll(z,mu,v).sum(),mu)[0]
            expected=2*(mu-posterior_field(z,mu,v))/v
            self.assertTrue(torch.isfinite(grad).all())
            torch.testing.assert_close(grad,expected,rtol=3e-5,atol=2e-5)
            qgrad=torch.autograd.grad(posterior_field(z,mu,v).real.sum(),mu)[0]
            self.assertTrue(torch.isfinite(qgrad).all())

    def test_em_preconditioner(self):
        mu=self.g@self.s+self.b; q=posterior_field(self.z,mu,self.v); gram=self.s@self.s.mH
        em=right_solve((q-self.b)@self.s.mH,gram)
        expr=self.g-right_solve(self.v[:,None,None]*wirtinger(self.g,self.z,self.s,self.b,self.v),gram)
        torch.testing.assert_close(em,expr,rtol=1e-12,atol=1e-12)

    def test_repository_parity(self):
        from codex_verification.solver import initial,fit
        g0=spectral_initial(self.z,self.s,self.b)
        result=AdaptiveHankelEstimator(ModelConfig(structure='none',stages=10)).double()(*self.data.args())
        for i in range(2):
            z,s,b=self.z[i].numpy(),self.s[i].numpy(),self.b[i].numpy()
            np.testing.assert_allclose(g0[i].numpy(),initial(s,z,b),rtol=1e-11,atol=1e-11)
            np.testing.assert_allclose(result[i].detach().numpy(),fit(s,z,b,float(self.v[i]),iterations=10)[0],rtol=1e-10,atol=1e-10)

    def test_hankel_inverse_and_shrinkage(self):
        h=hankel(self.g.mT)
        torch.testing.assert_close(unhankel(h),self.g.mT,rtol=1e-14,atol=1e-14)
        out,tau=smooth_shrink(h,torch.full((2,3),.2,dtype=torch.float64))
        u,s,vh=torch.linalg.svd(h,full_matrices=False)
        expected=(u*(s**3/(s*s+tau[...,None]**2))[...,None,:])@vh
        torch.testing.assert_close(out,expected,rtol=1e-11,atol=1e-11)

    def test_repeated_spectrum_gradients(self):
        for delta in (0.,1e-10):
            h=(torch.eye(4,dtype=torch.complex128)+delta*torch.randn(4,4,dtype=torch.complex128)).requires_grad_()
            sv=torch.linalg.svdvals(h); p=sv/sv.sum()
            t=.1+.01*(-(p*p.log()).sum()).exp()
            out,_=smooth_shrink(h,t)
            grad=torch.autograd.grad(out.abs().square().sum(),h)[0]
            self.assertTrue(torch.isfinite(grad).all())
        self.assertTrue(torch.autograd.gradcheck(lambda x: smooth_shrink(x,torch.tensor(.2))[0],(h,),eps=1e-6,atol=1e-5))

    def test_zero_gate_and_user_permutation(self):
        for core in ('em','gd'):
            bare=AdaptiveHankelEstimator(ModelConfig(core=core,structure='none')).double()
            adaptive=AdaptiveHankelEstimator(ModelConfig(core=core)).double()
            # Nonconstant outputs exercise the feature path too.
            with torch.no_grad(): adaptive.net[-1].weight.normal_(0,.02)
            torch.testing.assert_close(bare(*self.data.args()),adaptive(*self.data.args(),gate_override=0),rtol=1e-12,atol=1e-12)
            order=[2,0,1]
            torch.testing.assert_close(adaptive(self.z,self.s[:,order],self.b,self.v),adaptive(*self.data.args())[:,:,order],rtol=1e-9,atol=1e-9)

    def test_heldout_isolation(self):
        a,b=split_observations(self.data,40)
        model=AdaptiveHankelEstimator().double()
        estimate=model(*a.args())
        b.z.mul_(7)
        torch.testing.assert_close(estimate,model(*a.args()),rtol=0,atol=0)

    def test_no_truth_training_interface(self):
        with tempfile.TemporaryDirectory() as tmp:
            generate(tmp,'train',2,seed=983,N=8,K=2,P=12)
            obs=load_observations(Path(tmp)/'train_observations.npz',torch.float64)
            a,b=split_observations(obs,43)
            model=AdaptiveHankelEstimator().double()
            prediction=model(*a.args()); score=nll(b.z,prediction@b.s+b.b,b.sigma2).mean()
            np.savez(Path(tmp)/'train_truth.npz',g=np.zeros((2,8,2)))
            again=load_observations(Path(tmp)/'train_observations.npz',torch.float64)
            aa,bb=split_observations(again,43)
            pred2=model(*aa.args()); score2=nll(bb.z,pred2@bb.s+bb.b,bb.sigma2).mean()
            torch.testing.assert_close(prediction,pred2,rtol=0,atol=0)
            self.assertEqual(float(score.detach()),float(score2.detach()))

    def test_float32_against_float64(self):
        for core in ('em','gd'):
            torch.manual_seed(48)
            model=AdaptiveHankelEstimator(ModelConfig(core=core))
            args=tuple(x.to(torch.complex64 if x.is_complex() else torch.float32) for x in self.data.args())
            out=model(*args); loss=nll(args[0],out@args[1]+args[2],args[3]).mean();loss.backward()
            gradients=[p.grad.clone() for p in model.parameters()]
            model.zero_grad();model.double()
            out64=model(*self.data.args()); loss64=nll(self.z,out64@self.s+self.b,self.v).mean();loss64.backward()
            torch.testing.assert_close(out.double() if not out.is_complex() else out.cdouble(),out64,rtol=2e-4,atol=2e-5)
            for a,p in zip(gradients,model.parameters()):
                self.assertTrue(torch.isfinite(a).all())
                torch.testing.assert_close(a.double(),p.grad,rtol=3e-3,atol=2e-5)

if __name__=='__main__': unittest.main()
