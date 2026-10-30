import copy
import json
import unittest
import numpy as np
import torch
from sahure_expansion.data import world
from sahure.data import Observations,split_observations
from sahure.math import nll
from codex_verification.solver import fit
from paper2_optshrink_loop.estimator import estimate,weights
from .model import OptController,smooth_gate,opt_filter,opt_weights,OLD_SELECTION


def obs(worlds):
    return Observations(torch.tensor(np.stack([w.Z for w in worlds])),
        torch.tensor(np.stack([w.S for w in worlds])),torch.tensor(np.stack([w.B for w in worlds])),
        torch.tensor([w.sigma2 for w in worlds],dtype=torch.float64))


class Checks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.snrs=(-10,-5,0,5,10,15,20,60)
        cls.worlds=[world(i,202610116801,snr=s)[0] for i,s in enumerate(cls.snrs)]
        cls.obs=obs(cls.worlds)
        cls.old=json.loads(OLD_SELECTION.read_text())['noise_aware']

    def test_strong_rank_and_iterations_numpy_parity(self):
        fixed=OptController('fixed').double()
        actual,diag=fixed(*self.obs.args(),diagnostics=True)
        for i,(snr,w) in enumerate(zip(self.snrs,self.worlds)):
            cfg=self.old[str(min(snr,20))]['loop']
            expected=estimate(w.Z,w.S,w.B,w.sigma2,cfg)
            np.testing.assert_allclose(actual[i].detach().numpy(),expected,rtol=2e-8,atol=2e-9)
            self.assertEqual(float(diag[i,:,0,2].sum()),cfg['iterations'])
            idx=int((fixed.levels-min(snr,20)).abs().argmin())
            self.assertEqual(int(fixed.ranks[idx]),cfg['rank'])

    def test_mixed_batch_equals_single_cases(self):
        m=OptController('adaptive').double()
        with torch.no_grad():m.net[-1].bias.fill_(-.3)
        grouped=m(*self.obs.args())
        for i in range(len(self.obs)):
            single=m(*self.obs.batch(slice(i,i+1)).args())
            torch.testing.assert_close(grouped[i],single[0],rtol=2e-8,atol=2e-9)

    def test_zero_correction_is_em(self):
        w=self.worlds[2]
        actual=OptController('fixed',stages=20).double()(*obs([w]).args(),strength_override=0.)
        np.testing.assert_allclose(actual.detach().numpy()[0],fit(w.S,w.Z,w.B,w.sigma2,iterations=20)[0],rtol=2e-9,atol=2e-10)

    def test_initialization_controls(self):
        fixed=OptController('fixed').double()
        expected=fixed(*self.obs.args())
        torch.testing.assert_close(OptController('clipped').double()(*self.obs.args()),expected,rtol=1e-10,atol=1e-11)
        idx=(10*torch.log10(3/self.obs.sigma2)[:,None]-fixed.levels).abs().argmin(-1)
        near=fixed(*self.obs.args(),strength_override=fixed.strengths[idx,None].clamp(.05,.95))
        for name in ('adaptive','global'):
            torch.testing.assert_close(OptController(name).double()(*self.obs.args()),near,rtol=2e-9,atol=2e-10)

    def test_smooth_full_range_and_nonzero_slope(self):
        raw=torch.tensor([-12.,0.,12.],dtype=torch.float64,requires_grad=True)
        beta,slope=smooth_gate(raw,torch.ones_like(raw))
        self.assertLess(float(beta[0].detach()),.001)
        self.assertGreater(float(beta[-1].detach()),.999)
        self.assertTrue(torch.all(slope>0))
        beta.sum().backward();torch.testing.assert_close(raw.grad,slope)
        self.assertTrue(torch.autograd.gradcheck(lambda x:smooth_gate(x,torch.ones_like(x))[0],(raw,)))

    def test_native_filter_forward_gradient_scale(self):
        gen=torch.Generator().manual_seed(87)
        h=torch.randn(5,4,dtype=torch.complex128,generator=gen,requires_grad=True)
        fn=lambda x:opt_filter(x,torch.tensor(2))[0]
        self.assertTrue(torch.autograd.gradcheck(fn,(h,),eps=1e-6,atol=2e-5,rtol=2e-4))
        torch.testing.assert_close(fn(3*h),3*fn(h),rtol=1e-10,atol=1e-11)
        for x in (torch.zeros_like(h),torch.eye(5,4,dtype=torch.complex128)):
            self.assertTrue(torch.isfinite(fn(x)).all())
        s=torch.linalg.svdvals(h).detach()
        np.testing.assert_allclose(opt_weights(s,5,4,torch.tensor(2)).numpy(),weights(s.numpy(),5,4,2),rtol=1e-11,atol=1e-12)

    def test_unrolled_directional_derivative(self):
        a,b=split_observations(obs([self.worlds[2]]),421)
        m=OptController('adaptive',stages=3).double()
        def loss():return nll(b.z,m(*a.args())@b.s+b.b,b.sigma2).mean()
        loss().backward();p=m.net[-1].bias;derivative=float(p.grad[0]);eps=1e-5
        with torch.no_grad():p.fill_(eps)
        plus=float(loss().detach())
        with torch.no_grad():p.fill_(-eps)
        minus=float(loss().detach())
        np.testing.assert_allclose(derivative,(plus-minus)/(2*eps),rtol=2e-4,atol=2e-6)

    def test_gradients_every_snr_and_mask_isolation(self):
        for i in range(7):
            a,b=split_observations(obs([self.worlds[i]]),311)
            m=OptController('adaptive').double()
            g,diag=m(*a.args(),diagnostics=True)
            self.assertTrue(torch.all(diag[...,3][diag[...,2]>0]>0))
            loss=nll(b.z,g@b.s+b.b,b.sigma2).mean();loss.backward()
            norm=0.
            for p in m.parameters():
                if p.grad is not None:
                    self.assertTrue(torch.isfinite(p.grad).all());norm+=float(p.grad.square().sum())
            self.assertGreater(norm,1e-20)
            b.z.mul_(2)
            torch.testing.assert_close(m(*a.args()),g,rtol=0,atol=0)

    def test_optimizer_resume(self):
        a,b=split_observations(obs([self.worlds[2]]),512)
        m=OptController('adaptive',stages=2).double();o=torch.optim.Adam(m.parameters(),lr=.001)
        def step(model,optimizer):
            optimizer.zero_grad(set_to_none=True)
            nll(b.z,model(*a.args())@b.s+b.b,b.sigma2).mean().backward();optimizer.step()
        step(m,o);state=copy.deepcopy(m.state_dict());opt=copy.deepcopy(o.state_dict());step(m,o)
        resumed=OptController('adaptive',stages=2).double();resumed.load_state_dict(state)
        ro=torch.optim.Adam(resumed.parameters(),lr=.001);ro.load_state_dict(opt);step(resumed,ro)
        for key,v in m.state_dict().items():torch.testing.assert_close(v,resumed.state_dict()[key],rtol=0,atol=0)


if __name__=='__main__':unittest.main()
