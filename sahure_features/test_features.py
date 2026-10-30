"""Run before freezing. Tests never load channel labels or old test data."""
import copy
import tempfile
from pathlib import Path
from unittest.mock import patch
import unittest
import torch
from sahure.data import Observations, split_observations
from sahure.model import AdaptiveHankelEstimator
from sahure.train import heldout_loss, train_epoch
from .model import FeatureEstimator,MASKS

def sample():
    torch.manual_seed(814)
    return Observations(torch.rand(8,8,12)+1,torch.randn(8,2,12,dtype=torch.complex64),
                        torch.ones(8,8,12,dtype=torch.complex64)*2,torch.ones(8))

class FeatureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_full_forward_and_backward_equivalence(self):
        torch.manual_seed(22);old=AdaptiveHankelEstimator().double()
        torch.manual_seed(22);new=FeatureEstimator('full').double()
        # Nonzero controller weights avoid a trivial constant-output comparison.
        with torch.no_grad():old.net[-1].weight.normal_(0,.03)
        new.load_state_dict(old.state_dict())
        d=sample();args=(d.z.double(),d.s.to(torch.complex128),d.b.to(torch.complex128),d.sigma2.double())
        x=old(*args);y=new(*args)
        torch.testing.assert_close(x,y,rtol=0,atol=0)
        x.abs().square().mean().backward();y.abs().square().mean().backward()
        for p,q in zip(old.parameters(),new.parameters()):torch.testing.assert_close(p.grad,q.grad,rtol=0,atol=0)

    def test_masked_inputs_are_invisible(self):
        for arm,mask in MASKS.items():
            model=FeatureEstimator(arm)
            with torch.no_grad():model.net[-1].weight.normal_()
            x=torch.randn(3,8,requires_grad=True);alter=x.detach().clone()
            alter[:,torch.tensor(mask)==0]+=50
            torch.testing.assert_close(model.net(x),model.net(alter),rtol=0,atol=0)
            model.net(x).sum().backward()
            self.assertTrue(torch.all(x.grad[:,torch.tensor(mask)==0]==0))
            self.assertEqual(sum(p.numel() for p in model.parameters()),188)

    def test_masks_and_order_and_resume(self):
        d=sample()
        for arm in MASKS:
            torch.manual_seed(81);model=FeatureEstimator(arm);opt=torch.optim.Adam(model.parameters(),lr=.001)
            train_epoch(model,opt,d,0,81,4,float('inf'))
            saved=copy.deepcopy((model.state_dict(),opt.state_dict()))
            train_epoch(model,opt,d,1,81,4,float('inf'))
            restored=FeatureEstimator(arm);other=torch.optim.Adam(restored.parameters(),lr=.001)
            restored.load_state_dict(saved[0]);other.load_state_dict(saved[1])
            train_epoch(restored,other,d,1,81,4,float('inf'))
            for p,q in zip(model.parameters(),restored.parameters()):torch.testing.assert_close(p,q,rtol=0,atol=0)

    def test_split_and_finite_gradients(self):
        d=sample();a,b=split_observations(d,192)
        # Use unique pilot identifiers to verify disjointness directly.
        identifiers=torch.arange(12).float().reshape(1,1,12).expand(8,8,12)
        ia,ib=split_observations(Observations(identifiers,d.s,d.b,d.sigma2),192)
        for row in range(8):self.assertFalse(set(ia.z[row,0].tolist())&set(ib.z[row,0].tolist()))
        for arm in MASKS:
            model=FeatureEstimator(arm);loss=heldout_loss(model,a,b);loss.backward()
            self.assertTrue(torch.isfinite(loss))
            self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_end_to_end_isolated_smoke(self):
        import numpy as np
        from . import train,evaluate,protocol
        from sahure_expansion.protocol import write_json
        d=sample();a,b=split_observations(d,11)
        cell=dict(id='smoke',N=8,K=2,P=12,snr=5.,rsr=12.,L=3,channel='ula',primary=True)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            write_json(root/'protocol.json',dict(smoke=True))
            with patch.dict(protocol.CONFIG,dict(seeds=[7],epochs=1,train_cases=8,val_cases=8,test_cases=4,test_banks=[1],bootstrap_resamples=32)), \
                 patch.object(train,'ROOT',root),patch.object(evaluate,'ROOT',root), \
                 patch.object(protocol,'ROOT',root),patch.object(train,'data',return_value=(d,a,b,np.zeros(8,dtype=int))), \
                 patch.object(evaluate,'check_sources',return_value=dict(sources={},inputs={})), \
                 patch.object(evaluate,'cells',return_value=[cell]):
                self.assertTrue(train.run(float('inf')))
                self.assertTrue(evaluate.run(float('inf')))
                evaluate.audit();evaluate.summarize()
                self.assertTrue((root/'REPORT.md').exists())
                self.assertEqual(len(list((root/'evaluation').rglob('*.npz'))),8)

if __name__=='__main__':unittest.main()
