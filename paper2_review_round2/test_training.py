import unittest
import tempfile
import time
from pathlib import Path
import numpy as np
import torch
from sahure.data import Observations
from .models import ReviewEstimator,ARMS
from .training_study import epoch_order,train_epoch,validate,atomic_torch


class TrainingChecks(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42);n=32
        s=torch.randn(n,3,30,dtype=torch.complex64);b=torch.ones(n,32,30,dtype=torch.complex64)*3
        g=torch.randn(n,32,3,dtype=torch.complex64)
        z=abs(g@s+b+torch.randn(n,32,30,dtype=torch.complex64))
        self.data=Observations(z,s,b,torch.ones(n));self.rsr=torch.tensor([0,5,12,20]*8)
    def test_curriculum_total_exposure(self):
        exposure=torch.zeros(32,dtype=torch.long)
        for e in range(12):
            order=epoch_order(self.data,self.rsr,e,91,'curriculum')
            self.assertEqual(len(order),32)
            exposure+=torch.bincount(order,minlength=32)
        torch.testing.assert_close(exposure,torch.full((32,),12,dtype=torch.long))
        for arm in ('baseline','depth5','width32','experts'):
            torch.testing.assert_close(epoch_order(self.data,self.rsr,2,91,arm),epoch_order(self.data,self.rsr,2,91,'baseline'))
    def test_checkpoint_resume_is_exact(self):
        torch.manual_seed(5);m=ReviewEstimator(ARMS['baseline']);o=torch.optim.Adam(m.parameters(),lr=.001)
        train_epoch(m,o,self.data,self.rsr,0,91,'baseline',time.monotonic()+30)
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'c.pt';atomic_torch(p,dict(model=m.state_dict(),optimizer=o.state_dict()))
            a=train_epoch(m,o,self.data,self.rsr,1,91,'baseline',time.monotonic()+30)
            r=ReviewEstimator();ro=torch.optim.Adam(r.parameters(),lr=.001)
            ck=torch.load(p,weights_only=False);r.load_state_dict(ck['model']);ro.load_state_dict(ck['optimizer'])
            b=train_epoch(r,ro,self.data,self.rsr,1,91,'baseline',time.monotonic()+30)
            self.assertEqual(a,b)
            for key in m.state_dict():torch.testing.assert_close(m.state_dict()[key],r.state_dict()[key],rtol=0,atol=0)
            self.assertTrue(np.isfinite(validate(r,self.data,np.arange(32)%4,81)))

if __name__=='__main__':unittest.main()
