import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from .data import Observations,split_observations
from .model import AdaptiveHankelEstimator,ModelConfig
from .train import heldout_loss,validation_score,choose_checkpoint,train_models,load_model
from .classical import gradient_fit,linearized_ls,likelihood
from .evaluate import constant_control

class PipelineTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(62);torch.set_num_threads(1)
        self.s=torch.randn(4,2,10,dtype=torch.complex128)
        self.g=torch.randn(4,8,2,dtype=torch.complex128)
        self.b=torch.ones(4,8,10,dtype=torch.complex128)*4
        self.z=(self.g@self.s+self.b+.1*torch.randn(4,8,10,dtype=torch.complex128)).abs()
        self.data=Observations(self.z,self.s,self.b,torch.full((4,),.01,dtype=torch.float64))

    def test_masks_reproducible_disjoint_and_resampled(self):
        markers=torch.arange(10).expand(4,8,-1).clone()
        d=Observations(markers,self.s,self.b,self.data.sigma2)
        a,b=split_observations(d,1);aa,bb=split_observations(d,1);c,_=split_observations(d,2)
        torch.testing.assert_close(a.z,aa.z);torch.testing.assert_close(b.z,bb.z)
        self.assertFalse(torch.equal(a.z,c.z))
        for i in range(4): self.assertEqual(len(set(a.z[i,0].tolist())&set(b.z[i,0].tolist())),0)

    def test_single_world_public_interface(self):
        model=AdaptiveHankelEstimator().double()
        out=model(*self.data.args())
        one,trace=model.estimate(self.z[0].numpy(),self.s[0].numpy(),self.b[0].numpy(),.01,diagnostics=True)
        torch.testing.assert_close(out[0],one,rtol=1e-10,atol=1e-10)
        self.assertEqual(one.shape,(8,2));self.assertEqual(trace[-1]['beta'].shape,(2,))

    def test_cli_can_set_threads_before_import(self):
        import subprocess,sys
        subprocess.run([sys.executable,'-c',"import sys,sahure; assert 'torch' not in sys.modules; assert 'numpy' not in sys.modules"],check=True)

    def test_gd_parity(self):
        model=AdaptiveHankelEstimator(ModelConfig(core='gd',structure='none')).double()
        out=model(*self.data.args()).detach().numpy()
        for i in range(4):
            z,s,b,v=(x[i].numpy() for x in self.data.args())
            expected,_=gradient_fit(z,s,b,float(v),iterations=10)
            np.testing.assert_allclose(out[i],expected[0],rtol=1e-10,atol=1e-10)

    def test_linearized_exact_linear_system(self):
        g,s,b=self.g[0].numpy(),self.s[0].numpy(),self.b[0].numpy()
        z=abs(b)+(np.exp(-1j*np.angle(b))*(g@s)).real
        np.testing.assert_allclose(linearized_ls(z,s,b),g,rtol=1e-12,atol=1e-12)

    def test_validation_constant_intervention(self):
        adaptive=AdaptiveHankelEstimator().double()
        control,means=constant_control(adaptive,self.data)
        # Initially the network is constant, so intervention must be identical.
        torch.testing.assert_close(adaptive(*self.data.args()),control(*self.data.args()),rtol=1e-10,atol=1e-10)
        self.assertAlmostEqual(means[0][0],.1,places=6)

    def test_actual_training_and_reload(self):
        import time
        data=Observations(*(x.to(torch.complex64 if x.is_complex() else torch.float32) for x in self.data.args()))
        with tempfile.TemporaryDirectory() as tmp:
            config=dict(seed=4,stages=2,lr=.001,batch_size=2,epochs=1)
            self.assertTrue(train_models(data,data,Path(tmp),config,time.monotonic()+60))
            for core in ('em','gd'):
                for structure in ('none','global','adaptive'):
                    name=f'{core}_{structure}'
                    model,checkpoint=load_model(Path(tmp)/f'{name}_best.pt')
                    score=validation_score(model,self.data,30004)
                    self.assertAlmostEqual(score,checkpoint['best_score'],delta=1e-3)
                    self.assertTrue(torch.isfinite(model(*self.data.args())).all())
            self.assertTrue(choose_checkpoint(1.,2.));self.assertFalse(choose_checkpoint(2.,1.))

    def test_bootstrap_pairing(self):
        from .report import paired_summary
        error=np.array([1.,2.,4.,8.]);energy=np.ones(4)*10
        indices=np.random.default_rng(13).integers(4,size=(100,4))
        x=paired_summary(error,2*error,energy,indices)
        self.assertAlmostEqual(x['gain_vs_hs_db'],10*np.log10(2))
        np.testing.assert_allclose(x['gain_vs_hs_ci'],[10*np.log10(2)]*2,atol=1e-12)

if __name__=='__main__': unittest.main()
