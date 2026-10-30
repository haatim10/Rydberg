import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key]='1'
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from sahure.data import Observations,split_observations
from sahure.model import AdaptiveHankelEstimator
from sahure.train import heldout_loss,train_epoch
from sahure_expansion.data import world
from sahure_expansion.train import save_checkpoint
from .train import model_config,load_labels,channel_loss,supervised_epoch,order_hash
from .protocol import jobs,cells,CONFIG
from .evaluate import summarize_arrays,selected_models

def fixture():
    worlds=[world(i,2026092909,snr=s,rsr=r,L=L)[0] for i,(s,r,L) in enumerate([(-10,5,16),(20,20,2),(5,12,4),(0,5,12)])]
    data=Observations(torch.tensor(np.stack([w.Z for w in worlds]),dtype=torch.float32),
        torch.tensor(np.stack([w.S for w in worlds]),dtype=torch.complex64),
        torch.tensor(np.stack([w.B for w in worlds]),dtype=torch.complex64),torch.tensor([w.sigma2 for w in worlds]))
    return data,torch.tensor(np.stack([w.G for w in worlds]),dtype=torch.complex64)

class Checks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_scope(self):
        self.assertEqual(len(jobs()),15);self.assertEqual(len(cells()),9)
        self.assertEqual(sum(c['primary'] for c in cells()),7)
        self.assertEqual(sum(20 for j in jobs()),300)

    def test_label_guard(self):
        for arm in ['adaptive','global','none','final_only','gate_one']:
            with patch('numpy.load',side_effect=AssertionError('should not load')):
                with self.assertRaises(ValueError):load_labels('matched',arm,'train')

    def test_supervised_loss_definition(self):
        truth=torch.ones(4,32,3,dtype=torch.complex64)
        torch.testing.assert_close(channel_loss(truth,truth),torch.zeros(4))
        torch.testing.assert_close(channel_loss(torch.zeros_like(truth),truth),torch.ones(4))

    def test_only_final_structural_call(self):
        data,_=fixture();model=AdaptiveHankelEstimator(model_config('final_only'));seen=[]
        hook=model.net.register_forward_hook(lambda *args:seen.append(1))
        try:
            g,trace=model(*data.args(),diagnostics=True)
        finally:hook.remove()
        self.assertEqual(len(seen),1)
        for t in trace[:-1]:self.assertEqual(float(t['correction'].sum()),0.)
        self.assertTrue(torch.isfinite(g).all())

    def test_fixed_gate_matches_override(self):
        data,_=fixture();torch.manual_seed(9101)
        m=AdaptiveHankelEstimator(model_config('adaptive'))
        g=AdaptiveHankelEstimator(model_config('gate_one'));g.load_state_dict(m.state_dict())
        torch.testing.assert_close(m(*data.args(),gate_override=1.),g(*data.args()),rtol=0,atol=0)

    def test_arms_finite_loss_and_gradients(self):
        data,truth=fixture();a,b=split_observations(data,9101)
        for arm in ['supervised','final_only','gate_one','adaptive','global','none']:
            model=AdaptiveHankelEstimator(model_config(arm))
            loss=channel_loss(model(*a.args()),truth).mean() if arm=='supervised' else heldout_loss(model,a,b)
            loss.backward();self.assertTrue(torch.isfinite(loss))
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))

    def test_resume_exact(self):
        data,truth=fixture()
        for arm in ['supervised','final_only','gate_one']:
            torch.manual_seed(9101);m=AdaptiveHankelEstimator(model_config(arm));o=torch.optim.Adam(m.parameters(),lr=.001)
            def step(model,opt,epoch):
                if arm=='supervised':return supervised_epoch(model,opt,data,truth,epoch,9101,float('inf'))
                return train_epoch(model,opt,data,epoch,9101,32,float('inf'))
            step(m,o,0)
            with tempfile.TemporaryDirectory() as d:
                p=Path(d)/'checkpoint.pt';save_checkpoint(dict(model=m.state_dict(),opt=o.state_dict()),p)
                step(m,o,1)
                state=torch.load(p,weights_only=False);re=AdaptiveHankelEstimator(model_config(arm));ro=torch.optim.Adam(re.parameters(),lr=.001)
                re.load_state_dict(state['model']);ro.load_state_dict(state['opt']);step(re,ro,1)
                for key,value in m.state_dict().items():torch.testing.assert_close(value,re.state_dict()[key],rtol=0,atol=0)

    def test_seed_predictions_not_independent_worlds(self):
        e=np.array([[1.,2.,4.]]);energy=np.ones(3);counts=np.array([[1,1,1],[0,1,2],[2,1,0]])
        a,b=summarize_arrays(e,energy,counts);c,d=summarize_arrays(np.repeat(e,3,0),energy,counts)
        np.testing.assert_allclose(b,d);self.assertAlmostEqual(float(a.mean()),float(c.mean()),places=14)

    def test_bank_selection_and_shared_order(self):
        frozen=dict(models={'m':dict(cohort='matched'),'a':dict(cohort='rep1'),'b':dict(cohort='rep2')})
        self.assertEqual(set(selected_models(frozen,1)),{'m','a'})
        self.assertEqual(order_hash(20,9101,1),order_hash(20,9101,1))
        self.assertNotEqual(order_hash(20,9101,1),order_hash(20,9101,2))

    def test_training_driver_resume_and_labels(self):
        from . import train
        data,truth=fixture()
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'protocol.json').write_text('{}')
            inp=root/'inputs'/'data';inp.mkdir(parents=True)
            for split in ['train','val']:
                np.savez_compressed(inp/f'{split}_observations.npz',**{k:getattr(data,k).numpy() for k in ['z','s','b','sigma2']})
                np.savez_compressed(inp/f'{split}_truth.npz',g=truth.numpy())
            np.save(inp/'val_conditions.npy',np.arange(4))
            with patch.object(train,'ROOT',root),patch.object(train,'check_sources',return_value={}),patch.object(train,'data_root',return_value=root/'inputs'):
                for arm in ['supervised','final_only','gate_one']:
                    job=dict(id=arm,cohort='matched',arm=arm,seed=9101)
                    with patch.object(train,'load_labels',wraps=train.load_labels) as loader:
                        self.assertTrue(train.run(job,1,60));self.assertTrue(train.run(job,2,60))
                        self.assertEqual(loader.call_count,4 if arm=='supervised' else 0)
                    state=torch.load(root/'checkpoints'/arm/'latest.pt',weights_only=False)
                    self.assertEqual(state['epoch'],2);self.assertEqual(len(state['history']),2)

    def test_evaluation_resume(self):
        from . import evaluate
        data,_=fixture();data=Observations(data.z.double(),data.s.to(torch.complex128),data.b.to(torch.complex128),data.sigma2.double())
        truth=AdaptiveHankelEstimator(model_config('none')).double()(*data.args()).detach().numpy()
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);base=root/'test1'/'data';base.mkdir(parents=True)
            np.savez_compressed(base/'snr_5_truth.npz',g=truth)
            model=AdaptiveHankelEstimator(model_config('adaptive')).double();path=root/'model.pt'
            from dataclasses import asdict
            save_checkpoint(dict(model=model.state_dict(),model_config=asdict(model.config)),path)
            frozen=dict(models={'tiny':dict(path=str(path),cohort='matched')})
            with patch.object(evaluate,'ROOT',root),patch.object(evaluate,'freeze',return_value=frozen),patch.object(evaluate,'observations',return_value=data),patch.object(evaluate,'CLASSICAL',['em_10']):
                evaluate.run_cell(1,'snr_5');first=(root/'evaluation/bank1/snr_5/tiny.npz').read_bytes()
                evaluate.run_cell(1,'snr_5');self.assertEqual(first,(root/'evaluation/bank1/snr_5/tiny.npz').read_bytes())
                rows=(root/'evaluation/bank1/snr_5/classical.jsonl').read_text().splitlines()
                self.assertEqual(len(rows),4)

if __name__=='__main__':unittest.main()
