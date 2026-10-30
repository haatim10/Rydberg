import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import tempfile
import json
import numpy as np
import torch
from sahure.data import Observations,split_observations
from sahure.model import AdaptiveHankelEstimator,ModelConfig
from sahure.train import heldout_loss
from .data import allocation,condition_grid,world
from .train import condition_mean
from .protocol import CONFIG,cells,write_json


class ExpansionChecks(unittest.TestCase):
    def test_seed_replication_does_not_narrow_world_interval(self):
        from .report import seed_nmse_and_bootstrap
        rng=np.random.default_rng(14);energy=np.ones(40)
        error=rng.exponential(size=(1,40))
        counts=rng.multinomial(40,np.ones(40)/40,size=1000)
        _,single=seed_nmse_and_bootstrap(error,energy,counts)
        _,repeated=seed_nmse_and_bootstrap(np.repeat(error,3,0),energy,counts)
        np.testing.assert_allclose(single[0],repeated.mean(0),rtol=1e-12,atol=1e-12)

    def test_mean_seed_db_is_not_prediction_ensemble(self):
        from .report import seed_nmse_and_bootstrap
        point,_=seed_nmse_and_bootstrap(np.array([[1.,1.],[100.,100.]]),np.ones(2),np.array([[1.,1.]]))
        self.assertAlmostEqual(float(point.mean()),10.)

    def test_concurrent_json_writers(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'config.json'
            with ThreadPoolExecutor(max_workers=3) as pool:
                list(pool.map(lambda _:write_json(path,CONFIG),range(30)))
            self.assertEqual(json.loads(path.read_text()),CONFIG)

    def test_validation_weights(self):
        v=torch.tensor([1.,1.,1.,9.],requires_grad=True)
        score,means=condition_mean(v,[0,0,0,1])
        self.assertEqual(float(score.detach()),5.)
        score.backward()
        torch.testing.assert_close(v.grad,torch.tensor([1/6,1/6,1/6,1/2]))

    def test_allocation(self):
        grid=condition_grid()
        self.assertEqual(len(grid),126)
        for n in [16384,2048]:
            ids=allocation(n,CONFIG['train_data_seed'])
            self.assertEqual(sum(grid[i]['L'] is None for i in ids),n//2)
            self.assertEqual(len(np.unique(ids)),126)
            np.testing.assert_array_equal(ids,allocation(n,CONFIG['train_data_seed']))

    def test_test_axis_deduplication(self):
        points,axes=cells()
        base=[p['id'] for p in points if p['N']==32 and p['K']==3 and p['P']==30 and p['snr']==5 and p['rsr']==12 and p['L'] is None and p['channel']=='ula' and p['noise_offset']==0]
        self.assertEqual(len(base),1)
        for axis in axes.values(): self.assertTrue(all(x['cell'] in {p['id'] for p in points} for x in axis))

    def test_mixed_extremes_precision_and_gradients(self):
        worlds=[world(i,2026092255,snr=snr,rsr=rsr,L=L)[0]
                for i,(snr,rsr,L) in enumerate([(-10,5,16),(20,20,2),(-10,20,12),(20,5,8)])]
        for core in ['em','gd']:
            torch.manual_seed(901)
            m=AdaptiveHankelEstimator(ModelConfig(core=core));md=AdaptiveHankelEstimator(ModelConfig(core=core)).double()
            md.load_state_dict(m.state_dict())
            outputs=[];grads=[]
            for model,rd,cd in [(m,torch.float32,torch.complex64),(md,torch.float64,torch.complex128)]:
                data=Observations(torch.tensor(np.stack([w.Z for w in worlds]),dtype=rd),
                    torch.tensor(np.stack([w.S for w in worlds]),dtype=cd),
                    torch.tensor(np.stack([w.B for w in worlds]),dtype=cd),
                    torch.tensor([w.sigma2 for w in worlds],dtype=rd))
                a,b=split_observations(data,901)
                loss=heldout_loss(model,a,b);loss.backward()
                g=torch.cat([p.grad.flatten() for p in model.parameters()])
                self.assertTrue(torch.isfinite(g).all())
                outputs.append(model(*a.args()).detach().to(torch.complex128));grads.append(g.double())
            torch.testing.assert_close(outputs[0],outputs[1],rtol=.005,atol=2e-4)
            torch.testing.assert_close(grads[0],grads[1],rtol=.05,atol=2e-3)


if __name__=='__main__':
    torch.set_num_threads(1);unittest.main()
