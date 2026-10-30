import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import paper2_hankel_confirmation as s
from rydberg_sim.track_b_proposed import hs_gs


class ConfirmationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(1)

    def test_matching_and_degenerations(self):
        with s.configured(s.MODES[0],s.SEEDS[0]):
            data=s.Worlds(s.MODES[0],'train',4); batch=s.base.collate([data[i] for i in range(4)])
            args=[batch[k] for k in ('G0','Z','S','B','sigma2')]; hashes=[]
            for arm in s.ARMS:
                model,_=s.build(arm); hashes.append(s.base.weight_hash(model))
                with torch.no_grad():
                    torch.testing.assert_close(model(*args),s.Fixed(10,arm.endswith('_H'))(*args),rtol=0,atol=0)
                s.train_step(model,torch.optim.Adam(model.parameters(),lr=.001),batch,s.base.cfg())
                for former in model.formers:
                    grad=former.output[-1].weight.grad
                    self.assertTrue(torch.isfinite(grad).all()); self.assertGreater(float(grad.abs().sum()),0.)
            self.assertEqual(len(set(hashes)),1)

    def test_data_and_test_lock(self):
        starts=[]
        for mode in s.MODES:
            for split in s.COUNTS:
                d=s.Worlds(mode,split,12); starts.append(int(d[0]['trial']))
                for k,v in d[3].items(): torch.testing.assert_close(v,d[3][k],rtol=0,atol=0)
                if split!='train':
                    bins=[int((float(d[i]['snr_db'])+10)//5) for i in range(12)]
                    self.assertEqual(bins,list(range(6))*2)
        self.assertEqual(len(set(starts)),8)
        with tempfile.TemporaryDirectory() as root,patch.object(s,'OUT',Path(root)):
            with self.assertRaises(AssertionError): s.cache(s.MODES[0],'test')

    def test_paper1_abstention(self):
        b=s.Worlds(s.MODES[0],'train',1)[0]; args=[b[k].numpy() for k in ('S','Z','B')]
        structured=hs_gs(*args,float(b['sigma2']),L_hat=16,max_iter=50).G_hat
        unstructured,_=s.native_record(b,'paper1_emgs50')
        np.testing.assert_array_equal(structured,unstructured)

    def test_resume_weighted_h(self):
        with tempfile.TemporaryDirectory() as root:
            states=[]
            for name in ('full','resume'):
                with patch.object(s,'OUT',Path(root)/name),patch.object(s,'COUNTS',dict(train=32,val=12,test=6,tune=6)):
                    data={split:s.cache(s.MODES[0],split) for split in ('train','val')}
                    spec=dict(train=32,val=12,epochs=2,start=s.STARTS[s.MODES[0]],snr=None)
                    with s.configured(s.MODES[0],s.SEEDS[0]):
                        if name=='resume': s.base.train_arm('development','weighted_H',spec,data,stop_after=1)
                        s.base.train_arm('development','weighted_H',spec,data)
                    states.append(torch.load(s.folder(s.MODES[0],s.SEEDS[0],'weighted_H')/'checkpoint.pt',weights_only=False))
            for k in states[0]['model']: torch.testing.assert_close(states[0]['model'][k],states[1]['model'][k],rtol=0,atol=0)

    def test_budget_guard(self):
        with tempfile.TemporaryDirectory() as root,patch.multiple(s,OUT=Path(root),SEGMENT_START=0.,PRIOR_SECONDS=0.,WORKER_LIMIT=1.):
            with self.assertRaises(s.BudgetReached): s.check_budget(force=True)

    def test_end_to_end_small(self):
        with tempfile.TemporaryDirectory() as root,patch.multiple(s,OUT=Path(root),MODES=('literal_independent',),
             SEEDS=(2026092601,),COUNTS=dict(train=32,val=6,test=6,tune=6),EPOCHS=1):
            mode=s.MODES[0]; seed=s.SEEDS[0]
            data={split:s.cache(mode,split) for split in ('train','val')}
            spec=dict(train=32,val=6,epochs=1,start=s.STARTS[mode],snr=None)
            with s.configured(mode,seed):
                for arm in s.ARMS: s.base.train_arm('development',arm,spec,data)
            audit=s.audit_training(); selection=s.freeze_selection(); s.test_all(selection)
            audit.update(s.audit_test()); result=s.analyze(selection); s.report(result,audit)
            self.assertEqual(audit['runs'],4); self.assertEqual(audit['test_files_verified'],15)
            self.assertTrue((Path(root)/'RESULTS_REPORT.md').exists())
            self.assertEqual(s.freeze_selection(),selection)


if __name__=='__main__': unittest.main()
