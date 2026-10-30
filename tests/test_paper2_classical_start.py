import sys
import tempfile
import unittest
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import paper2_classical_start as study


class ClassicalStartTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def batch(self,snr=None):
        spec = dict(study.SPEC,snr=snr)
        return study.base.collate([study.base.LearningData('development','train',spec)[i] for i in range(4)])

    def test_exact_classical_initialization(self):
        with study.configured():
            model,_ = study.build('classical_start')
            for snr in (None,-10.,20.,60.):
                b = self.batch(snr); args = [b[k] for k in ('G0','Z','S','B','sigma2')]
                with torch.no_grad():
                    torch.testing.assert_close(model(*args),study.base.Classical(10)(*args),rtol=0,atol=0)

    def test_matching_nonoutput_weights_and_ordinary(self):
        with study.configured():
            ordinary,_ = study.build('ordinary'); candidate,_ = study.build('classical_start')
            for old,new in zip(ordinary.layers,candidate.formers):
                for name,value in old.former.state_dict().items():
                    if not name.startswith('output.2.'):
                        torch.testing.assert_close(value,new.state_dict()[name],rtol=0,atol=0)
            original,_ = study.base.repro.build_model(study.base.cfg())
            self.assertEqual(study.base.weight_hash(ordinary),study.base.weight_hash(original))

    def test_gradients_all_layers_and_disjoint_ids(self):
        with study.configured():
            model,_ = study.build('classical_start'); b = self.batch()
            opt = torch.optim.Adam(model.parameters(),lr=.001)
            study.base.repro.train_step(model,opt,b,study.base.cfg())
            for former in model.formers:
                self.assertTrue(torch.isfinite(former.output[-1].weight.grad).all())
                self.assertGreater(float(former.output[-1].weight.grad.abs().sum()),0.)
            study.base.repro.train_step(model,opt,b,study.base.cfg())
            self.assertGreater(float(model.formers[0].embed.weight.grad.abs().sum()),0.)
            train = study.base.LearningData('development','train',study.SPEC)
            val = study.base.LearningData('development','val',study.SPEC)
            tune = study.base.LearningData('development','val',dict(study.SPEC,start=25000000,val=256))
            self.assertEqual([int(d[0]['trial']) for d in (train,val,tune)],[24000000,25000000,26000000])
            with self.assertRaises(ValueError):
                study.base.LearningData('development','test',study.SPEC)

    def test_exact_resume(self):
        spec = dict(study.SPEC,train=32,val=32,epochs=2)
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            for label in ('full','resume'):
                with study.configured(root/label):
                    data = study.base.cache('development',spec)
                    if label == 'resume':
                        study.base.train_arm('development','classical_start',spec,data,stop_after=1)
                    study.base.train_arm('development','classical_start',spec,data)
            states = [torch.load(root/name/'development/classical_start/checkpoint.pt',weights_only=False)
                      for name in ('full','resume')]
            for key in states[0]['model']:
                torch.testing.assert_close(states[0]['model'][key],states[1]['model'][key],rtol=0,atol=0)
            self.assertEqual([h['validation_nmse_db'] for h in states[0]['history']],
                             [h['validation_nmse_db'] for h in states[1]['history']])


if __name__ == '__main__':
    unittest.main()
