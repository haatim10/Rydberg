import sys
import tempfile
import unittest
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import paper2_loss_weighting as study


class LossWeightingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_formula_gradient_and_boundary(self):
        errors=torch.tensor([.2,.1,.02,.03],requires_grad=True)
        snr=torch.tensor([-10.,14.999,15.,20.])
        weights=torch.tensor([1.,1.,10.,10.])
        mean=weights.mean()
        loss=study.objective(errors,snr,mean)
        torch.testing.assert_close(loss,(errors*weights/mean).mean())
        torch.testing.assert_close(torch.autograd.grad(loss,errors)[0],weights/mean/4)
        torch.testing.assert_close(study.objective(errors,snr,1.,factor=1),errors.mean())

    def test_unchanged_initial_model_and_control_step(self):
        with study.configured():
            a,_=study.build('standard'); b,_=study.build('high_snr_weighted')
            self.assertEqual(study.base.weight_hash(a),study.base.weight_hash(b))
            original,_=study.prior.build('classical_start')
            self.assertEqual(study.base.weight_hash(a),study.base.weight_hash(original))
            data=study.base.LearningData('development','train',study.SPEC)
            batch=study.base.collate([data[i] for i in range(4)])
            args=[batch[k] for k in ('G0','Z','S','B','sigma2')]
            with torch.no_grad():
                torch.testing.assert_close(b(*args),study.base.Classical(10)(*args),rtol=0,atol=0)
            study.train_step(a,torch.optim.Adam(a.parameters(),lr=.001),batch,study.base.cfg())
            study.ORIGINAL_STEP(original,torch.optim.Adam(original.parameters(),lr=.001),batch,study.base.cfg())
            self.assertEqual(study.base.weight_hash(a),study.base.weight_hash(original))

    def test_exact_weighted_resume(self):
        spec=dict(study.SPEC,train=32,val=32,epochs=2)
        with tempfile.TemporaryDirectory() as root:
            root=Path(root)
            for name in ('full','resume'):
                with study.configured(root/name):
                    data=study.base.cache('development',spec)
                    study.WEIGHT_MEAN=float(study.weights(data['train'].arrays['snr_db']).mean())
                    if name=='resume':
                        study.base.train_arm('development','high_snr_weighted',spec,data,stop_after=1)
                    study.base.train_arm('development','high_snr_weighted',spec,data)
            states=[torch.load(root/name/'development/high_snr_weighted/checkpoint.pt',weights_only=False)
                    for name in ('full','resume')]
            for k in states[0]['model']:
                torch.testing.assert_close(states[0]['model'][k],states[1]['model'][k],rtol=0,atol=0)
            self.assertEqual([h['validation_nmse_db'] for h in states[0]['history']],
                             [h['validation_nmse_db'] for h in states[1]['history']])
            study.WEIGHT_MEAN=1.


if __name__=='__main__':
    unittest.main()
