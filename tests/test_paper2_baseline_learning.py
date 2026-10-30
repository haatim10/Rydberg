import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import paper2_baseline_learning as study
from trackD_urformer.torch_forward import em_gs_layer, em_kappa


class LearningChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_units_preserve_physics(self):
        batch = study.collate([study.LearningData('development', 'train')[i] for i in range(4)])
        args = [batch[k].to(torch.complex128 if batch[k].is_complex() else torch.float64)
                for k in ('G0', 'Z', 'S', 'B', 'sigma2')]
        scale = np.sqrt(32.)
        norm = study.normalize_inputs(*args, scale)
        def kappa(v):
            g,z,s,b,var = v
            return 2*z*abs(g@s+b)/var[:,None,None]
        torch.testing.assert_close(kappa(args), kappa(norm), rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(em_kappa(args[1],args[0]@args[2]+args[3],args[4],1e-8),
            em_kappa(norm[1],norm[0]@norm[2]+norm[3],norm[4],1e-8/scale),rtol=1e-12,atol=1e-12)
        torch.testing.assert_close(em_gs_layer(*args, eps=1e-8),
            scale*em_gs_layer(*norm, eps=1e-8/scale), rtol=1e-7, atol=1e-7)

    def test_loss_and_gradient_units(self):
        target = torch.randn(4,32,4, dtype=torch.complex128)
        pred = torch.randn_like(target).requires_grad_()
        def loss(p,t):
            return (abs(p-t).square().sum((1,2))/abs(t).square().sum((1,2))).mean()
        a = loss(pred, target)
        b = loss(pred/np.sqrt(32.), target/np.sqrt(32.))
        torch.testing.assert_close(a,b)
        torch.testing.assert_close(torch.autograd.grad(a,pred)[0], torch.autograd.grad(b,pred)[0])

    def test_matching_weights_and_ids(self):
        a,_ = study.build('physical'); b,_ = study.build('normalized')
        self.assertEqual(study.weight_hash(a), study.weight_hash(b))
        ids = [study.LearningData(s,p)[0]['trial'].item()
               for s in study.STUDIES for p in ('train','val')]
        self.assertEqual(ids, [20000000,21000000,22000000,23000000])
        with self.assertRaises(ValueError):
            study.LearningData('development','test')

    def test_physical_arm_is_unchanged_reconstruction(self):
        model,_ = study.build('physical')
        original,_ = study.repro.build_model(study.cfg())
        batch = study.collate([study.LearningData('development','val')[i] for i in range(2)])
        args = [batch[k] for k in ('G0','Z','S','B','sigma2')]
        model.eval(); original.eval()
        with torch.no_grad():
            torch.testing.assert_close(model(*args),original(*args),rtol=0,atol=0)

    def test_exact_resume_and_cache(self):
        spec = dict(train=32,val=32,epochs=2,start=20000000,snr=60.)
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            with patch.object(study, 'OUT', root/'full'):
                data = study.cache('memorization',spec)
                cached = study.cache('memorization',spec)
                torch.testing.assert_close(data['train'].arrays['Z'], cached['train'].arrays['Z'])
                complete = study.train_arm('memorization','normalized',spec,data)
                full = torch.load(root/'full/memorization/normalized/checkpoint.pt',weights_only=False)
            with patch.object(study, 'OUT', root/'resume'):
                data = study.cache('memorization',spec)
                study.train_arm('memorization','normalized',spec,data,stop_after=1)
                resumed = study.train_arm('memorization','normalized',spec,data)
                state = torch.load(root/'resume/memorization/normalized/checkpoint.pt',weights_only=False)
                self.assertEqual(complete['final_validation_nmse_db'],resumed['final_validation_nmse_db'])
                for k in full['model']:
                    torch.testing.assert_close(full['model'][k],state['model'][k],rtol=0,atol=0)
                self.assertEqual([h['data_order_sha256'] for h in complete['history']],
                                 [h['data_order_sha256'] for h in resumed['history']])
                self.assertEqual(study.train_arm('memorization','normalized',spec,data),resumed)


if __name__ == '__main__':
    unittest.main()
