import sys
import unittest
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from paper2_reproduction import ReproductionData,PaperEncoder,build_model,cfg_for,curve,SNRS
from trackD_urformer.dataset import collate


class ReproductionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_printed_encoder(self):
        layer=PaperEncoder().double()
        v=torch.randn(2,4,64,dtype=torch.float64)
        h=layer.ln(v)
        a=layer.attn(h,h,h,need_weights=False)[0]+v
        torch.testing.assert_close(layer(v),layer.ffn(a)+a,rtol=1e-12,atol=1e-12)
        self.assertEqual(sum(isinstance(x,torch.nn.LayerNorm) for x in layer.modules()),1)

    def test_layer_against_direct_complex_equations(self):
        cfg=cfg_for(1)
        model,_=build_model(cfg)
        layer=model.layers[0].double()
        batch=collate([ReproductionData('train')[i] for i in range(2)])
        g,z,s,b,sig=[batch[k].to(torch.complex128 if batch[k].is_complex() else torch.float64)
            for k in ('G0','Z','S','B','sigma2')]
        y=g@s+b
        direct=z*torch.exp(1j*torch.angle(y))
        kappa=2*z*abs(y)/sig[:,None,None]
        f=layer.filter_net
        filtered=torch.sigmoid(f.fc2(torch.relu(f.fc1(kappa.unsqueeze(-1))))).squeeze(-1)
        rec=direct*(.5*filtered+.5)
        linear=(rec-b)@torch.linalg.pinv(s)
        expected=linear+layer.former(linear)
        # Production guards use eps=1e-8; independent formula above uses angle.
        torch.testing.assert_close(layer(g,z,s,b,sig),expected,rtol=1e-7,atol=1e-7)
        self.assertFalse(layer.use_hankel)

    def test_fresh_data_and_fixed_snr_calibration(self):
        test=ReproductionData('test')
        for i,snr in enumerate(SNRS):
            item=test[2000*i+17]
            self.assertEqual(int(item['trial']),9000017)
            self.assertEqual(float(item['snr_db']),snr)
            self.assertAlmostEqual(10*np.log10(test.signal_power/float(item['sigma2'])),snr,places=5)
            torch.testing.assert_close(item['G_true'],test[17]['G_true'],rtol=0,atol=0)
        ids=[int(ReproductionData(split)[0]['trial']) for split in ('train','val','test')]
        self.assertEqual(ids,[7000000,8000000,9000000])
        self.assertAlmostEqual(10*np.log10(np.mean(abs(test.B)**2)/test.signal_power),10)

    def test_metric_is_mean_linear_nmse_then_db(self):
        r={'numerator':np.array([1.,9.]),'denominator':np.array([1.,3.]),'snr_db':np.array([5.,5.])}
        self.assertAlmostEqual(curve(r)[0]['nmse_db'],10*np.log10(2))


if __name__=='__main__':unittest.main()
