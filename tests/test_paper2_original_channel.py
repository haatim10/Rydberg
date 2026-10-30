import sys
import unittest
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from paper2_original_channel import PaperChannelData,draw_channel,cfg_for,MODES,rng_for


class OriginalChannelTests(unittest.TestCase):
    def test_equation_matches_independent_loop(self):
        trial=9876543
        theta=rng_for(trial,1).uniform(-np.pi/2,np.pi/2,(4,4,10))
        r=rng_for(trial,4)
        g=(r.standard_normal((4,4,10))+1j*r.standard_normal((4,4,10)))/np.sqrt(2)
        expected=np.zeros((32,4),dtype=complex)
        for k in range(4):
            for l in range(4):
                for c in range(10):
                    expected[:,k]+=np.sqrt(32/40)*g[k,l,c]*np.exp(1j*np.pi*np.arange(32)*np.sin(theta[k,l,c]))
        np.testing.assert_allclose(draw_channel(trial,MODES[0]),expected,atol=1e-12,rtol=1e-12)

    def test_calibration_and_forward(self):
        for mode in MODES:
            ds=PaperChannelData('train',mode)
            trial,snr,sigma2,g,ex=ds.raw(0)
            self.assertAlmostEqual(10*np.log10(ds.signal_power/sigma2),snr)
            self.assertAlmostEqual(10*np.log10(np.mean(abs(ds.B)**2)/ds.signal_power),10)
            np.testing.assert_allclose(ex.Z,abs(g@ds.S+ds.B+ex.W),atol=1e-12)
            self.assertTrue(all(torch.equal(v,ds[0][k]) for k,v in ds[0].items()))

    def test_pair_inputs_and_disjoint_splits(self):
        a,b=[PaperChannelData('train',m)[0] for m in MODES]
        for key in ('trial','snr_db','sigma2','S','B','G0'):
            self.assertTrue(torch.equal(a[key],b[key]))
        self.assertFalse(torch.equal(a['G_true'],b['G_true']))
        ids=[int(PaperChannelData(s,MODES[0])[0]['trial']) for s in ('train','val','test')]
        self.assertEqual(ids,[4000000,5000000,6000000])
        c=cfg_for(1,True)
        self.assertEqual((c.system.K,c.system.N,c.system.P,c.data.n_train,c.train.epochs),(4,32,20,20000,50))
        self.assertEqual(c.train.init,'random')
        self.assertEqual(c.model.hankel_rank,7)


if __name__=='__main__': unittest.main()
