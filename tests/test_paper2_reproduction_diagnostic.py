import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import paper2_reproduction_diagnostic as d


class DiagnosticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_matched_data_and_noise(self):
        for split in ('train','val'):
            a,b=[d.DiagnosticData(split,arm,8) for arm in d.ARMS]
            for i in (0,7):
                x,y=a[i],b[i]
                for key in ('G_true','G0','S','B','trial'):
                    torch.testing.assert_close(x[key],y[key],rtol=0,atol=0)
                ar,br=a.raw(i),b.raw(i)
                np.testing.assert_allclose(ar[-1].W/np.sqrt(ar[2]),br[-1].W/np.sqrt(br[2]))
                self.assertEqual(float(y['snr_db']),5)
                if split=='val':
                    for key in x:torch.testing.assert_close(x[key],y[key],rtol=0,atol=0)
        with self.assertRaises(ValueError):d.DiagnosticData('test','mixed')

    def test_initial_weights_and_order(self):
        a,_=d.reproduction.build_model(d.cfg());b,_=d.reproduction.build_model(d.cfg())
        self.assertEqual(d.weights_hash(a),d.weights_hash(b))
        datasets=[d.DiagnosticData('train',arm,64) for arm in d.ARMS]
        for epoch in (0,1):
            orders=[torch.cat([x['trial'] for x in d.loader(ds,epoch)]) for ds in datasets]
            torch.testing.assert_close(*orders,rtol=0,atol=0)

    def test_tiny_pipeline_and_safe_completed_skip(self):
        saved=d.OUT,d.COUNTS,d.EPOCHS
        try:
            with tempfile.TemporaryDirectory(prefix='urformer-diagnostic-check-') as tmp:
                d.OUT=Path(tmp);d.COUNTS={'train':64,'val':64};d.EPOCHS=2
                for arm in d.ARMS:d.train_arm(arm)
                d.summary()
                result=json.loads((d.OUT/'summary.json').read_text())
                self.assertTrue(result['matched_orders'])
                self.assertTrue(result['identical_validation'])
                self.assertTrue(result['matched_latent_train'])
                for arm in d.ARMS:
                    checkpoint=d.OUT/arm/'checkpoint.pt'
                    digest=d.engine.hash_file(checkpoint)
                    d.train_arm(arm)
                    self.assertEqual(d.engine.hash_file(checkpoint),digest)
                    value=torch.load(checkpoint,weights_only=False)
                    self.assertEqual(value['scheduler']['T_max'],50)
                    self.assertEqual(len(value['history']),2)
        finally:
            d.OUT,d.COUNTS,d.EPOCHS=saved


if __name__=='__main__':unittest.main()
