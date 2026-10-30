import sys
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import paper2_original_progress as p


class ConfirmationProgress(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(); self.root=Path(self.tmp.name)
        self.ctx=patch.object(p,'CONFIRM',self.root); self.ctx.start()
        self.live=patch.object(p,'alive',return_value=True); self.live.start()
        self.write('manifest.json',{'modes':['literal_independent'],'seeds':[2026092601],
          'arms':['standard_U','standard_H','weighted_U','weighted_H'],'epochs':18,'counts':{'train':16000,'val':1200,'test':1800}})
    def tearDown(self): self.live.stop();self.ctx.stop();self.tmp.cleanup()
    def write(self,name,value):
        path=self.root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))
    def test_batch_mapping_and_budget(self):
        self.write('status.json',{'state':'training','mode':'literal_independent','seed':2026092601,'arm':'standard_U'})
        self.write('literal_independent/seed2026092601/status.json',{'arm':'standard_U','epoch':2,'batch':100,'batches':500})
        self.write('literal_independent/seed2026092601/development/standard_U/history.json',
           [{'epoch':1,'train_online_nmse':.3,'validation_nmse_db':-4.,'seconds':30.}])
        self.write('budget.json',{'worker_seconds':3600,'limit_seconds':54000})
        v=p.confirmation_snapshot(); self.assertEqual(v['active']['epoch'],2)
        self.assertEqual(v['active']['last_validation_db'],-4.)
        self.assertAlmostEqual(v['percent'],100*1.2/72);self.assertEqual(v['worker_hours_left'],14.)
    def test_stale_arm_is_not_counted(self):
        self.write('status.json',{'state':'training','mode':'literal_independent','seed':2026092601,'arm':'standard_H'})
        self.write('literal_independent/seed2026092601/status.json',{'arm':'standard_U','epoch':18,'batch':500,'batches':500})
        v=p.confirmation_snapshot();self.assertEqual(v['active']['batch'],0)
        self.assertEqual(v['active']['epoch'],1)
        self.assertEqual(v['percent'],0)
    def test_budget_stop_is_incomplete(self):
        self.write('status.json',{'state':'budget_exhausted','reason':'Limit reached'})
        v=p.confirmation_snapshot();self.assertIn('incomplete',v['label']);self.assertIsNone(v['eta_hours'])
        self.assertEqual(v['error'],'Limit reached')
    def test_classical_phase_and_complete(self):
        self.write('status.json',{'state':'classical_evaluation','mode':'literal_independent','method':'paper1_auto','split':'val','done':8,'total':1200})
        self.assertIn('8/1,200',p.confirmation_snapshot()['phase'])
        self.write('status.json',{'state':'complete'})
        self.assertIsNone(p.confirmation_snapshot()['eta_hours'])


if __name__=='__main__':unittest.main()
