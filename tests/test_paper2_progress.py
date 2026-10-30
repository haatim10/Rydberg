import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import paper2_original_progress as progress


class ProgressTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.root=Path(self.tmp.name)
        self.directory=patch.object(progress,'DIAGNOSTIC',self.root)
        self.directory.start()
        self.live=patch.object(progress,'alive',return_value=True)
        self.live.start()

    def tearDown(self):
        self.live.stop();self.directory.stop();self.tmp.cleanup()

    def write(self,name,value):
        path=self.root/name;path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(value))

    def test_ten_epoch_progress_and_validation_mapping(self):
        self.write('status.json',{'state':'training','arm':'mixed','epoch':2,'batch':100,'batches':625})
        self.write('mixed/history.json',[{'epoch':1,'validation_5db_nmse_db':-8.,'train_loss':.2,'seconds':60.}])
        value=progress.diagnostic_snapshot()
        self.assertEqual(value['epochs_total'],20)
        self.assertEqual(value['active']['epoch_budget'],10)
        self.assertEqual(value['active']['last_validation_db'],-8)
        self.assertAlmostEqual(value['percent'],100*1.16/20)
        self.assertIsNotNone(value['eta_hours'])

    def test_saved_epoch_not_double_counted(self):
        self.write('status.json',{'state':'training','arm':'mixed','epoch':1,'batch':625,'batches':625})
        self.write('mixed/history.json',[{'epoch':1,'validation_5db_nmse_db':-8.,'train_loss':.2,'seconds':60.}])
        value=progress.diagnostic_snapshot()
        self.assertEqual(value['active']['batch'],0)
        self.assertAlmostEqual(value['percent'],5)

    def test_completed_and_failed_states(self):
        self.write('status.json',{'state':'complete'})
        for arm in ('mixed','fixed5'):
            self.write(f'{arm}/result.json',{})
            self.write(f'{arm}/history.json',[{'epoch':i+1,'validation_5db_nmse_db':-8.,'train_loss':.2,'seconds':60.} for i in range(10)])
        self.write('summary.json',{'checkpoint_results':{'final':{'fixed5_gain_db':1.}}})
        value=progress.diagnostic_snapshot()
        self.assertEqual(value['runs_done'],2)
        self.assertEqual(value['percent'],100)
        self.assertIsNone(value['eta_hours'])
        self.assertIn('final',value['diagnostic_results'])
        self.write('status.json',{'state':'failed','error':'example'})
        self.assertEqual(progress.diagnostic_snapshot()['error'],'example')

    def test_preparation_and_missing_data(self):
        self.assertIsNone(progress.diagnostic_snapshot()['active'])
        self.write('status.json',{'state':'preparing','arm':'fixed5'})
        self.write('fixed5/status.json',{'state':'generating_data','split':'train','samples':1000,'total':20000})
        self.assertIn('1,000 / 20,000',progress.diagnostic_snapshot()['preparation'])


if __name__=='__main__':unittest.main()
