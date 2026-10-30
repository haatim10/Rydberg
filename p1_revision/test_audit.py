"""Integrity-regression tests using isolated copies of the development store."""
import json
from pathlib import Path
import shutil
import tempfile
import unittest
import numpy as np
from p1_revision.audit import REPO, run_audit, selection_checks


class CampaignAuditTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='p1_audit_')
        self.root = Path(self.tmp.name)/'smoke'
        shutil.copytree(REPO/'results/p1_revision/smoke', self.root)

    def tearDown(self):
        self.tmp.cleanup()

    def mutate_npz(self, relative, change):
        path = self.root/relative
        with np.load(path, allow_pickle=False) as f:
            data = {k:f[k].copy() for k in f.files}
        change(data)
        np.savez_compressed(path, **data)

    def test_complete_smoke_is_labelled_and_counted(self):
        result = run_audit(self.root, replay=False)
        self.assertEqual(result['scope'], 'smoke')
        self.assertEqual(result['condition_trials'], 30)
        self.assertEqual(result['independent_base_worlds'], 12)
        self.assertEqual(result['state'], 'integrity_checks_passed_replay_skipped')

    def test_missing_completion_never_passes(self):
        (self.root/'COMPLETE.json').unlink()
        with self.assertRaisesRegex(ValueError, 'incomplete'):
            run_audit(self.root, replay=False)

    def test_false_completion_never_passes(self):
        p=self.root/'COMPLETE.json'; d=json.loads(p.read_text());d['complete']=False
        p.write_text(json.dumps(d))
        with self.assertRaisesRegex(ValueError, 'not complete'):
            run_audit(self.root, replay=False)

    def test_source_tampering_is_detected(self):
        p=self.root/'source_snapshot/p1_revision/scenarios.py'
        p.write_text(p.read_text()+'\n# altered source\n')
        with self.assertRaisesRegex(ValueError, 'source hash mismatch'):
            run_audit(self.root, replay=False)

    def test_missing_chunk_is_detected(self):
        (self.root/'raw/baselines_L3_S0/00000.npz').unlink()
        with self.assertRaisesRegex(ValueError, 'missing/extra chunks'):
            run_audit(self.root, replay=False)

    def test_wrong_trial_boundary_is_detected(self):
        def change(data):data['trial'][0]+=1
        self.mutate_npz('raw/baselines_L3_S0/00000.npz',change)
        with self.assertRaisesRegex(ValueError, 'trial ids/boundaries'):
            run_audit(self.root,replay=False)

    def test_changed_selected_rank_is_detected(self):
        def change(data):
            d=json.loads(str(data['diagnostics_json']))
            item=d[0]['methods']['hs']
            item['selected_rank']=item['selected_rank']%16+1
            data['diagnostics_json']=json.dumps(d)
        self.mutate_npz('raw/baselines_L3_S0/00000.npz',change)
        with self.assertRaisesRegex(ValueError, 'argmin/tie rule'):
            run_audit(self.root,replay=False)

    def test_broken_split_pairing_is_detected(self):
        def change(data):data['num_em'][0]*=1.1
        self.mutate_npz('raw/holdout_P10_S0_F0.1/00000.npz',change)
        with self.assertRaisesRegex(ValueError, 'holdout fractions must share'):
            run_audit(self.root,replay=False)

    def test_exact_fallback_required_not_approximate(self):
        cell=dict(tag='synthetic_integrity_check',kind='holdout',N=8)
        data=dict(trial=np.array([1]),scores=np.array([[3.,2.,1.,0.]]),
                  rank=np.array([4]),num_em=np.array([1.]),num_hs=np.array([1.]))
        for key in ('seconds_select','seconds_em','seconds_hs_fit'):
            data[key]=np.array([.5])
        data['seconds_hs_total']=np.array([1.])
        diag={k:(v[0].tolist() if hasattr(v[0],'tolist') else v[0])
              for k,v in data.items() if k not in ('trial','num_em','num_hs')}
        counts=selection_checks(cell,data,[diag])
        self.assertEqual(counts['hs'],1)
        data['num_hs'][0]=np.nextafter(1.,2.)
        with self.assertRaisesRegex(ValueError,'fallback error differs'):
            selection_checks(cell,data,[diag])


if __name__=='__main__':
    unittest.main()
