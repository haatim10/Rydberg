"""Numerical/provenance gates for existing-result reuse, without rank search."""
import unittest
from . import reuse
import numpy as np


class ReuseTests(unittest.TestCase):
    def test_identical_paired_errors_have_zero_gain_interval(self):
        error=np.arange(1,41,dtype=float)
        result=reuse.paired_summary(error,error,np.ones(40),np.random.default_rng(1),300)
        for key in ('ratio_gain_db','median_gain_db'):
            for suffix in ('','_lo','_hi'):
                self.assertEqual(result[key+suffix],0.)

    def test_known_factor_gain(self):
        error=np.arange(1,41,dtype=float)
        result=reuse.paired_summary(2*error,error,np.ones(40),np.random.default_rng(1),300)
        self.assertAlmostEqual(result['ratio_gain_db'],10*np.log10(2))
        self.assertAlmostEqual(result['median_gain_db'],10*np.log10(2))

    def test_invalid_pair_is_rejected(self):
        with self.assertRaises(ValueError):
            reuse.paired_summary(np.ones(3),np.ones(4),np.ones(3),np.random.default_rng(1),10)

    def test_protocol_cell_count(self):
        cells=reuse.selected_cells()
        self.assertEqual(len(cells),26)
        self.assertEqual(sum(c['n'] for c in cells),10400)
        self.assertEqual(len(reuse.selected_cells('matched')),8)

    def test_fingerprint_and_actual_replay(self):
        reuse.replay_sources()
        cell=next(c for c in reuse.selected_cells() if c['tag']=='matchedrho_N32_L8')
        saved=reuse.load_npz(reuse.P1/'trials'/f'{cell["tag"]}.npz')
        row=reuse.replay_one(cell,0,saved)
        self.assertLess(row['replay_error_relative'],2e-11)
        self.assertEqual(row['rho_em_columns'].shape,(3,))
        self.assertTrue(np.isfinite(row['g_em']).all())

    def test_constant_diagnostic_is_undefined(self):
        self.assertIsNone(reuse.correlation(np.ones(4),np.arange(4)))

    def test_rank_bootstrap_respects_ties_and_constant_variables(self):
        x=np.asarray([0,0,1,1,2,2,3,3],float)
        intervals=reuse.correlation_intervals(dict(x=x,y=2*x,z=np.ones(8)),
            dict(same=('x','y'),constant=('x','z')),np.random.default_rng(2),300)
        self.assertAlmostEqual(intervals['same_lo'],1.)
        self.assertAlmostEqual(intervals['same_hi'],1.)
        self.assertIsNone(intervals['constant_lo'])
        self.assertEqual(intervals['constant_valid_resamples'],0)


if __name__=='__main__':
    unittest.main()
