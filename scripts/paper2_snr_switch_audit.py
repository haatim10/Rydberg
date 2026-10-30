"""Independently recalculate stored switching results; no model/data mutations."""
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.stats import t

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/paper2_snr_switch'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run():
    manifest = json.loads((OUT/'manifest.json').read_text())
    for key, file in [('script_sha256','scripts/paper2_snr_switch.py'),
                      ('plan_sha256','reports/paper2_snr_switch_plan.md'),
                      ('source_driver_sha256','scripts/paper2_long_training.py'),
                      ('model_code_sha256','trackD_urformer/urformer.py')]:
        assert digest(ROOT/file) == manifest[key]
    assert digest(OUT/'fresh_test_data.npz') == manifest['fresh_test']['sha256']
    for models in manifest['models'].values():
        for item in models.values():
            assert digest(item['path']) == item['sha256']
    selection = json.loads((OUT/'threshold.json').read_text())
    costs = {}
    for threshold in (-11., 0., 2.5, 5., 7.5, 10., 21.):
        costs[threshold] = []
        for seed in (1,2,3):
            u = np.load(OUT/f'validation_seed{seed}_U.npz')
            h = np.load(OUT/f'validation_seed{seed}_H.npz')
            costs[threshold].append(np.where(u['snr_db'] >= threshold,
                                            h['numerator'],u['numerator'])/u['denominator'])
        costs[threshold] = float(np.mean(costs[threshold]))
        np.testing.assert_allclose(costs[threshold], selection['validation_scores_mean_normalized_error'][str(threshold)],atol=1e-12)
    chosen = min(costs, key=lambda x:(costs[x],-x))
    assert chosen == selection['threshold_db']
    summary = json.loads((OUT/'summary.json').read_text())
    assert chosen == summary['threshold_db']
    per_seed = []
    for seed in (1,2,3):
        saved = json.loads((OUT/f'seed{seed}.json').read_text())
        z = np.load(OUT/f'test_seed{seed}.npz')
        assert all(np.isfinite(z[k]).all() for k in z.files)
        assert len(np.unique(z['trial'])) == 2000
        assert min(z['trial']) == 3000000 and max(z['trial']) == 3001999
        for key in z.files:
            if key.startswith('mask_'):
                name = key.removeprefix('mask_')
                np.testing.assert_array_equal(z[name], np.where(z[key],z['H'],z['U']))
                assert float(z[key].mean()) == summary['fraction_routed_H'][name]
        np.testing.assert_array_equal(z['mask_router'],z['snr_db']>=chosen)
        np.testing.assert_array_equal(z['same_weights_toggle'],np.where(z['mask_router'],z['H'],z['H_disabled']))
        groups = {'all':np.ones(2000,dtype=bool), 'low':z['snr_db']<5, 'high':z['snr_db']>=5}
        groups.update({f'[{lo},{lo+5})':(z['snr_db']>=lo)&(z['snr_db']<lo+5) for lo in range(-10,20,5)})
        for method, references in saved['metrics'].items():
            for reference, subsets in references.items():
                for group, expected in subsets.items():
                    mask = groups[group]
                    a,b,d = z[method][mask],z[reference][mask],z['denominator'][mask]
                    actual = {'n':int(mask.sum()),'candidate_nmse_db':10*np.log10(np.mean(a/d)),
                              'paired_median_gain_db':np.median(10*np.log10(b/a)),
                              'raw_error_sum_gain_db':10*np.log10(b.sum()/a.sum()),
                              'normalized_error_sum_gain_db':10*np.log10((b/d).sum()/(a/d).sum())}
                    for metric, value in actual.items():
                        np.testing.assert_allclose(value,expected[metric],atol=1e-12)
        assert max(saved['routing_replay_max_abs_error'].values()) == 0.0
        per_seed.append(saved)
    for method, refs in summary['aggregate'].items():
        for reference, subsets in refs.items():
            for group, metrics in subsets.items():
                for metric, expected in metrics.items():
                    vals = np.array([x['metrics'][method][reference][group][metric] for x in per_seed])
                    sd = vals.std(ddof=1)
                    margin = t.ppf(.975,2)*sd/np.sqrt(3)
                    np.testing.assert_allclose(vals,expected['per_seed'],atol=1e-12)
                    np.testing.assert_allclose([vals.mean(),sd,vals.mean()-margin,vals.mean()+margin],
                        [expected['mean'],expected['sd'],*expected['nominal_seed_ci95']],atol=1e-12)
    audit = {'passed':True,'checkpoint_and_source_hashes_match':True,
             'validation_threshold_recomputed':chosen,'per_trial_branch_identities':True,
             'all_seed_and_group_metrics_recomputed':True,'all_seed_intervals_recomputed':True,
             'recorded_direct_routing_replay_max_abs_error':0.0,
             'note':'Replays were executed by the experiment driver; this audit independently checks saved arrays and arithmetic, not the receiver physics.'}
    (OUT/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps(audit,indent=2))


if __name__ == '__main__':
    run()
