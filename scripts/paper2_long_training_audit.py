"""Post-run audit; never modifies training artifacts or the frozen driver."""
import json
from pathlib import Path
import datetime
import numpy as np
import torch
from scipy.stats import t
from paper2_long_training import (OUT, ROOT, BUDGETS, SEEDS, cfg_for,
    CachedData, dataset, build_model, test_model, hash_file, select_epoch,
    DataLoader, collate)

torch.set_num_threads(1)
manifest = json.loads((OUT / 'manifest.json').read_text())
assert hash_file(ROOT / 'scripts/paper2_long_training.py') == manifest['script_sha256']
assert json.loads((OUT / 'status.json').read_text()) == {'state': 'complete', 'runs': 18}
assert len(list(OUT.glob('seed*/result.json'))) == 18
summary = json.loads((OUT / 'summary.json').read_text())
audit = {'timestamp': datetime.datetime.now(datetime.timezone.utc).isoformat(),
         'driver_hash_matches': True, 'datasets': {}, 'runs': [], 'pairs': []}
for split in ('train', 'val', 'test'):
    folder = OUT / 'data' / split
    meta = json.loads((folder / 'complete.json').read_text())
    for key, expected in meta['sha256'].items():
        assert hash_file(folder / f'{key}.npy') == expected, (split, key)
    cached = CachedData(split)
    original = dataset(cfg_for(1, False, 13), split)
    for idx in (0, len(cached)//2, len(cached)-1):
        assert all(torch.equal(value, cached[idx][key]) for key, value in original[idx].items())
    audit['datasets'][split] = {'count': len(cached), 'hashes_match': True,
                                'representative_generator_matches': True}
print('Dataset hashes and generator spot checks passed.', flush=True)
loader = DataLoader(CachedData('test'), batch_size=32, shuffle=False, collate_fn=collate)
records = {}
for seed in SEEDS:
    for budget in BUDGETS:
        for arm in ('U', 'H'):
            name = f'seed{seed}_{arm}_{budget}ep'
            folder = OUT / name
            r = json.loads((folder / 'result.json').read_text())
            cfg = cfg_for(seed, arm == 'H', budget)
            assert r['config'] == json.loads(json.dumps(cfg.to_dict()))
            hist = json.loads((folder / 'history.json').read_text())
            assert hist == r['history'] and len(hist) == budget
            assert [x['epoch'] for x in hist] == list(range(1, budget+1))
            assert all((folder / f'epoch_{ep:03d}.pt').exists() for ep in range(1, budget+1))
            # These are trusted, locally generated checkpoints from this experiment.
            ck = torch.load(folder / 'checkpoint.pt', map_location='cpu', weights_only=False)
            assert ck['epoch'] == budget-1 and ck['history'] == hist
            val = np.asarray(ck['validations'])
            assert val.shape == (budget, 2000) and np.isfinite(val).all() and (val > 0).all()
            curve = np.asarray([x['validation_db'] for x in hist])
            np.testing.assert_allclose(10*np.log10(val.mean(axis=1)), curve, atol=1e-6, rtol=0)
            chosen, selection = select_epoch(curve, val[int(curve.argmin())], 'one_se')
            assert chosen+1 == r['chosen_epoch_1based'] and selection == r['selection']
            model, _ = build_model(cfg, 'arm1b_full_random')
            errors = {}
            for label, ep in (('selected', chosen+1), ('final', budget)):
                weights = torch.load(folder / f'epoch_{ep:03d}.pt', weights_only=True)['model']
                if label == 'final':
                    assert all(torch.equal(v, ck['model'][k]) for k, v in weights.items())
                model.load_state_dict(weights)
                replay = test_model(model, loader)
                with np.load(folder / f'test_{label}.npz') as z:
                    stored = {key: z[key].copy() for key in z.files}
                assert all(a.shape == (2000,) and np.isfinite(a).all() for a in stored.values())
                assert (stored['numerator'] > 0).all() and (stored['denominator'] > 0).all()
                assert len(np.unique(stored['trial'])) == 2000
                for key in stored:
                    np.testing.assert_allclose(replay[key], stored[key], atol=1e-7, rtol=1e-6)
                errors[label] = float(np.max(np.abs(replay['numerator']-stored['numerator'])))
                records[(seed, budget, arm, label)] = stored
            best = hist[int(curve.argmin())]
            audit['runs'].append({'run': name, 'selected_epoch': chosen+1,
                'best_validation_epoch': best['epoch'], 'best_validation_db': best['validation_db'],
                'final_validation_db': hist[-1]['validation_db'],
                'training_loss_at_best_validation': best['train_loss'],
                'final_training_loss': hist[-1]['train_loss'],
                'test_replay_max_absolute_error': errors})
            print(name, 'checkpoint selection and both test replays passed', flush=True)
        ru, rh = [json.loads((OUT/f'seed{seed}_{m}_{budget}ep/result.json').read_text()) for m in ('U','H')]
        assert [x['data_order_sha256'] for x in ru['history']] == [x['data_order_sha256'] for x in rh['history']]
        audit['pairs'].append({'seed': seed, 'budget': budget, 'all_epoch_orders_match': True})
        for label in ('selected', 'final'):
            u, h = [records[(seed,budget,m,label)] for m in ('U','H')]
            for key in ('trial','snr_db','denominator'):
                assert np.array_equal(u[key],h[key])
            masks = {'snr_ge5':u['snr_db']>=5, 'all':np.ones(2000,dtype=bool)}
            masks.update({f'[{lo},{lo+5})':(u['snr_db']>=lo)&(u['snr_db']<lo+5) for lo in range(-10,20,5)})
            expected = next(x for x in summary[f'{budget}ep_{label}']['pairs'] if x['seed']==seed)['bins']
            for key, mask in masks.items():
                values = {'n': int(mask.sum()),
                    'paired_median_db': float(np.median(10*np.log10(u['numerator'][mask]/h['numerator'][mask]))),
                    'ratio_of_squared_error_sums_db': float(10*np.log10(u['numerator'][mask].sum()/h['numerator'][mask].sum())),
                    'ratio_of_normalized_error_sums_db':float(10*np.log10((u['numerator'][mask]/u['denominator'][mask]).sum()/(h['numerator'][mask]/h['denominator'][mask]).sum()))}
                for metric, value in values.items():
                    np.testing.assert_allclose(value, expected[key][metric], atol=1e-10, rtol=0)
for budget in BUDGETS:
    for label in ('selected','final'):
        s = summary[f'{budget}ep_{label}']
        assert s['n_seeds'] == 3
        gains = np.array([x['bins']['snr_ge5']['paired_median_db'] for x in s['pairs']])
        sd = gains.std(ddof=1)
        margin = t.ppf(.975,2)*sd/np.sqrt(3)
        np.testing.assert_allclose([gains.mean(),sd,gains.mean()-margin,gains.mean()+margin],
            [s['mean_gain_db'],s['sd_db'],*s['seed_ci95_db']], atol=1e-10, rtol=0)
        if budget != 13:
            old = {x['seed']:x['bins']['snr_ge5']['paired_median_db'] for x in summary[f'13ep_{label}']['pairs']}
            change = np.array([x['bins']['snr_ge5']['paired_median_db']-old[x['seed']] for x in s['pairs']])
            margin = t.ppf(.975,2)*change.std(ddof=1)/np.sqrt(3)
            np.testing.assert_allclose(change,s['paired_changes_from_13ep_db'],atol=1e-10,rtol=0)
            np.testing.assert_allclose([change.mean(),change.mean()-margin,change.mean()+margin],
                [s['mean_paired_change_db'],*s['paired_change_seed_ci95_db']],atol=1e-10,rtol=0)
audit.update(passed=True, completed_runs=18, epoch_checkpoints=546, test_replays=36,
             matched_pairs=9, per_bin_and_aggregate_metrics_verified=True,
             paired_changes_and_seed_intervals_verified=True)
(OUT/'completion_audit.json').write_text(json.dumps(audit,indent=2,allow_nan=False)+'\n')
print('ALL CHECKS PASSED', flush=True)
