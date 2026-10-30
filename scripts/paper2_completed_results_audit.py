"""Read-only audit of completed experiments; prints JSON, never retrains or edits results."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from scipy.stats import t

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'results'


def read(path):
    return json.loads(path.read_text())


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def near(a, b):
    np.testing.assert_allclose(a, b, rtol=1e-10, atol=1e-10)


def db(x):
    return float(10 * np.log10(np.mean(x)))


def main():
    audit = {'scope': 'Recompute saved per-trial statistics and validation checkpoint selection; not fresh model inference.',
             'checks': {}, 'main': {}, 'training': []}
    for exp in ('paper2_original_channel', 'paper2_reproduction', 'paper2_reproduction_diagnostic'):
        p = RESULTS / exp
        m = read(p / 'manifest.json')
        assert read(p / 'status.json')['state'] == 'complete'
        for filename, expected in m['source_hashes'].items():
            assert sha(ROOT / filename) == expected, filename
        caches = list(p.glob('**/data/*/complete.json'))
        n = 0
        for c in caches:
            meta = read(c)
            for key, expected in meta['sha256'].items():
                assert sha(c.parent / f'{key}.npy') == expected, str(c)
                n += 1
            assert len(np.load(c.parent / 'trial.npy')) == meta['count']
        for data in {c.parent.parent for c in caches}:
            ids = {s: np.unique(np.load(data / s / 'trial.npy')) for s in ('train', 'val', 'test')
                   if (data / s / 'trial.npy').exists()}
            for s in ids:
                for q in ids:
                    if s < q:
                        assert np.intersect1d(ids[s], ids[q]).size == 0
        audit['checks'][exp] = {'complete': True, 'source_hashes_verified': len(m['source_hashes']),
                                'cache_hashes_verified': n, 'disjoint_split_ids': True}

    folders = list((RESULTS / 'paper2_original_channel').glob('*/seed*_50ep'))
    folders += list((RESULTS / 'paper2_reproduction').glob('seed*_50ep'))
    folders += [RESULTS / 'paper2_reproduction_diagnostic' / a for a in ('mixed', 'fixed5')]
    assert len(folders) == 17
    for p in sorted(folders):
        r = read(p / 'result.json')
        hist = r['history']
        n = r.get('budget', r.get('epochs'))
        assert len(hist) == n and [v['epoch'] for v in hist] == list(range(1, n + 1))
        assert hist == read(p / 'history.json') and len(list(p.glob('epoch_*.pt'))) == n
        b = torch.load(p / 'checkpoint.pt', map_location='cpu', weights_only=False)
        assert b['epoch'] == n - 1 and b['history'] == hist
        # JSON serializes Python tuple config fields as lists; canonicalize before comparison.
        assert json.loads(json.dumps(b['config'])) == r['config']
        v = np.asarray(b['validations'])
        assert v.shape == (n, 2000) and np.isfinite(v).all() and (v >= 0).all()
        curves = 10 * np.log10(v.mean(axis=1))
        key = 'validation_db' if 'validation_db' in hist[0] else 'validation_5db_nmse_db'
        near(curves, [h[key] for h in hist])
        best = int(curves.argmin())
        rng = np.random.default_rng(20260830)
        boot = np.array([rng.choice(v[best], 2000, replace=True).mean() for _ in range(2000)])
        se = float(np.std(10 * np.log10(boot), ddof=1))
        chosen = int(np.argmax(curves <= curves[best] + se))
        assert chosen + 1 == r['chosen_epoch_1based']
        near(se, r['selection']['se_db'])
        final = torch.load(p / f'epoch_{n:03d}.pt', map_location='cpu', weights_only=True)['model']
        assert all(torch.equal(final[k], x) for k, x in b['model'].items())
        selected = torch.load(p / f'epoch_{chosen+1:03d}.pt', map_location='cpu', weights_only=True)['model']
        assert all(torch.isfinite(x).all() for x in selected.values())
        audit['training'].append({'run': str(p.relative_to(RESULTS)), 'epochs': n,
            'selected_epoch': chosen + 1, 'best_validation_epoch': best + 1,
            'best_validation_db': float(curves[best]), 'final_validation_db': float(curves[-1]),
            'train_first': hist[0]['train_loss'], 'train_final': hist[-1]['train_loss'],
            'median_epoch_seconds': float(np.median([h['seconds'] for h in hist]))})
        del b, final, selected

    for mode in ('literal_independent', 'clustered_5deg'):
        base = RESULTS / 'paper2_original_channel' / mode
        summary = read(base / 'summary.json')
        audit['main'][mode] = {}
        for seed in range(1, 4):
            rr = [read(base / f'seed{seed}_{a}_50ep/result.json') for a in ('U', 'H')]
            assert [h['data_order_sha256'] for h in rr[0]['history']] == [h['data_order_sha256'] for h in rr[1]['history']]
            assert rr[0]['config']['train'] == rr[1]['config']['train']
        for label in ('selected', 'final'):
            assert summary[label]['n_seeds'] == 3
            groups = {}
            for seed in range(1, 4):
                u, h = [np.load(base / f'seed{seed}_{a}_50ep' / f'test_{label}.npz') for a in ('U', 'H')]
                for key in ('trial', 'snr_db', 'denominator'):
                    np.testing.assert_array_equal(u[key], h[key])
                assert len(u['trial']) == len(np.unique(u['trial'])) == 2000
                for a in (u, h):
                    assert all(np.isfinite(a[k]).all() for k in a.files)
                    assert (a['numerator'] > 0).all() and (a['denominator'] > 0).all()
                masks = {'all': np.ones(2000, dtype=bool), 'snr_ge5': u['snr_db'] >= 5, 'snr_lt5': u['snr_db'] < 5}
                masks.update({f'[{lo},{lo+5})': (u['snr_db'] >= lo) & (u['snr_db'] < lo+5) for lo in range(-10, 20, 5)})
                assert sum(m.sum() for k, m in masks.items() if k.startswith('[')) == 2000
                for group, mask in masks.items():
                    a, b, d = u['numerator'][mask], h['numerator'][mask], u['denominator'][mask]
                    stat = {'n': int(mask.sum()), 'paired_median_db': float(np.median(10*np.log10(a/b))),
                        'raw_error_sum_gain_db': float(10*np.log10(a.sum()/b.sum())),
                        'normalized_error_sum_gain_db': float(10*np.log10((a/d).sum()/(b/d).sum())),
                        'U_nmse_db': db(a/d), 'H_nmse_db': db(b/d)}
                    if group != 'snr_lt5':
                        ref = next(p for p in summary[label]['pairs'] if p['seed'] == seed)['bins'][group]
                        for k, value in stat.items():
                            near(value, ref[k])
                    groups.setdefault(group, []).append({**stat, 'u_mean': float((a/d).mean()),
                        'h_mean': float((b/d).mean()), 'h_win_fraction': float((b < a).mean())})
            ag = {}
            for group, rows in groups.items():
                vals = np.array([x['paired_median_db'] for x in rows])
                margin = float(t.ppf(.975, 2)*vals.std(ddof=1)/np.sqrt(3))
                um, hm = [np.mean([x[k] for x in rows]) for k in ('u_mean', 'h_mean')]
                ag[group] = {'n_channels': rows[0]['n'], 'mean_seed_paired_median_db': float(vals.mean()),
                    'nominal_seed_ci95': [float(vals.mean()-margin), float(vals.mean()+margin)],
                    'pooled_mean_nmse_gain_db': float(10*np.log10(um/hm)),
                    'pooled_U_nmse_db': db(um), 'pooled_H_nmse_db': db(hm), 'H_over_U_error_ratio': float(hm/um),
                    'seed_mean_nmse_gains_db': [x['normalized_error_sum_gain_db'] for x in rows],
                    'seed_paired_median_gains_db': vals.tolist(),
                    'mean_seed_h_win_fraction': float(np.mean([x['h_win_fraction'] for x in rows]))}
            near(ag['snr_ge5']['mean_seed_paired_median_db'], summary[label]['mean_gain_db'])
            near(ag['snr_ge5']['nominal_seed_ci95'], summary[label]['nominal_seed_ci95'])
            audit['main'][mode][label] = ag

    p = RESULTS / 'paper2_reproduction'
    s, target = read(p / 'summary.json'), read(p / 'figure3_targets.json')
    audit['reproduction'] = {}
    for label in ('selected', 'final'):
        rows = []
        for snr in target['snr_db']:
            means = []
            for seed in range(1, 4):
                a = np.load(p / f'seed{seed}_U_50ep' / f'test_{label}.npz')
                assert len(a['trial']) == 14000
                mask = a['snr_db'] == snr
                assert mask.sum() == 2000
                value = np.mean(a['numerator'][mask] / a['denominator'][mask])
                assert np.isfinite(value) and value > 0
                means.append(value)
                ref = next(r for r in s['results'] if r['seed'] == seed and r['checkpoint'] == label)
                point = next(x for x in ref['points'] if x['snr_db'] == snr)
                near(db(value), point['nmse_db'])
            paper = target['curves_nmse_db']['URformer'][target['snr_db'].index(snr)]
            rows.append({'snr_db': snr, 'ours_pooled_db': db(means), 'paper_plot_db': paper, 'gap_db': db(means)-paper})
        audit['reproduction'][label] = rows
    audit['classical'] = read(p / 'classical.json')
    for name, points in audit['classical'].items():
        a = np.load(p / f'classical_{name}.npz')
        for point in points:
            mask = a['snr_db'] == point['snr_db']
            near(db(a['numerator'][mask]/a['denominator'][mask]), point['nmse_db'])

    p = RESULTS / 'paper2_reproduction_diagnostic'
    rr = [read(p / a / 'result.json') for a in ('mixed', 'fixed5')]
    vv = [np.load(p / a / 'validation_errors.npz') for a in ('mixed', 'fixed5')]
    assert rr[0]['initial_hash'] == rr[1]['initial_hash']
    assert [h['data_order_sha256'] for h in rr[0]['history']] == [h['data_order_sha256'] for h in rr[1]['history']]
    for key in ('G_true', 'G0', 'S', 'B', 'trial'):
        assert sha(p / 'mixed/data/train' / f'{key}.npy') == sha(p / 'fixed5/data/train' / f'{key}.npy')
    for f in (p / 'mixed/data/val').glob('*.npy'):
        assert sha(f) == sha(p / 'fixed5/data/val' / f.name)
    d = read(p / 'summary.json')
    for label in ('final', 'selected'):
        errors = [v['errors'][-1 if label == 'final' else r['chosen_epoch_1based']-1] for v, r in zip(vv, rr)]
        near(db(errors[0]), d['checkpoint_results'][label]['mixed_db'])
        near(db(errors[1]), d['checkpoint_results'][label]['fixed5_db'])
        near(db(errors[0])-db(errors[1]), d['checkpoint_results'][label]['fixed5_gain_db'])
    audit['diagnostic'] = d
    audit['checks']['totals'] = {'run_results': 17, 'epoch_records': 770, 'main_test_files': 24,
        'main_matched_50_epoch_orders': 6, 'main_summary_bins_replayed': 96,
        'reproduction_test_files': 6, 'reproduction_curve_points': 42, 'classical_curve_points': 14,
        'validation_checkpoint_selections_recomputed': 17}
    audit['passed'] = True
    print(json.dumps(audit, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
