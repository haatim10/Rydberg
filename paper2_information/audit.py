"""Read-only independent checks of the frozen 40-run confirmation study.

All new outputs belong in the new experiment folder, never in the old study.
Uses legacy model constructors for checkpoint replay, not legacy statistics.
"""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.stats import t

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT/'results/paper2_hankel_confirmation'


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def db(values):
    return float(10*np.log10(np.mean(values)))


def verify_sources():
    manifest = read(OLD/'manifest.json')
    for path, expected in manifest['source_hashes'].items():
        if digest(ROOT/path) != expected:
            raise RuntimeError(f'Frozen source changed: {path}')
    return len(manifest['source_hashes'])


def audit_completed():
    if read(OLD/'status.json')['state'] != 'complete':
        raise RuntimeError('Existing confirmation study is not complete')
    import torch
    from torch.utils.data import DataLoader
    torch.set_num_threads(1)
    sys.path.insert(0, str(ROOT/'scripts'))
    import paper2_hankel_confirmation as old
    source_count = verify_sources()
    manifest, selection, summary = (read(OLD/name) for name in ('manifest.json', 'selection.json', 'summary.json'))
    assert selection['source_hashes'] == manifest['source_hashes']
    counts = {'training_records': 0, 'epoch_histories': 0, 'test_files': 0,
              'data_hashes': 0, 'checkpoint_replays': 0, 'data_regeneration_cases': 0}
    result = {'source_hashes_verified': source_count, 'modes': {}}
    seen_trials = set()
    for mode in manifest['modes']:
        for split, count in manifest['counts'].items():
            folder = OLD/mode/'data'/split
            meta = read(folder/'complete.json')
            assert meta['count'] == count
            for name, expected in meta['sha256'].items():
                assert digest(folder/f'{name}.npy') == expected, (mode, split, name)
                counts['data_hashes'] += 1
            ids = np.load(folder/'trial.npy')
            assert len(np.unique(ids)) == count and not seen_trials.intersection(ids.tolist())
            seen_trials.update(ids.tolist())
            disk, generator = old.DiskData(mode, split), old.Worlds(mode, split)
            for index in (0, count//2, count-1):
                for name, value in generator[index].items():
                    np.testing.assert_array_equal(value.numpy(), disk[index][name].numpy())
                counts['data_regeneration_cases'] += 1
        data = old.DiskData(mode, 'test')
        trial = np.asarray(data.arrays['trial'])
        snr = np.asarray(data.arrays['snr_db'])
        true_energy = np.sum(np.abs(np.asarray(data.arrays['G_true'], dtype=np.complex128))**2, axis=(1, 2))
        masks = {'all': np.ones(len(trial), bool), 'snr_ge5': snr >= 5}
        masks.update({f'[{lo},{lo+5})': (snr >= lo)&(snr < lo+5) for lo in range(-10, 20, 5)})
        arrays, means = {}, {arm: [] for arm in old.ARMS}
        for seed in manifest['seeds']:
            hashes, orders = [], []
            # Recreate the actual DataLoader permutation, including its base-seed draw.
            train_ids = np.load(OLD/mode/'data/train/trial.npy')
            expected_orders = []
            for epoch in range(18):
                batches = DataLoader(range(len(train_ids)), batch_size=32, shuffle=True,
                                    generator=torch.Generator().manual_seed(seed+epoch))
                order_hash = hashlib.sha256()
                for indices in batches:
                    order_hash.update(train_ids[np.asarray(indices)].tobytes())
                expected_orders.append(order_hash.hexdigest())
            for arm in old.ARMS:
                folder = old.folder(mode, seed, arm)
                record = read(folder/'result.json')
                history = record['history']
                val = np.load(folder/'validation_errors.npz')['errors']
                assert val.shape == (18, 1200) and np.isfinite(val).all() and (val >= 0).all()
                assert len(history) == 18 and len(list(folder.glob('epoch_*.pt'))) == 18
                curve = 10*np.log10(val.mean(axis=1))
                chosen, rule = old.base.select_epoch(curve, val[curve.argmin()], 'one_se')
                assert chosen+1 == record['chosen_epoch'] == selection['modes'][mode]['epochs'][f'{seed}/{arm}']
                assert rule == record['selection']
                np.testing.assert_allclose(curve, [x['validation_nmse_db'] for x in history], atol=1e-12, rtol=0)
                assert all(np.isfinite(x['max_gradient_norm']) for x in history)
                actual_order = [x['data_order_sha256'] for x in history]
                assert actual_order == expected_orders
                orders.append(actual_order); hashes.append(record['initial_hash'])
                means[arm].append(float(val[chosen].mean()))
                counts['training_records'] += 1; counts['epoch_histories'] += 18
                for label in ('selected', 'final'):
                    path = folder/f'test_{label}.npz'
                    f = np.load(path)
                    for name, expected in (('trial', trial), ('snr_db', snr)):
                        np.testing.assert_array_equal(f[name], expected)
                    assert f['numerator'].shape == (1800,) and np.isfinite(f['numerator']).all() and (f['numerator'] >= 0).all()
                    assert np.isfinite(f['denominator']).all() and (f['denominator'] > 0).all()
                    np.testing.assert_allclose(f['denominator'], true_energy, rtol=1e-6)
                    assert float(f['elapsed_seconds']) > 0
                    arrays[(seed, arm, label)] = f['numerator']/f['denominator']
                    counts['test_files'] += 1
                    # First training seed, every arm, both checkpoint choices, first full batch.
                    if seed == manifest['seeds'][0]:
                        with old.configured(mode, seed):
                            model, _ = old.build(arm)
                            assert old.base.weight_hash(model) == record['initial_hash']
                            ep = chosen+1 if label == 'selected' else 18
                            model.load_state_dict(torch.load(folder/f'epoch_{ep:03d}.pt', weights_only=True)['model'])
                            replay = old.evaluate(model, [data[i] for i in range(32)])
                        for name in ('numerator', 'denominator'):
                            np.testing.assert_allclose(replay[name], f[name][:32], rtol=2e-5, atol=1e-5)
                        counts['checkpoint_replays'] += 1
            assert len(set(hashes)) == 1 and all(x == orders[0] for x in orders)
        means = {a: float(np.mean(v)) for a, v in means.items()}
        sel = selection['modes'][mode]
        for arm, val in means.items():
            np.testing.assert_allclose(val, sel['model_validation_means'][arm], rtol=1e-12)
        assert sel['best_U'] == min(('standard_U', 'weighted_U'), key=means.get)
        assert sel['best_H'] == min(('standard_H', 'weighted_H'), key=means.get)
        classical_val, classical_errors = {}, {}
        output = {'classical': {}, 'contrasts': {}, 'absolute': {}}
        for method in old.CLASSICAL:
            vf = np.load(OLD/mode/'classical/val'/f'{method}.npz')
            classical_val[method] = float(np.mean(vf['numerator']/vf['denominator']))
            f = np.load(OLD/mode/'classical/test'/f'{method}.npz')
            np.testing.assert_array_equal(f['trial'], trial)
            np.testing.assert_array_equal(f['snr_db'], snr)
            e = f['numerator']/f['denominator']
            assert e.shape == (1800,) and np.isfinite(e).all() and (e >= 0).all() and (f['denominator'] > 0).all()
            np.testing.assert_allclose(f['denominator'], true_energy, rtol=1e-6)
            classical_errors[method] = e
            counts['test_files'] += 1
            output['classical'][method] = {key: {'nmse_db': db(e[mask]), 'severe_error_rate': float(np.mean(e[mask] > 1))} for key, mask in masks.items()}
            output['classical'][method]['seconds_per_case'] = float(f['elapsed_seconds'])/len(e)
            for key in masks:
                for metric, value in output['classical'][method][key].items():
                    np.testing.assert_allclose(value, summary['modes'][mode]['classical'][method]['bins'][key][metric], atol=1e-10)
        assert sel['best_classical'] == min(old.CLASSICAL, key=classical_val.get)
        tuned = {}
        for steps in old.STOPS:
            f = np.load(OLD/mode/'classical/tune'/f'emgs{steps}.npz')
            tuned[steps] = float(np.mean(f['numerator']/f['denominator']))
        assert sel['tuned_emgs_steps'] == min(old.STOPS, key=tuned.get)
        for label in ('selected', 'final'):
            for arm in old.ARMS:
                errors = np.stack([arrays[(seed, arm, label)] for seed in old.SEEDS])
                output['absolute'][f'{label}/{arm}'] = {key: {'nmse_db': db(errors[:, mask]), 'severe_error_rate': float(np.mean(errors[:, mask] > 1))} for key, mask in masks.items()}
                for key in masks:
                    for metric, value in output['absolute'][f'{label}/{arm}'][key].items():
                        np.testing.assert_allclose(value, summary['modes'][mode]['absolute'][f'{label}/{arm}'][key][metric], atol=1e-10)
            for loss in ('standard', 'weighted'):
                contrasts = {}
                for key, mask in masks.items():
                    gains = np.array([db(arrays[(seed, loss+'_U', label)][mask])-db(arrays[(seed, loss+'_H', label)][mask]) for seed in old.SEEDS])
                    margin = t.ppf(.9875, 4)*gains.std(ddof=1)/np.sqrt(5)
                    observed = summary['modes'][mode]['contrasts'][f'{label}/{loss}'][key]
                    np.testing.assert_allclose(gains, observed['per_seed_gain_db'], atol=1e-10)
                    np.testing.assert_allclose([gains.mean()-margin, gains.mean()+margin], observed['seed_ci975'], atol=1e-10)
                    np.testing.assert_allclose(gains.mean(), observed['mean_gain_db'], atol=1e-10)
                    np.testing.assert_allclose(gains.std(ddof=1), observed['seed_sd_db'], atol=1e-10)
                    medians = [float(np.median(10*np.log10(arrays[(seed, loss+'_U', label)][mask]/arrays[(seed, loss+'_H', label)][mask]))) for seed in old.SEEDS]
                    np.testing.assert_allclose(medians, observed['per_seed_paired_median_db'], atol=1e-10)
                    contrasts[key] = {'gain_db': float(gains.mean()), 'seed_ci975': [float(gains.mean()-margin), float(gains.mean()+margin)]}
                output['contrasts'][f'{label}/{loss}'] = contrasts
            best_h = np.stack([arrays[(seed, sel['best_H'], label)] for seed in old.SEEDS])
            classical_gain = db(classical_errors[sel['best_classical']][masks['snr_ge5']])-db(best_h[:, masks['snr_ge5']])
            output[f'{label}_bestH_vs_selected_classical_high_gain_db'] = classical_gain
            np.testing.assert_allclose(classical_gain, summary['modes'][mode][f'{label}_bestH_vs_selected_classical_high_gain_db'], atol=1e-10)
        primary = output['contrasts']['selected/'+sel['primary_loss']]
        screen = {'primary_loss': sel['primary_loss'],
                  'gain_at_least_05': primary['snr_ge5']['gain_db'] >= .5,
                  'adjusted_interval_positive': primary['snr_ge5']['seed_ci975'][0] > 0,
                  'full_range_preserved': primary['all']['gain_db'] >= -.2,
                  'beats_selected_classical_by_05': output['selected_bestH_vs_selected_classical_high_gain_db'] >= .5}
        screen['all_pass'] = all(v for k, v in screen.items() if k != 'primary_loss')
        assert screen == summary['modes'][mode]['screen']
        output['screen'] = screen
        result['modes'][mode] = output
    assert counts['training_records'] == 40 and counts['epoch_histories'] == 720 and counts['test_files'] == 94
    result.update(passed=True, counts=counts, legacy_summary_sha256=digest(OLD/'summary.json'),
                  limitations='Checkpoint replay is representative (16 x 32 cases), not every checkpoint on every test case. Seed intervals do not include dataset/pilot uncertainty. Numerical audit is not a novelty or publication verdict.')
    return result


def report_text(result):
    lines = ['# Completed 40-run study: independent numerical audit', '',
             f"Audit passed: {result['passed']}. Verified counts: {result['counts']}.", '',
             'Positive gain means Hankel reduced error. Gains below average paired per-training-seed dB mean-NMSE differences, not per-case medians.', '',
             '| Channel | Checkpoint / objective | Full-range gain (dB) | SNR ≥5 dB gain (dB) |',
             '|---|---|---:|---:|']
    for mode, values in result['modes'].items():
        for label, contrast in values['contrasts'].items():
            lines.append(f"| {mode} | {label} | {contrast['all']['gain_db']:.3f} | {contrast['snr_ge5']['gain_db']:.3f} |")
    lines += ['', '## Classical controls and the declared screen', '']
    for mode, values in result['modes'].items():
        lines.append(f"- {mode}: declared screen passed = {values['screen']['all_pass']}; selected learned Hankel versus the validation-selected classical comparator at high SNR = {values['selected_bestH_vs_selected_classical_high_gain_db']:.3f} dB.")
    lines += ['', 'Read legacy_audit.json for every bin, severe-error frequency, timing, selected/final results and seed intervals. Read the old study’s summary.json for its conditional case bootstrap outputs.', '',
              result['limitations'], '',
              'Five training seeds share the same dataset and pilot matrix. Their intervals do not establish replication across channel populations. The comparison is an independently implemented classical-start learned estimator, not certified reproduction of published URformer. Rank seven and the identity-backward approximation remain features of this old protocol.', '',
              'This audit checks numerical consistency; it does not establish a universal Hankel advantage, explain every failure causally, or certify novelty. The new feasibility work uses fresh development worlds and separate success rules. No historical output or manuscript has been edited.']
    return '\n'.join(lines)+'\n'
