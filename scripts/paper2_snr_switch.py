"""Inference-only follow-up; existing experiment artifacts remain read-only."""
from __future__ import annotations

import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from scipy.stats import t

from paper2_long_training import (
    ROOT, OUT as SOURCE, CachedData, DataLoader, SEEDS, build_model,
    cfg_for, collate, hash_file, save_json, test_model,
)
from trackD_urformer.dataset import TrackDDataset

OUT = ROOT / 'results/paper2_snr_switch'
THRESHOLDS = (-11.0, 0.0, 2.5, 5.0, 7.5, 10.0, 21.0)


def choose_threshold(u_errors, h_errors, snr):
    """Inputs are normalized validation errors, seed by case. No test input."""
    scores = {threshold: float(np.where(snr[None, :] >= threshold,
                                        h_errors, u_errors).mean())
              for threshold in THRESHOLDS}
    selected = min(scores, key=lambda threshold: (scores[threshold], -threshold))
    return selected, scores


def route_mask(snr, threshold):
    return np.asarray(snr) >= threshold


def noise_snr(batch):
    # K=3, beta=1 and c=1, frozen physical configuration.
    return 10 * torch.log10(3.0 / batch['sigma2'].double())


def forward(model, batch):
    return model(*(batch[key] for key in ('G0', 'Z', 'S', 'B', 'sigma2')))


@torch.no_grad()
def routed_forward(u, h, batch, threshold, same_weights=False):
    """Only execute the selected branch; never compute both and pick by error."""
    use_h = noise_snr(batch) >= threshold
    result = torch.empty_like(batch['G0'])
    for mask, model, disable in ((~use_h, h if same_weights else u, same_weights),
                                 (use_h, h, False)):
        if not bool(mask.any()):
            continue
        part = {key: value[mask] for key, value in batch.items()}
        previous = [layer._disable_hankel for layer in model.layers]
        try:
            for layer in model.layers:
                layer._disable_hankel = disable
            result[mask] = forward(model, part)
        finally:
            for layer, old in zip(model.layers, previous):
                layer._disable_hankel = old
    return result


class Router(torch.nn.Module):
    def __init__(self, u, h, threshold, same_weights=False):
        super().__init__()
        self.u, self.h = u, h
        self.threshold, self.same_weights = threshold, same_weights

    def forward(self, G0, Z, S, B, sigma2):
        return routed_forward(self.u, self.h, dict(G0=G0, Z=Z, S=S, B=B,
                                                  sigma2=sigma2),
                              self.threshold, self.same_weights)


def load_pair(seed):
    models, provenance = {}, {}
    for arm in ('U', 'H'):
        folder = SOURCE / f'seed{seed}_{arm}_52ep'
        result = json.loads((folder / 'result.json').read_text())
        path = folder / f'epoch_{result["chosen_epoch_1based"]:03d}.pt'
        model, _ = build_model(cfg_for(seed, arm == 'H', 52), 'arm1b_full_random')
        model.load_state_dict(torch.load(path, map_location='cpu', weights_only=True)['model'])
        models[arm] = model.eval()
        provenance[arm] = {'path': str(path), 'sha256': hash_file(path),
                           'epoch': result['chosen_epoch_1based'],
                           'parameters': sum(p.numel() for p in model.parameters()),
                           'parameter_bytes': sum(p.numel()*p.element_size()
                                                  for p in model.parameters())}
    return models, provenance


def validate_record(record):
    assert all(x.shape == (2000,) and np.isfinite(x).all() for x in record.values())
    assert (record['numerator'] > 0).all() and (record['denominator'] > 0).all()
    assert len(np.unique(record['trial'])) == 2000


def metrics(candidate, reference, denominator, mask):
    a, b, d = candidate[mask], reference[mask], denominator[mask]
    return {
        'n': int(mask.sum()),
        'candidate_nmse_db': float(10*np.log10(np.mean(a/d))),
        'paired_median_gain_db': float(np.median(10*np.log10(b/a))),
        'raw_error_sum_gain_db': float(10*np.log10(b.sum()/a.sum())),
        'normalized_error_sum_gain_db': float(10*np.log10((b/d).sum()/(a/d).sum())),
    }


def interval(values):
    values = np.asarray(values)
    sd = float(values.std(ddof=1))
    mean = float(values.mean())
    margin = float(t.ppf(.975, len(values)-1)*sd/np.sqrt(len(values)))
    return {'per_seed': values.tolist(), 'mean': mean, 'sd': sd,
            'nominal_seed_ci95': [mean-margin, mean+margin]}


@torch.no_grad()
def timing(models, threshold, batches):
    u, h = models['U'], models['H']
    methods = {
        'U': lambda b: forward(u, b), 'H': lambda b: forward(h, b),
        'router': lambda b: routed_forward(u, h, b, threshold),
        'same_weights_toggle': lambda b: routed_forward(u, h, b, threshold, True),
    }
    for fn in methods.values():
        fn(batches[0])
    measured = {key: [] for key in methods}
    names = list(methods)
    for repeat in range(3):
        for key in names[repeat:] + names[:repeat]:
            start = time.perf_counter()
            for batch in batches:
                methods[key](batch)
            measured[key].append(1000*(time.perf_counter()-start)/256)
    return {key: {'ms_per_case_repeats': vals, 'median_ms_per_case': float(np.median(vals))}
            for key, vals in measured.items()}


def run():
    torch.set_num_threads(1)
    # Refuse accidental reruns/overwrites. Each study has exactly one test pass.
    OUT.mkdir(parents=True, exist_ok=False)
    save_json(OUT / 'status.json', {'state': 'validation'})
    manifest = {'script_sha256': hash_file(Path(__file__)),
                'plan_sha256': hash_file(ROOT / 'reports/paper2_snr_switch_plan.md'),
                'torch': torch.__version__, 'models': {},
                'source_driver_sha256': hash_file(ROOT / 'scripts/paper2_long_training.py'),
                'model_code_sha256': hash_file(ROOT / 'trackD_urformer/urformer.py')}
    source_hash = manifest['source_driver_sha256']
    val_ds = CachedData('val')
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, collate_fn=collate)
    val_errors = {'U': [], 'H': []}
    val_snr = 10*np.log10(3.0 / val_ds.arrays['sigma2'].astype(np.float64))
    np.testing.assert_allclose(val_snr, val_ds.arrays['snr_db'], atol=2e-6, rtol=0)
    for seed in SEEDS:
        models, manifest['models'][str(seed)] = load_pair(seed)
        for arm, model in models.items():
            record = test_model(model, val_loader)
            validate_record(record)
            val_errors[arm].append(record['numerator']/record['denominator'])
            np.savez(OUT / f'validation_seed{seed}_{arm}.npz', **record)
        print(f'Validation evaluated for seed {seed}', flush=True)
    threshold, scores = choose_threshold(np.array(val_errors['U']),
                                         np.array(val_errors['H']), val_snr)
    save_json(OUT / 'threshold.json', {'threshold_db': threshold,
              'validation_scores_mean_normalized_error': scores,
              'selection_uses_test': False})
    print(f'Validation-selected threshold frozen at {threshold} dB', flush=True)
    # Generate fresh test only AFTER freezing threshold.
    cfg = cfg_for(1, False, 52)
    cfg = replace(cfg, system=replace(cfg.system, master_seed=20260919),
                  data=replace(cfg.data, test_seed_range=(3000000, 4000000)))
    fresh = TrackDDataset('test', sysc=cfg.system, datac=cfg.data,
                         numeric=cfg.numeric, init='spectral')
    save_json(OUT / 'status.json', {'state': 'generating_fresh_test'})
    items = []
    for i in range(len(fresh)):
        items.append(fresh[i])
        if (i+1) % 500 == 0:
            print(f'Fresh test cases generated: {i+1}/2000', flush=True)
    arrays = {key: torch.stack([it[key] for it in items]).numpy() for key in items[0]}
    np.savez(OUT / 'fresh_test_data.npz', **arrays)
    manifest['fresh_test'] = {'config': cfg.to_dict(),
                             'sha256': hash_file(OUT / 'fresh_test_data.npz')}
    for split in ('train', 'val', 'test'):
        assert not np.intersect1d(arrays['trial'], CachedData(split).arrays['trial']).size
    snr = 10*np.log10(3.0 / arrays['sigma2'].astype(np.float64))
    np.testing.assert_allclose(snr, arrays['snr_db'], atol=2e-6, rtol=0)
    mask = route_mask(snr, threshold)
    rng = np.random.default_rng(2026091901)
    gaussian = rng.normal(size=len(items))
    masks = {'router': mask, 'router_fixed5': route_mask(snr, 5.0),
             'router_snr_sd1': route_mask(snr+gaussian, threshold),
             'router_snr_sd3': route_mask(snr+3*gaussian, threshold),
             'router_snr_bias_minus3': route_mask(snr-3, threshold),
             'router_snr_bias_plus3': route_mask(snr+3, threshold)}
    groups = {'all': np.ones(len(items), dtype=bool), 'low': snr < 5, 'high': snr >= 5}
    groups.update({f'[{lo},{lo+5})': (snr >= lo) & (snr < lo+5) for lo in range(-10,20,5)})
    loader = DataLoader(items, batch_size=32, shuffle=False, collate_fn=collate)
    batches = [collate(items[i:i+32]) for i in range(0,256,32)]
    results = []
    for seed in SEEDS:
        save_json(OUT / 'status.json', {'state': 'evaluating', 'seed': seed})
        models, current_provenance = load_pair(seed)
        assert current_provenance == manifest['models'][str(seed)]
        records = {arm: test_model(model, loader) for arm, model in models.items()}
        h = models['H']
        try:
            for layer in h.layers:
                layer._disable_hankel = True
            records['H_disabled'] = test_model(h, loader)
        finally:
            for layer in h.layers:
                layer._disable_hankel = False
        for record in records.values():
            validate_record(record)
            for key in ('denominator', 'trial', 'snr_db'):
                np.testing.assert_array_equal(record[key], records['U'][key])
        uerr, herr, offerr = [records[key]['numerator'] for key in ('U', 'H', 'H_disabled')]
        errors = {'U': uerr, 'H': herr, 'H_disabled': offerr,
                  'same_weights_toggle': np.where(mask, herr, offerr)}
        errors.update({key: np.where(m, herr, uerr) for key, m in masks.items()})
        for extreme, expected in ((-11, herr), (21, uerr)):
            np.testing.assert_array_equal(np.where(route_mask(snr, extreme), herr, uerr), expected)
        replay_diffs = {}
        for name, same_weights in (('router', False), ('same_weights_toggle', True)):
            replay = test_model(Router(models['U'], models['H'], threshold, same_weights), loader)
            np.testing.assert_allclose(replay['numerator'], errors[name], atol=1e-6, rtol=2e-5)
            replay_diffs[name] = float(np.max(np.abs(replay['numerator']-errors[name])))
        den = records['U']['denominator']
        scores_by_method = {name: {reference: {group: metrics(err, errors[reference], den, m)
                            for group, m in groups.items()} for reference in ('U', 'H')}
                            for name, err in errors.items()}
        np.savez(OUT / f'test_seed{seed}.npz', **errors, denominator=den,
                 snr_db=snr, trial=arrays['trial'], **{f'mask_{k}':v for k,v in masks.items()})
        result = {'seed': seed, 'metrics': scores_by_method,
                  'routing_replay_max_abs_error': replay_diffs,
                  'timing': timing(models, threshold, batches)}
        save_json(OUT / f'seed{seed}.json', result)
        results.append(result)
        print(f'Seed {seed} complete; full-range router gain vs U: '
              f'{scores_by_method["router"]["U"]["all"]["normalized_error_sum_gain_db"]:.4f} dB', flush=True)
    aggregated = {}
    for method in results[0]['metrics']:
        aggregated[method] = {reference: {group: {metric: interval([
            row['metrics'][method][reference][group][metric] for row in results])
            for metric in ('candidate_nmse_db', 'paired_median_gain_db',
                           'raw_error_sum_gain_db', 'normalized_error_sum_gain_db')}
            for group in groups} for reference in ('U', 'H')}
    save_json(OUT / 'summary.json', {'threshold_db': threshold, 'n_test': len(items),
              'n_seeds': 3, 'groups_n': {k:int(v.sum()) for k,v in groups.items()},
              'fraction_routed_H': {k:float(v.mean()) for k,v in masks.items()},
              'aggregate': aggregated})
    assert source_hash == hash_file(ROOT / 'scripts/paper2_long_training.py')
    save_json(OUT / 'manifest.json', manifest)
    save_json(OUT / 'status.json', {'state': 'complete', 'seeds': 3,
              'fresh_test_cases': 2000, 'threshold_db': threshold,
              'branch_replay_and_data_separation_checks_passed': True})
    print('All switching evaluations and branch checks passed.', flush=True)


if __name__ == '__main__':
    run()
