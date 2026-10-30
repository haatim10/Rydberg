"""Numerical audit and predeclared conditional paired comparisons."""
import json

import numpy as np
import torch

from fair_optshrink.estimator import lookup_key, tensors
from optshrink_rank_check.estimator import parameters
from paper2_optshrink_loop.estimator import estimate as numpy_estimate
from paper2_optshrink_loop.estimator import trajectory as numpy_trajectory
from sahure.classical import likelihood
from sahure_expansion.data import world

from .evaluate import fixed_config, frozen_parent
from .models import load_models
from .protocol import CONFIG, ROOT, SPECS, load, save, sha, write


def reference_loop(row, parent):
    """Independent NumPy candidate fitting, held-out scoring and all-pilot refit."""
    obs = tensors(row)
    base = parent['choices'][lookup_key(obs)[0]]['loop']
    z, s, b, variance = [row[key][0] for key in ('z', 's', 'b', 'sigma2')]
    params = parameters(base['strength'])
    count = base['iterations']
    candidates = numpy_trajectory(z[:, :21], s[:, :21], b[:, :21],
                                  variance, params, (count,))[count]
    scores = likelihood(z[:, 21:], candidates @ s[:, 21:] + b[:, 21:],
                        variance).mean((-2, -1))
    index = int(scores.argmin())
    config = dict(**params[index], iterations=count, placement='loop')
    return numpy_estimate(z, s, b, variance, config), scores, config


@torch.no_grad()
def audit(check):
    freeze_path = ROOT / 'frozen_selection.json'
    frozen = json.loads(freeze_path.read_text())
    digest = sha(freeze_path)
    if 'manifest_sha256' in frozen and frozen['manifest_sha256'] != sha(ROOT / 'manifest.json'):
        raise RuntimeError('frozen experiment manifest changed')
    manifest = json.loads((ROOT / 'manifest.json').read_text())
    for path, value in manifest['sources'].items():
        if sha(path) != value:
            raise RuntimeError('frozen experiment source/data changed: ' + path)
    models, parent = load_models(), frozen_parent()
    ids, hashes = set(), {}
    replays, classical_replays, maximum_difference = 0, 0, 0.
    # Fixed before results: every eighth condition, in both fresh banks.
    reference_cells = set(range(0, len(SPECS), 8))
    for bank in range(len(CONFIG['banks'])):
        for cell, spec in enumerate(SPECS):
            check('audit', bank=bank, cell=cell)
            folder = ROOT / 'data' / f'bank{bank}' / spec['id']
            meta = json.loads((folder / 'manifest.json').read_text())
            if meta['selection_sha256'] != digest or meta['generated_unix'] < frozen['frozen_unix']:
                raise RuntimeError('test generation preceded selection freeze')
            if meta['seed'] != CONFIG['banks'][bank]:
                raise RuntimeError('test bank seed mismatch')
            expected_indices = list(range(cell * CONFIG['cases'], (cell + 1) * CONFIG['cases']))
            if meta['indices'] != expected_indices or meta['count'] != CONFIG['cases']:
                raise RuntimeError('unexpected world indices')
            for path, value in meta['hashes'].items():
                if sha(path) != value:
                    raise RuntimeError('input hash mismatch: ' + path)
                hashes[path] = value
            observed = load(folder / 'observations.npz')
            truth = load(folder / 'truth.npz')
            path = ROOT / 'predictions' / f'bank{bank}_{spec["id"]}.npz'
            result = load(path)
            hashes[str(path)] = sha(path)
            names = result['names'].tolist()
            if str(result['selection_sha256']) != digest:
                raise RuntimeError('prediction freeze mismatch')
            expected_ids = np.array([f'compact:{meta["seed"]}:{i}' for i in expected_indices])
            for current_ids in (observed['trial_ids'], truth['trial_ids'], result['trial_ids']):
                np.testing.assert_array_equal(current_ids, expected_ids)
            if ids.intersection(expected_ids):
                raise RuntimeError('duplicate test-world identifiers')
            ids.update(expected_ids)
            for case, index in enumerate(expected_indices):
                w, _ = world(index, meta['seed'], **spec)
                for key, value in zip(('z', 's', 'b', 'sigma2'), (w.Z, w.S, w.B, w.sigma2)):
                    np.testing.assert_array_equal(observed[key][case], value)
                np.testing.assert_array_equal(truth['g'][case], w.G)
            row = {key: observed[key][:1] for key in ('z', 's', 'b', 'sigma2')}
            obs = tensors(row)
            for name, model in models.items():
                expected = model(*obs.args()).numpy()[0]
                actual = result['estimates'][names.index(name), 0]
                np.testing.assert_allclose(actual, expected, rtol=3e-8, atol=3e-9)
                maximum_difference = max(maximum_difference, float(abs(actual - expected).max()))
                replays += 1
            if cell in reference_cells:
                expected, scores, config = reference_loop(row, parent)
                np.testing.assert_allclose(result['rank_scores'][0], scores, rtol=3e-8, atol=3e-9)
                np.testing.assert_array_equal(result['rank_configs'][0],
                                              [config['rank'], config['strength'], config['iterations']])
                actual = result['estimates'][names.index('rank_loop'), 0]
                np.testing.assert_allclose(actual, expected, rtol=3e-8, atol=3e-9)
                maximum_difference = max(maximum_difference, float(abs(actual - expected).max()))
                classical_replays += 1
                config = fixed_config(obs, parent)
                expected = numpy_estimate(row['z'][0], row['s'][0], row['b'][0],
                                          float(row['sigma2'][0]), config)
                actual = result['estimates'][names.index('opt_selected'), 0]
                np.testing.assert_allclose(actual, expected, rtol=3e-8, atol=3e-9)
                maximum_difference = max(maximum_difference, float(abs(actual - expected).max()))
                classical_replays += 1
            timing = json.loads((ROOT / 'timing' / f'{spec["id"]}.json').read_text())
            if timing['selection_sha256'] != digest or set(timing['seconds']) != set(names):
                raise RuntimeError('timing methods/freeze mismatch')
            if any(len(values) != 3 or not np.isfinite(values).all() or min(values) <= 0
                   for values in timing['seconds'].values()):
                raise RuntimeError('invalid timing records')
    result = dict(status='passed', regenerated_test=len(ids), neural_replays=replays,
                  independent_classical_replays=classical_replays,
                  replay_max_abs_error=maximum_difference, hashes=hashes,
                  scope='all-world input regeneration; one neural replay per cell/bank; NumPy classical replay every eighth cell/bank')
    write(ROOT / 'audit.json', result)
    return result


def analyze(check=lambda *args, **kwargs: None):
    verification = audit(check)
    errors, powers = [], []
    names = None
    for bank in range(len(CONFIG['banks'])):
        bank_errors, bank_power = [], []
        for spec in SPECS:
            result = load(ROOT / 'predictions' / f'bank{bank}_{spec["id"]}.npz')
            if names is None:
                names = result['names'].tolist()
            else:
                np.testing.assert_array_equal(result['names'], names)
            truth = load(ROOT / 'data' / f'bank{bank}' / spec['id'] / 'truth.npz')['g']
            bank_errors.append((abs(result['estimates'] - truth[None]) ** 2).sum((-2, -1)))
            bank_power.append((abs(truth) ** 2).sum((-2, -1)))
        errors.append(bank_errors)
        powers.append(bank_power)
    errors, power = np.asarray(errors), np.asarray(powers)
    if not np.isfinite(errors).all() or not np.isfinite(power).all() or np.any(power <= 0):
        raise FloatingPointError('invalid evaluation arrays')
    ratio = errors / power[:, :, None, :]
    primary_cells = 126
    groups = {arm: [names.index(f'{arm}_{seed}') for seed in CONFIG['seeds']]
              for arm in ('none', 'global', 'adaptive', 'recurrent')}
    groups.update({name: [names.index(name)] for name in ('opt_selected', 'rank_loop')})
    save(ROOT / 'all_errors.npz', squared_error=errors, channel_power=power,
         names=np.array(names), cell_ids=np.array([spec['id'] for spec in SPECS]))
    db = 10 * np.log10(errors.sum((0, 3)) / power.sum((0, 2))[:, None])
    bank_db = 10 * np.log10(errors.sum(-1) / power.sum(-1)[..., None])
    primary = {group: float(db[:primary_cells, indexes].mean()) for group, indexes in groups.items()}
    cells = {}
    for cell, spec in enumerate(SPECS):
        cells[spec['id']] = {
            group: dict(nmse_db=float(db[cell, indexes].mean()),
                        mean_case_nmse=float(ratio[:, cell, indexes, :].mean()),
                        exceedance={str(t): float((ratio[:, cell, indexes, :] > t).mean())
                                    for t in (.5, 1., 2.)})
            for group, indexes in groups.items()}
    save(ROOT / 'error_distributions.npz',
         **{group: np.sort(ratio[:, :primary_cells, indexes, :].ravel())
            for group, indexes in groups.items()})
    rng = np.random.default_rng(CONFIG['bootstrap_seed'])
    comparators, draws = ('none', 'global', 'recurrent'), []
    for start in range(0, CONFIG['bootstrap_samples'], 100):
        check('bootstrap', done=start)
        count = min(100, CONFIG['bootstrap_samples'] - start)
        weights = rng.multinomial(CONFIG['cases'], np.ones(CONFIG['cases']) / CONFIG['cases'],
                                  size=(count, len(CONFIG['banks']), primary_cells))
        error = np.einsum('rbcn,bcmn->rcm', weights, errors[:, :primary_cells], optimize=True)
        norm = np.einsum('rbcn,bcn->rc', weights, power[:, :primary_cells], optimize=True)
        sampled_db = 10 * np.log10(error / norm[..., None])
        adaptive_db = sampled_db[..., groups['adaptive']].mean(-1)
        draws.append(np.stack([(sampled_db[..., groups[group]].mean(-1) - adaptive_db).mean(1)
                               for group in comparators], -1))
    draws = np.concatenate(draws)
    alpha = .05 / (2 * len(comparators))
    contrasts = {
        group: dict(gain_db=primary[group] - primary['adaptive'],
                    simultaneous95_macro_ci=np.quantile(draws[:, index], [alpha, 1 - alpha]).tolist(),
                    worst_cell_deterioration_db=float(np.max(
                        db[:primary_cells, groups['adaptive']].mean(-1) -
                        db[:primary_cells, groups[group]].mean(-1))))
        for index, group in enumerate(comparators)}
    timing_rows = [json.loads((ROOT / 'timing' / f'{spec["id"]}.json').read_text()) for spec in SPECS]
    timing = {}
    for group, indexes in groups.items():
        values = [value for row in timing_rows[:primary_cells] for index in indexes
                  for value in row['seconds'][names[index]]]
        timing[group] = dict(mean_ms=float(1000 * np.mean(values)), median_ms=float(1000 * np.median(values)))
    models = load_models()
    counts = {group: sum(parameter.numel() for parameter in models[f'{group}_{CONFIG["seeds"][0]}'].parameters())
              for group in ('none', 'global', 'adaptive', 'recurrent')}
    summary = dict(primary=primary, cells=cells, contrasts=contrasts, timing=timing,
        parameters=counts, audit=verification,
        seed_primary={group: {str(seed): float(db[:primary_cells, names.index(f'{group}_{seed}')].mean())
                              for seed in CONFIG['seeds']}
                      for group in ('none', 'global', 'adaptive', 'recurrent')},
        bank_primary={group: bank_db[:, :primary_cells, indexes].mean((1, 2)).tolist()
                      for group, indexes in groups.items()},
        primary_threshold_rates={group: {str(t): float((ratio[:, :primary_cells, indexes, :] > t).mean())
                                         for t in (.5, 1., 2.)}
                                 for group, indexes in groups.items()},
        ratio_of_sums_pooled={group: float(np.mean(10 * np.log10(
            errors[:, :primary_cells, indexes].sum((0, 1, 3)) / power[:, :primary_cells].sum())))
            for group, indexes in groups.items()},
        mean_case_linear={group: float(ratio[:, :primary_cells, indexes].mean())
                          for group, indexes in groups.items()},
        classical_adaptive_gain_db={group: primary[group] - primary['adaptive']
                                    for group in ('opt_selected', 'rank_loop')},
        caveats=[
            'The recurrent arm is a NA-ALISTA-inspired adaptation to this Rician EM/Hankel model, not a reproduction of NA-ALISTA.',
            'Three training seeds share the original development data. Paired intervals are conditional on these fitted models and frozen settings.',
            'The four learned arms share observation budget, training data/orders, epoch budget and validation criterion, not parameter count or controller features.',
            'Classical settings were frozen in the preceding validation-only study. Classical comparisons here are descriptive, not part of the three adjusted tests.',
            'Fresh held-out worlds do not establish hardware performance or novelty, and must not be pooled with preceding tests.'])
    for group, indexes in groups.items():
        scalar = np.mean([10 * np.log10(
            sum(float(errors[bank, cell, method, case]) for bank in range(len(CONFIG['banks'])) for case in range(CONFIG['cases'])) /
            sum(float(power[bank, cell, case]) for bank in range(len(CONFIG['banks'])) for case in range(CONFIG['cases'])))
            for cell in range(primary_cells) for method in indexes])
        np.testing.assert_allclose(scalar, primary[group], rtol=1e-12, atol=1e-12)
    summary['independent_headline_recalculation'] = 'passed'
    write(ROOT / 'summary.json', summary)
    lines = ['FOCUSED COMPACT-CONTROLLER COMPARISON', '',
        f'{errors.shape[0] * errors.shape[1] * errors.shape[-1]} fresh paired worlds; '
        '126 primary conditions plus two separately reported stress conditions.',
        'Original 10/30/188-parameter arms reused; three recurrent controls trained with the matched budget.', '',
        'PRIMARY MACRO NMSE (lower is better), MEAN SERIAL TIME, TRAINABLE PARAMETERS']
    lines += [f'{group}: {primary[group]:.3f} dB; {timing[group]["mean_ms"]:.3f} ms; '
              f'{counts.get(group, 0)} parameters' for group in groups]
    lines += ['', '188-PARAMETER GAINS (positive favors the 188-parameter model); adjusted 95% paired intervals']
    lines += [f'{group}: {entry["gain_db"]:.3f} dB; {entry["simultaneous95_macro_ci"]}; '
              f'worst-cell deterioration {entry["worst_cell_deterioration_db"]:.3f} dB'
              for group, entry in contrasts.items()]
    lines += ['', 'SEVERE-ERROR RATE (NMSE > 1)']
    lines += [f'{group}: {100 * summary["primary_threshold_rates"][group]["1.0"]:.3f}%'
              for group in groups]
    lines += ['', 'STRESS CONDITIONS (excluded from the primary average)']
    lines += [spec['id'] + ': ' + ', '.join(
        f'{group} {cells[spec["id"]][group]["nmse_db"]:.3f} dB' for group in groups)
        for spec in SPECS[primary_cells:]]
    lines += ['', 'Input regeneration, checkpoint replay, independent classical replay and scalar statistic audit passed.',
              '', *summary['caveats']]
    (ROOT / 'RESULTS.txt').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines), flush=True)
    return summary
