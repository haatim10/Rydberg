"""Fresh paired development screen, timing, replay audit and fixed analysis."""
import json
import os
from pathlib import Path
import tempfile
import time

import numpy as np
import torch

from compact_baseline.analyze import reference_loop
from fair_optshrink.estimator import estimate, lookup_key, tensors
from optshrink_rank_check.estimator import fit_candidates, select_and_estimate
from paper2_optshrink_loop.estimator import estimate as numpy_estimate
from sahure.train import load_model
from sahure_expansion.data import world

from .protocol import ROOT, CONFIG, SPECS, sha, load, save, write, initialize


MODEL_NAMES = ('holdout_best', 'holdout_final', 'allpilot_best',
               'allpilot_final', 'mixed_best', 'mixed_final')
NAMES = ('rank_loop', 'opt_selected') + MODEL_NAMES
PARENT_SELECTION = Path('results/fair_optshrink/frozen_selection.json')
GENERATOR_SOURCE = Path('sahure_expansion/data.py')


def frozen_record():
    path = ROOT / 'frozen_selection.json'
    if not path.exists():
        raise RuntimeError('freeze checkpoints before generating fresh screen data')
    record = json.loads(path.read_text())
    if record['manifest_sha256'] != sha(ROOT / 'manifest.json'):
        raise RuntimeError('frozen experiment manifest changed')
    if set(record['models']) != set(MODEL_NAMES):
        raise RuntimeError('unexpected frozen model names')
    return record


def frozen_parent():
    frozen_record()
    manifest = json.loads((ROOT / 'manifest.json').read_text())
    if sha(PARENT_SELECTION) != manifest['sources'][str(PARENT_SELECTION)]:
        raise RuntimeError('historical OptShrink settings changed')
    return json.loads(PARENT_SELECTION.read_text())


def load_models():
    entries = frozen_record()['models']
    models = {}
    for name in MODEL_NAMES:
        entry = entries[name]
        if sha(entry['path']) != entry['sha256']:
            raise RuntimeError('checkpoint hash changed: ' + name)
        model, _ = load_model(entry['path'], dtype=torch.float64)
        if sum(p.numel() for p in model.parameters()) != 188:
            raise RuntimeError('not the matched 188-parameter architecture')
        models[name] = model.eval()
    return models


def fixed_config(obs, parent):
    keys = lookup_key(obs)
    if len(set(keys)) != 1:
        raise ValueError('OptShrink lookup requires a homogeneous cell')
    return parent['choices'][keys[0]]['selected']


def data(bank, cell):
    freeze = frozen_record()
    if not 0 <= bank < len(CONFIG['banks']) or not 0 <= cell < len(SPECS):
        raise IndexError('unknown screen bank or condition')
    digest = sha(ROOT / 'frozen_selection.json')
    generator_digest = sha(GENERATOR_SOURCE)
    seed = CONFIG['banks'][bank]
    indices = list(range(cell * CONFIG['cases'], (cell + 1) * CONFIG['cases']))
    ids = np.array([f'rician_mixed:{seed}:{i}' for i in indices])
    folder = ROOT / 'data' / f'bank{bank}' / SPECS[cell]['id']
    path = folder / 'manifest.json'
    if path.exists():
        meta = json.loads(path.read_text())
        if (meta['selection_sha256'] != digest or meta['generator_sha256'] != generator_digest
                or meta['indices'] != indices or meta['seed'] != seed
                or meta['spec'] != SPECS[cell] or meta['generated_unix'] < freeze['frozen_unix']):
            raise RuntimeError('cached screen data identity changed')
        for filename, value in meta['hashes'].items():
            if sha(filename) != value:
                raise RuntimeError('screen data modified: ' + filename)
        observed = load(folder / 'observations.npz')
        np.testing.assert_array_equal(observed['trial_ids'], ids)
        return observed
    arrays = {key: [] for key in ('z', 's', 'b', 'sigma2')}
    truth = []
    for index in indices:
        item, _ = world(index, seed, **SPECS[cell])
        for key, value in zip(arrays, (item.Z, item.S, item.B, item.sigma2)):
            arrays[key].append(value)
        truth.append(item.G)
    observed = {key: np.stack(value) for key, value in arrays.items()}
    if not all(np.isfinite(value).all() for value in observed.values()):
        raise FloatingPointError('nonfinite generated observation')
    paths = (folder / 'observations.npz', folder / 'truth.npz')
    save(paths[0], **observed, trial_ids=ids)
    save(paths[1], g=np.stack(truth), trial_ids=ids)
    # Manifest written last: incomplete cells are regenerated deterministically.
    write(path, dict(seed=seed, indices=indices, count=len(indices), bank=bank,
          spec=SPECS[cell], selection_sha256=digest,
          generator_sha256=generator_digest, generated_unix=time.time(),
          hashes={str(p): sha(p) for p in paths}))
    return load(paths[0])


@torch.no_grad()
def evaluate(parent, models, check):
    digest, parent_digest = sha(ROOT / 'frozen_selection.json'), sha(PARENT_SELECTION)
    for bank in range(len(CONFIG['banks'])):
        for cell, spec in enumerate(SPECS):
            observed = data(bank, cell)
            path = ROOT / 'predictions' / f'bank{bank}_{spec["id"]}.npz'
            if path.exists():
                old = load(path)
                if (str(old['selection_sha256']) != digest
                        or str(old['parent_selection_sha256']) != parent_digest):
                    raise RuntimeError('prediction freeze changed')
                np.testing.assert_array_equal(old['trial_ids'], observed['trial_ids'])
                np.testing.assert_array_equal(old['names'], NAMES)
                if not np.isfinite(old['estimates']).all():
                    raise FloatingPointError('nonfinite cached prediction')
                continue
            obs = tensors(observed)
            if obs.s.shape[-1] != 30:
                raise ValueError('this protocol requires the fixed 21/9 pilot split')
            predictions, configs, scores = [], [], []
            tick = time.perf_counter()
            for case in range(len(obs)):
                check('testing', bank=bank, cell=cell, cells=len(SPECS), case=case)
                row = obs.batch(slice(case, case + 1))
                base = parent['choices'][lookup_key(row)[0]]['loop']
                config, score = fit_candidates(row, base)
                if config['placement'] != 'loop':
                    raise RuntimeError('rank comparator must use in-loop OptShrink')
                predictions.append(estimate(row, config).numpy()[0])
                configs.append([config['rank'], config['strength'], config['iterations']])
                scores.append(score.numpy())
            outputs = dict(rank_loop=np.stack(predictions))
            costs = dict(rank_loop=time.perf_counter() - tick)
            check('testing', bank=bank, cell=cell, method='opt_selected')
            tick = time.perf_counter()
            outputs['opt_selected'] = estimate(obs, fixed_config(obs, parent)).numpy()
            costs['opt_selected'] = time.perf_counter() - tick
            for name in MODEL_NAMES:
                check('learned_testing', bank=bank, cell=cell, method=name)
                tick = time.perf_counter()
                outputs[name] = models[name](*obs.args()).numpy()
                costs[name] = time.perf_counter() - tick
            values = np.stack([outputs[name] for name in NAMES])
            if values.shape != (len(NAMES), len(obs), obs.z.shape[1], obs.s.shape[1]):
                raise RuntimeError('incorrect prediction dimensions')
            if not np.isfinite(values).all():
                raise FloatingPointError('nonfinite channel prediction')
            save(path, estimates=values, names=np.array(NAMES),
                 trial_ids=observed['trial_ids'], rank_configs=np.array(configs),
                 rank_scores=np.array(scores), selection_sha256=digest,
                 parent_selection_sha256=parent_digest,
                 batch_evaluation_seconds=np.array([costs[name] for name in NAMES]))
            print(f'Screen bank {bank + 1}/{len(CONFIG["banks"])} cell '
                  f'{cell + 1}/{len(SPECS)} complete: {spec["id"]}', flush=True)


@torch.no_grad()
def timing(parent, models, check):
    digest, parent_digest = sha(ROOT / 'frozen_selection.json'), sha(PARENT_SELECTION)
    for cell, spec in enumerate(SPECS):
        path = ROOT / 'timing' / f'{spec["id"]}.json'
        if path.exists():
            old = json.loads(path.read_text())
            if (old['selection_sha256'] != digest or old['parent_selection_sha256'] != parent_digest
                    or set(old['seconds']) != set(NAMES)):
                raise RuntimeError('timing freeze changed')
            continue
        item, _ = world(cell, CONFIG['timing_seed'], **spec)
        obs = tensors(dict(z=item.Z[None], s=item.S[None], b=item.B[None],
                           sigma2=np.array([item.sigma2])))
        methods = dict(rank_loop=lambda: select_and_estimate(obs, parent, 'loop')[0],
                       opt_selected=lambda: estimate(obs, fixed_config(obs, parent)))
        methods.update({name: (lambda model=models[name]: model(*obs.args())) for name in MODEL_NAMES})
        for name, fn in methods.items():
            check('timing_warmup', cell=cell, method=name)
            if not torch.isfinite(fn()).all():
                raise FloatingPointError('nonfinite timing prediction')
        seconds = {name: [] for name in NAMES}
        rng = np.random.default_rng(CONFIG['timing_seed'] + cell)
        for repeat in range(CONFIG['timing_repeats']):
            for name in rng.permutation(list(NAMES)):
                check('timing', cell=cell, method=str(name), repeat=repeat)
                tick = time.perf_counter()
                methods[name]()
                seconds[name].append(time.perf_counter() - tick)
        write(path, dict(seconds=seconds, spec=spec, selection_sha256=digest,
              parent_selection_sha256=parent_digest,
              backend='PyTorch CPU float64 one thread batch1',
              includes='initialization, all updates, controller, rank search and refit; '
                       'excludes training, offline tuning, loading and I/O'))


def run(check=lambda *args, **kwargs: None):
    if torch.get_num_threads() != 1:
        raise RuntimeError('screen requires one CPU thread')
    parent, models = frozen_parent(), load_models()
    evaluate(parent, models, check)
    timing(parent, models, check)


@torch.no_grad()
def audit(check):
    initialize()
    from .checks import audit_training
    training_audit = audit_training()
    frozen = frozen_record()
    digest, parent_digest = sha(ROOT / 'frozen_selection.json'), sha(PARENT_SELECTION)
    manifest = json.loads((ROOT / 'manifest.json').read_text())
    for path, value in manifest['sources'].items():
        if sha(path) != value:
            raise RuntimeError('frozen source or data changed: ' + path)
    parent, models = frozen_parent(), load_models()
    ids, hashes = set(), {}
    neural_replays, classical_replays, maximum = 0, 0, 0.
    for bank, seed in enumerate(CONFIG['banks']):
        for cell, spec in enumerate(SPECS):
            check('auditing', bank=bank, cell=cell)
            observed = data(bank, cell)
            folder = ROOT / 'data' / f'bank{bank}' / spec['id']
            meta = json.loads((folder / 'manifest.json').read_text())
            truth = load(folder / 'truth.npz')
            path = ROOT / 'predictions' / f'bank{bank}_{spec["id"]}.npz'
            result = load(path)
            expected = [f'rician_mixed:{seed}:{i}' for i in meta['indices']]
            for value in (observed['trial_ids'], truth['trial_ids'], result['trial_ids']):
                np.testing.assert_array_equal(value, expected)
            if ids.intersection(expected):
                raise RuntimeError('duplicate world IDs')
            ids.update(expected)
            if (str(result['selection_sha256']) != digest
                    or str(result['parent_selection_sha256']) != parent_digest):
                raise RuntimeError('prediction freeze mismatch')
            np.testing.assert_array_equal(result['names'], NAMES)
            if (not np.isfinite(result['estimates']).all() or not np.isfinite(truth['g']).all()
                    or not np.isfinite(result['rank_scores']).all()):
                raise FloatingPointError('nonfinite saved result')
            hashes.update(meta['hashes'])
            hashes[str(path)] = sha(path)
            # Every cell has one exact input/truth regeneration; replay a fixed subset.
            item, _ = world(meta['indices'][0], seed, **spec)
            for key, value in zip(('z', 's', 'b', 'sigma2'), (item.Z, item.S, item.B, item.sigma2)):
                np.testing.assert_array_equal(observed[key][0], value)
            np.testing.assert_array_equal(truth['g'][0], item.G)
            if cell % 16 == 0 or cell >= CONFIG['primary_cells']:
                row = {key: observed[key][:1] for key in ('z', 's', 'b', 'sigma2')}
                obs = tensors(row)
                for name, model in models.items():
                    actual = result['estimates'][NAMES.index(name), 0]
                    expected_g = model(*obs.args()).numpy()[0]
                    np.testing.assert_allclose(actual, expected_g, rtol=3e-8, atol=3e-9)
                    maximum = max(maximum, float(abs(actual - expected_g).max()))
                    neural_replays += 1
                expected_g, scores, config = reference_loop(row, parent)
                np.testing.assert_allclose(result['rank_scores'][0], scores, rtol=3e-8, atol=3e-9)
                np.testing.assert_allclose(result['estimates'][0, 0], expected_g, rtol=3e-8, atol=3e-9)
                np.testing.assert_array_equal(result['rank_configs'][0],
                                             [config['rank'], config['strength'], config['iterations']])
                expected_g = numpy_estimate(row['z'][0], row['s'][0], row['b'][0],
                                           float(row['sigma2'][0]), fixed_config(obs, parent))
                np.testing.assert_allclose(result['estimates'][1, 0], expected_g, rtol=3e-8, atol=3e-9)
                classical_replays += 2
            path = ROOT / 'timing' / f'{spec["id"]}.json'
            record = json.loads(path.read_text())
            if (record['selection_sha256'] != digest or record['parent_selection_sha256'] != parent_digest
                    or set(record['seconds']) != set(NAMES)):
                raise RuntimeError('timing identity mismatch')
            for values in record['seconds'].values():
                if (len(values) != CONFIG['timing_repeats'] or not np.isfinite(values).all()
                        or min(values) <= 0):
                    raise RuntimeError('invalid timing samples')
            hashes[str(path)] = sha(path)
    result = dict(passed=True, worlds=len(ids), neural_replays=neural_replays, training=training_audit,
                  independent_classical_replays=classical_replays,
                  maximum_neural_replay_error=maximum, hashes=hashes,
                  scope='All IDs, input hashes, frozen sources and finite outputs; '
                        'one input regeneration per cell; prediction replay every16th cell and both stress cells. '
                        'Training history, update counts and checkpoint selection checked separately; '
                        'not a full replay of all predictions or all training updates.')
    write(ROOT / 'audit.json', result)
    return result


def _stats(errors, power, cells):
    """Equal cell dB NMSE and separate conventional per-case tail statistics."""
    er, pw = errors[:, cells], power[:, cells]
    ratio = er / pw[:, :, None, :]
    cell_db = 10 * np.log10(er.sum((0, 3)) / pw.sum((0, 2))[:, None])
    return {name: dict(macro_nmse_db=float(cell_db[:, j].mean()),
                      pooled_nmse_db=float(10 * np.log10(er[:, :, j].sum() / pw.sum())),
                      mean_case_nmse=float(ratio[:, :, j].mean()),
                      severe_rate=float((ratio[:, :, j] > 1).mean()),
                      threshold_rates={str(t): float((ratio[:, :, j] > t).mean()) for t in (.5, 1., 2.)})
            for j, name in enumerate(NAMES)}


def _transitions(ratio, cells, comparator):
    a = ratio[:, cells, NAMES.index(comparator), :] > 1
    b = ratio[:, cells, NAMES.index('mixed_best'), :] > 1
    return dict(worlds=int(a.size), both_good=int((~a & ~b).sum()),
                corrected_by_mixed=int((a & ~b).sum()), new_failure_from_mixed=int((~a & b).sum()),
                both_severe=int((a & b).sum()), severe_rate_delta=float(b.mean() - a.mean()))


def _bootstrap(errors, power, check):
    pc, cases, banks = CONFIG['primary_cells'], CONFIG['cases'], len(CONFIG['banks'])
    er, pw = errors[:, :pc], power[:, :pc]
    ratio = er / pw[:, :, None, :]
    comparators = ('holdout_best', 'allpilot_best', 'rank_loop', 'opt_selected')
    methods = [NAMES.index(name) for name in comparators]
    mixed = NAMES.index('mixed_best')
    low = {str(s): np.array([i for i, spec in enumerate(SPECS[:pc]) if spec['snr'] == s]) for s in (-10, -5)}
    for snr, cells in low.items():
        if not len(cells):
            raise RuntimeError('missing prespecified low-SNR group: ' + snr)
    rng = np.random.default_rng(CONFIG['bootstrap_seed'])
    draws = {key: [] for key in ('gain_db', 'severe_delta', 'mean_nmse_delta')}
    low_draws = {key: dict(severe_delta=[], mean_nmse_delta=[]) for key in low}
    for start in range(0, CONFIG['bootstrap_samples'], 100):
        check('bootstrap', done=start, total=CONFIG['bootstrap_samples'])
        count = min(100, CONFIG['bootstrap_samples'] - start)
        weights = rng.multinomial(cases, np.full(cases, 1 / cases), size=(count, banks, pc))
        error_sum = np.einsum('rbcn,bcmn->rcm', weights, er, optimize=True)
        power_sum = np.einsum('rbcn,bcn->rc', weights, pw, optimize=True)
        db = 10 * np.log10(error_sum / power_sum[..., None])
        severity = np.einsum('rbcn,bcmn->rcm', weights, (ratio > 1).astype(float), optimize=True) / (banks * cases)
        mean_ratio = np.einsum('rbcn,bcmn->rcm', weights, ratio, optimize=True) / (banks * cases)
        draws['gain_db'].append(db[:, :, methods].mean(1) - db[:, :, mixed].mean(1)[:, None])
        draws['severe_delta'].append(severity[:, :, mixed].mean(1)[:, None] - severity[:, :, methods].mean(1))
        draws['mean_nmse_delta'].append(mean_ratio[:, :, mixed].mean(1)[:, None] - mean_ratio[:, :, methods].mean(1))
        allpilot = NAMES.index('allpilot_best')
        for snr, cells in low.items():
            low_draws[snr]['severe_delta'].append((severity[:, cells, mixed] - severity[:, cells, allpilot]).mean(1))
            low_draws[snr]['mean_nmse_delta'].append((mean_ratio[:, cells, mixed] - mean_ratio[:, cells, allpilot]).mean(1))
    draws = {key: np.concatenate(value) for key, value in draws.items()}
    low_draws = {snr: {key: np.concatenate(value) for key, value in values.items()} for snr, values in low_draws.items()}
    alpha = .05 / (2 * len(comparators))
    comparisons = {name: dict(
        gain_db_conditional95_ci=np.quantile(draws['gain_db'][:, j], [.025, .975]).tolist(),
        gain_db_bonferroni95_ci=np.quantile(draws['gain_db'][:, j], [alpha, 1 - alpha]).tolist(),
        severe_delta_conditional95_ci=np.quantile(draws['severe_delta'][:, j], [.025, .975]).tolist(),
        mean_nmse_delta_conditional95_ci=np.quantile(draws['mean_nmse_delta'][:, j], [.025, .975]).tolist())
        for j, name in enumerate(comparators)}
    low_intervals = {snr: {key + '_conditional95_ci': np.quantile(value, [.025, .975]).tolist()
                           for key, value in values.items()} for snr, values in low_draws.items()}
    save(ROOT / 'bootstrap_draws.npz', comparators=np.array(comparators), **draws,
         **{f'low_{snr}_{key}': value for snr, values in low_draws.items() for key, value in values.items()})
    return comparisons, low_intervals


def _write_text(path, text):
    path = Path(path)
    with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, prefix=path.name + '.', suffix='.tmp', delete=False) as stream:
        stream.write(text)
        temporary = stream.name
    os.replace(temporary, path)


def analyze(check=lambda *args, **kwargs: None):
    verification = audit(check)
    errors, powers = [], []
    for bank in range(len(CONFIG['banks'])):
        er, pw = [], []
        for spec in SPECS:
            pred = load(ROOT / 'predictions' / f'bank{bank}_{spec["id"]}.npz')
            truth = load(ROOT / 'data' / f'bank{bank}' / spec['id'] / 'truth.npz')['g']
            er.append((abs(pred['estimates'] - truth[None]) ** 2).sum((-2, -1)))
            pw.append((abs(truth) ** 2).sum((-2, -1)))
        errors.append(er)
        powers.append(pw)
    errors, power = np.asarray(errors), np.asarray(powers)
    if not np.isfinite(errors).all() or not np.isfinite(power).all() or (power <= 0).any():
        raise FloatingPointError('invalid channel-error arrays')
    save(ROOT / 'all_errors.npz', squared_error=errors, channel_power=power, names=np.array(NAMES))
    pc = CONFIG['primary_cells']
    primary_ids = np.arange(pc)
    ratio = errors / power[:, :, None, :]
    primary = _stats(errors, power, primary_ids)
    cells = {spec['id']: dict(spec=spec, methods=_stats(errors, power, [j])) for j, spec in enumerate(SPECS)}
    by = {}
    for dimension in ('snr', 'rsr', 'L'):
        values = sorted(set(spec[dimension] for spec in SPECS[:pc]), key=lambda value: (value is not None, value or 0))
        by[dimension] = {str(value): _stats(errors, power, [j for j, spec in enumerate(SPECS[:pc]) if spec[dimension] == value])
                         for value in values}
    contrasts, low_intervals = _bootstrap(errors, power, check)
    db = 10 * np.log10(errors.sum((0, 3)) / power.sum((0, 2))[:, None])
    for comparator, result in contrasts.items():
        result.update(gain_db=primary[comparator]['macro_nmse_db'] - primary['mixed_best']['macro_nmse_db'],
                      severe_delta=primary['mixed_best']['severe_rate'] - primary[comparator]['severe_rate'],
                      mean_nmse_delta=primary['mixed_best']['mean_case_nmse'] - primary[comparator]['mean_case_nmse'],
                      improved_cells=int((db[:pc, NAMES.index('mixed_best')] < db[:pc, NAMES.index(comparator)]).sum()),
                      worst_cell_deterioration_db=max(0., float((db[:pc, NAMES.index('mixed_best')] - db[:pc, NAMES.index(comparator)]).max())))
    timing_result = {}
    for name in NAMES:
        values = [seconds for spec in SPECS[:pc]
                  for seconds in json.loads((ROOT / 'timing' / f'{spec["id"]}.json').read_text())['seconds'][name]]
        timing_result[name] = dict(mean_ms=float(np.mean(values) * 1000), median_ms=float(np.median(values) * 1000))
    low_snr, transitions = {}, {}
    for snr in (-10, -5):
        key = str(float(snr))
        group = by['snr'][key]
        low_snr[str(snr)] = dict(methods=group, **low_intervals[str(snr)],
            mean_nmse_delta=group['mixed_best']['mean_case_nmse'] - group['allpilot_best']['mean_case_nmse'],
            severe_delta=group['mixed_best']['severe_rate'] - group['allpilot_best']['severe_rate'])
    for comparator in ('holdout_best', 'allpilot_best', 'rank_loop'):
        transitions[comparator] = dict(primary=_transitions(ratio, primary_ids, comparator),
            low_snr={str(snr): _transitions(ratio, [j for j, spec in enumerate(SPECS[:pc]) if spec['snr'] == snr], comparator)
                     for snr in (-10, -5)})
    runtime_ratio = timing_result['mixed_best']['mean_ms'] / timing_result['holdout_best']['mean_ms']
    criteria = dict(macro_gain_at_least_0p2_db=bool(contrasts['holdout_best']['gain_db'] >= .2),
                    lower_low_snr_mean_nmse=bool(all(value['mean_nmse_delta'] < 0 for value in low_snr.values())),
                    lower_low_snr_severe_rate=bool(all(value['severe_delta'] < 0 for value in low_snr.values())),
                    overall_severe_within_0p002_of_old=bool(contrasts['holdout_best']['severe_delta'] <= .002),
                    runtime_within_1p05_of_old=bool(runtime_ratio <= 1.05))
    final = {name: dict(macro_gain_db=primary[name]['macro_nmse_db'] - primary['mixed_final']['macro_nmse_db'],
                       severe_delta=primary['mixed_final']['severe_rate'] - primary[name]['severe_rate'])
             for name in ('holdout_final', 'allpilot_final', 'rank_loop', 'opt_selected')}
    # Independently recalculate the primary macro using scalar loops.
    for j, name in enumerate(NAMES):
        independent = np.mean([10 * np.log10(
            sum(float(errors[b, c, j, i]) for b in range(len(CONFIG['banks'])) for i in range(CONFIG['cases'])) /
            sum(float(power[b, c, i]) for b in range(len(CONFIG['banks'])) for i in range(CONFIG['cases']))) for c in range(pc)])
        np.testing.assert_allclose(independent, primary[name]['macro_nmse_db'], rtol=1e-12, atol=1e-12)
    result = dict(primary=primary, contrasts=contrasts, cells=cells, by_condition=by,
                  low_snr=low_snr, severe_transitions=transitions, timing=timing_result,
                  mixed_to_old_runtime_ratio=runtime_ratio, final_sensitivity=final,
                  criteria=criteria, screening_passed=bool(all(criteria.values())), audit=verification,
                  independent_headline_recalculation='passed',
                  caveats=['Exploratory development screen with one training seed and one fresh data bank; not independent confirmation.',
                           'Paired bootstrap intervals condition on the fitted models and the fixed condition grid; they do not measure training-population uncertainty.',
                           'Only the four macro-gain intervals receive Bonferroni adjustment; tail and low-SNR intervals are descriptive pointwise intervals.',
                           'The 0.002 severe-rate tolerance is a practical screening rule, not statistical noninferiority.',
                           'Selected-checkpoint results are primary; fixed-final-epoch comparisons are sensitivity checks.',
                           'Runtime uses the same CPU backend and includes initialization, rank search and refit; small differences can reflect timing variation.',
                           'Fresh development data cannot become untouched final confirmation data for a subsequently revised method.'])
    write(ROOT / 'summary.json', result)
    worlds = len(CONFIG['banks']) * CONFIG['cases'] * len(SPECS)
    lines = ['MIXED-OBJECTIVE DEVELOPMENT SCREEN', '',
             f'{worlds} paired fresh worlds; {len(CONFIG["banks"]) * CONFIG["cases"] * pc} primary worlds. One training seed.',
             'Lower NMSE is better. Gains are comparator minus mixed; positive gain favors mixed.', '',
             'Primary: validation-selected checkpoints.']
    for name, value in primary.items():
        lines.append(f'{name}: {value["macro_nmse_db"]:.4f} dB; mean per-case NMSE {value["mean_case_nmse"]:.5f}; '
                     f'severe NMSE>1 {value["severe_rate"]:.3%}; CPU {timing_result[name]["mean_ms"]:.3f} ms')
    lines += ['', 'Mixed selected checkpoint comparisons:']
    for name, value in contrasts.items():
        lines.append(f'{name}: gain {value["gain_db"]:.4f} dB, adjusted conditional95% CI '
                     f'{value["gain_db_bonferroni95_ci"]}; severe-rate change {value["severe_delta"]:+.4%}; '
                     f'wins {value["improved_cells"]}/{pc} cells; worst deterioration {value["worst_cell_deterioration_db"]:.4f} dB')
    lines += ['', 'Low-SNR mixed changes versus selected all-pilot:']
    for snr, value in low_snr.items():
        lines.append(f'{snr} dB: mean per-case NMSE change {value["mean_nmse_delta"]:+.6f}, '
                     f'conditional95% CI {value["mean_nmse_delta_conditional95_ci"]}; severe-rate change '
                     f'{value["severe_delta"]:+.4%}, conditional95% CI {value["severe_delta_conditional95_ci"]}')
    lines += ['', 'Fixed screening criteria:'] + [f'{name}: {value}' for name, value in criteria.items()]
    lines += [f'Overall screening passed: {result["screening_passed"]}',
              f'Mixed/old runtime ratio: {runtime_ratio:.4f}', '',
              'Paired severe transitions: ' + json.dumps(transitions), '',
              'Final-epoch sensitivity: ' + json.dumps(final), '', 'Stress cells (excluded from primary):']
    lines += [spec['id'] + ': ' + json.dumps(cells[spec['id']]['methods']) for spec in SPECS[pc:]]
    lines += ['', 'Full per-SNR, per-reference and per-path results, threshold0.5/1/2 rates and all cells are in summary.json.', '',
              'Limitations:'] + result['caveats']
    _write_text(ROOT / 'RESULTS.txt', '\n'.join(lines) + '\n')
    return result
