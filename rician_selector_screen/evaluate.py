"""Paired selector screen with unchanged observations under variance errors."""
import json
import os
from pathlib import Path
import tempfile
import time
import numpy as np
import torch
from fair_optshrink.estimator import tensors
from sahure.data import Observations
from sahure.train import load_model
from sahure_expansion.data import world
from .protocol import ROOT, CONFIG, SPECS, sha, load, save, write, initialize
from .selector import supplied_variance, route_mask, estimate

MODELS = ('holdout_best', 'allpilot_best')
NAMES = MODELS + ('selector',)
GENERATOR = Path('sahure_expansion/data.py')


def freeze():
    path = ROOT / 'frozen_selection.json'
    if not path.exists():
        raise RuntimeError('freeze observation-only selection before fresh generation')
    value = json.loads(path.read_text())
    if value['manifest_sha256'] != sha(ROOT / 'manifest.json') or set(value['models']) != set(MODELS):
        raise RuntimeError('selection manifest or model names changed')
    return value


def load_models():
    models = {}
    for name, entry in freeze()['models'].items():
        if sha(entry['path']) != entry['sha256']:
            raise RuntimeError('checkpoint changed: ' + name)
        model, _ = load_model(entry['path'], dtype=torch.float64)
        if sum(p.numel() for p in model.parameters()) != 188:
            raise RuntimeError('incorrect frozen architecture')
        models[name] = model.eval()
    return models


def data(bank, cell):
    frozen = freeze()
    digest, seed = sha(ROOT / 'frozen_selection.json'), CONFIG['banks'][bank]
    indices = list(range(cell * CONFIG['cases'], (cell + 1) * CONFIG['cases']))
    ids = np.array([f'rician_selector:{seed}:{i}' for i in indices])
    folder = ROOT / 'data' / f'bank{bank}' / SPECS[cell]['id']
    path = folder / 'manifest.json'
    if path.exists():
        meta = json.loads(path.read_text())
        if (meta['selection_sha256'] != digest or meta['indices'] != indices or meta['seed'] != seed
                or meta['spec'] != SPECS[cell] or meta['generator_sha256'] != sha(GENERATOR)
                or meta['generated_unix'] < frozen['frozen_unix']):
            raise RuntimeError('data generation identity changed')
        for filename, expected in meta['hashes'].items():
            if sha(filename) != expected:
                raise RuntimeError('input hash changed: ' + filename)
        observed = load(folder / 'observations.npz')
        np.testing.assert_array_equal(observed['trial_ids'], ids)
        return observed
    arrays, truth = {key: [] for key in ('z', 's', 'b', 'sigma2')}, []
    for index in indices:
        item, _ = world(index, seed, **SPECS[cell])
        for key, value in zip(arrays, (item.Z, item.S, item.B, item.sigma2)):
            arrays[key].append(value)
        truth.append(item.G)
    paths = (folder / 'observations.npz', folder / 'truth.npz')
    save(paths[0], **{key: np.stack(value) for key, value in arrays.items()}, trial_ids=ids)
    save(paths[1], g=np.stack(truth), trial_ids=ids)
    write(path, dict(seed=seed, indices=indices, spec=SPECS[cell], selection_sha256=digest,
          generator_sha256=sha(GENERATOR), generated_unix=time.time(), hashes={str(p): sha(p) for p in paths}))
    return load(paths[0])


@torch.no_grad()
def predictions(models, check):
    frozen, digest = freeze(), sha(ROOT / 'frozen_selection.json')
    threshold, power = frozen['threshold_db'], CONFIG['nominal_power']
    for bank in range(len(CONFIG['banks'])):
        for cell, spec in enumerate(SPECS):
            observed = data(bank, cell)
            path = ROOT / 'predictions' / f'bank{bank}_{spec["id"]}.npz'
            if path.exists():
                old = load(path)
                if str(old['selection_sha256']) != digest:
                    raise RuntimeError('prediction freeze changed')
                np.testing.assert_array_equal(old['trial_ids'], observed['trial_ids'])
                np.testing.assert_array_equal(old['names'], NAMES)
                np.testing.assert_array_equal(old['noise_offsets_db'], CONFIG['noise_offsets'])
                if not np.isfinite(old['estimates']).all():
                    raise FloatingPointError('nonfinite cached output')
                continue
            base = tensors(observed)
            outputs, masks, snrs, calls = [], [], [], []
            for offset in CONFIG['noise_offsets']:
                check('testing', bank=bank, cell=cell, cells=len(SPECS), noise_offset=offset)
                obs = supplied_variance(base, offset)
                old, allpilot = [models[name](*obs.args()) for name in MODELS]
                selected, info = estimate(obs, models, threshold, power, diagnostics=True)
                routed = torch.where(info['allpilot_mask'][:, None, None], allpilot, old)
                torch.testing.assert_close(selected, routed, rtol=3e-8, atol=3e-9)
                outputs.append(torch.stack((old, allpilot, selected)).numpy())
                masks.append(info['allpilot_mask'].numpy())
                snrs.append(info['nominal_snr_db'].numpy())
                calls.append(info['forward_calls'])
            values = np.stack(outputs)
            if not np.isfinite(values).all():
                raise FloatingPointError('nonfinite selector output')
            save(path, estimates=values, names=np.array(NAMES), trial_ids=observed['trial_ids'],
                 noise_offsets_db=np.array(CONFIG['noise_offsets']), allpilot_mask=np.stack(masks),
                 nominal_snr_db=np.stack(snrs), batch_forward_calls=np.array(calls), selection_sha256=digest)
            print(f'Selector screen cell {cell + 1}/{len(SPECS)} bank {bank + 1} complete: {spec["id"]}', flush=True)


@torch.no_grad()
def timing(models, check):
    frozen, digest = freeze(), sha(ROOT / 'frozen_selection.json')
    for cell, spec in enumerate(SPECS):
        path = ROOT / 'timing' / f'{spec["id"]}.json'
        if path.exists():
            old = json.loads(path.read_text())
            if old['selection_sha256'] != digest or old['noise_offsets_db'] != CONFIG['noise_offsets']:
                raise RuntimeError('timing identity changed')
            continue
        item, _ = world(cell, CONFIG['timing_seed'], **spec)
        base = tensors(dict(z=item.Z[None], s=item.S[None], b=item.B[None], sigma2=np.array([item.sigma2])))
        records = []
        for offset_index, offset in enumerate(CONFIG['noise_offsets']):
            obs = supplied_variance(base, offset)
            methods = {name: (lambda model=models[name]: model(*obs.args())) for name in MODELS}
            methods['selector'] = lambda: estimate(obs, models, frozen['threshold_db'], CONFIG['nominal_power'])
            _, info = estimate(obs, models, frozen['threshold_db'], CONFIG['nominal_power'], diagnostics=True)
            if info['forward_calls'] != 1:
                raise RuntimeError('batch-one selector must call exactly one model')
            for fn in methods.values():
                if not torch.isfinite(fn()).all():
                    raise FloatingPointError('nonfinite timing output')
            seconds = {name: [] for name in NAMES}
            rng = np.random.default_rng(CONFIG['timing_seed'] + 10 * cell + offset_index)
            for repeat in range(CONFIG['timing_repeats']):
                for name in rng.permutation(list(NAMES)):
                    check('timing', cell=cell, noise_offset=offset, method=str(name), repeat=repeat)
                    start = time.perf_counter()
                    methods[name]()
                    seconds[name].append(time.perf_counter() - start)
            records.append(dict(noise_offset_db=offset, seconds=seconds, allpilot=bool(info['allpilot_mask'][0]), forward_calls=1))
        write(path, dict(spec=spec, selection_sha256=digest, noise_offsets_db=CONFIG['noise_offsets'], offsets=records,
                         backend='PyTorch CPU float64 one thread batch1; initialization and routing included; checkpoint loading excluded'))


def run(check=lambda *args, **kwargs: None):
    if torch.get_num_threads() != 1:
        raise RuntimeError('selector screen requires one CPU thread')
    models = load_models()
    predictions(models, check)
    timing(models, check)


@torch.no_grad()
def audit(check):
    initialize()
    frozen, digest, models = freeze(), sha(ROOT / 'frozen_selection.json'), load_models()
    for path, value in json.loads((ROOT / 'manifest.json').read_text())['sources'].items():
        if sha(path) != value:
            raise RuntimeError('frozen source or data changed: ' + path)
    ids, hashes, replays, maximum, mixed_rows = set(), {}, 0, 0., []
    for bank, seed in enumerate(CONFIG['banks']):
        for cell, spec in enumerate(SPECS):
            check('auditing', bank=bank, cell=cell)
            observed = data(bank, cell)
            folder = ROOT / 'data' / f'bank{bank}' / spec['id']
            meta = json.loads((folder / 'manifest.json').read_text())
            truth = load(folder / 'truth.npz')
            path = ROOT / 'predictions' / f'bank{bank}_{spec["id"]}.npz'
            pred = load(path)
            expected_ids = [f'rician_selector:{seed}:{i}' for i in meta['indices']]
            for value in (observed['trial_ids'], truth['trial_ids'], pred['trial_ids']):
                np.testing.assert_array_equal(value, expected_ids)
            if ids.intersection(expected_ids) or str(pred['selection_sha256']) != digest:
                raise RuntimeError('duplicate IDs or prediction freeze mismatch')
            ids.update(expected_ids)
            np.testing.assert_array_equal(pred['names'], NAMES)
            np.testing.assert_array_equal(pred['noise_offsets_db'], CONFIG['noise_offsets'])
            if not np.isfinite(pred['estimates']).all() or not np.isfinite(truth['g']).all():
                raise FloatingPointError('nonfinite saved prediction or truth')
            hashes.update(meta['hashes']); hashes[str(path)] = sha(path)
            item, _ = world(meta['indices'][0], seed, **spec)
            for key, value in zip(('z', 's', 'b', 'sigma2'), (item.Z, item.S, item.B, item.sigma2)):
                np.testing.assert_array_equal(observed[key][0], value)
            np.testing.assert_array_equal(truth['g'][0], item.G)
            base = tensors({key: observed[key][:1] for key in ('z', 's', 'b', 'sigma2')})
            if bank == 0 and cell in (0, CONFIG['primary_cells'] - 1):
                mixed_rows.append(base)
            for oi, offset in enumerate(CONFIG['noise_offsets']):
                obs = supplied_variance(base, offset)
                actual, info = estimate(obs, models, frozen['threshold_db'], CONFIG['nominal_power'], diagnostics=True)
                if info['forward_calls'] != 1:
                    raise RuntimeError('selector replay made more than one forward call')
                for j, name in enumerate(NAMES):
                    expected = actual if name == 'selector' else models[name](*obs.args())
                    target = pred['estimates'][oi, j, 0]
                    np.testing.assert_allclose(target, expected.numpy()[0], rtol=3e-8, atol=3e-9)
                    maximum = max(maximum, float(abs(target - expected.numpy()[0]).max())); replays += 1
                allobs = supplied_variance(tensors(observed), offset)
                np.testing.assert_array_equal(pred['allpilot_mask'][oi], route_mask(allobs.sigma2, frozen['threshold_db'], CONFIG['nominal_power']).numpy())
            path = ROOT / 'timing' / f'{spec["id"]}.json'; record = json.loads(path.read_text())
            if record['selection_sha256'] != digest or record['noise_offsets_db'] != CONFIG['noise_offsets']:
                raise RuntimeError('timing freeze mismatch')
            if len(record['offsets']) != len(CONFIG['noise_offsets']):
                raise RuntimeError('missing variance-offset timing')
            for oi, row in enumerate(record['offsets']):
                if row['noise_offset_db'] != CONFIG['noise_offsets'][oi] or row['forward_calls'] != 1 or set(row['seconds']) != set(NAMES):
                    raise RuntimeError('invalid timing route record')
                if any(len(v) != CONFIG['timing_repeats'] or not np.isfinite(v).all() or min(v) <= 0 for v in row['seconds'].values()):
                    raise RuntimeError('invalid timing samples')
            hashes[str(path)] = sha(path)
    mixed_check = dict(passed=False, different_routes=False)
    if len(mixed_rows) == 2:
        batch = Observations(*(torch.cat([row.args()[i] for row in mixed_rows]) for i in range(4)))
        audit_threshold = frozen['threshold_db'] if abs(frozen['threshold_db']) != 100 else 0.
        output, info = estimate(batch, models, audit_threshold, CONFIG['nominal_power'], diagnostics=True)
        singles = torch.cat([estimate(row, models, audit_threshold, CONFIG['nominal_power']) for row in mixed_rows])
        torch.testing.assert_close(output, singles, rtol=3e-8, atol=3e-9)
        mixed_check = dict(passed=True, different_routes=bool(info['allpilot_mask'].any() and not info['allpilot_mask'].all()),
                           forward_calls=info['forward_calls'], cases_per_model=info['cases_per_model'],
                           diagnostic_threshold_db=audit_threshold)
    result = dict(passed=True, worlds=len(ids), replay_predictions=replays, maximum_replay_error=maximum,
                  mixed_batch_check=mixed_check, hashes=hashes,
                  scope='All paired IDs, inputs, sources and routing masks; first world regenerated and predictions replayed in every cell/offset. Not every prediction replayed.')
    write(ROOT / 'audit.json', result)
    return result


def stats(errors, power, cells):
    er, pw = errors[:, cells], power[:, cells]
    ratios = er / pw[:, :, None, :]
    db = 10 * np.log10(er.sum((0, 3)) / pw.sum((0, 2))[:, None])
    return {name: dict(macro_nmse_db=float(db[:, j].mean()), mean_case_nmse=float(ratios[:, :, j].mean()),
                      pooled_nmse_db=float(10 * np.log10(er[:, :, j].sum() / pw.sum())),
                      severe_rate=float((ratios[:, :, j] > 1).mean())) for j, name in enumerate(NAMES)}


def bootstrap(errors, power, check):
    pc, count = CONFIG['primary_cells'], CONFIG['cases']
    er, pw = errors[:, :pc], power[:, :pc]
    ratios = er / pw[:, :, None, None, :]
    rng = np.random.default_rng(CONFIG['bootstrap_seed']); gain, fail, mean = [], [], []
    for start in range(0, CONFIG['bootstrap_samples'], 100):
        check('bootstrap', done=start, total=CONFIG['bootstrap_samples'])
        size = min(100, CONFIG['bootstrap_samples'] - start)
        weights = rng.multinomial(count, np.full(count, 1 / count), size=(size, len(CONFIG['banks']), pc))
        e = np.einsum('rbcn,bcomn->rcom', weights, er, optimize=True)
        p = np.einsum('rbcn,bcn->rc', weights, pw, optimize=True)
        db = (10 * np.log10(e / p[:, :, None, None])).mean(1)
        severity = np.einsum('rbcn,bcomn->rom', weights, (ratios > 1).astype(float), optimize=True) / (len(CONFIG['banks']) * pc * count)
        average = np.einsum('rbcn,bcomn->rom', weights, ratios, optimize=True) / (len(CONFIG['banks']) * pc * count)
        gain.append(db[:, :, :2] - db[:, :, 2, None]); fail.append(severity[:, :, 2, None] - severity[:, :, :2])
        mean.append(average[:, :, 2, None] - average[:, :, :2])
    values = dict(gain_db=np.concatenate(gain), severe_delta=np.concatenate(fail), mean_nmse_delta=np.concatenate(mean))
    save(ROOT / 'bootstrap_draws.npz', noise_offsets_db=np.array(CONFIG['noise_offsets']), comparators=np.array(MODELS), **values)
    return [{name: {key + '_conditional95_ci': np.quantile(value[:, oi, j], [.025, .975]).tolist()
                    for key, value in values.items()} for j, name in enumerate(MODELS)} for oi in range(len(CONFIG['noise_offsets']))]


def analyze(check=lambda *args, **kwargs: None):
    verification, frozen = audit(check), freeze()
    errors, powers, routes = [], [], []
    for bank in range(len(CONFIG['banks'])):
        er, pw, masks = [], [], []
        for spec in SPECS:
            pred = load(ROOT / 'predictions' / f'bank{bank}_{spec["id"]}.npz')
            g = load(ROOT / 'data' / f'bank{bank}' / spec['id'] / 'truth.npz')['g']
            er.append((abs(pred['estimates'] - g[None, None]) ** 2).sum((-2, -1)))
            pw.append((abs(g) ** 2).sum((-2, -1))); masks.append(pred['allpilot_mask'])
        errors.append(er); powers.append(pw); routes.append(masks)
    errors, power, routes = np.asarray(errors), np.asarray(powers), np.asarray(routes)
    if not np.isfinite(errors).all() or not np.isfinite(power).all() or (power <= 0).any():
        raise FloatingPointError('invalid error arrays')
    save(ROOT / 'all_errors.npz', squared_error=errors, channel_power=power, allpilot_mask=routes, names=np.array(NAMES), noise_offsets_db=np.array(CONFIG['noise_offsets']))
    intervals, pc = bootstrap(errors, power, check), CONFIG['primary_cells']
    nominal = CONFIG['noise_offsets'].index(0.)
    offsets = {}
    for oi, offset in enumerate(CONFIG['noise_offsets']):
        er, ratios = errors[:, :, oi], errors[:, :, oi] / power[:, :, None, :]
        primary = stats(er, power, np.arange(pc))
        cells = {spec['id']: dict(spec=spec, methods=stats(er, power, [j]), allpilot_fraction=float(routes[:, j, oi].mean()),
                 switched_vs_correct_variance_fraction=float((routes[:, j, oi] != routes[:, j, nominal]).mean())) for j, spec in enumerate(SPECS)}
        snrs = sorted(set(spec['snr'] for spec in SPECS[:pc]))
        per_snr = {str(s): stats(er, power, [j for j, spec in enumerate(SPECS[:pc]) if spec['snr'] == s]) for s in snrs}
        contrasts = intervals[oi]
        for name in MODELS:
            before = ratios[:, :pc, NAMES.index(name)] > 1
            after = ratios[:, :pc, NAMES.index('selector')] > 1
            contrasts[name].update(gain_db=primary[name]['macro_nmse_db'] - primary['selector']['macro_nmse_db'],
                severe_delta=primary['selector']['severe_rate'] - primary[name]['severe_rate'],
                mean_nmse_delta=primary['selector']['mean_case_nmse'] - primary[name]['mean_case_nmse'],
                severe_transitions=dict(worlds=int(before.size), both_good=int((~before & ~after).sum()),
                    corrected_by_selector=int((before & ~after).sum()),
                    new_failure_from_selector=int((~before & after).sum()),
                    both_severe=int((before & after).sum())))
        costs = {}
        for name in NAMES:
            values = [v for spec in SPECS[:pc] for v in json.loads((ROOT / 'timing' / f'{spec["id"]}.json').read_text())['offsets'][oi]['seconds'][name]]
            costs[name] = dict(mean_ms=float(np.mean(values) * 1000), median_ms=float(np.median(values) * 1000))
        changed = routes[:, :pc, oi] != routes[:, :pc, nominal]
        nominal_route_error = np.where(routes[:, :pc, nominal], ratios[:, :pc, 1], ratios[:, :pc, 0])
        route_delta = np.where(changed, ratios[:, :pc, 2] - nominal_route_error, 0.)
        switch = dict(count=int(changed.sum()), fraction=float(changed.mean()),
                      beneficial=int((changed & (route_delta < 0)).sum()), harmful=int((changed & (route_delta > 0)).sum()),
                      tied=int((changed & (route_delta == 0)).sum()),
                      mean_nmse_change_due_to_routing_only=float(route_delta.mean()),
                      definition='At this same supplied variance, compare changed route with the route selected under correct variance; truth used only for this evaluation diagnostic.')
        # Independent scalar recalculation of the macro statistic.
        for j, name in enumerate(NAMES):
            manual = np.mean([10 * np.log10(er[:, c, j].sum() / power[:, c].sum()) for c in range(pc)])
            np.testing.assert_allclose(manual, primary[name]['macro_nmse_db'], rtol=1e-12, atol=1e-12)
        low = {str(s): dict(selector=per_snr[str(float(s))]['selector']['mean_case_nmse'], original=per_snr[str(float(s))]['holdout_best']['mean_case_nmse']) for s in (-10, -5)}
        offsets[str(offset)] = dict(primary=primary, contrasts=contrasts, per_snr=per_snr, cells=cells, timing=costs,
            low_snr_mean_nmse=low, routing=dict(allpilot_fraction=float(routes[:, :pc, oi].mean()), switches=switch),
            selector_to_original_runtime_ratio=costs['selector']['mean_ms'] / costs['holdout_best']['mean_ms'])
    nominal_result = offsets[str(CONFIG['noise_offsets'][nominal])]
    comparison = nominal_result['contrasts']['holdout_best']
    criteria = dict(gain_at_least_0p1_db=bool(comparison['gain_db'] >= .1),
                    severe_increase_at_most_0p002=bool(comparison['severe_delta'] <= .002),
                    low_snr_mean_nmse_not_worse=bool(all(v['selector'] <= v['original'] for v in nominal_result['low_snr_mean_nmse'].values())),
                    runtime_within_1p05=bool(nominal_result['selector_to_original_runtime_ratio'] <= 1.05))
    result = dict(threshold_db=frozen['threshold_db'], offsets=offsets, criteria=criteria, screening_passed=all(criteria.values()),
                  audit=verification, independent_headline_recalculation='passed',
                  caveats=['One-seed exploratory development screen, not confirmation or a new estimator architecture.',
                           'Confidence intervals are descriptive paired intervals conditional on the two fitted models, fixed grid and data bank; no multiplicity adjustment.',
                           'The three variance-input settings reuse identical physical worlds and are not independent test replicates.',
                           'Both baseline models and routing receive the same biased variance; actual measurement noise is unchanged.',
                           'The 0.002 error-rate tolerance is a practical screening rule, not statistical noninferiority.',
                           'Two 188-parameter models must be stored; only one model is evaluated per world.',
                           'Noise routing assumes known calibrated expected signal power3 in this benchmark.'])
    write(ROOT / 'summary.json', result)
    lines = ['FIXED SELECTOR DEVELOPMENT SCREEN', '', f'Frozen nominal-SNR threshold: {frozen["threshold_db"]} dB.',
             f'{verification["worlds"]} physical worlds, reused for all variance-input offsets. Primary cells: {pc}.', '']
    for offset, row in offsets.items():
        lines += [f'Supplied variance offset: {offset} dB']
        for name, value in row['primary'].items():
            lines.append(f'{name}: macro {value["macro_nmse_db"]:.4f} dB; meanNMSE {value["mean_case_nmse"]:.6f}; '
                         f'NMSE>1 {value["severe_rate"]:.3%}; CPU {row["timing"][name]["mean_ms"]:.3f} ms')
        for name, value in row['contrasts'].items():
            lines.append(f'Selector vs {name}: gain {value["gain_db"]:.4f} dB, conditional95%CI {value["gain_db_conditional95_ci"]}; '
                         f'failure-rate change {value["severe_delta"]:+.4%}, CI {value["severe_delta_conditional95_ci"]}')
        lines += ['Low SNR: ' + json.dumps(row['low_snr_mean_nmse']), 'Routing: ' + json.dumps(row['routing']), '']
    lines += ['Correct-variance screening criteria:'] + [f'{key}: {value}' for key, value in criteria.items()]
    lines += [f'Passed: {result["screening_passed"]}', '', 'Variance-error robustness is reported separately and does not change the frozen selection.',
              'Per-SNR, all stress/boundary cells and routing fractions are in summary.json.', ''] + result['caveats']
    target = ROOT / 'RESULTS.txt'
    with tempfile.NamedTemporaryFile(mode='w', dir=target.parent, prefix=target.name + '.', suffix='.tmp', delete=False) as stream:
        stream.write('\n'.join(lines) + '\n'); temporary = stream.name
    os.replace(temporary, target)
    return result
