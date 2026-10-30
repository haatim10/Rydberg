"""Audited screening statistics. Channel truth is opened only here."""
import json
import numpy as np
import torch

from sahure_expansion.data import world
from fair_optshrink.estimator import tensors
from . import evaluate, protocol

ROOT, CONFIG, SPECS = protocol.ROOT, protocol.CONFIG, protocol.SPECS
sha, write, save, load = protocol.sha, protocol.write, protocol.save, protocol.load
NAMES = evaluate.NAMES
GROUPS = {'original': [0, 1, 2]} | {name: [i] for i, name in enumerate(NAMES) if i >= 3}


def collect_diagnostics(value, totals):
    """Counts all candidate-fit and refit denoisings, not only selected output."""
    if isinstance(value, dict) and 'user_denoisings' in value:
        for key in ('calls', 'user_denoisings', 'sweeps', 'capped'):
            totals[key] += value[key]
        for key in ('max_sweeps', 'max_residual'):
            totals[key] = max(totals[key], value[key])
    elif isinstance(value, dict):
        for child in value.values():
            collect_diagnostics(child, totals)


@torch.no_grad()
def audit(check):
    manifest = protocol.initialize()
    selected = evaluate.frozen()
    digest = sha(ROOT/'frozen_selection.json')
    models = evaluate.load_models()
    parent = json.loads(evaluate.PARENT.read_text())
    seen, hashes, cap = set(), {}, {}
    replays = {'original': 0, 'lrhd_loop': 0}
    maxdiff = {'original': 0., 'lrhd_loop': 0.}
    prediction_seconds = {name: 0. for name in NAMES}
    ranks = {name: {} for name in ('rank_old_opt', 'matched_onepass', 'lrhd_loop', 'lrhd_final')}
    for bank, seed in enumerate(CONFIG['banks']):
        for cell, spec in enumerate(SPECS):
            check('auditing', bank=bank, cell=cell)
            observations = evaluate.data(bank, cell)
            folder = ROOT/'data'/f'bank{bank}'/spec['id']
            meta = json.loads((folder/'manifest.json').read_text())
            truth = load(folder/'truth.npz')
            if meta['selection_sha256'] != digest or meta['generated_unix'] < selected['frozen_unix']:
                raise RuntimeError('test generated before final selection freeze')
            expected_ids = [f'lrhd_screen:{seed}:{i}' for i in
                range(cell*CONFIG['cases'], (cell+1)*CONFIG['cases'])]
            np.testing.assert_array_equal(observations['trial_ids'], expected_ids)
            np.testing.assert_array_equal(truth['trial_ids'], expected_ids)
            if seen.intersection(expected_ids):
                raise RuntimeError('duplicate test IDs')
            seen.update(expected_ids)
            # Independent regeneration checks the simulator, pilot/noise pairing and truth.
            item, _ = world(cell*CONFIG['cases'], seed, **spec)
            for key, value in zip(('z', 's', 'b', 'sigma2'), (item.Z, item.S, item.B, item.sigma2)):
                np.testing.assert_array_equal(observations[key][0], value)
            np.testing.assert_array_equal(truth['g'][0], item.G)
            for case, trial_id in enumerate(expected_ids):
                result = evaluate.cached(bank, cell, case, trial_id)
                if result is None:
                    raise RuntimeError('incomplete test prediction')
                prediction, info = result
                path, commit = evaluate.paths(bank, spec, case)
                hashes[str(path)], hashes[str(commit)] = sha(path), sha(commit)
                for name in NAMES:
                    seconds = info['seconds'][name]
                    if not np.isfinite(seconds) or seconds <= 0:
                        raise RuntimeError('invalid evaluation processing time')
                    prediction_seconds[name] += seconds
                    if name.startswith('lrhd'):
                        totals = cap.setdefault(name, dict(calls=0, user_denoisings=0,
                            sweeps=0, capped=0, max_sweeps=0, max_residual=0.))
                        collect_diagnostics(info['diagnostics'][name], totals)
                    if name in ranks:
                        rank = str(info['diagnostics'][name]['config']['rank'])
                        ranks[name][rank] = ranks[name].get(rank, 0)+1
                if case == 0 and cell in (0, 5, 11, len(SPECS)-1):
                    obs = tensors({key: observations[key][:1] for key in ('z','s','b','sigma2')})
                    for name in evaluate.MODEL_NAMES + ('lrhd_loop',):
                        value, replay_info = evaluate.infer(name, obs, models, selected, parent)
                        actual = prediction['estimates'][NAMES.index(name)]
                        np.testing.assert_allclose(actual, value.numpy()[0], rtol=3e-10, atol=3e-11)
                        group = 'original' if name in evaluate.MODEL_NAMES else name
                        replays[group] += 1
                        maxdiff[group] = max(maxdiff[group], float(abs(actual-value.numpy()[0]).max()))
                        if name == 'lrhd_loop':
                            old_info = info['diagnostics'][name]
                            if old_info['config'] != replay_info['config']:
                                raise RuntimeError('LRHD selected rank replay changed')
                            np.testing.assert_allclose(old_info['scores'], replay_info['scores'], rtol=3e-10, atol=3e-11)
            timing = json.loads((ROOT/'timing'/f'{spec["id"]}.json').read_text())
            if (timing['selection_sha256'] != digest or timing['parent_selection_sha256'] != sha(evaluate.PARENT)
                    or set(timing['seconds']) != set(NAMES)):
                raise RuntimeError('timing identity mismatch')
            for values in timing['seconds'].values():
                if len(values) != CONFIG['timing_repeats'] or not np.isfinite(values).all() or min(values) <= 0:
                    raise RuntimeError('incomplete or invalid timings')
    for path, expected_hash in selected['development_hashes'].items():
        if sha(path) != expected_hash:
            raise RuntimeError('development record changed after selection: '+path)
    development_diagnostics = dict(calls=0, user_denoisings=0, sweeps=0,
                                   capped=0, max_sweeps=0, max_residual=0.)
    for entry in selected['development_diagnostics']:
        if entry['kind'] == 'lrhd':
            collect_diagnostics(entry, development_diagnostics)
    record = dict(passed=True, worlds=len(seen), primary_worlds=len(CONFIG['banks'])*CONFIG['primary_cells']*CONFIG['cases'],
        source_files_checked=len(manifest['sources']), replays=replays, maximum_replay_error=maxdiff,
        development_hashes_checked=len(selected['development_hashes']),
        development_lrhd_diagnostics=development_diagnostics,
        lrhd_diagnostics=cap, selected_rank_counts=ranks, evaluation_processing_seconds=prediction_seconds,
        hashes=hashes, scope='all source/checkpoint/data/prediction hashes and IDs; one generated world per bank/cell; '
        'three original models and online LRHD replayed at first case in four cells per bank. '
        'Cap counts include all candidate fits/refits; timing warmups/repeats are excluded.')
    write(ROOT/'audit.json', record)
    return record


def run(check=lambda *args, **kwargs: None):
    verification = audit(check)
    errors, powers = [], []
    for bank in range(len(CONFIG['banks'])):
        bank_errors, bank_powers = [], []
        for spec in SPECS:
            truth = load(ROOT/'data'/f'bank{bank}'/spec['id']/'truth.npz')['g']
            estimates = np.stack([load(evaluate.paths(bank, spec, case)[0])['estimates']
                                  for case in range(CONFIG['cases'])], axis=1)
            bank_errors.append(np.square(abs(estimates-truth[None])).sum((-2,-1)))
            bank_powers.append(np.square(abs(truth)).sum((-2,-1)))
        errors.append(bank_errors)
        powers.append(bank_powers)
    errors, power = np.asarray(errors), np.asarray(powers)
    if not np.isfinite(errors).all() or not np.isfinite(power).all() or (power <= 0).any() or (errors <= 0).any():
        raise FloatingPointError('invalid channel error/power arrays')
    save(ROOT/'all_errors.npz', squared_error=errors, channel_power=power, names=np.array(NAMES))
    pc = CONFIG['primary_cells']
    db = 10*np.log10(errors.sum((0,3))/power.sum((0,2))[:,None])
    ratios = errors/power[:,:,None,:]
    bank_db = 10*np.log10(errors.sum(-1)/power.sum(-1)[...,None])
    primary = {name: float(db[:pc, indices].mean()) for name, indices in GROUPS.items()}
    cells = {spec['id']: {name: dict(nmse_db=float(db[cell,indices].mean()),
        severe_rate=float((ratios[:,cell,indices,:]>1).mean())) for name, indices in GROUPS.items()}
        for cell, spec in enumerate(SPECS)}
    comparators = ('original', 'matched_onepass', 'rank_old_opt')
    rng = np.random.default_rng(CONFIG['bootstrap_seed'])
    draws = []
    for start in range(0, CONFIG['bootstrap_samples'], 100):
        check('bootstrap', done=start, total=CONFIG['bootstrap_samples'])
        count = min(100, CONFIG['bootstrap_samples']-start)
        weights = rng.multinomial(CONFIG['cases'], np.full(CONFIG['cases'], 1/CONFIG['cases']),
            size=(count, len(CONFIG['banks']), pc))
        e = np.einsum('rbcn,bcmn->rcm', weights, errors[:,:pc], optimize=True)
        p = np.einsum('rbcn,bcn->rc', weights, power[:,:pc], optimize=True)
        sample_db = 10*np.log10(e/p[...,None])
        target = sample_db[...,GROUPS['lrhd_loop']].mean(-1)
        draws.append(np.stack([(sample_db[...,GROUPS[other]].mean(-1)-target).mean(1)
                               for other in comparators], -1))
    draws = np.concatenate(draws)
    contrasts = {}
    for i, other in enumerate(comparators):
        cell_gain = db[:pc,GROUPS[other]].mean(-1)-db[:pc,GROUPS['lrhd_loop']].mean(-1)
        contrasts[other] = dict(lrhd_gain_db=primary[other]-primary['lrhd_loop'],
            descriptive95_ci=np.quantile(draws[:,i], [.025,.975]).tolist(),
            improved_cells=int((cell_gain>0).sum()), worst_cell_degradation_db=max(0.,float(-cell_gain.min())))
    timing = {}
    for name, indices in GROUPS.items():
        values = np.array([value for spec in SPECS[:pc] for index in indices
            for value in json.loads((ROOT/'timing'/f'{spec["id"]}.json').read_text())['seconds'][NAMES[index]]])*1000
        timing[name] = dict(mean_ms=float(values.mean()), median_ms=float(np.median(values)),
            p10_ms=float(np.quantile(values,.1)), p90_ms=float(np.quantile(values,.9)),
            min_ms=float(values.min()), max_ms=float(values.max()), measurements=len(values))
    selected = evaluate.frozen()
    result = dict(primary=primary, contrasts=contrasts, cells=cells, timing=timing,
        seed_primary={name:float(db[:pc,NAMES.index(name)].mean()) for name in evaluate.MODEL_NAMES},
        bank_primary={name:bank_db[:,:pc,indices].mean((1,2)).tolist() for name, indices in GROUPS.items()},
        severe_rate={name:float((ratios[:,:pc,indices,:]>1).mean()) for name, indices in GROUPS.items()},
        frozen_choices=selected['choices'], development_processing_seconds=selected['development_seconds'],
        audit=verification,
        caveats=['Small development screen: 384 primary and 64 stress worlds, 32 worlds per cell; not a confirmation study.',
          'Original scores average three separately fitted models, not an ensemble; their training dataset is shared.',
          'Bootstrap resamples paired worlds within each bank/cell and preserves all three model predictions. '
          'Intervals are descriptive, unadjusted and conditional on fixed banks, selected settings and fitted models.',
          'Iterative LRHD is a complex adaptation inside Rician EM; the original real additive-noise guarantees do not transfer.',
          'The matched one-pass control has the same new tuning grid; the historical OptShrink control had a larger tuning budget.',
          'Primary LRHD is the frozen in-loop method. Final-only and offline-rank LRHD are diagnostics, not test-selected winners.',
          'Every online rank search, initializer and refit is timed on CPU float64 batch1. Offline tuning is not inference cost.',
          'Capped inner iterations are not converged outputs; inspect numerical diagnostics and per-cell results.',
          'These 12-cell scores must not be pooled with the earlier 126-cell primary studies. No hardware or novelty claim follows.'])
    for name, indices in GROUPS.items():
        independent = np.mean([10*np.log10(sum(float(errors[b,c,m,j]) for b in range(len(CONFIG['banks'])) for j in range(CONFIG['cases']))/
            sum(float(power[b,c,j]) for b in range(len(CONFIG['banks'])) for j in range(CONFIG['cases'])))
            for c in range(pc) for m in indices])
        np.testing.assert_allclose(independent, primary[name], rtol=1e-12, atol=1e-12)
    result['independent_headline_recalculation'] = 'passed'
    write(ROOT/'summary.json', result)
    lines = ['SMALL MATCHED LRHD SCREEN', '',
        '384 fresh primary worlds and 64 stress worlds. Lower NMSE is better.',
        'Primary: equal average of 12 cell-level dB ratios of summed error to summed channel power.',
        'Original estimator unchanged; no retraining. Three individual model scores are averaged, not ensembled.', '']
    for name in GROUPS:
        lines.append(f"{name}: {primary[name]:.4f} dB; severe NMSE > 1 {result['severe_rate'][name]:.3%}; "
            f"CPU mean {timing[name]['mean_ms']:.3f} ms, median {timing[name]['median_ms']:.3f} ms, "
            f"10th-90th percentile {timing[name]['p10_ms']:.3f}-{timing[name]['p90_ms']:.3f} ms")
    lines += ['', 'Primary in-loop LRHD comparisons (positive gain favors LRHD):']
    for name, value in contrasts.items():
        lines.append(f"Against {name}: {value['lrhd_gain_db']:.4f} dB; descriptive 95% interval "
            f"{value['descriptive95_ci']}; improved {value['improved_cells']}/12 cells; "
            f"worst cell degradation {value['worst_cell_degradation_db']:.4f} dB")
    lines += ['', 'Original model-seed scores: '+json.dumps(result['seed_primary']),
              f"Offline development processing time, both methods together: {result['development_processing_seconds']/60:.2f} minutes.",
              '', 'Per-cell results:']
    for spec in SPECS:
        label = 'STRESS, excluded from primary' if SPECS.index(spec) >= pc else 'PRIMARY'
        lines.append(spec['id']+' ['+label+']: '+json.dumps(cells[spec['id']]))
    lines += ['', 'LRHD numerical diagnostics (including candidate fitting):',
        json.dumps(verification['lrhd_diagnostics'], indent=2),
        '', 'LRHD development numerical diagnostics (all tuning candidates):',
        json.dumps(verification['development_lrhd_diagnostics'], indent=2), '', 'Limits:']+result['caveats']
    lines += ['', 'Source/checkpoint/data hashes, matched IDs, representative replay and independent metric recalculation passed.',
              'No manuscript change or automatic larger run. Review the accuracy, failures and cost together.']
    (ROOT/'RESULTS.txt').write_text('\n'.join(lines)+'\n')
    return result
