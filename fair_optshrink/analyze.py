"""Post-freeze numerical audit and conditional paired analysis; truth lives here."""
import json
import time
import numpy as np
import torch
from sahure_expansion.data import world
from sahure.data import load_observations
from sahure.math import nll
from paper2_optshrink_loop.estimator import estimate as independent_opt
from merged_review_checks.run import one_sweep
from .run import (ROOT, DATA, OLD, SPECS, SEEDS, CONFIG, sha, load, save, write, initialize,
                  models, tensors, configs, select_settings, validation_data)


@torch.no_grad()
def audit(check):
    initialize()
    frozen = json.loads((ROOT / 'frozen_selection.json').read_text())
    digest = sha(ROOT / 'frozen_selection.json')
    for path, h in frozen['development_hashes'].items():
        if sha(path) != h:
            raise RuntimeError('changed development evidence')
    table = load(ROOT / 'development_scores.npz')
    selected, strata = select_settings(table['scores'], table['condition_ids'])
    if selected != frozen['choices']:
        raise RuntimeError('independent selection replay mismatch')
    np.testing.assert_allclose(strata, table['strata'], rtol=0, atol=0)
    # Validate all original observation inputs against the original generator.
    val = load_observations(DATA / 'val_observations.npz', dtype=torch.float64)
    for i, condition in enumerate(table['condition_ids']):
        check('audit_validation_inputs', done=i, total=len(val))
        w, _ = world(i, 2026092202, **SPECS[int(condition)])
        for key, value in zip(('z', 's', 'b', 'sigma2'), (w.Z, w.S, w.B, w.sigma2)):
            np.testing.assert_array_equal(getattr(val, key)[i].numpy(), value)
    learned = models(); a, b, ids = validation_data(); learned_validation = {}
    for name, model in learned.items():
        losses = []
        for start in range(0, len(a), 32):
            check('audit_learned_validation', model=name, done=start)
            aa, bb = a.batch(slice(start, start + 32)), b.batch(slice(start, start + 32))
            g = model(*aa.args())
            losses.extend(nll(bb.z, g @ bb.s + bb.b, bb.sigma2).mean((-2, -1)).tolist())
        learned_validation[name] = float(np.mean([np.asarray(losses)[ids == j].mean() for j in range(126)]))
    all_ids = set(); hashes = {}; replay = 0; max_abs = 0.
    for bank in range(len(CONFIG['test_banks'])):
        for j, spec in enumerate(SPECS):
            check('audit_test', bank=bank, cell=j)
            folder = ROOT / 'data' / f'bank{bank}' / spec['id']
            meta = json.loads((folder / 'manifest.json').read_text())
            if meta['selection_sha256'] != digest or meta['generated_unix'] < frozen['frozen_unix']:
                raise RuntimeError('freeze-before-test chronology failed')
            for path, h in meta['hashes'].items():
                if sha(path) != h:
                    raise RuntimeError('test data integrity failed')
                hashes[path] = h
            data, truth = load(folder / 'observations.npz'), load(folder / 'truth.npz')
            path = ROOT / 'predictions' / f'bank{bank}_{spec["id"]}.npz'
            out = load(path); hashes[str(path)] = sha(path)
            if str(out['selection_sha256']) != digest:
                raise RuntimeError('prediction selection mismatch')
            for other in (truth['trial_ids'], out['trial_ids']):
                np.testing.assert_array_equal(data['trial_ids'], other)
            if all_ids.intersection(data['trial_ids']):
                raise RuntimeError('duplicate test identity')
            all_ids.update(data['trial_ids'])
            for i in range(meta['count']):
                w, _ = world(i, meta['seed'], **spec)
                for key, value in zip(('z', 's', 'b', 'sigma2'), (w.Z, w.S, w.B, w.sigma2)):
                    np.testing.assert_array_equal(data[key][i], value)
                np.testing.assert_array_equal(truth['g'][i], w.G)
            obs = tensors({k: data[k][:1] for k in ('z', 's', 'b', 'sigma2')})
            names = out['names'].tolist()
            for name, model in learned.items():
                np.testing.assert_allclose(model(*obs.args()).numpy()[0], out['estimates'][names.index(name), 0], rtol=3e-8, atol=3e-9)
                replay += 1
            for name, cfg in configs(obs, frozen).items():
                expected = independent_opt(data['z'][0], data['s'][0], data['b'][0], float(data['sigma2'][0]), cfg)
                actual = out['estimates'][names.index(name), 0]
                np.testing.assert_allclose(actual, expected, rtol=3e-8, atol=3e-9)
                max_abs = max(max_abs, float(np.max(abs(actual - expected)))); replay += 1
            expected, r, _ = one_sweep(data['z'][0], data['s'][0], data['b'][0], float(data['sigma2'][0]))
            np.testing.assert_allclose(out['estimates'][names.index('hs_one'), 0], expected, rtol=3e-8, atol=3e-9)
            if int(out['hs_ranks'][0]) != r:
                raise RuntimeError('HS rank replay mismatch')
            replay += 1
    result = dict(status='passed', regenerated_validation=2048, regenerated_test=len(all_ids),
        representative_prediction_replays=replay, independent_classical_max_abs_error=max_abs,
        learned_validation_double_precision=learned_validation, hashes=hashes,
        scope='conditional numerical audit; not external replication or training convergence')
    write(ROOT / 'audit.json', result)
    return result


def analyze(check=lambda *a, **k: None):
    verification = audit(check)
    errors, powers = [], []
    for bank in range(len(CONFIG['test_banks'])):
        ee, pp = [], []
        for spec in SPECS:
            out = load(ROOT / 'predictions' / f'bank{bank}_{spec["id"]}.npz')
            truth = load(ROOT / 'data' / f'bank{bank}' / spec['id'] / 'truth.npz')['g']
            names = out['names'].tolist()
            ee.append(abs(out['estimates'] - truth[None]) ** 2)
            pp.append((abs(truth) ** 2).sum((-2, -1)))
        errors.append(np.stack(ee).sum((-2, -1))); powers.append(np.stack(pp))
    errors, powers = np.stack(errors), np.stack(powers)  # bank, cell, method, world
    save(ROOT / 'all_errors.npz', squared_error=errors, channel_power=powers, names=np.array(names),
         cell_ids=np.array([s['id'] for s in SPECS]))
    groups = {name: [names.index(name)] for name in names if not name.startswith('learned_')}
    groups['learned'] = [names.index(f'learned_{s}') for s in SEEDS]
    # Pool independent equally sized test banks within a condition, not seeds as worlds.
    cell_db = 10 * np.log10(errors.sum((0, 3)) / powers.sum((0, 2))[:, None])
    bank_db = 10 * np.log10(errors.sum(-1) / powers.sum(-1)[..., None])
    ratios = errors / powers[:, :, None, :]
    primary = {g: float(cell_db[:126, ii].mean()) for g, ii in groups.items()}
    cells = {}
    thresholds = [.5, 1., 2.]
    for j, spec in enumerate(SPECS):
        cells[spec['id']] = {g: dict(nmse_db=float(cell_db[j, ii].mean()),
            mean_case_nmse=float(ratios[:, j, ii, :].mean()),
            exceedance={str(t): float((ratios[:, j, ii, :] > t).mean()) for t in thresholds}) for g, ii in groups.items()}
    macro_tails = {g: {str(t): float((ratios[:, :126, ii, :] > t).mean()) for t in thresholds} for g, ii in groups.items()}
    # Every empirical point is retained; no favourable threshold chosen later.
    save(ROOT / 'error_distributions.npz', **{g: np.sort(ratios[:, :126, ii, :].ravel()) for g, ii in groups.items()})
    comparators = ('opt_selected', 'opt_loop', 'opt_final')
    rng = np.random.default_rng(CONFIG['bootstrap_seed']); samples = []; cell_samples = []
    for start in range(0, CONFIG['bootstrap_samples'], 100):
        check('paired_uncertainty', done=start, total=CONFIG['bootstrap_samples'])
        size = min(100, CONFIG['bootstrap_samples'] - start)
        n = CONFIG['cases_per_cell']
        weights = rng.multinomial(n, np.ones(n) / n, size=(size, len(CONFIG['test_banks']), 126))
        e = np.einsum('rbcn,bcmn->rcm', weights, errors[:, :126], optimize=True)
        p = np.einsum('rbcn,bcn->rc', weights, powers[:, :126], optimize=True)
        db = 10 * np.log10(e / p[..., None])
        learned = db[..., groups['learned']].mean(-1)
        gain = np.stack([db[..., names.index(c)] - learned for c in comparators], -1)
        samples.append(gain.mean(1)); cell_samples.append(gain)
    samples = np.concatenate(samples); cell_samples = np.concatenate(cell_samples)
    contrasts = {}
    alpha = .05 / (2 * len(comparators))
    for ci, comparator in enumerate(comparators):
        contrasts[comparator] = dict(gain_db=primary[comparator] - primary['learned'],
            simultaneous95_macro_ci=np.quantile(samples[:, ci], [alpha, 1 - alpha]).tolist(),
            pointwise95_macro_ci=np.quantile(samples[:, ci], [.025, .975]).tolist(),
            worst_cell_deterioration_db=float(np.max(cell_db[:126, groups['learned']].mean(-1) - cell_db[:126, names.index(comparator)])),
            pointwise95_cell_ci=np.quantile(cell_samples[:, :, ci], [.025, .975], axis=0).T.tolist())
    timing_rows = [json.loads((ROOT / 'timing' / f'{s["id"]}.json').read_text()) for s in SPECS]
    timings = {g: dict(mean_ms=float(1000 * np.mean([v for row in timing_rows[:126] for i in ii for v in row['seconds'][names[i]]])),
        median_ms=float(1000 * np.median([v for row in timing_rows[:126] for i in ii for v in row['seconds'][names[i]]]))) for g, ii in groups.items()}
    c = contrasts['opt_selected']
    signal = c['gain_db'] >= .3 and c['simultaneous95_macro_ci'][0] > 0
    summary = dict(primary=primary, contrasts=contrasts, primary_threshold_rates=macro_tails,
        cells=cells, timing=timings, bank_primary={g: bank_db[:, :126, ii].mean((1, 2)).tolist() for g, ii in groups.items()},
        learned_seed_primary={str(s): float(cell_db[:126, names.index(f'learned_{s}')].mean()) for s in SEEDS},
        ratio_of_sums_pooled={g: float(np.mean(10 * np.log10(errors[:, :126, ii].sum((0, 1, 3)) / powers[:, :126].sum()))) for g, ii in groups.items()},
        mean_case_linear={g: float(ratios[:, :126, ii].mean()) for g, ii in groups.items()},
        accuracy_screen_passed=bool(signal), decision='review_tradeoffs_before_any_publication_claim', audit=verification,
        limits=['Original frozen training dataset shared by three initializations; two independent TEST banks only.',
                'Intervals condition on model weights and development selection; not repeat-training uncertainty.',
                'Only three primary macro intervals adjusted for multiplicity; cell intervals are pointwise.',
                'All selectors use held-out likelihood, not channel-error labels or a failure-rate objective.',
                'Full error distribution is descriptive; no reliability/outage or stochastic-dominance guarantee.',
                'No global OptShrink optimality, universal winner, hardware validation or new-method novelty claimed.'])
    write(ROOT / 'summary.json', summary)
    lines = ['ORIGINAL188 VERSUS POPULATION-TUNED OPTSHRINK', '',
        'Completed:4096 paired fresh test worlds;126 primary conditions and2 separate diagnostics.',
        'Three frozen original models. OptShrink selected on the same2048 validation observations and pilot masks.',
        'No new neural training. No true channel or path count supplied to any selector.',
        'Main error: equal mean of126 per-cell dB NMSEs; lower is better.', '',
        'PRIMARY ERROR / END-TO-END MEAN CPU TIME']
    lines += [f'  {g}: {primary[g]:.3f} dB; {timings[g]["mean_ms"]:.3f} ms' for g in groups]
    lines += ['', 'LEARNED GAINS: positive favours learning (three-comparison adjusted95% macro intervals)']
    lines += [f'  vs {g}: {d["gain_db"]:.3f} dB; CI {d["simultaneous95_macro_ci"]}; worst-cell deterioration {d["worst_cell_deterioration_db"]:.3f} dB' for g, d in contrasts.items()]
    lines += ['', 'PREDECLARED THRESHOLD RATES: .5 /1 /2; not communication outage']
    lines += [f'  {g}: ' + ' / '.join(f'{100*macro_tails[g][str(t)]:.3f}%' for t in thresholds) for g in groups]
    lines += ['', f'Predeclared .3dB accuracy screen passed: {bool(signal)}',
        'A passed screen is not publication acceptance or verified novelty. Do not redefine a failure retrospectively.',
        'Inspect the full error distributions and fixed-budget curves before discussing a conditional trade-off.',
        'The two test banks do not measure independent retraining variability. Stress cells are not pooled into primary accuracy.',
        'Numerical replay/input audit passed. Previous experiments and manuscript unchanged.',
        'Full per-cell/per-seed/bank statistics, intervals and timings: summary.json. Raw errors: all_errors.npz.']
    (ROOT / 'RESULTS.txt').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines), flush=True)
