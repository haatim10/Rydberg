"""Independent post-campaign integrity checks; never passes partial campaigns.

Run: python -m p1_revision.audit --root results/p1_revision/campaign
The complete 30-cell development store named smoke is allowed, explicitly
labelled as a smoke check. A reduced store never passes as the full campaign.
No frozen simulation or historical result is modified.
"""
import os
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
              'VECLIB_MAXIMUM_THREADS'):
    os.environ[_name] = '1'

import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np

REPO = Path(__file__).resolve().parents[1]
DEFAULT = REPO/'results/p1_revision/campaign'
BASE_METHODS = ('em', 'hs', 'rician_gd', 'rician_hankel', 'em_smooth', 'xu_linearized')
PAIR_METHODS = ('em', 'hs')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def equal(a, b, message):
    require(np.array_equal(np.asarray(a), np.asarray(b)), message)


def close(a, b, message, rtol=2e-11, atol=2e-12):
    require(np.allclose(a, b, rtol=rtol, atol=atol), message)


def source_checks(root, manifest):
    sources = manifest['source_sha256']
    require(bool(sources), 'missing source hashes')
    fingerprint = hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest()
    require(fingerprint == manifest['source_fingerprint'], 'manifest source fingerprint mismatch')
    for relative, expected in sources.items():
        path = Path(relative)
        require(not path.is_absolute() and '..' not in path.parts, 'invalid snapshot path')
        snapshot = root/'source_snapshot'/path
        require(snapshot.is_file(), f'missing frozen source: {relative}')
        require(sha(snapshot) == expected, f'frozen source hash mismatch: {relative}')
        require(sha(REPO/path) == expected, f'current source differs from frozen source: {relative}')
    return len(sources)


def protocol_checks(root, manifest, completion):
    from p1_revision.scenarios import configurations
    cells = manifest['cells']
    expected = configurations()
    require(len(cells) == 30 and len({c['tag'] for c in cells}) == 30,
            'expected 30 distinct declared conditions')
    full = all(c['trials'] == 400 for c in cells)
    smoke = (not full and root.name.startswith('smoke'))
    require(full or smoke, 'reduced trial counts are only accepted in an explicitly named smoke store')
    if smoke:
        counts = {c['trials'] for c in cells}
        require(len(counts) == 1 and 0 < next(iter(counts)) < 400,
                'smoke must use one positive reduced count for every declared condition')
        for c in expected:
            c['trials'] = cells[0]['trials']
    require(cells == expected, 'manifest differs from declared condition/seed/budget protocol')
    count = sum(c['trials'] for c in cells)
    require(completion.get('complete') is True, 'campaign is not complete')
    require(completion.get('completed_cell_trials') == count
            and completion.get('total_cell_trials') == count,
            'completion counts disagree with manifest')
    require(completion.get('source_fingerprint') == manifest['source_fingerprint'],
            'completion source fingerprint mismatch')
    # Only declared within-design pairing may reuse a base trial seed.
    owners = {}
    for c in cells:
        group = ((c['kind'], c['tag']) if c['kind'] == 'baselines' else
                 (c['kind'], c['P'], c['snr']) if c['kind'] == 'holdout' else
                 (c['kind'], c['L']))
        for trial in range(c['seed_base'], c['seed_base'] + c['trials']):
            require(trial not in owners or owners[trial] == group,
                    'unintended seed overlap between designs')
            owners[trial] = group
    return dict(scope='smoke' if smoke else 'full_campaign',
                condition_trials=count, independent_base_worlds=len(owners))


def read_cell(root, cell, fingerprint, raw_hashes):
    directory = root/'raw'/cell['tag']
    chunk_size = int(read_json(root/'manifest.json')['chunk_size'])
    expected_files = [f'{i:05d}.npz' for i in range(0, cell['trials'], chunk_size)]
    files = sorted(directory.glob('*.npz'))
    require([p.name for p in files] == expected_files, f'missing/extra chunks: {cell["tag"]}')
    data, diagnostics, keys = [], [], None
    for path in files:
        with np.load(path, allow_pickle=False) as store:
            require(str(store['fingerprint']) == fingerprint, f'chunk fingerprint: {path}')
            begin = int(path.stem)
            n = min(chunk_size, cell['trials'] - begin)
            require(int(store['count']) == n, f'chunk count: {path}')
            part = {k: store[k].copy() for k in store.files
                    if k not in ('fingerprint', 'count', 'diagnostics_json')}
            current = set(part)
            require(keys is None or keys == current, f'inconsistent arrays: {path}')
            keys = current
            require(all(a.ndim >= 1 and len(a) == n for a in part.values()),
                    f'array leading dimensions: {path}')
            equal(part['trial'], cell['seed_base'] + np.arange(begin, begin+n),
                  f'chunk trial ids/boundaries: {path}')
            diag = json.loads(str(store['diagnostics_json']))
            require(len(diag) == n, f'diagnostic count: {path}')
            data.append(part); diagnostics.extend(diag)
        raw_hashes[str(path.relative_to(root))] = sha(path)
    merged = {k: np.concatenate([p[k] for p in data]) for k in keys}
    equal(merged['trial'], np.arange(cell['seed_base'], cell['seed_base']+cell['trials']),
          f'duplicate/missing/out-of-order trial: {cell["tag"]}')
    require(all(np.isfinite(v).all() for v in merged.values()), 'nonfinite raw values')
    methods = BASE_METHODS if cell['kind'] == 'baselines' else PAIR_METHODS
    require({k[4:] for k in merged if k.startswith('num_')} == set(methods),
            f'missing/unexpected estimators: {cell["tag"]}')
    require((merged['den'] > 0).all(), 'nonpositive channel energy')
    require(all((merged['num_'+method] > 0).all() for method in methods),
            'nonpositive error cannot support the declared log summaries')
    cap = (cell['N']+1)//2
    for key in ('rho_true', 'rho_em'):
        require(merged[key].shape == (cell['trials'], cell['K']), 'per-user rho shape')
        require(((merged[key] >= 1/cap-1e-12) & (merged[key] <= 1+1e-12)).all(),
                'effective rank outside its allowed range')
    require(merged['L_k'].shape == (cell['trials'], cell['K']), 'path-count shape')
    equal(merged['L_k'], np.floor(merged['L_k']), 'path counts must be integers')
    if cell['L'] is None:
        require(((merged['L_k'] >= 3) & (merged['L_k'] <= 7)).all(), 'random-path support')
    else:
        require((merged['L_k'] == cell['L']).all(), 'fixed path counts')
    return merged, diagnostics


def selection_checks(cell, data, diagnostics):
    from p1_revision.baselines import THRESHOLD_FACTORS
    cap = (cell['N']+1)//2
    fallback = dict(hs=0, rician_hankel=0, em_smooth=0)
    for i, diag in enumerate(diagnostics):
        prefix = f'{cell["tag"]}, trial {data["trial"][i]}: '
        if cell['kind'] == 'baselines':
            require(set(diag['methods']) == set(BASE_METHODS), prefix+'method diagnostics')
            heldout = max(1, round(.3*cell['P']))
            equal(diag['context_columns'], np.arange(cell['P']-heldout), prefix+'context split')
            equal(diag['heldout_columns'], np.arange(cell['P']-heldout, cell['P']), prefix+'heldout split')
            for key in ('iterations', 'select_iterations', 'sweeps'):
                require(diag[key] == cell[key], prefix+key)
            require(diag['val_fraction'] == .3 and diag['final_fit'] == 'spectral reinitialization and all pilots',
                    prefix+'selection/refit protocol')
            for method in BASE_METHODS:
                item = diag['methods'][method]
                times = [item['seconds_select'], item['seconds_final_fit']]
                require(np.isfinite(times).all() and min(times) >= 0, prefix+'invalid workload times')
                if method not in fallback:
                    require(item['scores'] == [] and item['candidates'] == []
                            and item['selected_rank'] is None and item['selected_threshold'] is None,
                            prefix+'unexpected parameter search for '+method)
                    continue
                grid = list(THRESHOLD_FACTORS) if method == 'em_smooth' else list(range(1, cap+1))
                equal(item['candidates'], grid, prefix+'candidate grid '+method)
                scores = np.asarray(item['scores'])
                require(scores.shape == (len(grid),) and np.isfinite(scores).all()
                        and (scores >= 0).all(), prefix+'candidate scores '+method)
                chosen = item['selected_threshold'] if method == 'em_smooth' else item['selected_rank']
                require(chosen == grid[int(np.argmin(scores))], prefix+'argmin/tie rule '+method)
                inactive = chosen == (0 if method == 'em_smooth' else cap)
                if inactive:
                    reference = 'rician_gd' if method == 'rician_hankel' else 'em'
                    require(data['num_'+method][i] == data['num_'+reference][i],
                            prefix+'fallback error differs from corresponding core '+method)
                    fallback[method] += 1
        else:
            scores = data['scores'][i]
            require(scores.shape == (cap,) and (scores >= 0).all(), prefix+'score grid')
            chosen = data['rank'][i]
            require(chosen == np.argmin(scores)+1, prefix+'selected rank not score argmin')
            require(diag['rank'] == chosen, prefix+'diagnostic rank disagrees')
            equal(diag['scores'], scores, prefix+'saved score disagreement')
            for key in ('seconds_select', 'seconds_em', 'seconds_hs_fit', 'seconds_hs_total'):
                require(np.isfinite(diag[key]) and diag[key] >= 0 and diag[key] == data[key][i],
                        prefix+'workload timing disagreement')
            close(data['seconds_hs_total'][i], data['seconds_select'][i]+data['seconds_hs_fit'][i],
                  prefix+'selection excluded from total timing')
            if chosen == cap:
                require(data['num_hs'][i] == data['num_em'][i], prefix+'HS fallback error differs')
                fallback['hs'] += 1
    return fallback


def paired_world_checks(cells, raw):
    compared = dict(holdout_condition_pairs=0, mismatch_condition_pairs=0)
    for c in cells:
        if c['kind'] == 'holdout':
            ref = next(x for x in cells if x['kind'] == 'holdout' and x['P'] == c['P']
                       and x['snr'] == c['snr'] and x['val_fraction'] == .3)
            a, b = raw[c['tag']], raw[ref['tag']]
            for key in ('trial', 'den', 'num_em', 'rho_true', 'rho_em', 'L_k'):
                equal(a[key], b[key], 'holdout fractions must share world/baseline arrays: '+key)
            compared['holdout_condition_pairs'] += 1
        elif c['kind'] == 'mismatch':
            ref = next(x for x in cells if x['kind'] == 'mismatch'
                       and x['L'] == c['L'] and x['position_std'] == 0)
            # Perturbed G is the target: den and EM errors MUST NOT be forced
            # to equal the matched-geometry anchor.
            for key in ('trial', 'L_k'):
                equal(raw[c['tag']][key], raw[ref['tag']][key], 'mismatch pairing: '+key)
            compared['mismatch_condition_pairs'] += 1
    return compared


def boundary_world_checks(cells, raw):
    from p1_revision.scenarios import make
    from codex_verification.solver import effective_ranks
    matched_worlds = {}
    records = []
    for c in cells:
        for index in sorted({0, c['trials']-1}):
            w = make(c, index)
            require(w.G.shape == (c['N'], c['K']) and w.Z.shape == (c['N'], c['P'])
                    and w.S.shape == (c['K'], c['P']), 'generated world dimensions')
            # This protocol has c=1 and beta_k=1; these are nominal powers,
            # not realized channel-energy or realized-noise SNR estimates.
            close(w.sigma2, c['K']/10**(c['snr']/10), 'nominal complex noise variance', atol=1e-14)
            close(abs(w.B)**2, 10**(c['rsr']/10), 'single-user RSR convention', atol=1e-12)
            close(w.Z, abs(w.G@w.S+w.B+w.W), 'exact nonlinear observation', rtol=1e-13, atol=1e-13)
            saved = raw[c['tag']]
            close(np.sum(abs(w.G)**2), saved['den'][index], 'boundary channel energy')
            equal(w.L_k, saved['L_k'][index], 'boundary path realization')
            close(effective_ranks(w.G)/((c['N']+1)//2), saved['rho_true'][index], 'boundary true rho')
            key = (c['seed_base'], index)
            if key in matched_worlds:
                original = matched_worlds[key]
                for field in ('S', 'W', 'B'):
                    equal(getattr(w, field), getattr(original, field), 'generated paired inputs '+field)
                if c['kind'] == 'holdout':
                    for field in ('G', 'Z'):
                        equal(getattr(w, field), getattr(original, field), 'generated split pairing '+field)
            else:
                matched_worlds[key] = w
            records.append(dict(tag=c['tag'], index=index, trial=int(saved['trial'][index]), sigma2=float(w.sigma2)))
    return records


def replay_checks(cells, raw):
    from p1_revision.run import evaluate
    probes = [next(c for c in cells if c['kind'] == 'baselines' and c['L'] == 7 and c['snr'] == 10),
              next(c for c in cells if c['kind'] == 'holdout' and c['P'] == 10 and c['snr'] == 0 and c['val_fraction'] == .4),
              next(c for c in cells if c['kind'] == 'mismatch' and c['L'] == 12 and c['position_std'] == .02)]
    records = []
    for c in probes:
        index = min(137, c['trials']-1)
        row, _ = evaluate(c, index)
        errors = {}
        for key, value in row.items():
            if key.startswith('seconds_'):
                continue
            target = raw[c['tag']][key][index]
            close(value, target, 'deterministic replay: '+c['tag']+'/'+key)
            if key.startswith('num_'):
                errors[key] = float(abs(value-target))
        records.append(dict(tag=c['tag'], trial=c['seed_base']+index, absolute_error_differences=errors))
    return records


def summary_checks(root, cells, raw, raw_hashes):
    path = root/'analysis/summary.json'
    if not path.exists():
        return dict(state='not_yet_generated', checked=False)
    summary = read_json(path)
    require(summary['manifest_sha256'] == sha(root/'manifest.json'), 'summary manifest hash')
    require(summary['source_fingerprint'] == read_json(root/'manifest.json')['source_fingerprint'], 'summary source fingerprint')
    require(summary['analysis_source_sha256'] == sha(REPO/'p1_revision/analyze.py'), 'summary analysis source hash')
    require(summary.get('analysis_dependencies_sha256') == {'p1_revision/reuse.py':sha(REPO/'p1_revision/reuse.py')},
            'summary bootstrap dependency hash')
    require(summary['raw_sha256'] == raw_hashes, 'summary raw input hashes')
    require(summary['condition_trials'] == sum(c['trials'] for c in cells), 'summary total condition-trials')
    require(summary['independent_base_worlds'] == len({trial for c in cells
            for trial in range(c['seed_base'],c['seed_base']+c['trials'])}), 'summary independent base worlds')
    seen = set()
    expected = {(c['tag'], m) for c in cells for m in (BASE_METHODS if c['kind']=='baselines' else PAIR_METHODS)}
    for row in summary['records']:
        key = (row['tag'], row['method'])
        require(key in expected and key not in seen, 'duplicate/unknown summary row')
        seen.add(key); d = raw[row['tag']]; num = d['num_'+row['method']]
        require(row['n'] == len(num), 'summary trial count')
        values = dict(nmse_db=10*np.log10(num.sum()/d['den'].sum()),
                      gain_vs_em_db=10*np.log10(d['num_em'].sum()/num.sum()),
                      gain_vs_hs_db=10*np.log10(d['num_hs'].sum()/num.sum()),
                      median_gain_vs_em_db=np.median(10*np.log10(d['num_em']/num)),
                      median_gain_vs_hs_db=np.median(10*np.log10(d['num_hs']/num)))
        for field, value in values.items():
            close(row[field], value, 'summary point estimate/reference ordering: '+field, rtol=1e-12)
        for lo, hi in (('nmse_lo','nmse_hi'), ('gain_vs_em_lo','gain_vs_em_hi'), ('gain_vs_hs_lo','gain_vs_hs_hi')):
            require(np.isfinite([row[lo], row[hi]]).all() and row[lo] <= row[hi], 'invalid confidence interval')
    require(seen == expected, 'missing summary methods/conditions')
    split = {(c['P'],c['snr'],c['val_fraction']):c for c in cells if c['kind']=='holdout'}
    seen = set()
    for row in summary['split_comparison']:
        key=(row['P'],row['snr'],row['fraction']);require(key in split and key not in seen, 'split summary rows')
        seen.add(key); c=split[key]; ref=split[(c['P'],c['snr'],.3)]
        value=10*np.log10(raw[ref['tag']]['num_hs'].sum()/raw[c['tag']]['num_hs'].sum())
        close(row['gain_vs_30pct_db'],value,'split contrast direction/aggregation',rtol=1e-12)
        require(row['n']==c['trials'],'split contrast count')
    require(seen == set(split), 'missing split summary')
    return dict(state='point_estimates_verified', checked=True, rows=len(expected),
                interval_scope='finite ordered intervals; bootstrap formula separately reviewed, draws not regenerated')


def run_audit(root=DEFAULT, *, replay=True):
    root=Path(root).resolve(); start=time.perf_counter()
    require((root/'COMPLETE.json').is_file(), 'COMPLETE.json absent: campaign incomplete')
    manifest=read_json(root/'manifest.json'); completion=read_json(root/'COMPLETE.json')
    count_sources=source_checks(root,manifest)
    scope=protocol_checks(root,manifest,completion)
    cells=manifest['cells']; raw={}; hashes={}; fallback=dict(hs=0,rician_hankel=0,em_smooth=0)
    require({p.name for p in (root/'raw').iterdir() if p.is_dir()}=={c['tag'] for c in cells},
            'unexpected/missing raw condition directories')
    for c in cells:
        data,diags=read_cell(root,c,manifest['source_fingerprint'],hashes);raw[c['tag']]=data
        counts=selection_checks(c,data,diags)
        for method,value in counts.items():fallback[method]+=value
    pairs=paired_world_checks(cells,raw)
    worlds=boundary_world_checks(cells,raw)
    replays=replay_checks(cells,raw) if replay else []
    analysis=summary_checks(root,cells,raw,hashes)
    return dict(passed=True,state='passed' if replay else 'integrity_checks_passed_replay_skipped',
                **scope,cells=len(cells),source_files_verified=count_sources,
                source_fingerprint=manifest['source_fingerprint'],
                audit_source_sha256=sha(__file__),manifest_sha256=sha(root/'manifest.json'),
                completion_sha256=sha(root/'COMPLETE.json'),raw_sha256=hashes,
                unique_condition_trial_pairs=scope['condition_trials'],duplicate_condition_trials=0,
                fallback_exact_error_checks=fallback,pairing=pairs,boundary_worlds=worlds,
                deterministic_replays=replays,analysis=analysis,seconds=time.perf_counter()-start,
                limitations=['Bitwise fallback check uses saved squared errors; estimator arrays are not stored.',
                             'World regeneration covers the first/last trial in every cell, not every channel.',
                             'Full estimator replay covers one predeclared trial in each of the three study types.',
                             'Condition-trials within a shared seed group are paired, not independent worlds.'])


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=DEFAULT)
    parser.add_argument('--skip-replay',action='store_true',help='report this limitation explicitly')
    args=parser.parse_args(); dest=args.root/'audit.json'
    try:
        result=run_audit(args.root,replay=not args.skip_replay)
    except Exception as exc:
        result=dict(passed=False,state='failed',error=f'{type(exc).__name__}: {exc}',
                    audit_source_sha256=sha(__file__))
        dest.parent.mkdir(parents=True,exist_ok=True)
        dest.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
        print(json.dumps(result));raise SystemExit(1) from exc
    dest.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:result[k] for k in ('passed','state','scope','cells','condition_trials',
                                           'independent_base_worlds','fallback_exact_error_checks','seconds')}))


if __name__=='__main__':
    main()
