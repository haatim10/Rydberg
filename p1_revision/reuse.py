"""Provenanced reuse of classical results and EM-only replay for Paper 1.

No rank search is run here. Existing experiment files are read-only inputs.
Run from repository root: python -m p1_revision.reuse {rsr,preflight,proxy,all}
"""
import os
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
              'VECLIB_MAXIMUM_THREADS'):
    os.environ[_name] = '1'

import argparse
import csv
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from scipy.stats import spearmanr, rankdata

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / 'results/p1_revision/reused'
P1 = REPO / 'codex_verification/results'
P2 = REPO / 'results/sahure_expansion'
SEED = 20261007
N_BOOT = 4000


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def write_csv(path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def load_npz(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key].copy() for key in data.files}


def paired_summary(em, hs, energy, rng, n_boot=N_BOOT):
    em, hs, energy = map(np.asarray, (em, hs, energy))
    if em.shape != hs.shape or em.shape != energy.shape or em.ndim != 1:
        raise ValueError('paired one-dimensional arrays required')
    if not all(np.isfinite(a).all() and (a > 0).all() for a in (em, hs, energy)):
        raise ValueError('positive finite errors and energies required')
    indices = rng.integers(len(em), size=(n_boot, len(em)))
    em_sum, hs_sum, den_sum = (a[indices].sum(1) for a in (em, hs, energy))
    values = dict(em_nmse_db=10*np.log10(em.sum()/energy.sum()),
                  hs_nmse_db=10*np.log10(hs.sum()/energy.sum()),
                  ratio_gain_db=10*np.log10(em.sum()/hs.sum()),
                  median_gain_db=np.median(10*np.log10(em/hs)))
    samples = dict(em_nmse_db=10*np.log10(em_sum/den_sum),
                   hs_nmse_db=10*np.log10(hs_sum/den_sum),
                   ratio_gain_db=10*np.log10(em_sum/hs_sum),
                   median_gain_db=np.median(10*np.log10(em[indices]/hs[indices]), axis=1))
    out = {'n': len(em)}
    for key, value in values.items():
        lo, hi = np.quantile(samples[key], [.025, .975])
        out.update({key: float(value), key+'_lo': float(lo), key+'_hi': float(hi)})
    return out


def reused_rsr():
    frozen_path = P2/'frozen_selection.json'
    frozen = json.loads(frozen_path.read_text())
    audit_path = P2/'audit.json'
    if json.loads(audit_path.read_text())['state'] != 'passed':
        raise RuntimeError('source campaign audit has not passed')
    expected = {'N': 32, 'K': 3, 'P': 30, 'snr': 5.0, 'L': None, 'channel': 'ula'}
    rng = np.random.default_rng(SEED)
    records, sources = [], {}
    for axis in ('rsr', 'noise_offset'):
        for point in frozen['axes'][axis]:
            cell = point['cell']
            spec = next(s for s in frozen['cells'] if s['id'] == cell)
            for key, value in expected.items():
                if spec[key] != value:
                    raise RuntimeError(f'{cell}: incompatible {key}')
            directory = P2/'cells'/cell
            a_path, b_path = directory/'em_100.npz', directory/'hs_gs.npz'
            a, b = load_npz(a_path), load_npz(b_path)
            np.testing.assert_array_equal(a['energy'], b['energy'])
            if len(a['error']) != 400:
                raise RuntimeError('incomplete source cell')
            for path in (a_path, b_path, directory/'classical.jsonl', directory/'status.json'):
                sources[str(path.relative_to(REPO))] = sha(path)
            # Verify the stored HS settings and declared final estimator implementation.
            rows = [json.loads(s) for s in (directory/'classical.jsonl').read_text().splitlines()]
            ranks = [r['info']['rank'] for r in rows if r['method'] == 'hs_gs']
            if len(ranks) != 400 or any(not 1 <= r <= 16 for r in ranks):
                raise RuntimeError('invalid rank records')
            summary = paired_summary(a['error'], b['error'], a['energy'], rng)
            records.append(dict(axis=axis, value=point['value'], cell=cell,
                **summary, fallback_fraction=float(np.mean(np.asarray(ranks) == 16))))
    declared_sources = frozen['sources']
    for filename in ('sahure/classical.py', 'codex_verification/solver.py',
                     'sahure_expansion/data.py', 'sahure_expansion/protocol.py'):
        if sha(REPO/filename) != declared_sources[filename]:
            raise RuntimeError(f'source changed since freeze: {filename}')
        sources[filename] = declared_sources[filename]
    result = dict(source_campaign='sahure_expansion; distinct from the original 27300-trial P1 verification',
        source_frozen_selection_sha256=sha(frozen_path), source_audit_sha256=sha(audit_path),
        reanalysis_source_sha256=sha(__file__), source_files=sources,
        settings=dict(**expected, paths='independent per-user uniform integers 3..7',
            em_updates=100, hs_updates=100, candidate_updates=25, cadzow_sweeps=4,
            validation_fraction=.3, selection='shared scalar rank; held-out amplitude residual',
            final_refit='all 30 pilots', snr_definition='aggregate signal / total complex noise variance',
            rsr_definition='reference / nominal single-user signal power'),
        uncertainty=dict(resamples=N_BOOT, seed=SEED, unit='paired independent world',
            interval='pointwise percentile 95%; no simultaneous or between-condition guarantee'),
        scope='7 RSR conditions, 400 worlds each; noise-offset conditions reuse the RSR12 worlds. '
              'Only classical EM-GS/HS-GS errors are used; no learned predictions. '
              'Noise offsets alter supplied variance, not generated noise covariance.', records=records)
    write_json(OUT/'rsr_and_noise_summary.json', result)
    write_csv(OUT/'rsr_and_noise_summary.csv', records)
    plot_rsr(records)
    return result


def pyplot():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 10, 'axes.grid': True, 'grid.alpha': .22,
                         'pdf.fonttype': 42, 'ps.fonttype': 42})
    return plt


def save_figure(fig, stem):
    directory = OUT/'figures'
    directory.mkdir(parents=True, exist_ok=True)
    for extension in ('pdf', 'png'):
        fig.savefig(directory/f'{stem}.{extension}', dpi=180, bbox_inches='tight')


def plot_rsr(records):
    plt = pyplot()
    for axis, label, stem in [('rsr', 'Single-user RSR (dB)', 'rsr_classical'),
                              ('noise_offset', 'Assumed noise-variance offset (dB)', 'noise_variance_classical')]:
        rows = [r for r in records if r['axis'] == axis]
        x = np.asarray([r['value'] for r in rows])
        fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.8), layout='constrained')
        for key, name, color, marker in [('em_nmse_db', 'EM-GS', '#2875b7', 's'),
                                        ('hs_nmse_db', 'HS-GS', '#c84b32', 'o')]:
            y = np.asarray([r[key] for r in rows])
            lo, hi = (np.asarray([r[key+s] for r in rows]) for s in ('_lo', '_hi'))
            axes[0].plot(x, y, marker=marker, color=color, label=name)
            axes[0].fill_between(x, lo, hi, color=color, alpha=.14)
        for key, name, color, marker in [('ratio_gain_db', 'Ratio-of-sums gain', '#64449c', 'o'),
                                        ('median_gain_db', 'Paired-median gain', '#248568', '^')]:
            y = np.asarray([r[key] for r in rows])
            lo, hi = (np.asarray([r[key+s] for r in rows]) for s in ('_lo', '_hi'))
            axes[1].errorbar(x, y, yerr=[y-lo, hi-y], marker=marker, color=color,
                             label=name, capsize=3)
        axes[0].set_ylabel('Ratio-of-sums NMSE (dB)')
        axes[1].set_ylabel('HS-GS improvement over EM-GS (dB)')
        axes[1].axhline(0, color='black', linewidth=.7)
        for ax in axes:
            ax.set_xlabel(label); ax.set_xticks(x); ax.legend(fontsize=8)
        fig.suptitle('Classical estimators: N = 32, K = 3, P = 30, SNR = 5 dB\n'
                     '400 paired worlds per condition; pointwise 95% intervals', fontsize=11)
        save_figure(fig, stem); plt.close(fig)


def selected_cells(scope='all'):
    manifest = json.loads((P1/'manifest.json').read_text())
    arrays = [c for c in manifest['cells'] if c['tag'].startswith('array_') and c['P'] == 30]
    matched = [c for c in manifest['cells'] if c['tag'].startswith('matchedrho_N32_')]
    return (arrays+matched) if scope == 'all' else matched


def replay_sources():
    from codex_verification.run import source_hash
    original = json.loads((P1/'manifest.json').read_text())
    current = source_hash()
    if current != original['source_fingerprint']:
        raise RuntimeError('P1 estimator/generator source fingerprint changed')
    paths = [REPO/'codex_verification'/name for name in ('run.py', 'worlds.py', 'solver.py')]
    paths += list((REPO/'rydberg_sim').glob('*.py'))
    return dict(p1_source_fingerprint=current, p1_manifest_sha256=sha(P1/'manifest.json'),
                current_sources={str(p.relative_to(REPO)): sha(p) for p in paths},
                replay_source_sha256=sha(__file__))


def replay_one(cell, local_index, saved):
    from codex_verification.run import world, trial_snr
    from codex_verification.solver import fit, effective_ranks
    w = world(cell, local_index)
    estimate = fit(w.S, w.Z, w.B, w.sigma2, iterations=cell['iterations'])[0]
    error = float(np.sum(abs(estimate-w.G)**2))
    expected = float(saved['num_em'][local_index])
    np.testing.assert_allclose(error, expected, rtol=2e-11, atol=2e-12)
    cap = (cell['N']+1)//2
    rho_true = effective_ranks(w.G)/cap
    np.testing.assert_allclose(rho_true, saved['rho_columns'][local_index], rtol=1e-12, atol=1e-12)
    energy = float(np.sum(abs(w.G)**2))
    np.testing.assert_allclose(energy, saved['den'][local_index], rtol=1e-13, atol=1e-13)
    if int(saved['trial'][local_index]) != cell['seed_base']+local_index:
        raise RuntimeError('trial order differs from original manifest')
    return dict(g_em=estimate, rho_em_columns=effective_ranks(estimate)/cap,
                rho_true_columns=rho_true, paths=np.asarray(w.L_k, int),
                trial=int(saved['trial'][local_index]), snr=trial_snr(cell, local_index),
                num_em=expected, num_hs=float(saved['num_hs'][local_index]), den=energy,
                replay_error_abs=abs(error-expected),
                replay_error_relative=abs(error-expected)/max(expected, 1e-300),
                selected_rank=int(saved['rank'][local_index]))


def preflight():
    sources = replay_sources()
    cells = selected_cells()
    probes = [next(c for c in cells if c['N'] == n and c['P'] == 30 and c['snr'] == 5) for n in (8,16,32)]
    probes += [next(c for c in cells if c['tag'] == 'matchedrho_N32_L8')]
    rows = []
    for cell in probes:
        saved = load_npz(P1/'trials'/f'{cell["tag"]}.npz')
        durations, errors = [], []
        for index in (0,1,2):
            tick = time.perf_counter(); result = replay_one(cell, index, saved)
            durations.append(time.perf_counter()-tick); errors.append(result['replay_error_abs'])
        rows.append(dict(tag=cell['tag'], N=cell['N'], P=cell['P'], seconds=durations,
                         median_seconds=float(np.median(durations)), max_error=max(errors)))
    estimate = sum(c['n']*next(r['median_seconds'] for r in rows if r['N'] == c['N'] and r['P'] == c['P']) for c in cells)
    result = dict(sources=sources, passed=True, probes=rows, cells=len(cells), trials=sum(c['n'] for c in cells),
                  estimated_replay_seconds_single_worker=estimate,
                  scope='all' if estimate <= 900 else 'matched',
                  note='Three replay trials per size/design; estimates exclude plotting/summary overhead. No HS rank search.')
    write_json(OUT/'proxy_preflight.json', result)
    print(json.dumps(result), flush=True)
    return result


def replay_proxy(scope='auto'):
    check = json.loads((OUT/'proxy_preflight.json').read_text())
    if not check['passed']:
        raise RuntimeError('preflight did not pass')
    scope = check['scope'] if scope == 'auto' else scope
    sources = replay_sources(); cells = selected_cells(scope)
    if sources['p1_source_fingerprint'] != check['sources']['p1_source_fingerprint']:
        raise RuntimeError('source changed after preflight')
    directory = OUT/'proxy'; directory.mkdir(parents=True, exist_ok=True)
    run = dict(scope=scope, sources=sources, cells=cells, expected_trials=sum(c['n'] for c in cells),
               initializer='P1 spectral initializer on all pilots', em_updates='original cell.iterations',
               estimator_inputs='Z,S,B,sigma2 only; truth used for evaluation and replay checks',
               definition='per-user effective rank of EM estimate divided by balanced Hankel capacity',
               family_master_seeds={'TB': 20250820, 'TD': 20260827},
               snr_seed='random-SNR cells use SeedSequence([cell.seed_base, local_index,0x534e52])',
               resumed=False)
    write_json(OUT/'proxy_manifest.json', run)
    start = time.perf_counter()
    for cell in cells:
        input_path = P1/'trials'/f'{cell["tag"]}.npz'
        path = directory/f'{cell["tag"]}.npz'
        metadata = path.with_suffix('.json')
        expected = dict(cell=cell, original_trial_file_sha256=sha(input_path),
                        original_source_fingerprint=sources['p1_source_fingerprint'])
        if path.exists() and metadata.exists():
            prior = json.loads(metadata.read_text())
            if all(prior[k] == v for k,v in expected.items()) and prior['output_sha256'] == sha(path):
                print(f'Replay cached: {cell["tag"]}', flush=True); continue
            raise RuntimeError(f'cached replay mismatch: {path}')
        saved = load_npz(input_path); rows = []; tick = time.perf_counter()
        for index in range(cell['n']):
            rows.append(replay_one(cell, index, saved))
        arrays = {key: np.asarray([row[key] for row in rows]) for key in rows[0]}
        arrays['rho_em_mean'] = arrays['rho_em_columns'].mean(1)
        arrays['rho_true_mean'] = arrays['rho_true_columns'].mean(1)
        arrays['L_normalized_mean'] = arrays['paths'].mean(1)/((cell['N']+1)//2)
        temp = path.with_suffix('.tmp.npz'); np.savez_compressed(temp, **arrays); temp.replace(path)
        metadata_values = dict(**expected, replay_source_sha256=sha(__file__), output_sha256=sha(path),
            trials=len(rows), seconds=time.perf_counter()-tick,
            max_replay_error=float(arrays['replay_error_abs'].max()),
            max_replay_relative_error=float(arrays['replay_error_relative'].max()))
        write_json(metadata, metadata_values)
        print(f'Replay complete: {cell["tag"]}; {len(rows)} worlds; {metadata_values["seconds"]:.1f}s', flush=True)
    write_json(OUT/'proxy_status.json', dict(state='complete', scope=scope, trials=run['expected_trials'],
                                           cells=len(cells), wall_seconds=time.perf_counter()-start))
    return summarize_proxy(scope)


def correlation(x,y):
    x, y = map(np.asarray, (x,y))
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return None
    return float(spearmanr(x,y).statistic)


def correlation_intervals(variables, pairs, rng, n_boot=N_BOOT):
    """Resample complete trials and recompute tie-aware ranks in every draw.

    The scalar inputs already average users within a trial. The same bootstrap
    index array is shared across all correlations in a stratum. Constant
    diagnostics have undefined correlation and receive no numeric interval.
    """
    n = len(next(iter(variables.values())))
    if any(len(v) != n for v in variables.values()):
        raise ValueError('paired trial arrays required')
    constant = {key: np.ptp(value) == 0 for key,value in variables.items()}
    samples = {key: [] for key in pairs}
    for start in range(0, n_boot, 250):
        size = min(250, n_boot-start)
        indices = rng.integers(n, size=(size,n))
        centered = {}
        sums = {}
        for key, values in variables.items():
            if constant[key]:
                continue
            ranks = rankdata(np.asarray(values)[indices], axis=1, method='average')
            centered[key] = ranks-ranks.mean(1,keepdims=True)
            sums[key] = np.sum(centered[key]**2,axis=1)
        for name,(a,b) in pairs.items():
            if constant[a] or constant[b]:
                continue
            denominator = np.sqrt(sums[a]*sums[b])
            valid = denominator > 0
            values = np.sum(centered[a]*centered[b],axis=1)[valid]/denominator[valid]
            samples[name].append(np.clip(values,-1,1))
    out={}
    for name, draws in samples.items():
        values = np.concatenate(draws) if draws else np.empty(0)
        lo,hi = np.quantile(values,[.025,.975]) if len(values) else (None,None)
        out.update({name+'_lo': None if lo is None else float(lo),
                    name+'_hi': None if hi is None else float(hi),
                    name+'_valid_resamples': len(values)})
    return out


def proxy_stats(tag, data, rng, **settings):
    gain = 10*np.log10(data['num_em']/data['num_hs'])
    em, true, paths = (data[k] for k in ('rho_em_mean', 'rho_true_mean', 'L_normalized_mean'))
    result = dict(stratum=tag, n=len(gain), **settings,
        rho_proxy_mean=float(em.mean()), rho_true_mean=float(true.mean()),
        rho_bias=float(np.mean(em-true)), rho_mae=float(np.mean(abs(em-true))),
        median_gain_db=float(np.median(gain)),
        proxy_truth_spearman=correlation(em,true), proxy_gain_spearman=correlation(em,gain),
        truth_gain_spearman=correlation(true,gain), normalized_paths_gain_spearman=correlation(paths,gain),
        max_replay_absolute_error=float(data['replay_error_abs'].max()))
    result.update(correlation_intervals(dict(proxy=em,true=true,paths=paths,gain=gain),
        dict(proxy_truth_spearman=('proxy','true'), proxy_gain_spearman=('proxy','gain'),
             truth_gain_spearman=('true','gain'), normalized_paths_gain_spearman=('paths','gain')), rng))
    return result


def summarize_proxy(scope='all'):
    cells = selected_cells(scope); rows=[]; matched=[]
    rng = np.random.default_rng(SEED)
    for cell in cells:
        data = load_npz(OUT/'proxy'/f'{cell["tag"]}.npz')
        rows.append(proxy_stats(cell['tag'], data, rng, N=cell['N'], P=cell['P'],
            snr_low=cell['snr'], snr_high=cell['snr'], L=cell['L'],
            grouping='fixed_N_SNR' if cell['snr'] is not None else 'fixed_L_random_SNR'))
        if cell['tag'].startswith('matchedrho'):
            matched.append(data)
    joined = {key: np.concatenate([d[key] for d in matched]) for key in matched[0]}
    # Fixed bins chosen before replay; pool complexity within each SNR band.
    for lo, hi in zip((-10,-5,0,5,10,15), (-5,0,5,10,15,20)):
        mask = (joined['snr'] >= lo) & ((joined['snr'] < hi) if hi < 20 else (joined['snr'] <= hi))
        subset = {key: value[mask] for key,value in joined.items()}
        rows.append(proxy_stats(f'matched_snr_{lo}_{hi}', subset, rng, N=32, P=20,
            snr_low=lo, snr_high=hi, L=None, grouping='SNR_band_across_complexities'))
    summary = dict(scope=scope, definitions='Trial means of per-user normalized effective rank; Spearman correlation with per-trial dB gain.',
        limitations=['Descriptive correlations only, not a fitted or validated decision rule.',
            'True channels and path counts are evaluation-only diagnostics, not receiver features.',
            'Unstructured EM estimates are used; post-HS rank is intentionally excluded.',
            'Fixed-L pooled-random-SNR correlations can be driven by SNR; use the separate SNR-band rows.',
            'Five-dB SNR bands only partially control SNR; no causal or universal-threshold claim.',
            'Pointwise percentile intervals do not give simultaneous or causal guarantees.'],
        uncertainty=dict(resamples=N_BOOT,seed=SEED,unit='entire trial; users averaged before resampling',
            method='tie-aware Spearman ranks recomputed in each resample; paired indices shared within stratum',
            interval='pointwise percentile 95%',
            constant_diagnostics='undefined correlation and interval, recorded as null',
            timing='analysis-only resampling specified after point estimates were inspected; no predictor fitting'),
        analysis_source_sha256=sha(__file__),
        full_target_trials=10400, completed_trials=sum(c['n'] for c in cells), strata=rows)
    write_json(OUT/'proxy_summary.json', summary); write_csv(OUT/'proxy_summary.csv', rows)
    plot_proxy(rows)
    return summary


def plot_proxy(rows):
    plt = pyplot(); fig, axes = plt.subplots(1,2,figsize=(10.2,3.8),layout='constrained')
    main = [r for r in rows if r['grouping'] == 'fixed_N_SNR']
    for n, marker in zip((8,16,32), ('s','^','o')):
        rr = sorted([r for r in main if r['N'] == n], key=lambda r:r['snr_low'])
        if rr:
            x = np.asarray([r['snr_low'] for r in rr]); key='proxy_truth_spearman'
            y = np.asarray([r[key] for r in rr])
            lo,hi=(np.asarray([r[key+s] for r in rr]) for s in ('_lo','_hi'))
            axes[0].errorbar(x,y,yerr=[y-lo,hi-y],marker=marker,capsize=2,label=f'N = {n}')
    axes[0].set(xlabel='SNR (dB)', ylabel='Spearman correlation: EM proxy and true rho',
                title='(a) Fixed SNR, P = 30, random 3–7 paths')
    axes[0].set_ylim(-1,1)
    if main: axes[0].legend(fontsize=8)
    bands = [r for r in rows if r['grouping'] == 'SNR_band_across_complexities']
    centers = [(r['snr_low']+r['snr_high'])/2 for r in bands]
    for key,label,marker in [('proxy_gain_spearman','EM-based rho','o'),
                              ('truth_gain_spearman','True rho (diagnostic)','s'),
                              ('normalized_paths_gain_spearman','True L / capacity (diagnostic)','^')]:
        y=np.asarray([r[key] for r in bands])
        lo,hi=(np.asarray([r[key+s] for r in bands]) for s in ('_lo','_hi'))
        axes[1].errorbar(centers,y,yerr=[y-lo,hi-y],marker=marker,capsize=2,label=label)
    axes[1].set(xlabel='SNR band centre (dB; band width 5 dB)', ylabel='Spearman correlation with HS-GS gain',
                title='(b) N = 32, P = 20, L = 2, 4, ..., 16')
    axes[1].set_ylim(-1,1); axes[1].legend(fontsize=7.6)
    for ax in axes: ax.axhline(0,color='black',linewidth=.7)
    fig.suptitle('Practical effective-rank proxy from unstructured EM-GS\n'
                 'Pointwise 95% trial-bootstrap intervals; no fitted decision rule', fontsize=11)
    save_figure(fig,'rho_proxy_diagnostics'); plt.close(fig)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=('rsr','preflight','proxy','summary','all'))
    parser.add_argument('--scope',choices=('auto','all','matched'),default='auto')
    args=parser.parse_args()
    if args.phase in ('rsr','all'): reused_rsr()
    if args.phase in ('preflight','all'): preflight()
    if args.phase in ('proxy','all'): replay_proxy(args.scope)
    if args.phase == 'summary': summarize_proxy('all' if args.scope == 'auto' else args.scope)


if __name__ == '__main__':
    main()
