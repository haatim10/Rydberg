"""Small, budgeted mechanism screen. Never launches learning or a larger study."""
from dataclasses import asdict, replace
import argparse
import fcntl
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback

import numpy as np

from .audit import ROOT, OLD, digest, read, verify_sources, audit_completed, report_text
from .core import Config, estimate, _estimate
from .data import FAMILIES, SNRS, world, nmse, noise_cell

OUT = ROOT/'results/paper2_information_feasibility'
LIMIT = 7200.
TUNE_N, SCREEN_N = 16, 64


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    os.replace(temp, path)


def progress(state, **values):
    write(OUT/'status.json', {'state': state, 'updated_unix': time.time(), **values})


def catalog():
    configs = {f'emgs_{n}': Config(iterations=n) for n in (1, 3, 5, 10, 20, 50, 100)}
    for n in (10, 50):
        for ridge in (.01, .1, 1.):
            configs[f'ridge_{n}_{ridge}'] = Config(iterations=n, ridge=ridge)
        configs[f'hard7_{n}'] = Config(method='hard', iterations=n)
    for lam in (.1, .3, 1., 3.):
        for alpha in (.05, .2, .5, 1.):
            configs[f'scalar_{lam}_{alpha}'] = Config(method='soft', lam=lam, alpha=alpha)
        for tau in (.03, .1, 1., 10.):
            configs[f'info_{lam}_{tau}'] = Config(method='information', lam=lam, tau=tau)
            configs[f'iso_{lam}_{tau}'] = Config(method='isotropic_information', lam=lam, tau=tau)
    return configs


def sources():
    paths = sorted((ROOT/'paper2_information').glob('*.py'))
    paths += [ROOT/'tests/test_paper2_information.py', ROOT/'reports/paper2_information_protocol.md']
    return {str(p.relative_to(ROOT)): digest(p) for p in paths}


def specification():
    return {'version': 1, 'source_hashes': sources(), 'families': FAMILIES, 'snrs': SNRS,
            'tune_per_cell': TUNE_N, 'screen_per_cell': SCREEN_N, 'pilots': 20,
            'worker_compute_limit_seconds': LIMIT,
            'catalog': {k: asdict(v) for k, v in catalog().items()},
            'selection': 'Noise-bin tuning, equal family weight, mean-linear NMSE per cell in dB; no family label at inference.',
            'screen': {'gain_db': .3, 'max_family_degradation_db': .2},
            'numpy': np.__version__, 'platform': platform.platform(),
            'old_manifest_sha256': digest(OLD/'manifest.json')}


class BudgetReached(RuntimeError):
    pass


class Budget:
    def __init__(self):
        self.path = OUT/'budget.json'
        self.prior = read(self.path)['seconds'] if self.path.exists() else 0.
        self.start = time.monotonic()
    def check(self):
        spent = self.prior+time.monotonic()-self.start
        write(self.path, {'seconds': spent, 'limit_seconds': LIMIT})
        if spent >= LIMIT:
            raise BudgetReached('Two-hour compute limit reached; partial records retained. No automatic extension.')


def outcome(w, cfg, oracle=False):
    if cfg.method == 'paper1_native':
        # Native reference, including its own initializer and held-out selector.
        # Not an otherwise-matched operation ablation: label this difference.
        from rydberg_sim.track_b_proposed import hs_gs_auto
        start = time.perf_counter()
        result = hs_gs_auto(w.S, w.Z, w.B, w.variance, max_iter=50, select_iter=25, cadzow_iter=4)
        h = result.G_hat
        return {'nmse': nmse(h, w.channel), 'diagnostics': {
            'seconds': time.perf_counter()-start, 'iterations': 50,
            'rank_selected': int(result.L_hat), 'oracle_only': False,
            'pilot_slots': w.S.shape[1], 'observations': w.Z.size,
            'initializer': 'native Paper 1 spectral', 'selection_iterations': 25,
            'internal_pilot_holdout': 'native 30 percent; counted within P, no extra observations'}}, h
    if oracle:
        h, diagnostics = _estimate(w.Z, w.S, w.B, w.variance, cfg, oracle_channel=w.channel)
    else:
        h, diagnostics = estimate(w.Z, w.S, w.B, w.variance, cfg)
    return {'nmse': nmse(h, w.channel), 'diagnostics': diagnostics}, h


def evaluate_record(split, family, snr, index, configs, budget):
    w = world(split, family, snr, index)
    path = OUT/'cases'/split/family/str(snr)/f'{index:03d}.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    expected = {k: asdict(v) for k, v in configs.items()}
    record = read(path) if path.exists() else {'case_id': w.case_id, 'world_sha256': w.digest,
                'pilot_id': w.pilot_id, 'configurations': expected, 'methods': {}}
    assert record['world_sha256'] == w.digest and record['configurations'] == expected
    for name, cfg in configs.items():
        if name in record['methods']:
            previous = record['methods'][name]
            predpath = path.parent/f'{index:03d}_{name}.npy'
            assert digest(predpath) == previous['prediction_sha256'], 'Saved prediction changed'
            np.testing.assert_allclose(nmse(np.load(predpath), w.channel), previous['nmse'], rtol=1e-12)
            continue
        budget.check()
        result, prediction = outcome(w, cfg)
        # Each method is committed atomically; a restart never needs a new world.
        record['methods'][name] = result
        predpath = path.parent/f'{index:03d}_{name}.npy'
        with predpath.with_suffix('.tmp').open('wb') as stream:
            np.save(stream, prediction)
        os.replace(predpath.with_suffix('.tmp'), predpath)
        record['methods'][name]['prediction_sha256'] = digest(predpath)
        write(path, record)
    return record


def choose(tune):
    configs = catalog()
    groups = {
        'classical': [k for k in configs if k.startswith(('emgs_', 'ridge_'))],
        'scalar': ['emgs_10']+[k for k in configs if k.startswith('scalar_')],
        'information': ['emgs_10']+[k for k in configs if k.startswith('info_')],
        'isotropic_information': ['emgs_10']+[k for k in configs if k.startswith('iso_')],
    }
    selected, scores = {}, {}
    for snr in SNRS:
        scores[str(snr)] = {}
        for name in configs:
            cells = [[r['methods'][name]['nmse'] for r in tune[(family, snr)]] for family in FAMILIES]
            scores[str(snr)][name] = float(np.mean([10*np.log10(np.mean(x)) for x in cells]))
        selected[str(snr)] = {group: min(names, key=scores[str(snr)].get) for group, names in groups.items()}
    return {'rules': selected, 'tune_scores_db': scores, 'source_hashes': sources(),
            'frozen_unix': time.time(), 'screen_not_accessed': True}


def screen_configs(w, selection):
    names = selection['rules'][str(noise_cell(w.S, w.variance))]
    cat = catalog()
    result = {label: cat[name] for label, name in names.items()}
    result.update({f'emgs_{n}': cat[f'emgs_{n}'] for n in (10, 50, 100)})
    result.update(hard7_10=cat['hard7_10'], hard7_50=cat['hard7_50'])
    result['post_scalar'] = replace(result['scalar'], postprocess=True)
    result['post_information'] = replace(result['information'], postprocess=True)
    result['paper1_native'] = Config(method='paper1_native', iterations=50)
    return result


def summarize(records):
    cells = {}
    for family in FAMILIES:
        for snr in SNRS:
            rows = records[(family, snr)]
            cell = {}
            for method in rows[0]['methods']:
                e = np.array([r['methods'][method]['nmse'] for r in rows])
                seconds = [r['methods'][method]['diagnostics']['seconds'] for r in rows]
                cell[method] = {'nmse_db': float(10*np.log10(e.mean())),
                                'mean_linear_nmse': float(e.mean()),
                                'severe_error_frequency': float(np.mean(e > 1)),
                                'mean_seconds': float(np.mean(seconds)), 'n_cases': len(e)}
            cells[f'{family}/{snr}'] = cell
    methods = next(iter(cells.values()))
    macro = {m: float(np.mean([c[m]['nmse_db'] for c in cells.values()])) for m in methods}
    gain = macro['scalar']-macro['information']
    degradations = {f: float(np.mean([cells[f'{f}/{s}']['information']['nmse_db']-cells[f'{f}/{s}']['scalar']['nmse_db'] for s in SNRS])) for f in FAMILIES}
    passed = gain >= .3 and max(degradations.values()) <= .2
    return {'cells': cells, 'macro_db': macro, 'screen_gain_over_scalar_db': gain,
            'family_degradation_vs_scalar_db': degradations,
            'worst_cell_degradation_vs_scalar_db': max(c['information']['nmse_db']-c['scalar']['nmse_db'] for c in cells.values()),
            'screen_passed': passed, 'decision': 'eligible_for_review_not_automatic_training' if passed else 'stop_expansion_of_A',
            'limitations': 'Development screening only. No independent confirmation, no learned comparator on these new worlds, no hardware claim, no verified novelty. Error thresholds are not publication standards.'}


def diagnostics(selection, budget):
    # Small, separately labelled mechanism checks, never used in selection.
    path = OUT/'diagnostics.json'
    saved = read(path) if path.exists() else {}
    for family in FAMILIES:
        for snr in (15, 80):
            for index in range(3):
                key = f'{family}/{snr}/{index}'
                if key in saved:
                    continue
                w = world('diagnostic', family, snr, index)
                cfg = screen_configs(w, selection)['information']
                # Force a nonzero reference mechanism for interpretability if tuning abstained.
                cfg = Config(method='information', lam=1., tau=1.) if cfg.method == 'none' else cfg
                row = {'world_sha256': w.digest, 'config': asdict(cfg), 'not_used_in_tuning': True}
                for label, oracle in (('estimated', False), ('oracle_only', True)):
                    budget.check()
                    row[label], _ = outcome(w, cfg, oracle=oracle)
                budget.check()
                poor, diag = _estimate(w.Z, w.S, w.B, w.variance, cfg, initial=np.zeros_like(w.channel))
                row['poor_zero_initialization'] = {'nmse': nmse(poor, w.channel), 'diagnostics': diag}
                saved[key] = row
                write(path, saved)
    # Change pilots while keeping exactly the same channel and nominal SNR.
    # Repeated pilots deliberately make the local information rank deficient.
    from .data import World
    for family in FAMILIES:
        base = world('diagnostic', family, 15, 21)
        for kind in ('new_pilot_A', 'new_pilot_B', 'repeated_pilot'):
            key = f'geometry/{family}/{kind}'
            if key in saved:
                continue
            S = world('diagnostic', family, 15, 22 if kind == 'new_pilot_A' else 23).S
            if kind == 'repeated_pilot':
                S = np.repeat(S[:, :1], S.shape[1], axis=1)
            S = S*np.linalg.norm(base.S)/np.linalg.norm(S)
            rng = np.random.default_rng(2026092198)
            noise = np.sqrt(base.variance/2)*(rng.normal(size=base.B.shape)+1j*rng.normal(size=base.B.shape))
            Z = np.abs(base.channel@S+base.B+noise)
            w = World(base.channel, Z, S, base.B, base.variance, key, base.digest, kind)
            cfg = Config(method='information', lam=1., tau=1., ridge=1e-6)
            row = {'same_channel_source': base.case_id, 'same_noise_seed': 2026092198,
                   'pilot_energy_matched_to_base': True,
                   'config': asdict(cfg), 'diagnostic_only': True, 'matched_nominal_snr_db': 15}
            budget.check()
            row['information'], _ = outcome(w, cfg)
            budget.check()
            row['scalar'], _ = outcome(w, replace(cfg, method='soft', alpha=.2))
            saved[key] = row
            write(path, saved)
    return saved


def run_batch(budget):
    configs = catalog()
    selection_path = OUT/'selection.json'
    if not selection_path.exists():
        tune = {}
        for family in FAMILIES:
            for snr in SNRS:
                tune[(family, snr)] = []
                for index in range(TUNE_N):
                    progress('tuning', family=family, snr=snr, case=index, cases_per_cell=TUNE_N)
                    tune[(family, snr)].append(evaluate_record('tune', family, snr, index, configs, budget))
        write(selection_path, choose(tune))
    selection = read(selection_path)
    assert selection['source_hashes'] == sources()
    records = {}
    for family in FAMILIES:
        for snr in SNRS:
            records[(family, snr)] = []
            for index in range(SCREEN_N):
                progress('screening', family=family, snr=snr, case=index, cases_per_cell=SCREEN_N)
                w = world('screen', family, snr, index)
                records[(family, snr)].append(evaluate_record('screen', family, snr, index, screen_configs(w, selection), budget))
    summary = summarize(records)
    write(OUT/'summary.json', summary)
    diagnostics(selection, budget)
    budget.check()
    report = [
        '# Measurement-informed correction: development screen', '',
        f"Screen decision: **{summary['decision']}**.", '',
        f"Information filtering versus the validation-tuned scalar correction: {summary['screen_gain_over_scalar_db']:.3f} dB macro-average gain (positive is better).",
        f"Family-average degradations (positive is worse): {summary['family_degradation_vs_scalar_db']}.", '',
        'The tuning worlds and screening worlds are separate. The settings were selected using only tuning outcomes. No true channel or family label enters the estimator.', '',
        'See summary.json for every cell, error tails and time. See diagnostics.json for oracle-only and poor-initialization checks. These diagnostics are not deployable estimator results.', '',
        summary['limitations'], '',
        'No larger run is launched by this program. A pass requires scientific review of controls and remaining tests. A failure stops expansion of this mechanism. The cancelled recalibration project is not involved.',
    ]
    (OUT/'RESULTS_REPORT.md').write_text('\n'.join(report)+'\n')
    progress('complete', screen_passed=summary['screen_passed'], scientific_review_pending=True)


def dependency_ready():
    status = read(OLD/'status.json')
    if status['state'] in ('failed', 'budget_exhausted'):
        raise RuntimeError(f"Existing study requires attention: {status}")
    with (OLD/'run.lock').open('r') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
            running = False
        except BlockingIOError:
            running = True
    if status['state'] == 'complete':
        return not running
    if not running:
        raise RuntimeError('Existing study stopped before completion; not restarting or bypassing it.')
    return False


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT/'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert read(OUT/'manifest.json') == json.loads(json.dumps(specification())), 'Frozen feasibility source/configuration changed'
        write(OUT/'pid.json', {'pid': os.getpid(), 'started_unix': time.time()})
        try:
            while not dependency_ready():
                progress('waiting_for_existing_study', dependency=str(OLD), compute_budget_started=False)
                time.sleep(30)
            budget = Budget()
            budget.check()
            verify_sources()
            progress('auditing_existing_study')
            if not (OUT/'legacy_audit.json').exists():
                write(OUT/'legacy_audit.json', audit_completed())
            assert read(OUT/'legacy_audit.json')['passed']
            assert read(OUT/'legacy_audit.json')['legacy_summary_sha256'] == digest(OLD/'summary.json')
            (OUT/'LEGACY_AUDIT_REPORT.md').write_text(report_text(read(OUT/'legacy_audit.json')))
            budget.check()
            run_batch(budget)
        except BudgetReached as exc:
            progress('budget_exhausted', error=str(exc), partial_results_retained=True)
        except Exception:
            progress('blocked', error=traceback.format_exc())
            raise


def launch():
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT/'launch.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        with (OUT/'run.lock').open('a') as worker:
            try:
                fcntl.flock(worker, fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:
                print('Already running; no duplicate launched.'); return
        if (OUT/'status.json').exists() and read(OUT/'status.json')['state'] in ('complete', 'budget_exhausted', 'blocked'):
            print(json.dumps(read(OUT/'status.json'))); return
        preflight = read(OUT/'preflight.json')
        assert preflight['passed'] and preflight['source_hashes'] == sources(), 'Run preflight on these exact sources first'
        path = OUT/'manifest.json'
        expected = json.loads(json.dumps(specification()))
        if path.exists():
            assert read(path) == expected
        else:
            write(path, expected)
        env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1')
        with (OUT/'worker.log').open('ab') as log:
            p = subprocess.Popen([sys.executable, '-m', 'paper2_information.experiment', 'run'], cwd=ROOT,
                                 stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
        print(json.dumps({'pid': p.pid, 'output': str(OUT), 'dependency': str(OLD)}))


def preflight():
    OUT.mkdir(parents=True, exist_ok=True)
    verify_sources()
    begin = time.monotonic()
    check = subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', 'tests', '-p', 'test_paper2_information.py', '-v'], cwd=ROOT, capture_output=True, text=True)
    (OUT/'preflight_tests.log').write_text(check.stdout+check.stderr)
    if check.returncode:
        raise RuntimeError(check.stdout+check.stderr)
    # Only diagnostic data, never tune/screen outcomes; quote measured time.
    w = world('diagnostic', 'independent', 15, 987)
    start = time.perf_counter()
    for cfg in catalog().values():
        estimate(w.Z, w.S, w.B, w.variance, cfg)
    timing = time.perf_counter()-start
    value = {'passed': True, 'source_hashes': sources(), 'seconds': time.monotonic()-begin,
             'diagnostic_catalog_seconds': timing, 'catalog_size': len(catalog()),
             'projected_tune_seconds': timing*TUNE_N*len(FAMILIES)*len(SNRS),
             'gpu_agreement': 'not established on unavailable CUDA; no GPU training authorized'}
    write(OUT/'preflight.json', value)
    print(json.dumps(value, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('preflight', 'launch', 'run', 'status'))
    args = parser.parse_args()
    if args.command == 'status':
        print(json.dumps(read(OUT/'status.json'), indent=2))
    else:
        globals()[args.command]()


if __name__ == '__main__':
    main()
