"""A resumable, validation-only fair-comparison runner. No training or test tuning."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[key] = '1'
import argparse
import fcntl
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import platform
import time
import traceback
import unittest
import numpy as np
import torch
from sahure.data import load_observations, split_observations
from sahure.train import load_model
from sahure_expansion.data import world, condition_grid
from paper2_optshrink_loop.run import write, save, load
from .estimator import PARAMS, BUDGETS, tensors, estimate, development_scores, hard_one, lookup_key

ROOT = Path('results/fair_optshrink')
OLD = Path('results/paper2_optshrink_loop')
DATA = Path('results/sahure_expansion/data')
SEEDS = (9101, 9102, 9103)
SPECS = [dict(id=f'cell_{j:03d}', **spec) for j, spec in enumerate(condition_grid())]
SPECS += [dict(id='weak_reference', snr=5., rsr=0., L=None),
          dict(id='nearly_noiseless', snr=60., rsr=12., L=None)]
CONFIG = dict(version=1, test_banks=[202610210001, 202610310001], cases_per_cell=16,
    validation_count=2048, validation_mask_seed=2026092203, benchmark_seed=202610123001,
    timing_seed=202610410001, timing_repeats=3, bootstrap_seed=202610127001,
    bootstrap_samples=4000, seeds=list(SEEDS), budgets=list(BUDGETS), params=PARAMS,
    specs=SPECS, max_seconds=7200, validation_criterion='equal per-stratum held-out Rician NLL',
    primary_criterion='equal average of126 per-cell dB ratio-of-sums NMSEs')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sources():
    paths = sorted(Path('fair_optshrink').glob('*.py')) + [Path('fair_optshrink/PROTOCOL.txt'),
        Path('hybrid_optshrink/model.py'), OLD / 'frozen_selection.json', OLD / 'manifest.json',
        DATA / 'val_observations.npz', DATA / 'val_conditions.npy', DATA / 'val_generation.json']
    original = json.loads((OLD / 'manifest.json').read_text())
    for path, digest in original['sources'].items():
        if sha(path) != digest:
            raise RuntimeError('historical source changed: ' + path)
    for entry in original['models'].values():
        if sha(entry['path']) != entry['sha256']:
            raise RuntimeError('historical model changed: ' + entry['path'])
    return {**original['sources'], **{str(p): sha(p) for p in paths}}


def initialize():
    original = json.loads((OLD / 'manifest.json').read_text())
    record = dict(config=CONFIG, sources=sources(), models=original['models'],
        environment=dict(python=platform.python_version(), numpy=np.__version__,
            torch=torch.__version__, system=platform.platform(), threads=1, dtype='float64'))
    path = ROOT / 'manifest.json'
    if path.exists() and json.loads(path.read_text()) != record:
        raise RuntimeError('frozen configuration/source/environment changed')
    if not path.exists():
        write(path, record)
    return record


def models():
    entries = json.loads((OLD / 'manifest.json').read_text())['models']
    out = {}
    for seed in SEEDS:
        entry = entries[str(seed)]
        if sha(entry['path']) != entry['sha256']:
            raise RuntimeError('checkpoint hash changed')
        model, _ = load_model(entry['path'])
        if sum(p.numel() for p in model.parameters()) != 188:
            raise RuntimeError('not the original188-parameter architecture')
        out[f'learned_{seed}'] = model
    return out


def validation_data():
    obs = load_observations(DATA / 'val_observations.npz', dtype=torch.float64)
    ids = np.load(DATA / 'val_conditions.npy', allow_pickle=False)
    if len(obs) != CONFIG['validation_count'] or set(ids) != set(range(126)):
        raise RuntimeError('unexpected original validation population')
    a, b = split_observations(obs, CONFIG['validation_mask_seed'])
    return a, b, ids


def config_at(index):
    placement, t, p = index
    return dict(**PARAMS[p], iterations=BUDGETS[t], placement=('loop', 'final')[placement])


def choose(scores, placement=None, budget_index=None):
    options = [(p, t, r) for p in range(2) for t in range(len(BUDGETS)) for r in range(len(PARAMS))
               if (placement is None or p == placement) and (budget_index is None or t == budget_index)]
    # Deterministic ties: loop, then fewer iterations, then no-filter/lower rank.
    return config_at(min(options, key=lambda idx: float(scores[idx])))


def select_settings(scores, ids):
    # Aggregation is balanced across the six regimes; none enters the online key.
    strata = np.stack([scores[ids == j].mean(0) for j in range(126)])
    choices = {}
    for snr in range(-10, 21, 5):
        for rsr in (5, 12, 20):
            js = [j for j, spec in enumerate(SPECS[:126]) if spec['snr'] == snr and spec['rsr'] == rsr]
            mean = strata[js].mean(0)
            choices[f'{snr}:{rsr}'] = dict(selected=choose(mean), loop=choose(mean, 0),
                final=choose(mean, 1), fixed={str(t): choose(mean, budget_index=i) for i, t in enumerate(BUDGETS)})
    return choices, strata


def tune(check):
    frozen_path = ROOT / 'frozen_selection.json'
    if frozen_path.exists():
        frozen = json.loads(frozen_path.read_text())
        if frozen['manifest_sha256'] != sha(ROOT / 'manifest.json'):
            raise RuntimeError('frozen manifest mismatch')
        for path, digest in frozen['development_hashes'].items():
            if sha(path) != digest:
                raise RuntimeError('development result changed')
        return frozen
    if (ROOT / 'data').exists():
        raise RuntimeError('test data exists before selection freeze')
    a, b, ids = validation_data()
    rows, hashes = [], {}
    for i in range(len(a)):
        check('tuning', done=i, total=len(a), message='Original validation observations; no channel labels')
        path = ROOT / 'development' / f'case_{i:04d}.npz'
        if not path.exists():
            tick = time.perf_counter()
            scores = development_scores(a.batch(slice(i, i + 1)), b.batch(slice(i, i + 1)),
                progress=lambda t: check('tuning', done=i, total=len(a), update=t))
            save(path, scores=scores, case=i, stratum=ids[i], manifest_sha256=sha(ROOT / 'manifest.json'),
                 seconds=time.perf_counter() - tick)
        row = load(path)
        if int(row['case']) != i or int(row['stratum']) != ids[i] or str(row['manifest_sha256']) != sha(ROOT / 'manifest.json'):
            raise RuntimeError('development cache identity mismatch')
        if not np.isfinite(row['scores']).all():
            raise FloatingPointError('cached candidate loss not finite')
        rows.append(row['scores']); hashes[str(path)] = sha(path)
        if (i + 1) % 32 == 0:
            print(f'Validation tuning: {i+1}/{len(a)} observations complete', flush=True)
    choices, strata = select_settings(np.stack(rows), ids)
    score_path = ROOT / 'development_scores.npz'
    save(score_path, scores=np.stack(rows), strata=strata, condition_ids=ids)
    hashes[str(score_path)] = sha(score_path)
    frozen = dict(manifest_sha256=sha(ROOT / 'manifest.json'), choices=choices,
        development_hashes=hashes, frozen_unix=time.time(), test_not_generated=True,
        mask_sha256=hashlib.sha256(a.s.numpy().tobytes() + b.s.numpy().tobytes()).hexdigest(),
        criterion=CONFIG['validation_criterion'], candidate_count=2 * len(BUDGETS) * len(PARAMS),
        original_model_hashes={str(s): json.loads((OLD / 'manifest.json').read_text())['models'][str(s)]['sha256'] for s in SEEDS})
    write(frozen_path, frozen)
    print('All classical choices frozen. Generating new paired test observations next.', flush=True)
    return frozen


@lru_cache(None)
def old_settings():
    return json.loads((OLD / 'frozen_selection.json').read_text())['noise_aware']


def configs(obs, frozen):
    keys = lookup_key(obs)
    if len(set(keys)) != 1:
        raise ValueError('evaluate homogeneous cells, not a mixed lookup batch')
    chosen = frozen['choices'][keys[0]]
    old = old_settings()[keys[0].split(':')[0]]['loop']
    return dict(opt_selected=chosen['selected'], opt_loop=chosen['loop'], opt_final=chosen['final'],
        **{f'opt_budget{t}': chosen['fixed'][str(t)] for t in BUDGETS}, opt_old=old,
        em100=dict(rank=0, strength=0., iterations=100, placement='loop'))


def make_cell(bank, j, check):
    folder = ROOT / 'data' / f'bank{bank}' / SPECS[j]['id']
    record = folder / 'manifest.json'
    digest = sha(ROOT / 'frozen_selection.json')
    if record.exists():
        meta = json.loads(record.read_text())
        if meta['selection_sha256'] != digest:
            raise RuntimeError('data selection mismatch')
        for path, h in meta['hashes'].items():
            if sha(path) != h:
                raise RuntimeError('test data hash mismatch')
        return load(folder / 'observations.npz')
    seed = CONFIG['test_banks'][bank] + 100 * j
    data = {k: [] for k in ('z', 's', 'b', 'sigma2')}; truth = []
    for i in range(CONFIG['cases_per_cell']):
        check('generating_test', bank=bank, cell=j, case=i)
        w, _ = world(i, seed, **SPECS[j])
        for k, value in zip(data, (w.Z, w.S, w.B, w.sigma2)):
            data[k].append(value)
        truth.append(w.G)
    ids = np.array([f'fair:{seed}:{i}' for i in range(len(truth))])
    obsfile, truthfile = folder / 'observations.npz', folder / 'truth.npz'
    save(obsfile, **{k: np.stack(v) for k, v in data.items()}, trial_ids=ids)
    save(truthfile, g=np.stack(truth), trial_ids=ids)
    write(record, dict(seed=seed, count=len(truth), spec=SPECS[j], bank=bank,
        selection_sha256=digest, generated_unix=time.time(), hashes={str(p): sha(p) for p in (obsfile, truthfile)}))
    return load(obsfile)


@torch.no_grad()
def evaluate(frozen, learned, check):
    for bank in range(len(CONFIG['test_banks'])):
        for j, spec in enumerate(SPECS):
            data = make_cell(bank, j, check)
            path = ROOT / 'predictions' / f'bank{bank}_{spec["id"]}.npz'
            digest = sha(ROOT / 'frozen_selection.json')
            if path.exists():
                if str(load(path)['selection_sha256']) != digest:
                    raise RuntimeError('prediction selection mismatch')
                continue
            obs = tensors(data); outputs = {}; costs = {}; cache = {}
            for name, cfg in configs(obs, frozen).items():
                check('testing', bank=bank, cell=j, total_cells=len(SPECS), method=name)
                key = json.dumps(cfg, sort_keys=True)
                if key not in cache:
                    tick = time.perf_counter(); g = estimate(obs, cfg)
                    cache[key] = (g.numpy(), time.perf_counter() - tick)
                outputs[name], costs[name] = cache[key]
            for name, model in learned.items():
                check('testing', bank=bank, cell=j, total_cells=len(SPECS), method=name)
                tick = time.perf_counter(); outputs[name] = model(*obs.args()).numpy(); costs[name] = time.perf_counter() - tick
            check('testing', bank=bank, cell=j, total_cells=len(SPECS), method='hs_one')
            tick = time.perf_counter(); hs, ranks = hard_one(obs)
            outputs['hs_one'] = hs.numpy(); costs['hs_one'] = time.perf_counter() - tick
            estimates = np.stack(list(outputs.values()))
            if not np.isfinite(estimates).all():
                raise FloatingPointError('nonfinite predictions')
            save(path, estimates=estimates, names=np.array(list(outputs)), trial_ids=data['trial_ids'],
                selection_sha256=digest, hs_ranks=np.array(ranks),
                batch_evaluation_seconds=np.array([costs[n] for n in outputs]))
            print(f'Test bank{bank+1}, cell{j+1}/{len(SPECS)} complete: {spec["id"]}', flush=True)


@torch.no_grad()
def timing(frozen, learned, check):
    for j, spec in enumerate(SPECS):
        path = ROOT / 'timing' / f'{spec["id"]}.json'
        if path.exists():
            continue
        w, _ = world(j, CONFIG['timing_seed'], **spec)
        obs = tensors(dict(z=w.Z[None], s=w.S[None], b=w.B[None], sigma2=np.array([w.sigma2])))
        methods = {n: (lambda cfg=cfg: estimate(obs, configs(obs, frozen)[cfg])) for n, cfg in
                   ((name, name) for name in configs(obs, frozen))}
        methods.update({n: (lambda model=model: model(*obs.args())) for n, model in learned.items()})
        methods['hs_one'] = lambda: hard_one(obs)[0]
        for fn in methods.values():
            fn()
        results = {n: [] for n in methods}
        rng = np.random.default_rng(CONFIG['timing_seed'] + j)
        for repeat in range(CONFIG['timing_repeats']):
            for name in rng.permutation(list(methods)):
                check('timing', cell=j, method=str(name), repeat=repeat)
                tick = time.perf_counter(); methods[name](); results[name].append(time.perf_counter() - tick)
        write(path, dict(seconds=results, spec=spec, selection_sha256=sha(ROOT / 'frozen_selection.json'),
            backend='PyTorch CPU float64 one thread batch1', includes='initialization/lookup/selection/refit; not loading or I/O'))


@torch.no_grad()
def benchmark():
    learned = models(); costs = []
    for i, (s, r, l) in enumerate([(-10, 5, 16), (5, 12, None), (20, 20, 2)]):
        w, _ = world(i, CONFIG['benchmark_seed'], snr=s, rsr=r, L=l)
        obs = tensors(dict(z=w.Z[None], s=w.S[None], b=w.B[None], sigma2=np.array([w.sigma2])))
        a, b = split_observations(obs, 1761)
        tick = time.perf_counter(); development_scores(a, b); costs.append(time.perf_counter() - tick)
    worlds = [world(i, CONFIG['benchmark_seed'] + 100, snr=5, rsr=12, L=None)[0] for i in range(CONFIG['cases_per_cell'])]
    obs = tensors(dict(z=np.stack([w.Z for w in worlds]), s=np.stack([w.S for w in worlds]),
        b=np.stack([w.B for w in worlds]), sigma2=np.array([w.sigma2 for w in worlds])))
    tick = time.perf_counter(); hard_one(obs); hs = time.perf_counter() - tick
    tick = time.perf_counter()
    for model in learned.values():
        model(*obs.args())
    neural = time.perf_counter() - tick
    tick = time.perf_counter()
    for t in BUDGETS:
        estimate(obs, dict(rank=7, strength=.5, placement='loop', iterations=t))
    fixed = time.perf_counter() - tick
    tick = time.perf_counter(); estimate(obs, dict(rank=7, strength=.5, placement='final', iterations=100)); final = time.perf_counter() - tick
    # Assumes no cache benefit, two extra100-step loop comparators, final100, EM100.
    tick = time.perf_counter(); estimate(obs, dict(rank=7, strength=.5, placement='loop', iterations=100)); loop100 = time.perf_counter() - tick
    tick = time.perf_counter(); estimate(obs, dict(rank=0, strength=0., placement='loop', iterations=100)); em100 = time.perf_counter() - tick
    tune_seconds = float(np.mean(costs) * CONFIG['validation_count'])
    test_seconds = (hs + neural + fixed + final + 2 * loop100 + em100) * len(SPECS) * len(CONFIG['test_banks'])
    base = tune_seconds + test_seconds
    record = dict(sources=sources(), candidate_world_seconds=costs, hs_batch_seconds=hs,
        learned_three_batch_seconds=neural, five_budget_batch_seconds=fixed,
        estimated_tuning_seconds=tune_seconds, conservative_test_seconds=test_seconds,
        total_minutes_range=[base * 1.1 / 60 + 5, base * 1.5 / 60 + 10],
        note='CPU throughput estimate; includes allowance for audit/timing/I/O; two-hour cap saves partial progress',
        measured_unix=time.time())
    write(ROOT / 'benchmark.json', record)
    print(json.dumps({k: v for k, v in record.items() if k != 'sources'}, indent=2), flush=True)


def run(phase):
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    ROOT.mkdir(parents=True, exist_ok=True)
    with (ROOT / 'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        start = time.monotonic(); last = 0.
        previous = json.loads((ROOT / 'budget.json').read_text())['seconds'] if (ROOT / 'budget.json').exists() else 0.
        def check(state, **details):
            nonlocal last
            used = previous + time.monotonic() - start
            if used >= CONFIG['max_seconds']:
                raise TimeoutError('two-hour active-compute cap reached; completed units saved; further budget needs approval')
            if time.monotonic() - last > 2:
                write(ROOT / 'status.json', dict(state=state, phase=phase, pid=os.getpid(),
                    seconds=used, updated_unix=time.time(), **details))
                write(ROOT / 'budget.json', dict(seconds=used)); last = time.monotonic()
        write(ROOT / 'pid.json', dict(pid=os.getpid(), started_unix=time.time(), phase=phase))
        try:
            if phase == 'benchmark':
                benchmark()
            else:
                suite = unittest.defaultTestLoader.loadTestsFromName('fair_optshrink.test_checks')
                if not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful():
                    raise RuntimeError('preflight checks failed')
                initialize()
                if json.loads((ROOT / 'benchmark.json').read_text())['sources'] != sources():
                    raise RuntimeError('source changed since benchmark; benchmark again first')
                frozen = tune(check); learned = models()
                evaluate(frozen, learned, check); timing(frozen, learned, check)
                from .analyze import analyze
                analyze(check); initialize()
            write(ROOT / 'status.json', dict(state='benchmark_complete' if phase == 'benchmark' else 'complete',
                phase=phase, seconds=previous + time.monotonic() - start, updated_unix=time.time()))
        except TimeoutError as exc:
            write(ROOT / 'status.json', dict(state='budget_exhausted', error=str(exc), resumable=True, updated_unix=time.time()))
            print(str(exc), flush=True)
        except Exception:
            write(ROOT / 'status.json', dict(state='failed', error=traceback.format_exc(), updated_unix=time.time()))
            raise
        finally:
            write(ROOT / 'budget.json', dict(seconds=previous + time.monotonic() - start))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--phase', choices=('benchmark', 'run'), default='run')
    run(parser.parse_args().phase)
