"""Frozen, small LRHD comparison. No changes to historical results or models."""
import json
from pathlib import Path
import platform
import time
import numpy as np
import torch
from rician_full_comparison.protocol import sha, write, save, load

ROOT = Path('results/lrhd_screen')
SNRS = [-10, 0, 10, 20]
SPECS = [dict(id=f'snr{s}_paths{L if L else "random"}', snr=float(s), rsr=12., L=L)
         for s in SNRS for L in (2, None, 16)]
SPECS += [dict(id='weak_reference', snr=5., rsr=0., L=None),
          dict(id='nearly_noiseless', snr=60., rsr=12., L=None)]
BUDGETS = (5, 20, 50, 100)
PARAMS = [dict(rank=0, strength=0.)]+[
    dict(rank=r, strength=a) for r in range(1, 16) for a in (.25, .5, .75, 1.)]
CONFIG = dict(version=1, development_seed=20261015201001, development_cases=8,
    banks=[20261015211001, 20261015221001], cases=16, primary_cells=12,
    seeds=[9101, 9102, 9103], bootstrap_samples=2000, bootstrap_seed=20261015241001,
    timing_seed=20261015231001, timing_repeats=3, benchmark_seed=20261015181001,
    tolerance=1e-5, max_sweeps=1000, max_seconds=7200, threads=1,
    dtype='float64/complex128', budgets=list(BUDGETS), params=PARAMS, specs=SPECS,
    selection='equal three-family mean held-out Rician NLL per SNR and placement',
    metric='equal12-cell average dB ratio-of-summed errors to summed channel power')


def initialize():
    old = json.loads(Path('results/fair_optshrink/manifest.json').read_text())
    # Verify the inherited sources against their original immutable record.
    for p, h in old['sources'].items():
        if sha(p) != h:
            raise RuntimeError('historical source changed: '+p)
    compact = json.loads(Path('results/compact_baseline/frozen_selection.json').read_text())
    entries = {f'original_{s}': compact['models'][f'adaptive_{s}'] for s in CONFIG['seeds']}
    for value in entries.values():
        if sha(value['path']) != value['sha256']:
            raise RuntimeError('original checkpoint changed')
    paths = list(Path('lrhd_screen').glob('*.py'))+[Path('lrhd_screen/PROTOCOL.txt')]
    paths += [Path(p) for p in ('results/fair_optshrink/frozen_selection.json',
        'results/compact_baseline/frozen_selection.json', 'fair_optshrink/estimator.py',
        'optshrink_rank_check/estimator.py', 'rician_full_comparison/protocol.py')]
    paths += [Path(v['path']) for v in entries.values()]
    paths += list(Path('rydberg_sim').rglob('*.py'))
    record = dict(config=CONFIG, models=entries,
        sources={**old['sources'], **{str(p):sha(p) for p in paths}},
        environment=dict(python=platform.python_version(), numpy=np.__version__,
            torch=torch.__version__, platform=platform.platform(), threads=1))
    target = ROOT/'manifest.json'
    if target.exists() and json.loads(target.read_text()) != record:
        raise RuntimeError('source/config/checkpoint/environment changed after freeze')
    if not target.exists():
        write(target, record)
    return record


def models():
    from sahure.train import load_model
    record = json.loads((ROOT/'manifest.json').read_text())
    out = {}
    for name, value in record['models'].items():
        if sha(value['path']) != value['sha256']:
            raise RuntimeError('original checkpoint changed')
        model, _ = load_model(value['path'], dtype=torch.float64)
        if sum(p.numel() for p in model.parameters()) != 188:
            raise RuntimeError('wrong original architecture')
        model.eval()
        out[name] = model
    return out


def lookup_snr(obs):
    value = float(10*torch.log10(3/obs.sigma2[0]))
    return str(min(SNRS, key=lambda s: abs(s-value)))


@torch.no_grad()
def development_scores(obs, kind):
    from sahure.data import Observations
    from sahure.math import nll
    from fair_optshrink.estimator import repeat, trajectory as onepath, correction as onecorrect
    from .estimator import trajectory, correct, empty_diagnostics
    context = Observations(obs.z[..., :21], obs.s[..., :21], obs.b[..., :21], obs.sigma2)
    expanded = repeat(context, len(PARAMS))
    diagnostics = empty_diagnostics()
    options = dict(tolerance=CONFIG['tolerance'], max_sweeps=CONFIG['max_sweeps'])
    if kind == 'lrhd':
        saved, diagnostics = trajectory(expanded, PARAMS, budgets=BUDGETS, **options)
    else:
        saved = onepath(expanded, PARAMS, budgets=BUDGETS)
    rows = []
    for placement in ('loop', 'final'):
        scores = []
        for t in BUDGETS:
            if placement == 'loop':
                g = saved[t]
            else:
                base = saved[t][:1].expand(len(PARAMS), -1, -1)
                g = correct(base, PARAMS, diagnostics, **options) if kind == 'lrhd' else onecorrect(base, PARAMS)
            scores.append(nll(obs.z[..., 21:], g@obs.s[..., 21:]+obs.b[..., 21:],
                              obs.sigma2).mean((-2,-1)).numpy())
        rows.append(scores)
    values = np.array(rows)
    if not np.isfinite(values).all():
        raise FloatingPointError('invalid development likelihood')
    return values, diagnostics


def tune(check):
    from sahure_expansion.data import world
    from fair_optshrink.estimator import tensors
    target = ROOT/'frozen_selection.json'
    if target.exists():
        result = json.loads(target.read_text())
        if result['manifest_sha256'] != sha(ROOT/'manifest.json'):
            raise RuntimeError('selection manifest changed')
        for p, h in result['development_hashes'].items():
            if sha(p) != h:
                raise RuntimeError('development record changed')
        return result
    if (ROOT/'data').exists():
        raise RuntimeError('test data exists before development selection')
    hashes, collected = {}, {'lrhd':[], 'onepass':[]}
    diagnostics, elapsed = [], 0.
    for j, spec in enumerate(SPECS[:CONFIG['primary_cells']]):
        for i in range(CONFIG['development_cases']):
            index = j*CONFIG['development_cases']+i
            check('tuning', done=index, total=96, cell=j, case=i)
            folder = ROOT/'development'/f'case_{index:03d}'
            obsfile = folder/'observations.npz'
            if not obsfile.exists():
                w, _ = world(index, CONFIG['development_seed'], **spec)
                save(obsfile, z=w.Z[None], s=w.S[None], b=w.B[None], sigma2=np.array([w.sigma2]),
                     trial_ids=np.array([f'lrhd-dev:{CONFIG["development_seed"]}:{index}']))
            obs = tensors(load(obsfile))
            for kind in ('lrhd', 'onepass'):
                resultfile = folder/f'{kind}.npz'
                infofile = folder/f'{kind}.json'
                if not (resultfile.exists() and infofile.exists()):
                    tick = time.monotonic()
                    scores, info = development_scores(obs, kind)
                    save(resultfile, scores=scores, manifest_sha256=sha(ROOT/'manifest.json'))
                    write(infofile, dict(diagnostics=info, seconds=time.monotonic()-tick,
                        observation_sha256=sha(obsfile), result_sha256=sha(resultfile)))
                row, info = load(resultfile), json.loads(infofile.read_text())
                if (str(row['manifest_sha256']) != sha(ROOT/'manifest.json') or
                    info['observation_sha256'] != sha(obsfile) or info['result_sha256'] != sha(resultfile)):
                    raise RuntimeError('development provenance mismatch')
                collected[kind].append(row['scores'])
                elapsed += info['seconds']
                diagnostics.append(dict(kind=kind, index=index, **info['diagnostics']))
                hashes[str(resultfile)] = sha(resultfile)
                hashes[str(infofile)] = sha(infofile)
            hashes[str(obsfile)] = sha(obsfile)
            print(f'Development {index+1}/96 complete', flush=True)
    choices = {}
    for kind, values in collected.items():
        cells = np.array(values).reshape(12, CONFIG['development_cases'], 2, len(BUDGETS), len(PARAMS)).mean(1)
        choices[kind] = {}
        for si, snr in enumerate(SNRS):
            scores = cells[si*3:si*3+3].mean(0)
            choices[kind][str(snr)] = {}
            for pi, place in enumerate(('loop', 'final')):
                ti, ri = np.unravel_index(scores[pi].argmin(), scores[pi].shape)
                choices[kind][str(snr)][place] = dict(**PARAMS[ri], iterations=BUDGETS[ti], placement=place)
    value = dict(manifest_sha256=sha(ROOT/'manifest.json'), choices=choices,
        frozen_unix=time.time(), development_hashes=hashes, development_seconds=elapsed,
        development_diagnostics=diagnostics, test_not_generated=True)
    write(target, value)
    print('Matched tuning frozen; fresh testing can begin.', flush=True)
    return value
