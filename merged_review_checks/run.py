"""Bounded post-review comparator and paired tail audit; preserves old studies."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[key] = '1'
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from codex_verification.solver import fit, project
from sahure.classical import likelihood
from sahure.train import load_model
from paper2_optshrink_loop.estimator import hs_rician, estimate, noise_index
from paper2_optshrink_loop.run import load, save, write, observations, CONFIG, SPECS

ROOT = Path('results/merged_review_checks')
OLD = Path('results/paper2_optshrink_loop')
SEEDS = (9101, 9102, 9103)

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def one_sweep(z, s, b, variance, sweeps=1):
    ranks = np.arange(1, (len(z)+1)//2+1)
    candidates = fit(s[:, :21], z[:, :21], b[:, :21], variance,
                     iterations=25, ranks=ranks, sweeps=sweeps)
    scores = likelihood(z[:, 21:], candidates@s[:, 21:]+b[:, 21:], variance).mean((-2, -1))
    rank = int(ranks[scores.argmin()])
    g = fit(s, z, b, variance, iterations=100, ranks=[rank], sweeps=sweeps)[0]
    return g, rank, scores

def verify_sources():
    original = json.loads((OLD/'manifest.json').read_text())
    for path, digest in original['sources'].items():
        assert sha(path) == digest, ('changed frozen source', path)
    for model in original['models'].values():
        assert sha(model['path']) == model['sha256']
    inputs = {}
    for spec in SPECS:
        meta = json.loads((OLD/'test_data'/spec['id']/'manifest.json').read_text())
        for path, digest in meta['hashes'].items():
            assert sha(path) == digest, ('changed observations/truth', path)
            inputs[path] = digest
        for path in sorted((OLD/'test_outputs'/spec['id']).glob('trial_*.npz')):
            inputs[str(path)] = sha(path)
    record = dict(protocol_sha256=sha('merged_review_checks/PROTOCOL.txt'),
                  script_sha256=sha(__file__), source_hashes=original['sources'],
                  input_hashes=inputs, models=original['models'],
                  frozen_selection_sha256=sha(OLD/'frozen_selection.json'))
    path = ROOT/'manifest.json'
    if path.exists():
        assert json.loads(path.read_text()) == record, 'resume source/input mismatch'
    else:
        write(path, record)
    return original

def preflight():
    d = load(OLD/'test_data'/SPECS[0]['id']/'observations.npz')
    z,s,b,v = (d[k][0] for k in ('z','s','b','variance'))
    v = float(v)
    a,r,_ = one_sweep(z,s,b,v,4)
    ref,rr = hs_rician(z,s,b,v)
    np.testing.assert_allclose(a, ref, rtol=0, atol=0)
    assert r == rr
    cap = (len(z)+1)//2
    bypass1 = fit(s,z,b,v,iterations=5,ranks=[cap],sweeps=1)
    bypass4 = fit(s,z,b,v,iterations=5,ranks=[cap],sweeps=4)
    np.testing.assert_array_equal(bypass1, bypass4)
    np.testing.assert_array_equal(project(bypass1,[cap],1), bypass1)
    old = load(OLD/'test_outputs'/SPECS[0]['id']/'trial_000.npz')
    np.testing.assert_allclose(a, old['estimates'][list(old['names']).index('hs_rician')], atol=1e-12, rtol=1e-12)
    tick = time.perf_counter()
    g,_,scores = one_sweep(z,s,b,v)
    assert np.isfinite(g).all() and np.isfinite(scores).all()
    info = dict(passed=True, four_sweep_replay_max_abs=float(np.max(abs(a-ref))),
                exact_bypass=True, seconds_one_case=time.perf_counter()-tick)
    write(ROOT/'preflight.json', info)
    print(json.dumps(info), flush=True)

def evaluate():
    for j,spec in enumerate(SPECS):
        bank = OLD/'test_data'/spec['id']
        data,truth = load(bank/'observations.npz'),load(bank/'truth.npz')
        np.testing.assert_array_equal(data['trial_ids'],truth['trial_ids'])
        for i in range(len(data['trial_ids'])):
            out = ROOT/'predictions'/spec['id']/f'trial_{i:03d}.npz'
            if out.exists():
                cached = load(out)
                assert str(cached['trial_id']) == data['trial_ids'][i]
                continue
            z,s,b,v = (data[k][i] for k in ('z','s','b','variance'))
            tick = time.perf_counter()
            g,r,scores = one_sweep(z,s,b,float(v))
            assert np.isfinite(g).all() and np.isfinite(scores).all()
            save(out,estimate=g,rank=r,scores=scores,trial_id=data['trial_ids'][i],
                 seconds=time.perf_counter()-tick)
            if i%32 == 0:
                write(ROOT/'status.json',dict(state='running',cell=spec['id'],completed=j*128+i+1,total=1152))
        print('Completed '+spec['id'], flush=True)

def interval(x):
    return [float(v) for v in np.quantile(x,[.025,.975])]

def analyze():
    rng = np.random.default_rng(202610081777)
    cells = {}; nmse_boot = {}; tail_boot = {}; max_discrepancy = 0.
    methods = ['one_sweep','hs_rician','loop_selected','final_selected','learned']
    for spec in SPECS:
        bank = OLD/'test_data'/spec['id']
        data,truth = load(bank/'observations.npz'),load(bank/'truth.npz')
        errors = {k:[] for k in methods}; energies=[]
        for i in range(128):
            saved = load(OLD/'test_outputs'/spec['id']/f'trial_{i:03d}.npz')
            new = load(ROOT/'predictions'/spec['id']/f'trial_{i:03d}.npz')
            assert str(saved['trial_id']) == str(new['trial_id']) == data['trial_ids'][i] == truth['trial_ids'][i]
            assert str(saved['selection_sha256']) == sha(OLD/'frozen_selection.json')
            names = list(saved['names']); target = truth['g'][i]
            energy = float(np.sum(abs(target)**2)); energies.append(energy)
            recalculated = np.sum(abs(saved['estimates']-target)**2,axis=(-2,-1))
            np.testing.assert_allclose(recalculated,saved['errors'],rtol=1e-12,atol=1e-12)
            np.testing.assert_allclose(energy,saved['truth_energy'],rtol=1e-12,atol=1e-12)
            max_discrepancy = max(max_discrepancy,float(np.max(abs(recalculated-saved['errors']))))
            for method in methods:
                if method == 'learned':
                    row = [recalculated[names.index(f'adaptive10_s{s}')] for s in SEEDS]
                elif method == 'one_sweep':
                    row = [np.sum(abs(new['estimate']-target)**2)]
                else:
                    row = [recalculated[names.index(method)]]
                errors[method].append(row)
        energy = np.array(energies)
        idx = rng.integers(0,128,(4000,128))
        cells[spec['id']] = {}; nmse_boot[spec['id']]={};tail_boot[spec['id']]={}
        for method,rows in errors.items():
            err = np.array(rows); ratios=err/energy[:,None]
            byseed=10*np.log10(err.sum(0)/energy.sum())
            boot=(10*np.log10(err[idx].sum(1)/energy[idx].sum(1)[:,None])).mean(-1)
            nmse_boot[spec['id']][method]=boot
            tail_boot[spec['id']][method]={}
            tails={}
            for threshold in (.5,1.,2.):
                indicator=(ratios>threshold).astype(float)
                rate_boot=100*indicator[idx].mean((1,2))
                tail_boot[spec['id']][method][str(threshold)] = rate_boot
                tails[str(threshold)]=dict(percent=float(100*indicator.mean()),
                    seed_percent=(100*indicator.mean(0)).tolist(),ci95=interval(rate_boot))
            cells[spec['id']][method]=dict(nmse_db=float(byseed.mean()),seed_nmse_db=byseed.tolist(),tail=tails)
        for method in methods[:-1]:
            cells[spec['id']][method]['learned_gain_db'] = dict(
                value=cells[spec['id']][method]['nmse_db']-cells[spec['id']]['learned']['nmse_db'],
                ci95=interval(nmse_boot[spec['id']][method]-nmse_boot[spec['id']]['learned']))
            for threshold in (.5,1.,2.):
                key=str(threshold)
                cells[spec['id']][method]['tail'][key]['minus_learned_pp_ci95']=interval(
                    tail_boot[spec['id']][method][key]-tail_boot[spec['id']]['learned'][key])
    primary=[s['id'] for s in SPECS[:7]]
    macro={}
    for method in methods:
        macro[method]=dict(nmse_db=float(np.mean([cells[c][method]['nmse_db'] for c in primary])),tail={})
        if method != 'learned':
            boots=np.mean([nmse_boot[c][method]-nmse_boot[c]['learned'] for c in primary],0)
            macro[method]['learned_gain_db_ci95']=interval(boots)
        for threshold in (.5,1.,2.):
            key=str(threshold)
            value=float(np.mean([cells[c][method]['tail'][key]['percent'] for c in primary]))
            macro[method]['tail'][key]=dict(percent=value)
            if method != 'learned':
                diff=np.mean([tail_boot[c][method][key]-tail_boot[c]['learned'][key] for c in primary],0)
                macro[method]['tail'][key]['minus_learned_pp_ci95']=interval(diff)
    result=dict(cells=cells,macro=macro,worlds=1152,new_predictions=1152,
                saved_error_max_abs_difference=max_discrepancy,
                bootstrap='4000 paired whole-world resamples within cells; fixed models; pointwise; exploratory',
                seed=202610081777)
    write(ROOT/'summary.json',result)
    print(json.dumps(macro,indent=2),flush=True)

@torch.no_grad()
def timing(manifest):
    path=ROOT/'timing.json'
    if path.exists(): return
    models={s:load_model(manifest['models'][str(s)]['path'])[0].double().eval() for s in SEEDS}
    frozen=json.loads((OLD/'frozen_selection.json').read_text())
    bank=[observations(CONFIG['timing_seed'],i,SPECS[3])[1] for i in range(24)]
    names=['one_sweep','hs_rician','loop_selected','final_selected']+[f'learned_{s}' for s in SEEDS]
    def call(name,d):
        z,s,b,v=(d[k] for k in ('z','s','b','variance'));v=float(v)
        if name=='one_sweep': return one_sweep(z,s,b,v)[0]
        if name=='hs_rician': return hs_rician(z,s,b,v)[0]
        if name.startswith('learned_'): return models[int(name.split('_')[1])].estimate(z,s,b,v).numpy()
        level=CONFIG['snrs'][noise_index(v)]
        cfg=frozen['noise_aware'][str(level)]['loop' if name=='loop_selected' else 'final']
        return estimate(z,s,b,v,cfg)
    for name in names: call(name,bank[0])
    raw={name:[] for name in names}
    for rep in range(3):
        for i,data in enumerate(bank):
            offset=(rep*24+i)%len(names)
            for name in names[offset:]+names[:offset]:
                tick=time.perf_counter();out=call(name,data)
                raw[name].append(time.perf_counter()-tick)
                assert np.isfinite(out).all()
    summary={k:dict(mean_ms=1000*float(np.mean(v)),median_ms=1000*float(np.median(v))) for k,v in raw.items()}
    summary['learned']=dict(mean_ms=float(np.mean([summary[f'learned_{s}']['mean_ms'] for s in SEEDS])))
    write(path,dict(summary=summary,raw_seconds=raw,worlds=24,repeats=3,seed=CONFIG['timing_seed'],
        scope='one thread, float64, batch one; includes initialization and dispatch/selection; not model loading or I/O'))

def main():
    args=argparse.ArgumentParser();args.add_argument('--preflight-only',action='store_true');opt=args.parse_args()
    ROOT.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    with (ROOT/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        manifest=verify_sources();preflight()
        if opt.preflight_only: return
        evaluate();analyze();timing(manifest)
        write(ROOT/'status.json',dict(state='complete',completed=1152,total=1152))

if __name__=='__main__': main()
