"""Run the frozen, resumable comparison; never edits earlier experiments."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key] = '1'
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import platform
import tempfile
import time
import unittest
import numpy as np
import scipy
import torch
from sahure_expansion.data import world
from sahure_expansion.evaluate import estimator_sources
from sahure.train import load_model
from .estimator import BUDGETS,CANDIDATES,development_scores,estimate,hs_rician,noise_index

ROOT = Path('results/paper2_optshrink_loop')
CONFIG = dict(version=1,development_cases=32,test_cases=128,N=32,K=3,P=30,
    snrs=list(range(-10,21,5)),development_seed=202610081101,test_seed=202610081901,
    timing_seed=202610083901,timing_cases=24,timing_repeats=3,
    bootstrap_seed=202610082901,bootstrap_samples=4000,
    training_seeds=[9101,9102,9103],budgets=list(BUDGETS),candidates=CANDIDATES,
    primary_comparator='loop_selected',primary_learned='adaptive10',
    max_seconds_per_invocation=7200,selection='observation-only withheld Rician NLL',
    source_protocol='paper2_optshrink_loop/PROTOCOL.md')
SPECS = [dict(id=f'snr_{s:+d}',snr=s,rsr=12,L=None) for s in CONFIG['snrs']]
SPECS += [dict(id='weak_reference',snr=5,rsr=0,L=None),dict(id='two_paths',snr=5,rsr=12,L=2)]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path,obj):
    path = Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w',dir=path.parent,delete=False,suffix='.tmp') as f:
        json.dump(obj,f,indent=2,allow_nan=False);f.write('\n');name=f.name
    os.replace(name,path)


def save(path,**arrays):
    path = Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent,delete=False,suffix='.tmp') as f:
        np.savez_compressed(f,**arrays);name=f.name
    os.replace(name,path)


def load(path):
    with np.load(path,allow_pickle=False) as f:
        return {k:f[k].copy() for k in f.files}


def sources():
    paths = list(Path('paper2_optshrink_loop').glob('*.py'))+[Path(CONFIG['source_protocol']),
        Path('paper2_review_checks/optshrink_control.py')]
    return {**estimator_sources(),**{str(p):sha(p) for p in sorted(paths)}}


def initialize():
    ROOT.mkdir(parents=True,exist_ok=True)
    old = json.loads(Path('results/sahure_expansion/frozen_selection.json').read_text())
    if old['sources'] != estimator_sources():
        raise RuntimeError('Frozen estimator sources changed; stop and inspect')
    models = {str(seed):old['models'][f'em_adaptive_s{seed}'] for seed in CONFIG['training_seeds']}
    for entry in models.values():
        if sha(entry['path']) != entry['sha256']: raise RuntimeError('checkpoint hash mismatch')
    manifest = dict(config=CONFIG,sources=sources(),models=models,
        previous_frozen_selection_sha256=sha('results/sahure_expansion/frozen_selection.json'),
        environment=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,
                         torch=torch.__version__,system=platform.platform(),machine=platform.machine(),threads=1))
    path = ROOT/'manifest.json'
    if path.exists():
        old_manifest=json.loads(path.read_text())
        for key in ('config','sources','models','previous_frozen_selection_sha256'):
            if old_manifest[key] != manifest[key]: raise RuntimeError(f'Cannot resume: changed {key}')
    else:
        write(path,manifest);write(ROOT/'config.json',CONFIG)
    return manifest


def observations(seed,i,spec):
    w,_ = world(i,seed,N=32,K=3,P=30,**{k:spec[k] for k in ('snr','rsr','L')})
    return w,dict(z=np.array(w.Z),s=np.array(w.S),b=np.array(w.B),variance=np.array(w.sigma2))


def choose(scores,placement):
    t,c = np.unravel_index(np.argmin(scores),scores.shape)
    return dict(**CANDIDATES[c],iterations=BUDGETS[t],placement=placement)


def development(progress):
    frozen_path = ROOT/'frozen_selection.json'
    if frozen_path.exists():
        frozen = json.loads(frozen_path.read_text())
        for path,digest in frozen['development_hashes'].items():
            if sha(path) != digest: raise RuntimeError('development data changed')
        if frozen['source_hashes'] != sources(): raise RuntimeError('selection source mismatch')
        return frozen
    means = {'loop':[],'final':[]};hashes={}
    for j,spec in enumerate(SPECS[:7]):
        seed = CONFIG['development_seed']+100*j
        bank = ROOT/'development'/spec['id']
        rows=[]
        for i in range(CONFIG['development_cases']):
            progress('development',j*CONFIG['development_cases']+i,7*CONFIG['development_cases'])
            p=bank/f'trial_{i:03d}.npz'
            if not p.exists():
                _,data=observations(seed,i,spec)
                tick=time.perf_counter()
                scores=development_scores(data['z'],data['s'],data['b'],float(data['variance']))
                save(p,**data,**{f'{k}_scores':v for k,v in scores.items()},
                     seed=seed,trial=i,seconds=time.perf_counter()-tick)
            row=load(p)
            if int(row['seed'])!=seed or int(row['trial'])!=i: raise RuntimeError('development identity mismatch')
            rows.append(row);hashes[str(p)]=sha(p)
        for name in means:
            means[name].append(np.mean([r[f'{name}_scores'] for r in rows],axis=0))
    frozen=dict(noise_aware={},global_loop=choose(np.mean(means['loop'],axis=0),'loop'),
        development_hashes=hashes,source_hashes=sources(),frozen_unix=time.time(),
        test_not_generated=True,scores={name:np.stack(v).tolist() for name,v in means.items()})
    if (ROOT/'test_data').exists(): raise RuntimeError('Test data already exists before freeze')
    for j,snr in enumerate(CONFIG['snrs']):
        score=means['loop'][j]
        fixed={str(t):dict(**CANDIDATES[int(score[k].argmin())],iterations=t,placement='loop')
               for k,t in enumerate(BUDGETS)}
        frozen['noise_aware'][str(snr)]=dict(loop=choose(score,'loop'),
            final=choose(means['final'][j],'final'),fixed=fixed)
    write(frozen_path,frozen)
    print('Development selection frozen. Fresh test generation starts next.',flush=True)
    return frozen


def configs_for(variance,frozen):
    snr=CONFIG['snrs'][noise_index(variance)]
    selected=frozen['noise_aware'][str(snr)]
    return dict(em100=dict(rank=0,strength=0.,iterations=100,placement='loop'),
        loop_selected=selected['loop'],loop_global=frozen['global_loop'],
        final_selected=selected['final'],
        **{f'loop_{t}':selected['fixed'][str(t)] for t in BUDGETS})


def test_data(j,spec,frozen_digest):
    obs_path=ROOT/'test_data'/spec['id']/'observations.npz'
    truth_path=obs_path.with_name('truth.npz')
    record=obs_path.with_name('manifest.json')
    if record.exists():
        meta=json.loads(record.read_text())
        if meta['frozen_selection_sha256']!=frozen_digest: raise RuntimeError('test selection mismatch')
        for p in (obs_path,truth_path):
            if sha(p)!=meta['hashes'][str(p)]: raise RuntimeError('test data changed')
        return load(obs_path),load(truth_path)
    seed=CONFIG['test_seed']+100*j
    data=[];truth=[]
    for i in range(CONFIG['test_cases']):
        w,row=observations(seed,i,spec);data.append(row);truth.append(w.G)
    ids=np.array([f'{seed}:{i}' for i in range(len(data))])
    save(obs_path,**{key:np.stack([d[key] for d in data]) for key in data[0]},trial_ids=ids)
    save(truth_path,g=np.stack(truth),trial_ids=ids)
    write(record,dict(seed=seed,spec=spec,generated_unix=time.time(),
        frozen_selection_sha256=frozen_digest,hashes={str(p):sha(p) for p in (obs_path,truth_path)}))
    return load(obs_path),load(truth_path)


@torch.no_grad()
def evaluate(frozen,models,progress):
    digest=sha(ROOT/'frozen_selection.json')
    for j,spec in enumerate(SPECS):
        data,truth=test_data(j,spec,digest)
        for i in range(CONFIG['test_cases']):
            progress('test',j*CONFIG['test_cases']+i,9*CONFIG['test_cases'])
            p=ROOT/'test_outputs'/spec['id']/f'trial_{i:03d}.npz'
            if p.exists():
                old=load(p)
                if str(old['trial_id'])!=data['trial_ids'][i] or str(old['selection_sha256'])!=digest:
                    raise RuntimeError('cached result identity mismatch')
                continue
            z,s,b,v=(data[k][i] for k in ('z','s','b','variance'));v=float(v)
            tick=time.perf_counter();est={};cache={}
            for name,cfg in configs_for(v,frozen).items():
                key=json.dumps(cfg,sort_keys=True)
                if key not in cache: cache[key]=estimate(z,s,b,v,cfg)
                est[name]=cache[key]
            est['hs_rician'],rank=hs_rician(z,s,b,v)
            for seed,model in models.items():
                est[f'adaptive10_s{seed}']=model.estimate(z,s,b,v).numpy()
            names=list(est);gs=np.stack(list(est.values()))
            if not np.isfinite(gs).all(): raise FloatingPointError('nonfinite test estimate')
            save(p,names=np.array(names),estimates=gs,truth_energy=np.sum(abs(truth['g'][i])**2),
                errors=np.sum(abs(gs-truth['g'][i])**2,axis=(-2,-1)),hs_rank=rank,
                trial_id=data['trial_ids'][i],selection_sha256=digest,seconds=time.perf_counter()-tick)
        print(f'Test cell complete: {spec["id"]}',flush=True)


@torch.no_grad()
def timing(frozen,models,progress):
    path=ROOT/'timing.json'
    if path.exists(): return
    bank=[observations(CONFIG['timing_seed'],i,SPECS[3])[1] for i in range(CONFIG['timing_cases'])]
    cfgs=configs_for(float(bank[0]['variance']),frozen)
    names=list(cfgs)+['hs_rician']+[f'adaptive10_s{s}' for s in models]
    def call(name,data):
        z,s,b,v=(data[k] for k in ('z','s','b','variance'));v=float(v)
        if name=='hs_rician': return hs_rician(z,s,b,v)[0]
        if name.startswith('adaptive10_'): return models[name.split('_s')[1]].estimate(z,s,b,v).numpy()
        return estimate(z,s,b,v,cfgs[name])
    for name in names: call(name,bank[0])
    times={name:[] for name in names}
    for r in range(CONFIG['timing_repeats']):
        for i,data in enumerate(bank):
            progress('timing',r*len(bank)+i,CONFIG['timing_repeats']*len(bank))
            offset=(r*len(bank)+i)%len(names)
            for name in names[offset:]+names[:offset]:
                tick=time.perf_counter();g=call(name,data);seconds=time.perf_counter()-tick
                if not np.isfinite(g).all(): raise FloatingPointError('timing output')
                times[name].append(seconds)
    write(path,dict(raw_seconds=times,summary={name:dict(mean_ms=1000*float(np.mean(v)),
        median_ms=1000*float(np.median(v)),p10_ms=1000*float(np.quantile(v,.1)),
        p90_ms=1000*float(np.quantile(v,.9))) for name,v in times.items()},
        includes='initialization, online selection and updates; excludes load/I/O; batch one; single CPU thread',
        worlds=len(bank),repetitions=CONFIG['timing_repeats'],seed=CONFIG['timing_seed'],configurations=cfgs))


def run():
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    ROOT.mkdir(parents=True,exist_ok=True)
    with (ROOT/'run.lock').open('a') as lock:
        try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError: raise RuntimeError('Another comparison process owns the lock')
        manifest=initialize()
        if (ROOT/'status.json').exists() and json.loads((ROOT/'status.json').read_text()).get('state')=='complete':
            print('Comparison already complete; preserving results.');return
        suite=unittest.defaultTestLoader.loadTestsFromName('paper2_optshrink_loop.test_estimator')
        checked=unittest.TextTestRunner(verbosity=1).run(suite)
        if not checked.wasSuccessful(): raise RuntimeError('Numerical tests failed')
        write(ROOT/'unit_checks.json',dict(tests=checked.testsRun,passed=True))
        write(ROOT/'pid.json',dict(pid=os.getpid(),started_unix=time.time()))
        start=time.monotonic();last_print=[-1e9]
        def progress(stage,done,total):
            elapsed=time.monotonic()-start
            if elapsed>CONFIG['max_seconds_per_invocation']: raise TimeoutError('two-hour computation cap; saved partial results')
            state=dict(state='running',stage=stage,completed=done,total=total,
                       invocation_seconds=elapsed,updated_unix=time.time(),pid=os.getpid())
            write(ROOT/'status.json',state)
            if elapsed-last_print[0]>30:
                print(json.dumps(state),flush=True);last_print[0]=elapsed
        try:
            frozen=development(progress)
            models={s:load_model(entry['path'])[0] for s,entry in manifest['models'].items()}
            evaluate(frozen,models,progress)
            timing(frozen,models,progress)
            if sources()!=manifest['sources']: raise RuntimeError('sources changed during run')
            for entry in manifest['models'].values():
                if sha(entry['path'])!=entry['sha256']: raise RuntimeError('model changed during run')
            from .analyze import analyze
            analyze()
            write(ROOT/'status.json',dict(state='complete',test_worlds=9*CONFIG['test_cases'],
                development_worlds=7*CONFIG['development_cases'],invocation_seconds=time.monotonic()-start,
                completed_unix=time.time()))
            print('Comparison and independent result recalculation complete.',flush=True)
        except TimeoutError as e:
            write(ROOT/'status.json',dict(state='partial_compute_cap',error=str(e),updated_unix=time.time()))
            print(str(e),flush=True)
        except BaseException as e:
            write(ROOT/'status.json',dict(state='failed',error=repr(e),updated_unix=time.time()))
            raise

if __name__=='__main__':
    run()
