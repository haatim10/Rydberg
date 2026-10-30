import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key] = '1'
import argparse
import fcntl
import json
import time
import traceback
import unittest
from pathlib import Path
import numpy as np
import torch
from sahure_expansion.data import world
from fair_optshrink.run import SPECS, models, sha, save, load, write, old_settings
from fair_optshrink.estimator import tensors, estimate, hard_one, lookup_key
from .estimator import fit_candidates, select_and_estimate

ROOT=Path('results/optshrink_rank_check')
PARENT=Path('results/fair_optshrink')
CONFIG=dict(banks=[202610610001,202610710001],cases=16,timing_seed=202610810001,
    benchmark_seed=202610510001,bootstrap_seed=202610910001,bootstrap_samples=4000,
    max_seconds=7200,specs=SPECS,threads=1,dtype='float64',seeds=[9101,9102,9103])


def frozen_parent():
    return json.loads((PARENT/'frozen_selection.json').read_text())


def initialize():
    parent=json.loads((PARENT/'manifest.json').read_text())
    for p,h in parent['sources'].items():
        if sha(p)!=h: raise RuntimeError('historical source changed: '+p)
    for m in parent['models'].values():
        if sha(m['path'])!=m['sha256']: raise RuntimeError('model changed')
    paths=list(Path('optshrink_rank_check').glob('*.py'))+[Path('optshrink_rank_check/PROTOCOL.txt'),
        PARENT/'manifest.json',PARENT/'frozen_selection.json']
    record=dict(config=CONFIG,sources={str(p):sha(p) for p in paths},parent_sources=parent['sources'],models=parent['models'])
    path=ROOT/'manifest.json'
    if path.exists() and json.loads(path.read_text())!=record: raise RuntimeError('frozen source/config changed')
    if not path.exists(): write(path,record)
    freeze=ROOT/'frozen_selection.json'
    if not freeze.exists():
        if (ROOT/'data').exists(): raise RuntimeError('data before freeze')
        write(freeze,dict(manifest_sha256=sha(path),parent_sha256=sha(PARENT/'frozen_selection.json'),frozen_unix=time.time()))
    else:
        f=json.loads(freeze.read_text())
        if f['manifest_sha256']!=sha(path) or f['parent_sha256']!=sha(PARENT/'frozen_selection.json'):
            raise RuntimeError('freeze mismatch')


def fixed_configs(obs,parent):
    key=lookup_key(obs)[0]; cfg=parent['choices'][key]
    return dict(opt_selected=cfg['selected'],opt_loop=cfg['loop'],opt_final=cfg['final'],
        opt_old=old_settings()[key.split(':')[0]]['loop'],
        em100=dict(rank=0,strength=0.,iterations=100,placement='loop'))


def data(bank,j):
    folder=ROOT/'data'/f'bank{bank}'/SPECS[j]['id']; mf=folder/'manifest.json'
    digest=sha(ROOT/'frozen_selection.json')
    if mf.exists():
        meta=json.loads(mf.read_text())
        if meta['selection_sha256']!=digest: raise RuntimeError('data freeze mismatch')
        for p,h in meta['hashes'].items():
            if sha(p)!=h: raise RuntimeError('data changed')
        return load(folder/'observations.npz')
    seed=CONFIG['banks'][bank]+100*j; arrays={k:[] for k in ('z','s','b','sigma2')}; truth=[]
    for i in range(CONFIG['cases']):
        w,_=world(i,seed,**SPECS[j])
        for k,v in zip(arrays,(w.Z,w.S,w.B,w.sigma2)): arrays[k].append(v)
        truth.append(w.G)
    ids=np.array([f'rank:{seed}:{i}' for i in range(CONFIG['cases'])])
    paths=[folder/'observations.npz',folder/'truth.npz']
    save(paths[0],**{k:np.stack(v) for k,v in arrays.items()},trial_ids=ids)
    save(paths[1],g=np.stack(truth),trial_ids=ids)
    write(mf,dict(seed=seed,count=len(truth),spec=SPECS[j],selection_sha256=digest,
        generated_unix=time.time(),hashes={str(p):sha(p) for p in paths}))
    return load(paths[0])


@torch.no_grad()
def evaluate(parent,learned,check):
    for bank in range(2):
        for j,sp in enumerate(SPECS):
            d=data(bank,j); path=ROOT/'predictions'/f'bank{bank}_{sp["id"]}.npz'
            if path.exists():
                if str(load(path)['selection_sha256'])!=sha(ROOT/'frozen_selection.json'): raise RuntimeError('output freeze mismatch')
                continue
            obs=tensors(d); outputs={}; all_cfg=[]; all_scores=[]
            rank_outputs={p:[] for p in ('loop','final','joint')}
            for i in range(len(obs)):
                check('testing',bank=bank,cell=j,cells=len(SPECS),case=i)
                row=obs.batch(slice(i,i+1)); entry=parent['choices'][lookup_key(row)[0]]
                selected=[fit_candidates(row,entry[p]) for p in ('loop','final')]
                gs=[estimate(row,cfg) for cfg,_ in selected]
                which=int(float(selected[1][1].min())<float(selected[0][1].min()))
                for p,g in zip(('loop','final','joint'),gs+[gs[which]]): rank_outputs[p].append(g.numpy()[0])
                all_cfg.append([[cfg['rank'],cfg['strength'],cfg['iterations'],int(cfg['placement']=='final')] for cfg,_ in selected]+[[*([selected[which][0]['rank'],selected[which][0]['strength'],selected[which][0]['iterations'],which])]])
                all_scores.append(np.stack([s.numpy() for _,s in selected]))
            for p,gs in rank_outputs.items(): outputs['rank_'+p]=np.stack(gs)
            for name,cfg in fixed_configs(obs,parent).items():
                check('fixed_controls',bank=bank,cell=j,method=name); outputs[name]=estimate(obs,cfg).numpy()
            for name,model in learned.items(): outputs[name]=model(*obs.args()).numpy()
            check('hard_control',bank=bank,cell=j); h,r=hard_one(obs);outputs['hs_one']=h.numpy()
            ee=np.stack(list(outputs.values()))
            if not np.isfinite(ee).all(): raise FloatingPointError('nonfinite output')
            save(path,estimates=ee,names=np.array(list(outputs)),trial_ids=d['trial_ids'],
                rank_configs=np.array(all_cfg),rank_scores=np.array(all_scores),hs_ranks=np.array(r),
                selection_sha256=sha(ROOT/'frozen_selection.json'))
            print(f'Bank {bank+1}/2 cell {j+1}/128 complete: {sp["id"]}',flush=True)


@torch.no_grad()
def timing(parent,learned,check):
    for j,sp in enumerate(SPECS):
        path=ROOT/'timing'/f'{sp["id"]}.json'
        if path.exists(): continue
        w,_=world(j,CONFIG['timing_seed'],**sp)
        obs=tensors(dict(z=w.Z[None],s=w.S[None],b=w.B[None],sigma2=np.array([w.sigma2])))
        methods={f'rank_{p}':(lambda p=p:select_and_estimate(obs,parent,p)[0]) for p in ('loop','final','joint')}
        methods.update({n:(lambda n=n:estimate(obs,fixed_configs(obs,parent)[n])) for n in fixed_configs(obs,parent)})
        methods.update({n:(lambda m=m:m(*obs.args())) for n,m in learned.items()});methods['hs_one']=lambda:hard_one(obs)[0]
        for fn in methods.values(): fn()
        times={n:[] for n in methods};rng=np.random.default_rng(CONFIG['timing_seed']+j)
        for rep in range(3):
            for n in rng.permutation(list(methods)):
                check('timing',cell=j,method=str(n),repeat=rep); tick=time.perf_counter();methods[n]();times[n].append(time.perf_counter()-tick)
        write(path,dict(seconds=times,spec=sp,backend='PyTorch CPU float64 one thread batch1',
            includes='all candidate fits, lookup, initialization, selection and refit; excludes loading/I/O'))


@torch.no_grad()
def benchmark():
    parent=frozen_parent(); records=[]
    for i,(s,r,l) in enumerate([(-10,5,16),(5,12,None),(15,5,2),(20,20,16)]):
        w,_=world(i,CONFIG['benchmark_seed'],snr=s,rsr=r,L=l)
        obs=tensors(dict(z=w.Z[None],s=w.S[None],b=w.B[None],sigma2=np.array([w.sigma2])))
        tick=time.perf_counter(); select_and_estimate(obs,parent,'joint'); elapsed=time.perf_counter()-tick
        records.append(dict(snr=s,rsr=r,seconds=elapsed))
    base=np.mean([r['seconds'] for r in records])*4096
    rec=dict(samples=records,estimated_run_minutes=[base/60+15,base*1.5/60+25],
        includes='allowance for other estimators, timing and replay; manuscript work separate')
    write(ROOT/'benchmark.json',rec); print(json.dumps(rec,indent=2),flush=True)


def run(phase):
    torch.set_num_threads(1);torch.set_num_interop_threads(1); ROOT.mkdir(parents=True,exist_ok=True)
    with (ROOT/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        start=time.monotonic();last=0.;prev=json.loads((ROOT/'budget.json').read_text())['seconds'] if (ROOT/'budget.json').exists() else 0.
        def check(state,**kw):
            nonlocal last
            used=prev+time.monotonic()-start
            if used>=CONFIG['max_seconds']: raise TimeoutError('two-hour compute cap; progress preserved')
            if time.monotonic()-last>2:
                write(ROOT/'status.json',dict(state=state,pid=os.getpid(),seconds=used,updated_unix=time.time(),**kw))
                write(ROOT/'budget.json',dict(seconds=used));last=time.monotonic()
        write(ROOT/'pid.json',dict(pid=os.getpid(),phase=phase,started_unix=time.time()))
        try:
            if phase=='benchmark': benchmark()
            else:
                suite=unittest.defaultTestLoader.loadTestsFromName('optshrink_rank_check.test_checks')
                if not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful(): raise RuntimeError('preflight failed')
                initialize();parent=frozen_parent();learned=models()
                evaluate(parent,learned,check);timing(parent,learned,check)
                from .analyze import analyze
                analyze(check);initialize()
            write(ROOT/'status.json',dict(state='benchmark_complete' if phase=='benchmark' else 'complete',seconds=prev+time.monotonic()-start,updated_unix=time.time()))
        except Exception:
            write(ROOT/'status.json',dict(state='failed',error=traceback.format_exc(),updated_unix=time.time()));raise
        finally: write(ROOT/'budget.json',dict(seconds=prev+time.monotonic()-start))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=('benchmark','run'),default='run');run(p.parse_args().phase)
