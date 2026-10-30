"""Deterministic repeated-noise study. Only writes results/hankel_error_check."""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[name]='1'
import fcntl
import hashlib
import json
from pathlib import Path
import time
import traceback
import unittest
import numpy as np
from sahure_expansion.data import world
from sahure_expansion.evaluate import estimator_sources
from paper2_optshrink_loop.run import save,load,write
from .core import path,predictions,STAGES,STARTS,MODELS

ROOT=Path('results/hankel_error_check')
SPECS=[dict(id=f'snr_{s:+d}',snr=s,rsr=12,L=None) for s in (-10,5,20)]
SPECS += [dict(id='weak_reference',snr=5,rsr=0,L=None),
          dict(id='rich_16',snr=5,rsr=12,L=16),
          dict(id='nearly_noiseless',snr=60,rsr=12,L=None)]
CONFIG=dict(worlds_per_cell=4,repeats=256,anchors=8,stages=STAGES,starts=STARTS,models=MODELS,
            specs=SPECS,world_seed=202610091101,noise_seed=202610092101,max_seconds=7200)


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def initialize():
    ROOT.mkdir(parents=True,exist_ok=True)
    deps=estimator_sources()
    frozen=json.loads(Path('results/paper2_optshrink_loop/manifest.json').read_text())
    for p,h in frozen['sources'].items():
        if sha(p)!=h:raise RuntimeError(f'old frozen source changed: {p}')
    paths=[Path('hankel_error_check')/n for n in ('__init__.py','core.py','run.py','test_checks.py','PROTOCOL.txt','THEORY.txt')]
    paths += [Path('paper2_optshrink_loop/run.py')]
    manifest=dict(config=CONFIG,sources={**deps,**{str(p):sha(p) for p in paths}},
        prior_sources_verified=True,numpy=np.__version__,threads=1)
    # JSON normalization makes tuples stable across processes.
    manifest=json.loads(json.dumps(manifest))
    p=ROOT/'manifest.json'
    if p.exists():
        if json.loads(p.read_text())!=manifest:raise RuntimeError('source/config mismatch on resume')
    else:write(p,manifest)
    return manifest


def inputs(cell,index):
    spec=SPECS[cell];seed=CONFIG['world_seed']+100*cell
    w,_=world(index,seed,**{k:spec[k] for k in ('snr','rsr','L')})
    mu=w.G@w.S+w.B
    arrays=[]
    for stream,count in ((0,CONFIG['repeats']),(1,CONFIG['anchors'])):
        rng=np.random.default_rng(np.random.SeedSequence([CONFIG['noise_seed'],cell,index,stream]))
        noise=np.sqrt(w.sigma2/2)*(rng.normal(size=(count,*mu.shape))+1j*rng.normal(size=(count,*mu.shape)))
        arrays.append(abs(mu+noise))
    return w,arrays[0],arrays[1]


def run():
    ROOT.mkdir(parents=True,exist_ok=True)
    with (ROOT/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        tests=unittest.defaultTestLoader.loadTestsFromName('hankel_error_check.test_checks')
        result=unittest.TextTestRunner(verbosity=1).run(tests)
        if not result.wasSuccessful():raise RuntimeError('numerical gate failed')
        initialize()
        write(ROOT/'numerical_checks.json',dict(passed=True,tests=result.testsRun))
        old=json.loads((ROOT/'budget.json').read_text())['seconds'] if (ROOT/'budget.json').exists() else 0.
        start=time.monotonic();count=0
        try:
            for j,spec in enumerate(SPECS):
                for i in range(CONFIG['worlds_per_cell']):
                    if old+time.monotonic()-start>=CONFIG['max_seconds']:
                        raise TimeoutError('two-hour cumulative CPU-run cap reached')
                    p=ROOT/'worlds'/spec['id']/f'{i:02d}.npz'
                    record=p.with_suffix('.json')
                    if p.exists() and record.exists():
                        meta=json.loads(record.read_text())
                        if sha(p)!=meta['sha256'] or meta['manifest_sha256']!=sha(ROOT/'manifest.json'):
                            raise RuntimeError('cached world changed')
                        count+=1;continue
                    w,z,anchors=inputs(j,i);errors=[];covs=[];anchors_g=[];anchor_spread=[]
                    times=[];tick=time.perf_counter()
                    for mode in STARTS:
                        write(ROOT/'status.json',dict(state='running',cell=spec['id'],world=i,
                            start=mode,completed_worlds=count,total_worlds=24,pid=os.getpid(),
                            seconds=old+time.monotonic()-start))
                        t0=time.perf_counter();oracle=w.G if mode=='oracle' else None
                        pp=path(z,w.S,w.B,w.sigma2,mode,oracle)
                        pred,pa,_=predictions(anchors,w.S,w.B,w.sigma2,mode,oracle)
                        errors.append(np.stack([pp[t][1] for t in STAGES]))
                        covs.append(np.stack([pred[t].mean(1) for t in STAGES],axis=1))
                        anchors_g.append(np.stack([pa[t][1] for t in STAGES]))
                        anchor_spread.append(np.stack([np.trace(pred[t],axis1=-2,axis2=-1).sum(-1) for t in STAGES],axis=1))
                        times.append(time.perf_counter()-t0)
                    covs=np.array(covs)
                    if not np.isfinite(covs).all():raise FloatingPointError('covariance is nonfinite')
                    np.testing.assert_allclose(covs,covs.swapaxes(-1,-2),rtol=1e-10,atol=1e-11)
                    ev=np.linalg.eigvalsh(covs)
                    if ev.min() < -1e-8*max(1.,ev.max()):raise FloatingPointError('non-PSD covariance')
                    save(p,z_repeats=z,z_anchors=anchors,s=w.S,b=w.B,g=w.G,variance=w.sigma2,
                        estimates=np.array(errors),predicted_covariances=covs,anchor_estimates=np.array(anchors_g),
                        anchor_prediction_traces=np.array(anchor_spread),seconds_by_start=np.array(times),
                        starts=np.array(STARTS),models=np.array(MODELS),stages=np.array(STAGES),
                        world_id=f'{CONFIG["world_seed"]+100*j}:{i}',cell_index=j,world_index=i)
                    write(record,dict(sha256=sha(p),manifest_sha256=sha(ROOT/'manifest.json'),
                        seconds=time.perf_counter()-tick,completed_unix=time.time()))
                    count+=1;print(f'Completed {count}/24 fixed worlds: {spec["id"]}, {i}',flush=True)
            initialize()
            write(ROOT/'status.json',dict(state='complete',worlds=count,repeat_realizations=6144,
                anchor_realizations=192,seconds=old+time.monotonic()-start))
        except TimeoutError as e:
            write(ROOT/'status.json',dict(state='budget_exhausted',error=str(e),completed_worlds=count))
        except Exception:
            write(ROOT/'status.json',dict(state='failed',error=traceback.format_exc()));raise
        finally:write(ROOT/'budget.json',dict(seconds=old+time.monotonic()-start))


if __name__=='__main__':run()
