"""Tight pilots and paired weak-reference diagnostics; no retraining or tuning."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key]='1'
import fcntl
import json
from pathlib import Path
import time
import numpy as np
import torch
from sahure_expansion.data import world
from sahure_expansion.protocol import write_json
from codex_verification.solver import fit as classical_fit
from sahure.classical import likelihood
from paper2_optshrink_loop.estimator import estimate as opt_estimate,noise_index
from .calibration import fit,conditioning
from .common import freeze_phase,frozen_models,save,load,progress,aggregate,verify_sources,sha

CONFIG=dict(version=1,cases=32,N=32,K=3,pilots=[6,8,10,12,20,30],snrs=[-5,5,15],rsr=12,
    pilot_seed=202610087101,reference_seed=202610087201,reference_snrs=[5,20,60],
    reference_levels=['zero',-10,0,5,12],cap_seconds=1200,
    scope='frozen-model transfer, not pilot-specific retraining; truth initialization is oracle-only',
    hs_split='first round(0.7P) for rank fitting, remainder for Rician selection; all P for refit',
    weak_pairing='same G,S and complex noise across reference strengths within each SNR/trial')


def hs(z,s,b,v):
    nc=round(.7*s.shape[-1]);ranks=np.arange(1,17)
    bank=classical_fit(s[:,:nc],z[:,:nc],b[:,:nc],v,iterations=25,ranks=ranks,sweeps=4)
    scores=likelihood(z[:,nc:],bank@s[:,nc:]+b[:,nc:],v).mean((-2,-1));r=int(ranks[scores.argmin()])
    return classical_fit(s,z,b,v,iterations=100,ranks=[r],sweeps=4)[0],r


def summary(directory,protocol):
    cells={};hashes={}
    for cell in sorted((directory/'predictions').iterdir()):
        rows=[]
        for p in sorted(cell.glob('*.npz')):
            row=load(p);truth=load(directory/'data'/cell.name/p.name)
            np.testing.assert_allclose(row['errors'],np.sum(abs(row['estimates']-truth['g'])**2,axis=(-2,-1)),rtol=1e-12,atol=1e-12)
            rows.append(row);hashes[str(p)]=sha(p)
        if len(rows)!=CONFIG['cases']:raise RuntimeError('incomplete cell')
        errors=np.stack([r['errors'] for r in rows]).T;energy=np.array([r['energy'] for r in rows])
        names=rows[0]['names'].tolist();groups={'learned':[0,1,2],**{n:[j] for j,n in enumerate(names) if j>=3}}
        cells[cell.name]=dict(methods={n:aggregate(errors[ix],energy) for n,ix in groups.items()},
            minimum_local_rank=int(min(r['local_rank'].min() for r in rows)),
            median_minimum_singular_value=float(np.median([r['local_minimum'] for r in rows])),
            median_condition_ratio=float(np.median([r['local_ratio'] for r in rows])),
            seconds_per_case=float(np.mean([r['seconds'] for r in rows])))
    verify_sources(protocol);write_json(directory/'summary.json',dict(config=CONFIG,cells=cells,
        caveats=['32 independent worlds per cell; reference-strength conditions paired within SNR',
        'Jacobian values use true channel only as a diagnostic, not as estimator inputs',
        'P=6 is a necessary dimension count for unconstrained row estimation, not a sufficient uniqueness guarantee',
        'Hard structural assumptions couple antenna rows; rowwise dimension counting alone cannot establish their identifiability',
        'A truth-initialized result is an oracle diagnostic and not an implementable comparator',
        'Fixed model and OptShrink settings transferred from P30; no test-tuned settings']))
    write_json(directory/'artifact_hashes.json',hashes)


@torch.no_grad()
def run():
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    selection_path='results/paper2_optshrink_loop/frozen_selection.json'
    directory,protocol=freeze_phase('geometry',CONFIG,[__file__,'paper2_review_round2/calibration.py',
        'paper2_optshrink_loop/estimator.py',selection_path,'sahure_expansion/data.py'])
    with (directory/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (directory/'summary.json').exists():print('Geometry checks already complete');return
        models=frozen_models(protocol);selection=json.loads(Path(selection_path).read_text())
        start=time.monotonic();done=0;total=CONFIG['cases']*(18+15)
        try:
            for j,(p,snr) in enumerate((p,s) for p in CONFIG['pilots'] for s in CONFIG['snrs']):
                cell=f'pilots_P{p}_snr{snr:+d}'
                for i in range(CONFIG['cases']):
                    out=directory/'predictions'/cell/f'trial_{i:03d}.npz'
                    progress(directory,cell,done,total,start);done+=1
                    if out.exists():continue
                    if time.monotonic()-start>CONFIG['cap_seconds']:raise TimeoutError('bounded check; resumable')
                    w,_=world(i,CONFIG['pilot_seed']+j*100,N=32,K=3,P=p,snr=snr,rsr=12)
                    z,s,b,g,v=w.Z,w.S,w.B,w.G,w.sigma2
                    save(directory/'data'/cell/out.name,z=z,s=s,b=b,g=g,v=v)
                    tick=time.perf_counter();estimates=[m.estimate(z.copy(),s.copy(),b.copy(),v).numpy() for m in models.values()]
                    em=classical_fit(s,z,b,v,iterations=100)[0]
                    key=str(int(np.arange(-10,21,5)[noise_index(v)]))
                    opt=opt_estimate(z,s,b,v,selection['noise_aware'][key]['loop']);hh,rank=hs(z,s,b,v)
                    estimates=np.stack(estimates+[em,opt,hh]);cond=conditioning(g,s,b)
                    if not np.isfinite(estimates).all():raise FloatingPointError('nonfinite estimate')
                    save(out,estimates=estimates,errors=np.sum(abs(estimates-g)**2,axis=(-2,-1)),energy=np.sum(abs(g)**2),
                        names=np.array([f'learned_s{seed}' for seed in models]+['em100','opt_loop_transfer','hs_rician']),
                        local_rank=cond['rank'],local_minimum=cond['minimum'],local_ratio=cond['ratio'],hs_rank=rank,seconds=time.perf_counter()-tick)
            for j,snr in enumerate(CONFIG['reference_snrs']):
                for i in range(CONFIG['cases']):
                    w,_=world(i,CONFIG['reference_seed']+j*100,N=32,K=3,P=30,snr=snr,rsr=12)
                    # Reconstruct the original complex noise once; noise is shared across levels.
                    if not hasattr(w,'W'):raise RuntimeError('world must expose original complex noise for paired test')
                    for rsr in CONFIG['reference_levels']:
                        cell=f'reference_{rsr}_snr{snr:+d}';out=directory/'predictions'/cell/f'trial_{i:03d}.npz'
                        progress(directory,cell,done,total,start);done+=1
                        if out.exists():continue
                        if time.monotonic()-start>CONFIG['cap_seconds']:raise TimeoutError('bounded check; resumable')
                        b=np.zeros_like(w.B) if rsr=='zero' else w.B*10**((rsr-12)/20)
                        z=abs(w.G@w.S+b+w.W);s,g,v=w.S,w.G,w.sigma2
                        save(directory/'data'/cell/out.name,z=z,s=s,b=b,g=g,v=v)
                        tick=time.perf_counter();estimates=[m.estimate(z.copy(),s.copy(),b.copy(),v).numpy() for m in models.values()]
                        em,_=fit(z,s,b,v,mode='none',budgets=(100,));oracle,_=fit(z,s,b,v,mode='none',budgets=(100,),g0=g)
                        estimates=np.stack(estimates+[em[100]['g'],oracle[100]['g']]);cond=conditioning(g,s,b)
                        save(out,estimates=estimates,errors=np.sum(abs(estimates-g)**2,axis=(-2,-1)),energy=np.sum(abs(g)**2),
                            names=np.array([f'learned_s{seed}' for seed in models]+['em_spectral','em_truth_initial_ORACLE']),
                            local_rank=cond['rank'],local_minimum=cond['minimum'],local_ratio=cond['ratio'],seconds=time.perf_counter()-tick)
            summary(directory,protocol)
            write_json(directory/'status.json',dict(state='complete',cases=done,seconds=time.monotonic()-start))
        except BaseException as e:
            write_json(directory/'status.json',dict(state='partial' if isinstance(e,TimeoutError) else 'failed',error=repr(e)));raise

if __name__=='__main__':run()
