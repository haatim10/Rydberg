"""Fresh static calibration stress test, with held-out-only nuisance selection."""
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
from .common import freeze_phase,frozen_models,save,load,progress,aggregate,verify_sources,sha
from .calibration import fit,select_and_refit,GAIN_BOUNDS,VARIANCE_BOUNDS

CONFIG=dict(version=1,seed=202610084201,N=32,K=3,P=30,cases=32,snrs=[-5,5,15],rsrs=[0,12],
    modes=['none','noise','joint'],budgets=[5,20,100],gain_bounds=GAIN_BOUNDS,variance_bounds=VARIANCE_BOUNDS,
    selection='21 context / 9 withheld observations; Rician likelihood includes log variance; refit all 30',
    max_seconds=1200,scope='global positive reference gain and shared variance only; reference phase is not identified',
    bootstrap_seed=202610084299,bootstrap_samples=2000)
CONDITIONS=[dict(id='matched',amplitude=1.,noise_db=0,phase=0),
    dict(id='noise_minus6',amplitude=1.,noise_db=-6,phase=0),
    dict(id='noise_plus6',amplitude=1.,noise_db=6,phase=0),
    dict(id='amplitude_minus10pct',amplitude=.9,noise_db=0,phase=0),
    dict(id='amplitude_plus10pct',amplitude=1.1,noise_db=0,phase=0),
    dict(id='joint_plus',amplitude=1.1,noise_db=6,phase=0),
    dict(id='phase_plus10deg',amplitude=1.,noise_db=0,phase=10)]


@torch.no_grad()
def run():
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    directory,protocol=freeze_phase('calibration',dict(CONFIG,conditions=CONDITIONS),
        [__file__,'paper2_review_round2/calibration.py','paper2_review_round2/test_numerics.py'])
    with (directory/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (directory/'summary.json').exists():print('Calibration check already complete');return
        models=frozen_models(protocol);start=time.monotonic();done=0
        try:
            for j,(snr,rsr) in enumerate((s,r) for s in CONFIG['snrs'] for r in CONFIG['rsrs']):
                cell=f'snr_{snr:+d}_rsr_{rsr}'
                for i in range(CONFIG['cases']):
                    p=directory/'data'/cell/f'trial_{i:03d}.npz'
                    if not p.exists():
                        w,_=world(i,CONFIG['seed']+100*j,N=32,K=3,P=30,snr=snr,rsr=rsr,L=None)
                        save(p,z=w.Z,s=w.S,b=w.B,v=w.sigma2,g=w.G,seed=CONFIG['seed']+100*j,trial=i)
                    data=load(p);z,s,b,v,g=(data[k] for k in ('z','s','b','v','g'));v=float(v)
                    true_models=np.stack([model.estimate(z,s,b,v).numpy() for model in models.values()])
                    for cond in CONDITIONS:
                        out=directory/'predictions'/cell/cond['id']/f'trial_{i:03d}.npz'
                        progress(directory,cell+'/'+cond['id'],done,6*CONFIG['cases']*len(CONDITIONS),start);done+=1
                        if out.exists():continue
                        if time.monotonic()-start>CONFIG['max_seconds']:raise TimeoutError('calibration cap; resumable')
                        bb=b*cond['amplitude']*np.exp(1j*np.deg2rad(cond['phase']));vv=v*10**(cond['noise_db']/10)
                        tick=time.perf_counter()
                        calibrated,info=select_and_refit(z,s,bb,vv,modes=CONFIG['modes'],budgets=tuple(CONFIG['budgets']))
                        calibration_seconds=time.perf_counter()-tick
                        original=np.stack([model.estimate(z,s,bb,vv).numpy() for model in models.values()])
                        corrected=np.stack([model.estimate(z,s,bb*calibrated['gain'],calibrated['variance']).numpy() for model in models.values()])
                        raw_em,_=fit(z,s,bb,vv,mode='none',budgets=(100,))
                        null=min([row for row in info['candidates'] if row['mode']=='none'],key=lambda row:row['heldout_nll'])
                        stopped,_=fit(z,s,bb,vv,mode='none',budgets=(null['iterations'],))
                        estimates=np.concatenate([original,corrected,true_models,
                            np.stack([raw_em[100]['g'],stopped[null['iterations']]['g'],calibrated['g']])])
                        save(out,estimates=estimates,energy=np.sum(abs(g)**2),errors=np.sum(abs(estimates-g)**2,axis=(-2,-1)),
                            names=np.array([f'{name}_s{seed}' for name in ('supplied','corrected','true_input') for seed in models]
                                +['em100_supplied','em_stopped_supplied','em_calibrated']),
                            estimated_gain=calibrated['gain'],estimated_variance=calibrated['variance'],
                            reference_relative_error=np.linalg.norm(bb*calibrated['gain']-b)/np.linalg.norm(b),
                            variance_ratio=calibrated['variance']/v,chosen=json.dumps(info['selected']),
                            candidate_scores=json.dumps(info['candidates']),calibration_seconds=calibration_seconds,
                            seconds=time.perf_counter()-tick,seed=data['seed'],trial=i)
            verify_sources(protocol);summarize(directory)
            write_json(directory/'status.json',dict(state='complete',condition_cases=done,unique_worlds=192,seconds=time.monotonic()-start))
        except BaseException as e:
            write_json(directory/'status.json',dict(state='partial' if isinstance(e,TimeoutError) else 'failed',error=repr(e)))
            raise


def summarize(directory):
    rng=np.random.default_rng(CONFIG['bootstrap_seed']);summary={};hashes={}
    for snr in CONFIG['snrs']:
        for rsr in CONFIG['rsrs']:
            cell=f'snr_{snr:+d}_rsr_{rsr}'
            for cond in CONDITIONS:
                rows=[]
                for i in range(CONFIG['cases']):
                    p=directory/'predictions'/cell/cond['id']/f'trial_{i:03d}.npz';row=load(p)
                    truth=load(directory/'data'/cell/f'trial_{i:03d}.npz')
                    np.testing.assert_allclose(row['errors'],np.sum(abs(row['estimates']-truth['g'])**2,axis=(-2,-1)),rtol=1e-12,atol=1e-12)
                    rows.append(row);hashes[str(p)]=sha(p)
                errors=np.stack([r['errors'] for r in rows]).T;energy=np.array([r['energy'] for r in rows])
                groups=dict(supplied=[0,1,2],corrected=[3,4,5],true_input=[6,7,8],em100_supplied=[9],em_stopped_supplied=[10],em_calibrated=[11])
                methods={n:aggregate(errors[ix],energy) for n,ix in groups.items()}
                counts=rng.multinomial(len(rows),np.ones(len(rows))/len(rows),size=CONFIG['bootstrap_samples'])
                boot={n:np.mean(10*np.log10((errors[ix]@counts.T)/(counts@energy)[None]),axis=0) for n,ix in groups.items()}
                selected=[json.loads(str(r['chosen'])) for r in rows]
                summary[cell+'/'+cond['id']]=dict(snr=snr,rsr=rsr,condition=cond,methods=methods,
                    corrected_gain_db=methods['supplied']['nmse_db']-methods['corrected']['nmse_db'],
                    corrected_gain_interval=np.quantile(boot['supplied']-boot['corrected'],[.025,.975]).tolist(),
                    median_variance_ratio=float(np.median([r['variance_ratio'] for r in rows])),
                    mean_reference_relative_error=float(np.mean([r['reference_relative_error'] for r in rows])),
                    mean_calibration_ms=1000*float(np.mean([r['calibration_seconds'] for r in rows])),
                    selected_modes={mode:sum(r['mode']==mode for r in selected) for mode in CONFIG['modes']},
                    variance_floor_hits=sum(float(r['estimated_variance'])<=VARIANCE_BOUNDS[0]*1.001 for r in rows))
    write_json(directory/'summary.json',dict(cells=summary,config=CONFIG,
        caveats=['exploratory unadjusted intervals conditional on frozen models','32 independent cases per SNR/reference cell',
                 'phase is not estimated; no claim of general calibration recovery','no extra observations; 21+9 split reused and full-data refit cost counted',
                 'noise/gain estimation can harm correct-input cases; every condition is retained']))
    write_json(directory/'artifact_hashes.json',hashes)

if __name__=='__main__':run()
