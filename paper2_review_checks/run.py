"""Small fresh paired reviewer checks. Run from the repository root.

All prior source, checkpoint and result files are read only. Five-stage output
is a prefix of the unchanged ten-stage model, not a retrained depth variant.
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key] = '1'
import argparse
import hashlib
import json
from pathlib import Path
import platform
import time
import numpy as np
import scipy
import torch
from codex_verification.solver import fit
from sahure.classical import likelihood
from sahure.train import load_model
from sahure_expansion.data import world

ROOT = Path('results/paper2_review_checks')
SEEDS = [9101,9102,9103]
NCASE = 32
SNRS = [-10,-5,0,5,10,15,20]
CONFIG = dict(version=1, cases_per_condition=NCASE, seeds=SEEDS,
    base_seed=2026100807, N=32, K=3, P=30, rsr=12,
    main_snrs=SNRS, precision='float64/complex128', threads=1,
    bootstrap_samples=2000, bootstrap_seed=2026100819,
    scope='exploratory frozen-model diagnostics; no retraining, tuning, or independent confirmation',
    rank_selection='identical first 21 pilots, 25 EM iterations, ranks 1..16, four Cadzow sweeps; last nine pilots scored with squared magnitude residual or Rician NLL; refit 100 iterations on all 30',
    prefix='stage five of the frozen ten-stage forward pass; stage features remain t/10',
    mismatch='change estimator inputs only; saved true reference, observations, channel and true noise variance stay fixed',
    reference_amplitudes=[.9,1.1], reference_phases_degrees=[-10,10],
    reference_phase_ramp_end_degrees=[-10,10], noise_variance_offsets_db=[-6,6],
    phase_ramp='assumed B[n,p] = true B[n,p] * exp(i delta * p/(P-1)), p=0..P-1',
    rsr_convention='reference power relative to one user; expected aggregate RSR is nominal RSR minus 10 log10 K',
    metrics='primary cell NMSE is 10log10(sum squared errors / sum channel energies), computed per model seed then averaged in dB; seed predictions are not independent worlds; also save mean per-case NMSE and severe error frequency',
    runtime='computation accounting only, not a controlled latency benchmark',
    max_run_seconds=850)

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')

def specs():
    out=[dict(id=f'snr_{s:+d}',bank=f'snr_{s:+d}',snr=s,rsr=12,L=None) for s in SNRS]
    out.extend([dict(id='weak_reference',bank='weak_reference',snr=5,rsr=0,L=None),
                dict(id='two_paths',bank='two_paths',snr=5,rsr=12,L=2)])
    for kind,vals in [('amplitude',[.9,1.1]),('phase',[-10,10]),('phase_ramp',[-10,10]),('noise',[-6,6])]:
        out.extend(dict(id=f'{kind}_{v:+g}',bank='snr_+5',snr=5,rsr=12,L=None,mismatch=kind,value=v) for v in vals)
    return out

def rank_scores(z,s,b,sigma2,candidates):
    """No truth. Identical candidate channels; only the withheld score differs."""
    nv=max(1,int(round(.3*s.shape[-1])))
    mu=candidates@s[:,-nv:]+b[:,-nv:]
    zz=z[:,-nv:]
    return np.sum((zz-np.abs(mu))**2,axis=(-2,-1)),np.sum(likelihood(zz,mu,sigma2),axis=(-2,-1))

def paired_classical(z,s,b,sigma2):
    nv=max(1,int(round(.3*s.shape[-1])))
    ranks=np.arange(1,(z.shape[0]+1)//2+1)
    candidates=fit(s[:,:-nv],z[:,:-nv],b[:,:-nv],sigma2,iterations=25,ranks=ranks,sweeps=4)
    residual,nll=rank_scores(z,s,b,sigma2,candidates)
    selected=[int(ranks[residual.argmin()]),int(ranks[nll.argmin()])]
    unique=np.unique([int(ranks[-1])]+selected)
    estimates=fit(s,z,b,sigma2,iterations=100,ranks=unique,sweeps=4)
    lookup={int(r):g for r,g in zip(unique,estimates)}
    return dict(em100=lookup[int(ranks[-1])],hs_residual=lookup[selected[0]],hs_rician=lookup[selected[1]]),dict(ranks=selected,residual_scores=residual,nll_scores=nll)

def prepare():
    ROOT.mkdir(parents=True,exist_ok=True)
    if (ROOT/'config.json').exists():
        if json.loads((ROOT/'config.json').read_text())!=CONFIG: raise RuntimeError('configuration mismatch')
    else: write(ROOT/'config.json',CONFIG)
    unique=[s for s in specs() if s['id']==s['bank']]
    for j,spec in enumerate(unique):
        path=ROOT/'data'/f'{spec["bank"]}.npz'
        if path.exists(): continue
        rows=[world(i,CONFIG['base_seed']+100*j,N=32,K=3,P=30,snr=spec['snr'],rsr=spec['rsr'],L=spec['L'])[0] for i in range(NCASE)]
        path.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(path,z=np.stack([w.Z for w in rows]),s=np.stack([w.S for w in rows]),
            true_b=np.stack([w.B for w in rows]),true_sigma2=np.array([w.sigma2 for w in rows]),
            truth=np.stack([w.G for w in rows]),trial_ids=np.arange(NCASE),data_seed=CONFIG['base_seed']+100*j)

def load_inputs(spec):
    with np.load(ROOT/'data'/f'{spec["bank"]}.npz') as f: data={k:f[k].copy() for k in f.files}
    data['b']=data['true_b'].copy();data['sigma2']=data['true_sigma2'].copy()
    kind=spec.get('mismatch');v=spec.get('value',0)
    if kind=='amplitude': data['b']*=v
    elif kind=='phase': data['b']*=np.exp(1j*np.deg2rad(v))
    elif kind=='phase_ramp': data['b']*=np.exp(1j*np.deg2rad(v)*np.linspace(0,1,30))[None,None,:]
    elif kind=='noise': data['sigma2']*=10**(v/10)
    return data

@torch.no_grad()
def learned(model,data):
    raw=[]
    hook=model.net.register_forward_hook(lambda mod,args,out:raw.append(out.detach().numpy()))
    try: final,trace=model.estimate(data['z'],data['s'],data['b'],data['sigma2'],diagnostics=True)
    finally: hook.remove()
    relative_tau=np.stack([.001+np.logaddexp(0,v[...,0]) for v in raw],axis=1)
    diagnostics={key:np.stack([t[key].numpy() for t in trace],axis=1) for key in ('beta','tau','correction')}
    diagnostics['relative_tau']=relative_tau
    diagnostics['gamma']=np.array([float(t['gamma']) for t in trace])
    return final.numpy(),trace[4]['g'].numpy(),diagnostics

def save_method(path,estimates,truth,**extra):
    if not np.isfinite(estimates).all(): raise FloatingPointError(path)
    # Diagnostic only: truth-aligned error must never enter the estimator.
    phase=np.angle(np.sum(estimates.conj()*truth,axis=(-2,-1)))
    aligned=estimates*np.exp(1j*phase)[...,None,None]
    np.savez_compressed(path,estimates=estimates,error=np.sum(abs(estimates-truth)**2,axis=(-2,-1)),
        aligned_error=np.sum(abs(aligned-truth)**2,axis=(-2,-1)),energy=np.sum(abs(truth)**2,axis=(-2,-1)),**extra)

def run():
    start=time.monotonic();prepare()
    frozen=json.loads(Path('results/sahure_expansion/frozen_selection.json').read_text())
    entries={str(seed):frozen['models'][f'em_adaptive_s{seed}'] for seed in SEEDS}
    for entry in entries.values():
        if sha(entry['path'])!=entry['sha256']: raise RuntimeError('checkpoint changed')
    models={seed:load_model(entry['path'])[0] for seed,entry in entries.items()}
    source_paths=[Path('paper2_review_checks/run.py'),Path('paper2_review_checks/test_checks.py')]
    source_paths+=list(Path('sahure').glob('*.py'))+list(Path('sahure_expansion').glob('*.py'))+list(Path('rydberg_sim').glob('*.py'))+[Path('codex_verification/solver.py')]
    sources={str(p):sha(p) for p in source_paths}
    manifest=dict(config=CONFIG,conditions=specs(),models=entries,sources=sources,
        banks={str(p):sha(p) for p in sorted((ROOT/'data').glob('*.npz'))},
        libraries=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,torch=torch.__version__,platform=platform.platform()))
    write(ROOT/'manifest.json',manifest)
    for spec in specs():
        out=ROOT/'cells'/spec['id'];out.mkdir(parents=True,exist_ok=True)
        data=load_inputs(spec)
        for seed,model in models.items():
            if (out/f'adaptive10_s{seed}.npz').exists(): continue
            tick=time.monotonic();g,g5,diagnostics=learned(model,data)
            save_method(out/f'adaptive10_s{seed}.npz',g,data['truth'],**diagnostics)
            save_method(out/f'prefix5_s{seed}.npz',g5,data['truth'])
            write(out/f'learned_s{seed}_runtime.json',dict(seconds=time.monotonic()-tick,batch=NCASE))
        records=out/'classical.jsonl'
        done=[json.loads(x) for x in records.read_text().splitlines()] if records.exists() else []
        with records.open('a') as stream:
            for i in range(len(done),NCASE):
                if time.monotonic()-start>CONFIG['max_run_seconds']:
                    write(ROOT/'status.json',dict(state='partial_compute_cap',cell=spec['id'],trial=i,elapsed=time.monotonic()-start));return
                tick=time.monotonic()
                est,info=paired_classical(data['z'][i],data['s'][i],data['b'][i],float(data['sigma2'][i]))
                save_method(out/f'classical_trial_{i:03d}.npz',np.stack(list(est.values())),data['truth'][i],
                    ranks=info['ranks'],residual_scores=info['residual_scores'],nll_scores=info['nll_scores'])
                row=dict(trial=i,seconds=time.monotonic()-tick,ranks=info['ranks'])
                stream.write(json.dumps(row)+'\n');stream.flush();done.append(row)
        arrays=[np.load(out/f'classical_trial_{i:03d}.npz') for i in range(NCASE)]
        for j,method in enumerate(('em100','hs_residual','hs_rician')):
            save_method(out/f'{method}.npz',np.stack([f['estimates'][j] for f in arrays]),data['truth'])
        for f in arrays: f.close()
        write(out/'status.json',dict(state='complete',spec=spec,rank_disagreements=sum(r['ranks'][0]!=r['ranks'][1] for r in done)))
        print(json.dumps(dict(cell=spec['id'],state='complete',elapsed=time.monotonic()-start)),flush=True)
        write(ROOT/'status.json',dict(state='running',completed_cell=spec['id'],elapsed=time.monotonic()-start))
    if any(sha(path)!=digest for path,digest in sources.items()): raise RuntimeError('source changed during check')
    summarize()
    write(ROOT/'status.json',dict(state='complete',conditions=len(specs()),unique_worlds=9*NCASE,condition_cases=len(specs())*NCASE,elapsed=time.monotonic()-start))

def summarize():
    rng=np.random.default_rng(CONFIG['bootstrap_seed']);summary={};boot={}
    methods=['em100','hs_residual','hs_rician','adaptive10','prefix5']
    for spec in specs():
        out=ROOT/'cells'/spec['id'];rows={};samples={}
        counts=rng.multinomial(NCASE,np.full(NCASE,1/NCASE),size=CONFIG['bootstrap_samples'])
        for method in methods:
            names=[f'{method}_s{s}' for s in SEEDS] if method in ('adaptive10','prefix5') else [method]
            files=[np.load(out/f'{name}.npz') for name in names]
            errors=np.stack([f['error'] for f in files]);energy=files[0]['energy']
            aligned=np.stack([f['aligned_error'] for f in files])
            seednmse=10*np.log10(errors.sum(1)/energy.sum())
            bs=10*np.log10((errors@counts.T)/(counts@energy)[None]).mean(0)
            rows[method]=dict(nmse_db=float(seednmse.mean()),seed_nmse_db=seednmse.tolist(),
                interval=np.quantile(bs,[.025,.975]).tolist(),mean_case_nmse=float((errors/energy).mean()),
                severe_fraction=float((errors/energy>1).mean()),aligned_nmse_db=float((10*np.log10(aligned.sum(1)/energy.sum())).mean()))
            samples[method]=bs
            for f in files:f.close()
        contrasts={}
        for ref,method in [('hs_residual','hs_rician'),('hs_residual','adaptive10'),('hs_rician','adaptive10'),('em100','adaptive10'),('prefix5','adaptive10')]:
            key=f'{method}_gain_vs_{ref}'
            contrasts[key]=dict(gain_db=rows[ref]['nmse_db']-rows[method]['nmse_db'],
                interval=np.quantile(samples[ref]-samples[method],[.025,.975]).tolist())
        diag=[]
        for seed in SEEDS:
            with np.load(out/f'adaptive10_s{seed}.npz') as f:diag.append({key:f[key].copy() for key in ('beta','tau','relative_tau','correction','gamma')})
        trajectories={key:dict(mean=np.stack([d[key] for d in diag]).mean((0,1,3)).tolist(),
            q10=np.quantile(np.stack([d[key] for d in diag]).transpose(2,0,1,3).reshape(10,-1),.1,axis=1).tolist(),
            q90=np.quantile(np.stack([d[key] for d in diag]).transpose(2,0,1,3).reshape(10,-1),.9,axis=1).tolist()) for key in ('beta','tau','relative_tau','correction')}
        summary[spec['id']]=dict(spec=spec,methods=rows,contrasts=contrasts,controller=trajectories,
            rank_disagreements=json.loads((out/'status.json').read_text())['rank_disagreements'])
        boot[spec['id']]=samples
    macro={}
    for ref,method in [('hs_residual','hs_rician'),('hs_residual','adaptive10'),('hs_rician','adaptive10'),('em100','adaptive10'),('prefix5','adaptive10')]:
        ids=[f'snr_{s:+d}' for s in SNRS]
        diff=np.stack([boot[c][ref]-boot[c][method] for c in ids]).mean(0)
        macro[f'{method}_gain_vs_{ref}']=dict(gain_db=float(np.mean([summary[c]['methods'][ref]['nmse_db']-summary[c]['methods'][method]['nmse_db'] for c in ids])),interval=np.quantile(diff,[.025,.975]).tolist())
    write(ROOT/'summary.json',dict(scope=CONFIG['scope'],main_macro=macro,cells=summary,
        uncertainty='paired case-bootstrap conditional on fixed checkpoints; independent draws per SNR cell, same case indices across methods/seeds; exploratory unadjusted intervals, not a new confirmation'))

if __name__=='__main__':
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    parser=argparse.ArgumentParser();parser.add_argument('--summary-only',action='store_true');args=parser.parse_args()
    summarize() if args.summary_only else run()
