"""Small, independently tuned final-filter comparison; no trained weights changed.

The single final Hankel OptShrink operation is an adapted component control,
not a reproduction of the complete LRHD algorithm. Run as a module from repo.
"""
import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_key] = '1'
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from codex_verification.solver import fit, geometry
from sahure.classical import likelihood
from sahure_expansion.data import world

ROOT = Path('results/paper2_review_checks/optshrink')
BANK = Path('results/paper2_review_checks/data')
SNRS = [-10, -5, 0, 5, 10, 15, 20]
SPECS = [dict(id=f'snr_{s:+d}', snr=s, rsr=12, L=None) for s in SNRS]
SPECS += [dict(id='weak_reference', snr=5, rsr=0, L=None), dict(id='two_paths', snr=5, rsr=12, L=2)]
CONFIG = dict(version=1, development_seed=2026102807, development_cases_per_cell=8,
    N=32, K=3, P=30, context_pilots=21, heldout_pilots=9, em_iterations=100,
    ranks=list(range(1,16)), relative_tau=[.01,.03,.1,.3,1.,3.],
    selection='one global hyperparameter per method; equal-cell mean withheld Rician NLL; no truth used',
    no_op='first candidate for every method; earliest candidate wins exact tie',
    cubic_scale='per-user Frobenius Hankel norm / sqrt(min Hankel dimensions)',
    placement='single final shrinkage and anti-diagonal averaging after ordinary EM100',
    optshrink='Nadakuditi Algorithm 1, reduced tail dimensions n-r and m-r; see source URL',
    source='https://arxiv.org/html/1306.6042v4', doi='10.1109/TIT.2014.2311661',
    tie_policy='zero weight at a tail pole within 1e-13 relative squared singular-value gap, the zero-weight continuous limit',
    scope='small exploratory component diagnostic, not full LRHD, end-to-end learned comparison, or hardware model',
    computation_cap_seconds=480)

def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, allow_nan=False)+'\n')

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def opt_weights(s, n, m, rank):
    """Empirical D-transform, including zero directions of rectangular tail."""
    q=min(n,m)
    if not 0 < rank < q:
        raise ValueError('OptShrink requires a nonempty noise tail and positive rank')
    s=np.asarray(s, dtype=float)
    if s.shape != (q,) or np.any(s < 0) or np.any(np.diff(s) > 1e-12):
        raise ValueError('descending nonnegative singular values required')
    weights=np.zeros_like(s)
    if s[0] == 0:
        return weights
    # Normalize only to avoid overflow/underflow; the transform is homogeneous.
    scale=s[0]; ss=s/scale; tail=ss[rank:]
    n0=n-rank; m0=m-rank; ell=q-rank
    for i,z in enumerate(ss[:rank]):
        denom=z*z-tail*tail
        if z == 0 or np.any(denom <= 1e-13*max(z*z, 1e-300)):
            continue
        trace=np.sum(z/denom)
        derivative=np.sum(-(z*z+tail*tail)/(denom*denom))
        a=(trace+(n0-ell)/z)/n0
        b=(trace+(m0-ell)/z)/m0
        ap=(derivative-(n0-ell)/(z*z))/n0
        bp=(derivative-(m0-ell)/(z*z))/m0
        weights[i]=(-2*a*b/(ap*b+a*bp))*scale
    if not np.isfinite(weights).all() or np.any(weights < 0):
        raise FloatingPointError('invalid OptShrink weight')
    return weights

def candidate_channels(g):
    """All operations share exactly the same input, lift and averaging."""
    idx,gather,count=geometry(g.shape[0])
    h=g.T[:,idx]
    u,s,vh=np.linalg.svd(h, full_matrices=False)
    def reconstruct(weights):
        hh=(u*weights[:,None,:])@vh
        flat=hh.reshape(g.shape[1],-1)
        flat=np.concatenate([flat,np.zeros((g.shape[1],1),complex)],axis=-1)
        return (flat[:,gather].sum(-1)/count).T
    out={'none':[g.copy()], 'hard':[g.copy()], 'cubic':[g.copy()], 'optshrink':[g.copy()]}
    for r in CONFIG['ranks']:
        out['hard'].append(reconstruct(s*(np.arange(s.shape[-1])<r)))
        weights=np.stack([opt_weights(row,h.shape[-2],h.shape[-1],r) for row in s])
        out['optshrink'].append(reconstruct(weights))
    scale=np.sqrt(np.mean(s*s,axis=-1))
    for relative in CONFIG['relative_tau']:
        tau=relative*scale
        weights=np.divide(s**3,s*s+tau[:,None]**2,out=np.zeros_like(s),where=(s*s+tau[:,None]**2)>0)
        out['cubic'].append(reconstruct(weights))
    return {k:np.stack(v) for k,v in out.items()}

def unit_checks():
    out={}
    s=np.array([7.,3.,0.,0.])
    out['zero_tail_max_error']=float(np.max(abs(opt_weights(s,4,7,2)-s)))
    s=np.array([9.,5.,2.,1.])
    w=opt_weights(s,4,7,2)
    out['scaling_max_relative_error']=float(np.max(abs(opt_weights(s*1e4,4,7,2)/1e4-w)))
    out['transpose_max_error']=float(np.max(abs(opt_weights(s,7,4,2)-w)))
    r=2; t=np.zeros((4-r,7-r));np.fill_diagonal(t,s[r:]);exact=[]
    for z in s[:r]:
        n,m=t.shape; eye_n=np.eye(n);eye_m=np.eye(m)
        rn=np.linalg.inv(z*z*eye_n-t@t.T);rm=np.linalg.inv(z*z*eye_m-t.T@t)
        a=np.trace(z*rn)/n;b=np.trace(z*rm)/m
        ap=np.trace(rn-2*z*z*rn@rn)/n;bp=np.trace(rm-2*z*z*rm@rm)/m
        exact.append(-2*a*b/(ap*b+a*bp))
    out['rectangular_direct_transform_max_error']=float(np.max(abs(w[:r]-exact)))
    out['zero_input_finite']=bool(np.isfinite(opt_weights(np.zeros(4),4,7,2)).all())
    out['tied_boundary_zero_weight']=bool(opt_weights(np.array([2.,1.,1.,.2]),4,7,2)[1]==0)
    for key,value in out.items():
        if isinstance(value,bool): assert value,key
        else: assert value < 1e-10,(key,value)
    out['all_passed']=True
    return out

def run():
    started=time.monotonic();ROOT.mkdir(parents=True,exist_ok=True)
    if (ROOT/'summary.json').exists():
        raise RuntimeError('completed diagnostic already exists; preserving it')
    write(ROOT/'config.json', CONFIG)
    write(ROOT/'unit_checks.json',unit_checks())
    sources=[Path(__file__),Path('codex_verification/solver.py'),Path('sahure/classical.py'),Path('sahure_expansion/data.py')]
    source_hashes={str(p):sha(p) for p in sources}
    # Selection contains observations only. The generated w.G is never read here.
    scores={k:np.zeros(len(v)) for k,v in {'none':[None], 'hard':[None]+CONFIG['ranks'],
        'cubic':[None]+CONFIG['relative_tau'],'optshrink':[None]+CONFIG['ranks']}.items()}
    dev_sources={}
    for j,spec in enumerate(SPECS):
        seed=CONFIG['development_seed']+100*j
        obs=[]
        for i in range(CONFIG['development_cases_per_cell']):
            if time.monotonic()-started > CONFIG['computation_cap_seconds']: raise TimeoutError('development cap')
            w,_=world(i,seed,N=32,K=3,P=30,snr=spec['snr'],rsr=spec['rsr'],L=spec['L'])
            obs.append((w.Z,w.S,w.B,w.sigma2))
            g=fit(w.S[:,:21],w.Z[:,:21],w.B[:,:21],w.sigma2,iterations=100)[0]
            for name,gs in candidate_channels(g).items():
                scores[name]+=likelihood(w.Z[:,21:],gs@w.S[:,21:]+w.B[:,21:],w.sigma2).mean((-2,-1))/(len(SPECS)*CONFIG['development_cases_per_cell'])
        path=ROOT/'development'/f'{spec["id"]}.npz'
        path.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(path,z=np.stack([o[0] for o in obs]),s=np.stack([o[1] for o in obs]),b=np.stack([o[2] for o in obs]),sigma2=np.array([o[3] for o in obs]),data_seed=seed)
        dev_sources[str(path)]=sha(path)
    selected={name:int(np.argmin(v)) for name,v in scores.items()}
    parameters={'none':[None], 'hard':[None]+CONFIG['ranks'], 'optshrink':[None]+CONFIG['ranks'], 'cubic':[None]+CONFIG['relative_tau']}
    frozen=dict(selection=selected,parameters={k:parameters[k][v] for k,v in selected.items()},
        scores={k:v.tolist() for k,v in scores.items()},development_sources=dev_sources,
        freeze_before_test_read=True,source_hashes=source_hashes)
    write(ROOT/'frozen_selection.json',frozen)
    print('Frozen observation-only settings: '+json.dumps(frozen['parameters']),flush=True)
    cells={};test_sources={}
    rng=np.random.default_rng(2026102891);bootstrap={}
    for spec in SPECS:
        path=BANK/f'{spec["id"]}.npz'
        if not path.exists(): raise FileNotFoundError(f'Test bank is not ready: {path}')
        test_sources[str(path)]=sha(path)
        with np.load(path) as f: data={k:f[k].copy() for k in f.files}
        if int(data['data_seed']) in [CONFIG['development_seed']+100*j for j in range(len(SPECS))]:
            raise AssertionError('split seed collision')
        energy=np.sum(abs(data['truth'])**2,axis=(-2,-1))
        estimates={k:[] for k in selected};processing={k:0. for k in selected}
        tick=time.monotonic()
        for i in range(len(energy)):
            if time.monotonic()-started > CONFIG['computation_cap_seconds']: raise TimeoutError('test cap')
            g=fit(data['s'][i],data['z'][i],data['true_b'][i],float(data['true_sigma2'][i]),iterations=100)[0]
            candidates=candidate_channels(g)
            for name,index in selected.items(): estimates[name].append(candidates[name][index])
        rows={};bs={};count=rng.multinomial(len(energy),np.full(len(energy),1/len(energy)),size=2000)
        result={'cell':spec,'trial_ids':data['trial_ids'].tolist(),'energy':energy.tolist(), 'methods':{},'elapsed_seconds':time.monotonic()-tick}
        for name,gs in estimates.items():
            gs=np.stack(gs);err=np.sum(abs(gs-data['truth'])**2,axis=(-2,-1))
            nmse=10*np.log10(err.sum()/energy.sum());samples=10*np.log10((count@err)/(count@energy))
            result['methods'][name]=dict(error=err.tolist(),per_case_nmse=(err/energy).tolist(),nmse_db=float(nmse),
                severe_fraction=float(np.mean(err/energy>1)),interval=np.quantile(samples,[.025,.975]).tolist(),parameter=frozen['parameters'][name])
            bs[name]=samples
        result['contrasts']={name:dict(gain_vs_no_filter_db=result['methods']['none']['nmse_db']-result['methods'][name]['nmse_db'],
            interval=np.quantile(bs['none']-bs[name],[.025,.975]).tolist()) for name in ('hard','cubic','optshrink')}
        write(ROOT/'cells'/f'{spec["id"]}.json',result)
        cells[spec['id']]=result;bootstrap[spec['id']]=bs
        print('Completed '+spec['id'],flush=True)
    ids=[f'snr_{s:+d}' for s in SNRS]
    macro={name:dict(nmse_db=float(np.mean([cells[c]['methods'][name]['nmse_db'] for c in ids])),
        gain_vs_no_filter_db=float(np.mean([cells[c]['methods']['none']['nmse_db']-cells[c]['methods'][name]['nmse_db'] for c in ids])),
        interval=np.quantile(np.mean([bootstrap[c]['none']-bootstrap[c][name] for c in ids],axis=0),[.025,.975]).tolist()) for name in selected}
    write(ROOT/'summary.json',dict(config=CONFIG,selected=frozen['parameters'],main_snr_macro=macro,
        cells=cells,development_sources=dev_sources,test_sources=test_sources,source_hashes=source_hashes,
        elapsed_seconds=time.monotonic()-started,uncertainty='exploratory paired-case bootstrap, unadjusted, conditional on the independent development selection',
        limitations=['72 development worlds total; 32 test worlds per cell','all methods correct EM100 only once at the end',
        'not a matched comparison to the trained ten-stage estimator','not an LRHD reproduction or OptShrink optimality validation',
        'global parameters selected on 21 context pilots and deployed on all 30; no test tuning']))
    report=['Exploratory final Hankel-filter diagnostic','',f'Frozen settings: {frozen["parameters"]}',
        'Selection used 72 independent development worlds and withheld Rician likelihood only; no channel truth.',
        'All test estimates use ordinary EM100 on all 30 pilots, followed by one structural operation.',
        '', 'Seven-SNR equal-cell macro-average (dB NMSE; improvement over no final filter):']
    for name,row in macro.items(): report.append(f'{name}: {row["nmse_db"]:.3f} dB; gain {row["gain_vs_no_filter_db"]:.3f} dB; paired 95% interval {row["interval"]}')
    report+=['','These results isolate one post-processing component. They do not rank full LRHD, the trained network, or all structured estimators.',
        'The default no-correction option can win selection; all controls have that option. OptShrink needs rank selection and a nonempty singular-value tail.',
        'Source: Nadakuditi, IEEE TIT 60(5):3002-3018,2014, doi:10.1109/TIT.2014.2311661, Algorithm1/equations16.']
    (ROOT/'REPORT.txt').write_text('\n'.join(report)+'\n')
    print(json.dumps(macro,indent=2),flush=True)

if __name__=='__main__': run()
