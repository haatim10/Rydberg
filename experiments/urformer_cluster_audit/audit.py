"""Cluster-law audit. This module never writes outside its own directory."""
from pathlib import Path
import os
ROOT=Path(__file__).resolve().parent
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='1'
os.environ['MPLCONFIGDIR']=str(ROOT/'matplotlib_cache')
import json,hashlib,time,sys,platform,multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from scipy.special import i0e,i1e,ndtri,j0
REPO=ROOT.parents[1]
CFG=json.loads((ROOT/'config.json').read_text());M=32;K=4;MASTER=CFG['master_seed']

def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
def write(name,obj):
    p=ROOT/name;p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp')
    t.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');t.replace(p)
def cn(r,shape):return (r.standard_normal(shape)+1j*r.standard_normal(shape))/np.sqrt(2)
def rng(split,start,stream):return np.random.default_rng(np.random.SeedSequence([MASTER,CFG['splits'][split],start,stream]))
def pilots(p):return cn(np.random.default_rng(np.random.SeedSequence(CFG['pilot_seed'])),(4,35))[:,:p]
def power(p):return float(M*np.sum(np.abs(pilots(p))**2)/p)
def generators():
    return [dict(name='independent',distribution='independent',cluster_spread_deg=None)]+[dict(name=d+'_'+str(s).replace('.','p'),distribution=d,cluster_spread_deg=s) for d in CFG['offset_distributions'] for s in CFG['spread_degrees']]
def channels(split,start,n,cluster_spread_deg=None,distribution='independent'):
    """Fresh centres per channel/user; shared random inputs across candidate generators."""
    if distribution=='independent':
        assert cluster_spread_deg is None
        angles=rng(split,start,1).uniform(-np.pi/2,np.pi/2,(n,K,4,10))
    else:
        centers=rng(split,start,4).uniform(-np.pi/2,np.pi/2,(n,K,4,1))
        u=rng(split,start,6).uniform(np.finfo(float).eps,1-np.finfo(float).eps,(n,K,4,10))
        if distribution=='laplacian':standard=-np.sign(u-.5)*np.log1p(-2*np.abs(u-.5))/np.sqrt(2)
        elif distribution=='gaussian':standard=ndtri(u)
        else:raise ValueError(distribution)
        offsets=np.deg2rad(cluster_spread_deg)*standard
        angles=np.arcsin(np.sin(centers+offsets))
    gains=cn(rng(split,start,2),angles.shape)
    a=np.exp(1j*np.pi*np.arange(M)[None,:,None,None,None]*np.sin(angles)[:,None])
    h=np.sqrt(M/40)*np.sum(a*gains[:,None],axis=(-2,-1))
    return h,angles

def stats(x):
    x=np.asarray(x);return dict(mean=float(x.mean()),p05=float(np.quantile(x,.05)),p95=float(np.quantile(x,.95)),min=float(x.min()),max=float(x.max()))
def rank_metrics(eigenvalues):
    a=np.maximum(eigenvalues,0);p=a/a.sum(-1,keepdims=True)
    entropy=np.exp(-np.sum(p*np.log(np.maximum(p,np.finfo(float).tiny)),axis=-1))
    participation=1/np.sum(p*p,axis=-1)
    return entropy,participation

def common(split):
    path=ROOT/'data'/f'{split}_common.npz'
    if not path.exists():
        n=CFG[split+'_draws'];w=np.concatenate([cn(rng(split,start,3),(min(128,n-start),M,35)) for start in range(0,n,128)])
        init=cn(rng(split,0,5),(n,M,K));path.parent.mkdir(exist_ok=True)
        np.savez(path,w=w,init=init)
    with np.load(path) as f:return {k:f[k].copy() for k in f.files}

def raw_channels(spec,split):
    path=ROOT/'data'/f'{split}_{spec["name"]}.npz'
    if path.exists():
        with np.load(path) as f:return {k:f[k].copy() for k in f.files}
    n=CFG[split+'_draws'];hh=[];aa=[]
    for start in range(0,n,128):
        h,a=channels(split,start,min(128,n-start),spec['cluster_spread_deg'],spec['distribution']);hh.append(h);aa.append(a)
    raw=dict(h=np.concatenate(hh),angles=np.concatenate(aa));np.savez(path,**raw);return raw

def covariance(spec):
    path=ROOT/'covariance'/f'{spec["name"]}.npz'
    if path.exists():
        with np.load(path) as f:return f['R'].copy()
    n=CFG['covariance_draws'];sums=np.zeros((K,M,M),complex);means=np.zeros((M,K),complex);digest=hashlib.sha256()
    for start in range(0,n,128):
        h,a=channels('covariance',start,min(128,n-start),spec['cluster_spread_deg'],spec['distribution'])
        sums+=np.einsum('nmk,nlk->kml',h,h.conj(),optimize=True);means+=h.sum(0);digest.update(h.tobytes())
    per_user=sums/n;R=per_user.mean(0);lam=np.linalg.eigvalsh(R);er,pr=rank_metrics(lam)
    theory=M*j0(np.pi*(np.arange(M)[:,None]-np.arange(M)[None,:]))
    te,tp=rank_metrics(np.linalg.eigvalsh(theory));tol=M*np.finfo(float).eps*lam[-1]
    record=dict(generator=spec,draws=n,user_vectors=n*K,channel_stream_sha256=digest.hexdigest(),numerical_rank=int(np.count_nonzero(lam>tol)),effective_rank_entropy=float(er),effective_rank_participation=float(pr),
        mean_per_entry_power=float(np.trace(R).real/M),mean_channel_absolute=float(np.abs(means/n).max()),
        eigenvalues=lam.tolist(),relative_distance_to_identity=float(np.linalg.norm(R-M*np.eye(M))/np.linalg.norm(M*np.eye(M))),
        relative_distance_to_uniform_angle_theory=float(np.linalg.norm(R-theory)/np.linalg.norm(theory)),
        analytic_uniform_angle=dict(covariance='R[m,n]=32*J0(pi*(m-n))',effective_rank_entropy=float(te),effective_rank_participation=float(tp),adjacent_correlation=float(j0(np.pi)),source='https://dlmf.nist.gov/10.9.E1'))
    path.parent.mkdir(exist_ok=True);np.savez(path,R=R,per_user_R=per_user,sample_mean=means/n,theory=theory)
    write('covariance/'+spec['name']+'.json',record)
    print('Covariance',spec['name'],'power',round(record['mean_per_entry_power'],3),'effective rank',round(float(er),3),flush=True)
    return R

def structure(spec,raw,split):
    path=ROOT/'structure'/f'{split}_{spec["name"]}.json'
    if path.exists():return json.loads(path.read_text())
    ranks=[];ers=[];prs=[]
    for start in range(0,len(raw['h']),64):
        angle=raw['angles'][start:start+64].reshape(-1,K,40)
        A=np.exp(1j*np.pi*np.arange(M)[None,None,:,None]*np.sin(angle)[:,:,None,:])
        sv=np.linalg.svd(A,compute_uv=False);tol=40*np.finfo(float).eps*sv[...,0]
        ranks.append(np.sum(sv>tol[...,None],axis=-1));er,pr=rank_metrics(sv**2);ers.append(er);prs.append(pr)
    ranks=np.concatenate(ranks);ers=np.concatenate(ers);prs=np.concatenate(prs)
    hs=np.linalg.svd(raw['h'],compute_uv=False);her,hpr=rank_metrics(hs**2)
    hr=np.sum(hs>32*np.finfo(float).eps*hs[:,:1],axis=-1)
    out=dict(generator=spec,split=split,user_channels=int(ranks.size),conditional_steering_numerical_rank=stats(ranks),conditional_steering_rank_counts={str(int(x)):int((ranks==x).sum()) for x in np.unique(ranks)},conditional_effective_rank_entropy=stats(ers),conditional_effective_rank_participation=stats(prs),M_by_K_channel_numerical_rank=stats(hr),M_by_K_channel_effective_rank_entropy=stats(her),M_by_K_channel_effective_rank_participation=stats(hpr),channel_per_entry_power=stats(np.mean(np.abs(raw['h'])**2,axis=(-2,-1))))
    path.parent.mkdir(exist_ok=True);np.savez_compressed(path.with_suffix('.npz'),rank=ranks,entropy_rank=ers,participation_rank=prs,H_rank=hr,H_entropy_rank=her,H_participation_rank=hpr)
    write(str(path.relative_to(ROOT)),out);return out

def observation(h,shared,p,snr):
    s=pilots(p);v=power(p)/10**(snr/10);b=np.sqrt(10*power(p));signal=h@s;w=shared['w'][...,:p]*np.sqrt(v);y=signal+b+w
    sig=np.mean(np.abs(signal)**2,axis=(-2,-1));noise=np.mean(np.abs(w)**2,axis=(-2,-1))
    return dict(z=np.abs(y),y=y,s=s,v=v,b=b,calibration=dict(mean_per_entry_channel_power=float(np.mean(np.abs(h)**2)),aggregate_realized_snr_db=float(10*np.log10(sig.mean()/noise.mean())),mean_realized_snr_db=float(np.mean(10*np.log10(sig/noise))),aggregate_RSR_db=float(10*np.log10(b*b/sig.mean())),mean_RSR_db=float(np.mean(10*np.log10(b*b/sig))),noise_variance=v))

def classical(obs,init):
    s=obs['s'];op=np.linalg.pinv(s);v=obs['v'];b=obs['b'];z=obs['z'];gs=init.copy();em=init.copy()
    for _ in range(100):
        mu=gs@s+b;gs=(z*mu/np.maximum(np.abs(mu),1e-8)-b)@op
        mu=em@s+b;am=np.abs(mu);kappa=2*z*am/v
        em=(z*mu/np.maximum(am,1e-8)*(i1e(kappa)/i0e(kappa))-b)@op
    return dict(GS100=gs,EM100=em)

def genie(obs,R):
    s=obs['s'];ls=(obs['y']-obs['b'])@np.linalg.pinv(s)
    lam,U=np.linalg.eigh(R);d,V=np.linalg.eigh(s@s.conj().T);lam=np.maximum(lam,0)
    weight=(lam[:,None]*d[None,:])/(lam[:,None]*d[None,:]+obs['v'])
    transformed=U.conj().T@ls@V
    est=U@(transformed*weight)@V.conj().T
    return dict(genie_LS=ls,genie_LMMSE=est)

def ratio(pred,h):return np.sum(np.abs(pred-h)**2,axis=(-2,-1))/np.sum(np.abs(h)**2,axis=(-2,-1))
def score(x):
    m=x.mean();se=x.std(ddof=1)/np.sqrt(len(x))
    return dict(nmse_linear=float(m),nmse_db=float(10*np.log10(m)),mean_per_channel_db=float(np.mean(10*np.log10(x))),mean_linear_standard_error=float(se),approximate_mean_NMSE_95_db=[float(10*np.log10(max(m-1.96*se,1e-300))),float(10*np.log10(m+1.96*se))])
def refs():return json.loads((ROOT/'published_curves.json').read_text())['curves']
def reference_points():
    r=refs();return [(20,int(x['x']),x,'fig3') for x in r['fig3'] if x['x'] in CFG['snr_db']]+[(int(x['x']),5,x,'fig4') for x in r['fig4']]
def grid():return sorted(set((p,s) for p,s,r,f in reference_points()))

def worker(spec,split):
    R=covariance(spec);raw=raw_channels(spec,split);shared=common(split);structure(spec,raw,split)
    for p,snr in grid():
        name=f'{split}_{spec["name"]}_P{p}_snr{snr}';path=ROOT/'cells'/f'{name}.json'
        if path.exists():continue
        obs=observation(raw['h'],shared,p,snr);start=time.monotonic();pred=classical(obs,shared['init']);pred.update(genie(obs,R))
        errors={k:ratio(v,raw['h']) for k,v in pred.items()};assert all(np.isfinite(v).all() for v in errors.values())
        record=dict(generator=spec,split=split,P=p,snr_db=snr,n=len(raw['h']),metrics={k:score(v) for k,v in errors.items()},calibration=obs['calibration'],seconds=time.monotonic()-start,covariance_sha256=sha(ROOT/'covariance'/f'{spec["name"]}.npz'))
        path.parent.mkdir(exist_ok=True);np.savez_compressed(path.with_suffix('.npz'),**errors);write(str(path.relative_to(ROOT)),record)
        print('Cell',split,spec['name'],p,snr,{k:round(v['nmse_db'],3) for k,v in record['metrics'].items()},flush=True)
    return spec['name']

def gather(split,names=None):
    rows=[json.loads(p.read_text()) for p in sorted((ROOT/'cells').glob(split+'_*.json'))]
    return rows if names is None else [r for r in rows if r['generator']['name'] in names]
def assess(spec,rows):
    lookup={(r['P'],r['snr_db']):r for r in rows if r['generator']['name']==spec['name']};diffs=[];targets=[];allpoints=[]
    for p,s,r,f in reference_points():
        cell=lookup[p,s]
        for method in ('GS100','EM100'):
            measured=cell['metrics'][method]['nmse_db'];diff=measured-r[method];diffs.append(diff)
            allpoints.append(dict(figure=f,P=p,snr_db=s,method=method,computed_db=measured,published_db=r[method],absolute_error_db=abs(diff)))
        if f=='fig4' or s in (5,20):
            measured=cell['metrics']['genie_LMMSE']['nmse_db'];targets.append(dict(figure=f,P=p,snr_db=s,lmmse_db=measured,published_URformer_db=r['URformer'],gap_db=measured-r['URformer'],passes=measured<=r['URformer']))
    rmse=float(np.sqrt(np.mean(np.square(diffs))));mx=float(np.max(np.abs(diffs)));gaps=[r['gap_db'] for r in targets]
    classical_pass=rmse<=CFG['classical_tolerance']['joint_RMSE_db'] and mx<=CFG['classical_tolerance']['max_absolute_error_db']
    lmmse_pass=max(gaps)<=0
    return dict(generator=spec,classical_rmse_db=rmse,classical_max_error_db=mx,classical_pass=classical_pass,lmmse_pass=lmmse_pass,passes_both=classical_pass and lmmse_pass,worst_lmmse_target_gap_db=max(gaps),mean_lmmse_target_gap_db=float(np.mean(gaps)),lmmse_pass_with_02db_reading_allowance=max(gaps)<=.2,targets=targets,classical_points=allpoints)

def protected_snapshot():
    paths=[REPO/p for p in json.loads((REPO/'experiments/urformer_snr_audit/protected_before.json').read_text())]
    paths+=list((REPO/'experiments/urformer_snr_audit').rglob('*'))
    return {str(p.relative_to(REPO)):sha(p) for p in sorted(set(paths)) if p.is_file()}
def verify_protected():
    old=json.loads((ROOT/'protected_before.json').read_text());new=protected_snapshot();assert old==new,'Protected files changed'
    return len(new)

def main():
    assert not (ROOT/'compute_complete.json').exists(),'Completed audit is immutable; do not silently rerun.'
    receipt=ROOT/'protocol_receipt.json'
    if receipt.exists():
        old=json.loads(receipt.read_text());assert old['config_sha256']==sha(ROOT/'config.json') and old['audit_script_sha256']==sha(__file__)
    else:
        assert json.loads((ROOT/'preflight_checks.json').read_text())['passed']
        write('protocol_receipt.json',dict(config_sha256=sha(ROOT/'config.json'),audit_script_sha256=sha(__file__),checks_sha256=sha(ROOT/'checks.py'),started_unix=time.time(),python=sys.version,numpy=np.__version__,platform=platform.platform()))
        if not (ROOT/'protected_before.json').exists():write('protected_before.json',protected_snapshot())
        else:verify_protected()
    common('validation')
    with ProcessPoolExecutor(max_workers=CFG['workers'],mp_context=mp.get_context('spawn')) as pool:
        futures=[pool.submit(worker,s,'validation') for s in generators()]
        for f in as_completed(futures):print('Generator complete',f.result(),flush=True)
    rows=gather('validation');assessments=[assess(s,rows) for s in generators()]
    candidates=[a for a in assessments if a['generator']['name']!='independent']
    eligible=[a for a in candidates if a['passes_both']]
    classmatch=[a for a in candidates if a['classical_pass']]
    pool=eligible or classmatch or candidates
    selected=min(pool,key=lambda a:(a['worst_lmmse_target_gap_db'],a['mean_lmmse_target_gap_db'],a['generator']['name'])) if eligible or classmatch else min(pool,key=lambda a:a['classical_rmse_db'])
    selection=dict(selected_generator=selected['generator'],qualified_for_confirmation=bool(eligible),mode='qualification' if eligible else 'negative_confirmation_only',ranked_validation=sorted(assessments,key=lambda a:(not a['passes_both'],not a['classical_pass'],a['worst_lmmse_target_gap_db'])),selected_before_confirmation=True,selection_unix=time.time())
    write('validation_selection.json',selection)
    print('Validation selection',selection['selected_generator'],selection['mode'],flush=True)
    common('confirmation');confirm=[selected['generator'],generators()[0]]
    with ProcessPoolExecutor(max_workers=2,mp_context=mp.get_context('spawn')) as pool:
        futures=[pool.submit(worker,s,'confirmation') for s in confirm]
        for f in as_completed(futures):print('Confirmation complete',f.result(),flush=True)
    cr=gather('confirmation');ca=[assess(s,cr) for s in confirm]
    training_gate=selected['passes_both'] and ca[0]['passes_both']
    out=dict(config=CFG,published=refs(),validation_selection=selection,validation=assessments,confirmation=ca,rows=rows+cr,
      covariances=[json.loads(p.read_text()) for p in sorted((ROOT/'covariance').glob('*.json'))],structure=[json.loads(p.read_text()) for p in sorted((ROOT/'structure').glob('*.json'))],training_gate_passed=training_gate)
    write('numerical_results.json',out);count=verify_protected()
    write('compute_complete.json',dict(passed=True,training_gate_passed=training_gate,protected_files_unchanged=count,rows=len(rows+cr),finished_unix=time.time(),audit_script_sha256=sha(__file__),config_sha256=sha(ROOT/'config.json')))
    print('Non-learning audit complete; training gate:',training_gate,'protected:',count,flush=True)

if __name__=='__main__':main()
