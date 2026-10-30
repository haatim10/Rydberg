"""Prespecified non-learning audit. All writes remain inside this directory."""
import os
from pathlib import Path
ROOT=Path(__file__).resolve().parent
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='1'
os.environ['MPLCONFIGDIR']=str(ROOT/'matplotlib_cache')
import json,hashlib,time,sys,platform
import numpy as np
from scipy.special import i0e,i1e
from PIL import Image

REPO=ROOT.parents[1]
CFG=json.loads((ROOT/'config.json').read_text())
M=32;K=4;N=CFG['cases_per_split'];MASTER=CFG['master_seed']

def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for part in iter(lambda:f.read(2**20),b''):h.update(part)
    return h.hexdigest()

def write(p,x):
    p=ROOT/p;p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');tmp.replace(p)

def cn(r,shape):return (r.standard_normal(shape)+1j*r.standard_normal(shape))/np.sqrt(2)
def pilots(p):return cn(np.random.default_rng(np.random.SeedSequence([2026102240,77])),(4,35))[:,:p]
def power(p):return float(M*np.abs(pilots(p)).__pow__(2).sum()/p)
def stats(x):return dict(mean=float(np.mean(x)),p05=float(np.quantile(x,.05)),p95=float(np.quantile(x,.95)))
def ratio(g,h):return np.sum(np.abs(g-h)**2,axis=(-2,-1))/np.sum(np.abs(h)**2,axis=(-2,-1))
def score(x):return dict(nmse_linear=float(x.mean()),nmse_db=float(10*np.log10(x.mean())),mean_per_case_db=float(np.mean(10*np.log10(x))))

def snapshot():
    paths=[]
    for prefix in ('urformer_reproduction_20261022','matched_supervision_20261023','sahure'):
        paths+=list((REPO/prefix).glob('*.py'))
    for prefix in ('urformer_reproduction_20261022','matched_supervision_20261023','tcom_reliability_20261021'):
        base=REPO/'results'/prefix
        paths+=list(base.glob('*.json'))+list(base.glob('*.txt'))
        paths+=list((base/'training').rglob('*.pt'))+list((base/'test').glob('*.npz'))
    return {str(p.relative_to(REPO)):sha(p) for p in sorted(set(paths))}

def world(split):
    path=ROOT/'data'/f'{split}.npz'
    if path.exists():
        with np.load(path) as f:return {k:f[k].copy() for k in f.files}
    sid=CFG['splits'][split];hs=[];ws=[];aa=[]
    for start in range(0,N,128):
        n=min(128,N-start)
        rng=lambda stream:np.random.default_rng(np.random.SeedSequence([MASTER,sid,start,stream]))
        angles=rng(1).uniform(-np.pi/2,np.pi/2,(n,4,4,10));gains=cn(rng(2),angles.shape)
        steering=np.exp(1j*np.pi*np.arange(M)[None,:,None,None,None]*np.sin(angles)[:,None])
        hs.append(np.sqrt(M/40)*np.sum(steering*gains[:,None],(-2,-1)))
        ws.append(cn(rng(3),(n,M,35)));aa.append(angles)
    init=cn(np.random.default_rng(np.random.SeedSequence([MASTER,sid,5])),(N,M,K))
    out=dict(h=np.concatenate(hs),w=np.concatenate(ws),angles=np.concatenate(aa),init=init)
    path.parent.mkdir(exist_ok=True);np.savez(path,**out);return out

def generate(raw,p,snr_db,snr_convention):
    """One observation generator; H is used solely to simulate physical observations."""
    s=pilots(p);signal=raw['h']@s;realized=np.mean(np.abs(signal)**2,axis=(-2,-1));ens=power(p)
    rho=10**(snr_db/10)
    if snr_convention=='ours':v=realized/rho
    elif snr_convention in ('reproduction_ensemble','paper_vector'):v=np.full(len(signal),ens/rho)
    elif snr_convention=='paper_shifted':v=realized/(M*rho)
    elif snr_convention=='unit_channel_assumed':v=np.full(len(signal),ens/(M*rho))
    elif snr_convention=='array_signal_scalar_noise':v=np.full(len(signal),M*ens/rho)
    else:raise ValueError(snr_convention)
    b=np.sqrt(10*ens);noise=raw['w'][...,:p]*np.sqrt(v[:,None,None]);y=signal+b+noise
    empirical_noise=np.mean(np.abs(noise)**2,axis=(-2,-1))
    metadata=dict(nominal_snr_db=snr_db,per_entry_channel_power=stats(np.mean(np.abs(raw['h'])**2,axis=(-2,-1))),
        per_antenna_snr_expected_noise_db=stats(10*np.log10(realized/v)),
        per_antenna_snr_realized_noise_db=stats(10*np.log10(realized/empirical_noise)),
        aggregate_realized_snr_db=float(10*np.log10(realized.mean()/empirical_noise.mean())),
        realized_RSR_db=stats(10*np.log10(b*b/realized)),aggregate_RSR_db=float(10*np.log10(b*b/realized.mean())),
        noise_variance=stats(v),ensemble_signal_power=ens)
    return dict(z=np.abs(y),s=s,b=b,v=v,y=y,meta=metadata)

def digitize():
    old=REPO/'results/urformer_reproduction_20261022'
    existing=json.loads((old/'published_curve_approx.json').read_text())
    refs={};specs={
       'fig3':dict(x0=105.5,x1=1035.,xmin=-10,xmax=20,y0=179.,scale=-10/(315.5-179.),x=[-10,-5,0,5,10,15,20],gs_y=[38,109,180,251,327,395,457]),
       'fig4':dict(x0=105.5,x1=1034.,xmin=10,xmax=35,y0=159.,scale=-5/(252.-159.),x=[10,15,20,25,30,35],gs_y=[145,212,260,294,318,337])}
    for name,c in specs.items():
        path=old/'reference'/f'{name}.png';a=np.array(Image.open(path).convert('RGB')).astype(int)
        mask=(np.abs(a-np.array([0,113,175])).max(-1)<10);out=[]
        for x,yy in zip(c['x'],c['gs_y']):
            xp=c['x0']+(x-c['xmin'])/(c['xmax']-c['xmin'])*(c['x1']-c['x0'])
            ys,xs=np.where(mask & (np.abs(np.arange(a.shape[1])[None]-xp)<3)&(np.abs(np.arange(a.shape[0])[:,None]-yy)<12))
            assert len(ys)>0,(name,x)
            center=(float(ys.min())+float(ys.max()))/2
            other=next(r for r in existing['curves'][name] if r['x']==x)
            out.append(dict(x=x,GS100=round((center-c['y0'])*c['scale'],1),EM100=other['em_nmse_db_approx'],URformer=other['nmse_db_approx'],GS_pixel_y=center))
        refs[name]=out
    write('reference/curve_readings.json',dict(source='https://arxiv.org/pdf/2509.12586v3',uncertainty_db_approx=.2,settings=specs,curves=refs,
         note='Approximate raster readings, not raw author data. Fig4 P15 URformer is about -20.2 dB, not -25 dB.'))
    return refs

def task1():
    raw=world('oracle');h=raw['h'];p=20;s=pilots(p);op=np.linalg.pinv(s)
    e=raw['w'][...,:p]@op;projected_e=np.empty_like(e);projected_h=np.empty_like(h);ranks=[]
    for i in range(N):
        for k in range(K):
            a=np.exp(1j*np.pi*np.arange(M)[:,None]*np.sin(raw['angles'][i,k].reshape(1,40)))
            u,d,_=np.linalg.svd(a,full_matrices=False);rank=int(np.count_nonzero(d>max(a.shape)*np.finfo(float).eps*d[0]));ranks.append(rank)
            q=u[:,:rank];projected_e[i,:,k]=q@(q.conj().T@e[i,:,k]);projected_h[i,:,k]=q@(q.conj().T@h[i,:,k])
    rows=[]
    for snr in CFG['snr_db']:
        obs=generate(raw,p,snr,'reproduction_ensemble');v=obs['v'];ls=(obs['y']-obs['b'])@op
        np.testing.assert_allclose(ls,h+np.sqrt(v[:,None,None])*e,rtol=1e-11,atol=1e-11)
        oracle=projected_h+np.sqrt(v[:,None,None])*projected_e
        conditional=M*v*np.linalg.norm(op)**2/np.sum(np.abs(h)**2,axis=(-2,-1))
        ref=K/((p-K)*10**(snr/10))
        rr=dict(snr_db=snr,genie_LS=score(ratio(ls,h)),oracle_DoA_projection=score(ratio(oracle,h)),
                fixed_pilot_conditional_theory_linear=float(conditional.mean()),fixed_pilot_conditional_theory_db=float(10*np.log10(conditional.mean())),
                requested_random_pilot_theory_linear=ref,requested_random_pilot_theory_db=float(10*np.log10(ref)),calibration=obs['meta'])
        rows.append(rr)
        np.savez_compressed(ROOT/'data'/f'oracle_snr{snr}.npz',LS=ratio(ls,h),projection=ratio(oracle,h),conditional=conditional)
    out=dict(rows=rows,steering_rank_counts={str(r):int(np.count_nonzero(np.array(ranks)==r)) for r in sorted(set(ranks))},
         projection_truth_max_relative_error=float(np.max(np.linalg.norm(projected_h-h,axis=(-2,-1))/np.linalg.norm(h,axis=(-2,-1)))),
         reachability='Inconclusive: LS and subspace-projected LS are not lower bounds for biased/Bayesian learned estimators. These tests alone cannot prove -21.6 dB impossible.',
         theory_note='K/((P-K)*SNR) is the usual random-Gaussian-pilot ensemble reference (and a ratio-of-expected-energies approximation), not exact for this one fixed pilot bank and mean-per-channel NMSE. Exact conditional fixed-pilot expectation is also given.')
    write('task1_oracle.json',out);print('Task1 complete',[(r['snr_db'],r['genie_LS']['nmse_db'],r['oracle_DoA_projection']['nmse_db']) for r in rows],flush=True)
    return out

def classical(obs,init):
    s=obs['s'];op=np.linalg.pinv(s);v=obs['v'][:,None,None];b=obs['b'];z=obs['z'];gs=init.copy();em=init.copy()
    for _ in range(100):
        mu=gs@s+b;gs=(z*mu/np.maximum(np.abs(mu),1e-8)-b)@op
        mu=em@s+b;amplitude=np.abs(mu);kap=2*z*amplitude/v
        em=(z*mu/np.maximum(amplitude,1e-8)*(i1e(kap)/i0e(kap))-b)@op
    return dict(GS100=gs,EM100=em)

def references(refs):
    return [(20,int(r['x']),r,'fig3') for r in refs['fig3'] if r['x'] in CFG['snr_db']]+[(int(r['x']),5,r,'fig4') for r in refs['fig4']]

def task23(refs):
    grid=sorted(set((p,s) for p,s,r,f in references(refs)));all_rows=[];ranking=None
    for split in ('validation','confirmation'):
        raw=world(split)
        a=generate(raw,20,5,'reproduction_ensemble');b=generate(raw,20,5,'paper_vector')
        assert np.array_equal(a['z'],b['z']) and np.array_equal(a['v'],b['v'])
        for convention in CFG['conventions']:
            for p,snr in grid:
                filename=f'{split}_{convention}_P{p}_snr{snr}';dest=ROOT/'cells'/f'{filename}.json'
                if dest.exists():all_rows.append(json.loads(dest.read_text()));continue
                obs=generate(raw,p,snr,convention);start=time.monotonic()
                if convention=='paper_vector':
                    source=next(x for x in all_rows if x['split']==split and x['convention']=='reproduction_ensemble' and x['P']==p and x['snr_db']==snr)
                    rr=dict(source,convention=convention,equivalent_to='reproduction_ensemble')
                    values=np.load(ROOT/'cells'/f'{split}_reproduction_ensemble_P{p}_snr{snr}.npz')
                    errors={n:values[n] for n in ('GS100','EM100')}
                else:
                    pred=classical(obs,raw['init']);errors={k:ratio(g,raw['h']) for k,g in pred.items()}
                    assert all(np.isfinite(v).all() for v in errors.values())
                    rr=dict(split=split,convention=convention,P=p,snr_db=snr,metrics={k:score(v) for k,v in errors.items()},calibration=obs['meta'],seconds=time.monotonic()-start)
                (ROOT/'cells').mkdir(exist_ok=True);np.savez_compressed(ROOT/'cells'/f'{filename}.npz',**errors)
                write('cells/'+filename+'.json',rr);all_rows.append(rr)
                print('Classical',split,convention,p,snr,{k:round(v['nmse_db'],3) for k,v in rr['metrics'].items()},flush=True)
        if split=='validation':
            ranking=rank(all_rows,refs,'validation');write('validation_selection.json',ranking)
            print('Validation-only convention ranking',ranking,flush=True)
    result=dict(rows=all_rows,validation_ranking=ranking,confirmation_ranking=rank(all_rows,refs,'confirmation'))
    write('task23_classical.json',result);return result

def rank(rows,refs,split):
    out=[]
    for c in CFG['conventions']:
        diffs=[];cells=[]
        for p,s,r,f in references(refs):
            row=next(x for x in rows if x['split']==split and x['convention']==c and x['P']==p and x['snr_db']==s)
            for n in ('GS100','EM100'):
                diff=row['metrics'][n]['nmse_db']-r[n];diffs.append(diff)
                cells.append(dict(figure=f,P=p,snr_db=s,method=n,measured_db=row['metrics'][n]['nmse_db'],published_approx_db=r[n],signed_error_db=diff,absolute_error_db=abs(diff)))
        rmse=float(np.sqrt(np.mean(np.square(diffs))));mx=float(max(map(abs,diffs)))
        out.append(dict(convention=c,rmse_db=rmse,mean_abs_error_db=float(np.mean(np.abs(diffs))),max_abs_error_db=mx,
             within_predeclared_tolerance=rmse<=1.5 and mx<=3.,points=cells))
    return sorted(out,key=lambda x:x['rmse_db'])

def main():
    assert not (ROOT/'complete.json').exists(),'Preserve completed study; do not silently rerun.'
    old=ROOT/'protected_before.json'
    before=json.loads(old.read_text()) if old.exists() else snapshot()
    if not old.exists():write('protected_before.json',before)
    write('protocol_receipt.json',dict(config_sha256=sha(ROOT/'config.json'),script_sha256=sha(__file__),python=sys.version,numpy=np.__version__,platform=platform.platform(),started_unix=time.time()))
    refs=digitize();oracle=task1();result=task23(refs)
    after=snapshot();assert before==after,'An earlier source/checkpoint/result changed'
    write('protected_verification.json',dict(passed=True,files=len(before)))
    write('compute_complete.json',dict(passed=True,oracle_draws=N,validation_draws=N,confirmation_draws=N,config_sha256=sha(ROOT/'config.json'),script_sha256=sha(__file__),finished_unix=time.time()))
    print('All non-learning computations completed. Earlier files unchanged.',flush=True)

if __name__=='__main__':main()
