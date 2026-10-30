"""Frozen, resumable paper-setting reproduction attempt (not author code)."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[k] = '1'
import argparse, hashlib, json, platform, time
from pathlib import Path
import numpy as np
import torch
from .model import URformer, Layer, ratios

REPO = Path(__file__).resolve().parents[1]
CODE = Path(__file__).resolve().parent
OUT = REPO/'results/urformer_reproduction_20261022'
SEEDS = (2026102201, 2026102202, 2026102203)
COUNTS = {'train':20000, 'val':2000, 'test':2000}
MASTER = 2026102240
SNR = (-10., -5., 0., 5., 10., 15., 20.)
PILOTS = (10, 15, 20, 25, 30, 35)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(p, value):
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n'); tmp.replace(p)


def rng(*keys):
    return np.random.default_rng(np.random.SeedSequence([MASTER, *keys]))


def cn(r, shape):
    return (r.standard_normal(shape)+1j*r.standard_normal(shape))/np.sqrt(2)


def pilots(p):
    # One fixed Gaussian pilot bank, prefix coupled across the pilot sweep.
    return cn(rng(77), (4, 35))[:, :p]


def power(p):
    return 32.*np.square(np.abs(pilots(p))).sum()/p


def checkpoint(p, value):
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix('.tmp.pt'); torch.save(value, tmp); tmp.replace(p)


def verify_protocol():
    p = json.loads((OUT/'protocol.json').read_text())
    for name, h in p['source_hashes'].items():
        assert sha(REPO/name) == h, name
    for name, h in p['protected_hashes'].items():
        assert sha(REPO/name) == h, name
    return p


def freeze():
    assert not (OUT/'protocol.json').exists()
    prior = ['results/tcom_reliability_20261021/protocol.json',
             'results/tcom_reliability_20261021/completion.json',
             'results/tcom_review_checks_20261022/completion.json',
             'results/tcom_review_checks_20261022/urformer/protocol.json']
    files = sorted(CODE.glob('*.py'))
    reference = sorted((OUT/'reference').glob('*'))
    protocol = dict(
        purpose='Independent reproduction attempt of URformer v3 explicit equations and published physical/training settings; not an exact author-code replication.',
        paper='https://arxiv.org/html/2509.12586v3',
        published=dict(M=32,K=4,clusters=4,rays_per_cluster=10,doa='Uniform[-pi/2,pi/2]',
            steering='Eq2 positive exponential, unnormalized lengthM',channel='Eq1 sqrt(M/40) sum g*a',
            T_UR=10,L_enc=3,d_model=64,n_train=20000,batch=32,epochs=50,Adam_lr=.001,
            scheduler='CosineAnnealingLR Tmax50',loss='Eq20 mean normalized per-example squared channel error',
            initializer='Independent zero-mean complex Gaussian normalized variance',
            reference_ratio_db=10,ratio_denominator='Expected total multiuser signal power E|Hs|^2',
            figure3=dict(P=20,snr_db=SNR),figure4=dict(snr_db=5,P=PILOTS),classical_iterations=100),
        assumptions_not_specified_by_paper=dict(
            code_search='No author repository or weights linked in arXiv source/abstract; exact title+URformer+GitHub searches found none. Not proof that none exist.',
            gains='Independent CN(0,1); no extra channel Frobenius normalization; expected per-entry channel power M=32 with printed unnormalized steering',
            channel_angles='All40 ray angles independent uniform as TableI; no invented cluster centers or angular spread',
            spacing='d=lambda/2',gain='Known transduction scalar divided out of observations (G=1); NMSE invariant to known scaling',
            pilots='One fixed CN(0,1)4x35 bank; use firstP columns. No Gram screening, orthogonalization, or oracle pilot design.',
            reference='Real positive broadside constant across antennas/pilots; amplitude sqrt(10*expected signal power) for eachP',
            snr='Ensemble expected signal power M/P sum|S|^2, not realized per-channel truth; variance=power/10^(SNR/10)',
            training_snr='One model per seed trained on20k examples with uniform continuous SNR[-10,20] and P20. Distribution not given by authors.',
            training_initialization='CN(0,1), independently redrawn each epoch; fixed independent draws for validation/test, shared across all methods',
            transformer='4heads, FFNwidth256/GELU, zero dropout, linear output projection with bias. Standard random PyTorch weights; positionalembeddingN(0,.02²).',
            exact_printed_encoder='A=MHSA(LN(V))+V; V=FFN(A)+A. No extra FFN LayerNorm or final LayerNorm.',
            filter='Raw kappa input (no log transform), hidden32/ReLU, sigmoid output; random weights',
            gates='g initialized0, alpha=.5, untied weights across10layers',
            optimizer='Adam defaults betas(.9,.999),eps1e-8,no weightdecay,gradientnormclipping1; no warmstarts/deep supervision',
            validation='2000 independently generated cases; primary final50epoch weights, best validation checkpoint additionally reported. Never choose using test.',
            test='2000 independent channels reused across13figure positions, with common noise draws/initializers; observedP20,SNR5point used by bothfigures',
            figure4='Diagnostic transfer of sameP20-trainedweights toP10..35; paper does not say whether separate models were trained perP',
            inference='Float64 batched accuracy; same randominit forURformer,GS100,EM100; standalone URformer no EMtail/Hankel/recovery',
            curves='Compare URformer and two specified classical methods. CNN/blackboxTransformer architectures not specified enough for exact reproduction; excluded.'),
        seeds=SEEDS,counts=COUNTS,master=MASTER,
        acceptance='Do not label published performance reproduced merely because training completes. Report differences to digitized published curves and unknown settings honestly. No test-guided parameter changes.',
        source_hashes={str(p.relative_to(REPO)):sha(p) for p in files},
        reference_hashes={p.name:sha(p) for p in reference},
        protected_hashes={p:sha(REPO/p) for p in prior},
        environment=dict(python=platform.python_version(),torch=torch.__version__,cpu=True,threads_per_worker=1),created_unix=time.time())
    write(OUT/'protocol.json', protocol)
    print('Protocol frozen', sha(OUT/'protocol.json'), flush=True)


def generate():
    verify_protocol()
    folder = OUT/'data'; folder.mkdir(parents=True, exist_ok=True)
    for split_id,(split,n) in enumerate(COUNTS.items(),1):
        path=folder/f'{split}.npz'
        if path.exists(): continue
        hs=[]; ws=[]
        for start in range(0,n,128):
            count=min(128,n-start)
            angles=rng(split_id,start,1).uniform(-np.pi/2,np.pi/2,(count,4,4,10))
            gains=cn(rng(split_id,start,2),angles.shape)
            steering=np.exp(1j*np.pi*np.arange(32)[None,:,None,None,None]*np.sin(angles)[:,None])
            h=np.sqrt(32/40)*np.sum(steering*gains[:,None],axis=(-2,-1))
            hs.append(h);ws.append(cn(rng(split_id,start,3),(count,32,35)))
        h=np.concatenate(hs);w=np.concatenate(ws)
        snr=rng(split_id,4).uniform(-10,20,n)
        init=cn(rng(split_id,5),(n,32,4))
        np.savez(path,h=h.astype(np.complex64),w=w.astype(np.complex64),
                 snr=snr.astype(np.float32),init=init.astype(np.complex64))
        print('Generated',split,n,flush=True)
    files={p.name:sha(p) for p in folder.glob('*.npz')}
    write(folder/'manifest.json',dict(files=files,protocol_sha256=sha(OUT/'protocol.json')))


def data(split):
    manifest=json.loads((OUT/'data/manifest.json').read_text())
    path=OUT/'data'/f'{split}.npz'; assert sha(path)==manifest['files'][path.name]
    with np.load(path) as a:return {k:torch.from_numpy(a[k].copy()) for k in a.files}


def observation(a,p=20,snr=None,double=False):
    cdtype=torch.complex128 if double else torch.complex64
    rdtype=torch.float64 if double else torch.float32
    s=torch.tensor(pilots(p),dtype=cdtype)
    b=torch.tensor(np.sqrt(10*power(p)),dtype=rdtype)
    signal_snr=a['snr'] if snr is None else torch.full((len(a['h']),),float(snr))
    variance=(power(p)/10**(signal_snr.double()/10)).to(rdtype)
    h=a['h'].to(cdtype);w=a['w'][...,:p].to(cdtype)*variance[:,None,None].sqrt()
    z=(h@s+b+w).abs();pinv=torch.linalg.pinv(s)
    return z,s,b,variance,pinv


@torch.no_grad()
def validation(model,a):
    model.eval();ob=observation(a);all_ratios=[]
    for start in range(0,len(a['h']),32):
        ids=slice(start,start+32)
        z,s,b,v,pinv=ob
        pred=model(a['init'][ids],z[ids],s,b,v[ids],pinv)
        all_ratios.append(ratios(pred,a['h'][ids]))
    r=torch.cat(all_ratios).double()
    assert torch.isfinite(r).all()
    return float(r.mean())


def train(seed):
    verify_protocol();torch.manual_seed(seed)
    a=data('train');va=data('val');ob=observation(a)
    m=URformer();opt=torch.optim.Adam(m.parameters(),lr=.001)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=50)
    folder=OUT/'training'/str(seed);latest=folder/'latest.pt'
    if latest.exists():
        state=torch.load(latest,map_location='cpu',weights_only=False)
        assert state['protocol_sha256']==sha(OUT/'protocol.json')
        m.load_state_dict(state['model']);opt.load_state_dict(state['optimizer']);sched.load_state_dict(state['scheduler'])
    else:
        score=validation(m,va)
        state=dict(seed=seed,epoch=0,best_score=score,history=[],protocol_sha256=sha(OUT/'protocol.json'),
                   model=m.state_dict(),optimizer=opt.state_dict(),scheduler=sched.state_dict())
        checkpoint(folder/'best.pt',state);checkpoint(latest,state)
    for epoch in range(state['epoch'],50):
        tick=time.monotonic();m.train()
        order=torch.randperm(len(a['h']),generator=torch.Generator().manual_seed(seed+1000+epoch))
        r=torch.Generator().manual_seed(seed+2000+epoch)
        init=torch.complex(torch.randn(len(a['h']),32,4,generator=r),torch.randn(len(a['h']),32,4,generator=r))/np.sqrt(2)
        total=0.;gradmax=0.
        for start in range(0,len(order),32):
            ids=order[start:start+32];z,s,b,v,pinv=ob
            opt.zero_grad(set_to_none=True)
            pred=m(init[ids],z[ids],s,b,v[ids],pinv)
            loss=ratios(pred,a['h'][ids]).mean();assert torch.isfinite(loss)
            loss.backward();norm=torch.nn.utils.clip_grad_norm_(m.parameters(),1.,error_if_nonfinite=True)
            gradmax=max(gradmax,float(norm));opt.step();total+=float(loss.detach())*len(ids)
            if start%4096==0:
                write(folder/'status.json',dict(state='training',seed=seed,epoch=epoch+1,completed=start,total=len(order),seconds=time.monotonic()-tick))
        sched.step();score=validation(m,va);best=score<state['best_score']
        entry=dict(epoch=epoch+1,train_nmse=total/len(order),validation_nmse=score,validation_db=10*np.log10(score),
                   seconds=time.monotonic()-tick,max_preclip_grad=gradmax,learning_rate=sched.get_last_lr()[0])
        state['history'].append(entry);state.update(epoch=epoch+1,best_score=min(state['best_score'],score),
            model=m.state_dict(),optimizer=opt.state_dict(),scheduler=sched.state_dict())
        checkpoint(latest,state)
        if best:checkpoint(folder/'best.pt',state)
        write(folder/'history.json',dict(seed=seed,history=state['history']))
        print('Seed',seed,'epoch',epoch+1,'val',entry['validation_db'],'dB',entry['seconds'],'seconds',flush=True)
    checkpoint(folder/'final.pt',state)
    write(folder/'status.json',dict(state='complete',epoch=50,seed=seed,best_score=state['best_score']))
    verify_protocol()


def preflight():
    verify_protocol();torch.manual_seed(777)
    a=data('val');a={k:v[:256] for k,v in a.items()};z,s,b,v,pinv=observation(a,double=True)
    g=a['init'].double() if not a['init'].is_complex() else a['init'].to(torch.complex128)
    layer=Layer().double();y=g@s+b
    direct=z*torch.exp(1j*torch.angle(y))
    gs=(direct-b)@torch.linalg.pinv(s)
    em=(direct*(torch.special.i1e(2*z*y.abs()/v[:,None,None])/torch.special.i0e(2*z*y.abs()/v[:,None,None]))-b)@torch.linalg.pinv(s)
    diffs={name:float((layer(g,z,s,b,v,pinv,name)-expected).abs().max()) for name,expected in [('gs',gs),('em',em)]}
    assert max(diffs.values())<1e-10,diffs
    real_design=torch.cat((s.real.T,-s.imag.T),-1)
    assert torch.linalg.matrix_rank(real_design)==8
    # Empirical calibration is diagnostic: generation uses expectation, never target normalization.
    full=data('train');s20=torch.tensor(pilots(20),dtype=torch.complex64)
    empirical=float((full['h']@s20).abs().square().mean());expected=power(20)
    assert abs(empirical/expected-1)<.03
    assert abs(float(full['h'].abs().square().mean())/32-1)<.03
    assert float((pinv@s-torch.eye(20,dtype=torch.complex128)).abs().max())>0 # rectangular projection
    np.testing.assert_allclose((s@pinv).numpy(),np.eye(4),atol=1e-12)
    m=URformer();opt=torch.optim.Adam(m.parameters(),lr=.001);small={k:v[:32] for k,v in a.items()}
    zz,ss,bb,vv,pp=observation(small);losses=[];tick=time.monotonic()
    for _ in range(5):
        opt.zero_grad();loss=ratios(m(small['init'],zz,ss,bb,vv,pp),small['h']).mean()
        loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1.,error_if_nonfinite=True);opt.step();losses.append(float(loss.detach()))
    # A smoke fit is discarded and cannot select a protocol setting.
    assert np.isfinite(losses).all()
    with torch.no_grad():
        m.eval();batch=m(small['init'],zz,ss,bb,vv,pp)
        one=torch.cat([m(small['init'][i:i+1],zz[i:i+1],ss,bb,vv[i:i+1],pp) for i in range(3)])
    batch_error=float((batch[:3]-one).abs().max());assert batch_error<1e-3
    write(OUT/'preflight.json',dict(passed=True,classical_max_abs_errors=diffs,parameters=sum(p.numel() for p in m.parameters()),
        channel_power=float(full['h'].abs().square().mean()),expected_channel_power=32.,
        signal_power_expected=expected,signal_power_empirical=empirical,reference_db_empirical=10*np.log10(10*expected/empirical),
        pilot_complex_rank=int(torch.linalg.matrix_rank(s)),real_jacobian_linear_design_rank=int(torch.linalg.matrix_rank(real_design)),
        discarded_smoke_losses=losses,seconds_per_train_step=(time.monotonic()-tick)/5,batch_error=batch_error))
    print('Preflight passed; parameters',sum(p.numel() for p in m.parameters()),flush=True)


@torch.no_grad()
def evaluate():
    verify_protocol();a=data('test');folder=OUT/'test';folder.mkdir(parents=True,exist_ok=True)
    methods={}
    for seed in SEEDS:
        for label in ('final','best'):
            c=torch.load(OUT/'training'/str(seed)/f'{label}.pt',map_location='cpu',weights_only=False)
            m=URformer().double().eval();m.load_state_dict(c['model']);methods[f'URformer_{seed}_{label}']=m
    grid=[(20,s) for s in SNR]+[(p,5.) for p in PILOTS if p!=20]
    for p,snr in grid:
        path=folder/f'P{p}_snr{snr:g}.npz'
        if path.exists():continue
        z,s,b,v,pinv=observation(a,p,snr,double=True);h=a['h'].to(torch.complex128);g0=a['init'].to(torch.complex128)
        results={}
        for name,m in methods.items():
            preds=[]
            for start in range(0,len(h),64):
                sl=slice(start,start+64);preds.append(m(g0[sl],z[sl],s,b,v[sl],pinv))
            pred=torch.cat(preds);assert torch.isfinite(pred).all();results[name]=ratios(pred,h).numpy()
        layer=Layer().double()
        for name,classic in [('GS100','gs'),('EM100','em')]:
            pred=g0.clone()
            for _ in range(100):pred=layer(pred,z,s,b,v,pinv,classic)
            assert torch.isfinite(pred).all();results[name]=ratios(pred,h).numpy()
        np.savez(path,**results,truth_energy=h.abs().square().sum((-2,-1)).numpy(),P=p,snr=snr)
        print('Test',p,snr,{k:round(float(10*np.log10(v.mean())),3) for k,v in results.items()},flush=True)
    verify_protocol();write(folder/'completed.json',dict(passed=True,cells=len(grid)))


if __name__=='__main__':
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    q=argparse.ArgumentParser();q.add_argument('phase',choices=('freeze','generate','preflight','train','evaluate'));q.add_argument('--seed',type=int)
    args=q.parse_args();train(args.seed) if args.phase=='train' else globals()[args.phase]()
