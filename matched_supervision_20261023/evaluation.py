"""Fresh blind inference, paired NMSE/BER, shared-backend isolated timing."""
import argparse,time
import numpy as np
import torch
from sahure.math import posterior_field
from scipy.special import i0e
from tcom_followups_20261017.ber_detector import draw_data,detect,hypotheses
from .common import *
from .models import build,ratio


def load_models():
    models={}
    for arch in ARCHES:
        for mode in MODES:
            for seed in SEEDS:
                for label in ('final','best'):
                    state=torch.load(OUT/'training'/f'{arch}_{mode}'/str(seed)/f'{label}.pt',map_location='cpu',weights_only=False)
                    m=build(arch,seed,torch.float64).eval();m.load_state_dict(state['model'])
                    models[f'{arch}_{mode}_{seed}_{label}']=m
    return models


@torch.no_grad()
def classic(args,name):
    g,z,s,b,v,op=args;g=g.clone()
    steps=10 if name=='EM10' else 100
    for _ in range(steps):
        mu=g@s+b
        q=z*mu/mu.abs().clamp_min(1e-8) if name=='GS100' else posterior_field(z,mu,v)
        g=(q-b)@op
    return g


@torch.no_grad()
def accuracy():
    frozen();models=load_models()
    with np.load(OUT/'data/test_raw.npz') as f:raw={k:f[k].copy() for k in f.files}
    h=torch.from_numpy(raw['h']).to(torch.complex128)
    for p,snr in GRID:
        dest=OUT/'test'/f'P{p}_snr{snr:g}.npz'
        if dest.exists():continue
        z,s,b,v=test_observations(raw,p,snr);results={};predictions={}
        for name,m in models.items():
            parts=[]
            for start in range(0,len(h),64):
                sl=slice(start,start+64);parts.append(m(*frontend(z[sl],s[sl],b[sl],v[sl])))
            pred=torch.cat(parts);assert torch.isfinite(pred).all()
            results[name]=ratio(pred,h).numpy();predictions[name]=pred.numpy()
        for name in ('GS100','EM10','EM100'):
            parts=[]
            for start in range(0,len(h),64):
                sl=slice(start,start+64);parts.append(classic(frontend(z[sl],s[sl],b[sl],v[sl]),name))
            pred=torch.cat(parts);assert torch.isfinite(pred).all()
            results[name]=ratio(pred,h).numpy();predictions[name]=pred.numpy()
        dest.parent.mkdir(parents=True,exist_ok=True)
        temp=dest.with_suffix('.tmp.npz')
        np.savez_compressed(temp,**results,truth_energy=h.abs().square().sum((-2,-1)).numpy(),P=p,snr=snr);temp.replace(dest)
        # Predictions are retained to score the exact same channel estimates in BER.
        pp=OUT/'predictions'/dest.name;pp.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(pp,**predictions)
        print('Matched accuracy',p,snr,{n:round(float(10*np.log10(r.mean())),3) for n,r in results.items() if n.endswith('_final') or n in ('GS100','EM10','EM100')},flush=True)
    frozen();write(OUT/'accuracy_complete.json',dict(passed=True,cells=12,physical_channels=2000))


def ber():
    frozen()
    with np.load(OUT/'data/test_raw.npz') as f:h=f['h'][:512].astype(np.complex128)
    for snr in SNRS:
        dest=OUT/'ber'/f'P20_snr{snr:g}.npz'
        if dest.exists():continue
        v=power(20)/10**(snr/10);b=np.full((32,1),np.sqrt(10*power(20)))
        # Reset per-anchor data streams across SNR for paired comparisons;
        # draw_data scales independent unit data noise by each variance.
        observations=[];truth=[]
        for i in range(512):
            r=np.random.default_rng(np.random.SeedSequence([DATA_MASTER,i]))
            z,bits,_=draw_data(h[i],b,v,r,vectors=32);observations.append(z);truth.append(bits)
        observations=np.stack(observations);truth=np.stack(truth)
        pp=OUT/'predictions'/f'P20_snr{snr:g}.npz'
        with np.load(pp) as f:
            names=list(f.files)+['known_channel'];errors=[]
            for name in names:
                g=h if name=='known_channel' else f[name][:512]
                counts=[]
                for i in range(512):counts.append(np.count_nonzero(detect(observations[i],g[i],b,v)!=truth[i]))
                errors.append(counts)
        dest.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(dest,method_names=np.array(names),bit_errors=np.array(errors),bits_per_channel=256,snr=snr)
        print('Matched BER',snr,'complete',flush=True)
    frozen();write(OUT/'ber_complete.json',dict(passed=True,cells=7,physical_channels=512,data_vectors=32,bits_per_channel=256))


@torch.no_grad()
def timing():
    frozen();models={n:m for n,m in load_models().items() if n.endswith('_final')}
    names=list(models)+['GS100','EM10','EM100']
    for window in (1,2):
        path=OUT/'timing'/f'window{window}.json'
        assert not path.exists(),'Timing windows must not be retried to pass.'
        raw=raw_worlds(TIMING_MASTER,window,32);cases=[]
        for p,snr in GRID:
            obs=test_observations(raw,p,snr)
            for i in range(32):cases.append((p,snr,i,tuple(x[i:i+1].numpy().copy() for x in obs)))
        # Warm up the same fixed set of call types before both windows.
        for p,snr,i,arrays in (cases[0],cases[-1]):
            args=frontend(*(torch.as_tensor(x) for x in arrays))
            for name in names:models[name](*args) if name in models else classic(args,name)
        rng=np.random.default_rng(TIMING_MASTER+window);records=[]
        for ci in rng.permutation(len(cases)):
            p,snr,i,arrays=cases[ci]
            for j in rng.permutation(len(names)):
                name=names[j];tick=time.perf_counter_ns()
                args=frontend(*(torch.as_tensor(x) for x in arrays))
                pred=models[name](*args) if name in models else classic(args,name)
                ms=(time.perf_counter_ns()-tick)/1e6
                assert torch.isfinite(pred).all()
                records.append(dict(method=name,P=p,snr=snr,anchor=i,ms=ms))
        scores={name:dict(p90_ms=float(np.quantile([r['ms'] for r in records if r['method']==name],.9)),
            mean_ms=float(np.mean([r['ms'] for r in records if r['method']==name])),
            max_ms=max(r['ms'] for r in records if r['method']==name),
            over_budget=sum(r['ms']>10.74 for r in records if r['method']==name)) for name in names}
        write(path,dict(window=window,records=records,scores=scores,one_thread=True,backend='PyTorch all methods'))
        print('Matched timing window',window,'complete',scores,flush=True)
    frozen()


if __name__=='__main__':
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    p=argparse.ArgumentParser();p.add_argument('phase',choices=('accuracy','ber','timing'));a=p.parse_args();globals()[a.phase]()
