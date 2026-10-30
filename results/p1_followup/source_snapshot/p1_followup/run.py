import argparse, hashlib, json, os, platform, time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from codex_verification.worlds import make_world
from p1_revision.baselines import fit_candidates, select_candidates, THRESHOLD_FACTORS
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/p1_followup'
CELLS=[dict(L=l,snr=s,seed=81_000_000+j*10000,n=100) for j,(l,s) in enumerate((l,s) for l in (3,7,14) for s in (0,10))]

def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False));tmp.replace(path)

def hashes():
    files=[Path(__file__),ROOT/'p1_followup/__init__.py',ROOT/'p1_revision/baselines.py',ROOT/'sahure/classical.py',ROOT/'codex_verification/worlds.py',ROOT/'codex_verification/solver.py']+sorted((ROOT/'rydberg_sim').glob('*.py'))
    return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}

def select_mask(s,z,b,sigma2,order,kind,sweeps,iterations=25):
    values=np.arange(1,(z.shape[0]+1)//2+1) if kind=='hard' else np.array(THRESHOLD_FACTORS)
    option={'ranks':values} if kind=='hard' else {'factors':values}
    index,scores=select_candidates(s[:,order],z[:,order],b[:,order],sigma2,iterations=iterations,sweeps=sweeps,**option)
    return float(values[index]),scores

def fit_selected(s,z,b,sigma2,value,kind,sweeps,iterations=100):
    option={'ranks':[int(value)]} if kind=='hard' else {'factors':[value]}
    return fit_candidates(s,z,b,sigma2,iterations=iterations,sweeps=sweeps,**option)[0]

def evaluate(cell,i,expected):
    if hashes()!=expected:raise RuntimeError('Sources changed after freeze')
    trial=cell['seed']+i;dest=OUT/'raw'/f'{trial}.json'
    if dest.exists():
        saved=json.loads(dest.read_text())
        if saved['source_sha256']!=expected:raise RuntimeError('Incompatible saved trial')
        return trial
    w=make_world(trial,N=32,K=3,P=30,L=cell['L'],snr=cell['snr'],rsr=12.)
    s,z,b=w.S,w.Z,w.B
    em=fit_selected(s,z,b,w.sigma2,16,'hard',4)
    rows=[];cache={}
    for sweeps in (1,2,4,8):
        for kind in ('hard','smooth'):
            value,scores=select_mask(s,z,b,w.sigma2,np.arange(30),kind,sweeps)
            estimate=fit_selected(s,z,b,w.sigma2,value,kind,sweeps)
            error=float(np.sum(abs(estimate-w.G)**2));cache[kind,sweeps,value]=error
            rows.append(dict(kind=kind,sweeps=sweeps,split=0,selected=value,error=error,scores=scores.tolist(),heldout=list(range(21,30))))
    for split in (1,2,3):
        order=np.random.default_rng(np.random.SeedSequence([trial,split,0x53504C])).permutation(30)
        value,scores=select_mask(s,z,b,w.sigma2,order,'hard',4)
        key=('hard',4,value)
        if key not in cache:
            estimate=fit_selected(s,z,b,w.sigma2,value,'hard',4)
            cache[key]=float(np.sum(abs(estimate-w.G)**2))
        rows.append(dict(kind='hard',sweeps=4,split=split,selected=value,error=cache[key],scores=scores.tolist(),heldout=order[21:].tolist()))
    write(dest,dict(trial=trial,L=cell['L'],snr=cell['snr'],den=float(np.sum(abs(w.G)**2)),em_error=float(np.sum(abs(em-w.G)**2)),rows=rows,source_sha256=expected))
    return trial

def freeze():
    protocol=dict(cells=CELLS,N=32,K=3,P=30,rsr=12,iterations=100,select_iterations=25,sweeps=[1,2,4,8],random_masks=3,validation_fraction=.3,independent_worlds=600,bootstrap=4000,
      comparisons='Each sweep count retunes its own rank or factor; random masks use hard processing and four sweeps. All final fits use original-order full pilots. Repeated masks are clustered within world.',
      timing='Parallel campaign time is not an inference benchmark; separate serial timing uses 12 fresh worlds.',
      stopping='Fixed 100 worlds per cell, no outcome-dependent stopping.',source_sha256=hashes(),python=platform.python_version(),numpy=np.__version__)
    path=OUT/'protocol.json'
    if path.exists():
        old=json.loads(path.read_text())
        if old!=protocol:raise RuntimeError('Protocol mismatch')
    else:
        write(path,protocol)
        for name in protocol['source_sha256']:
            dst=OUT/'source_snapshot'/name;dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes((ROOT/name).read_bytes())
    return protocol

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--preflight',action='store_true');args=parser.parse_args()
    if args.preflight:
        start=time.perf_counter();w=make_world(80_000_001,N=32,K=3,P=30,L=7,snr=5.,rsr=12.)
        for q in (1,2,4,8):
            for kind in ('hard','smooth'):
                value,_=select_mask(w.S,w.Z,w.B,w.sigma2,np.arange(30),kind,q)
                fit_selected(w.S,w.Z,w.B,w.sigma2,value,kind,q)
        print(json.dumps({'eight_fits_seconds':time.perf_counter()-start}));return
    protocol=freeze();done=0;start=time.perf_counter()
    with ProcessPoolExecutor(max_workers=4) as pool:
        fs=[pool.submit(evaluate,c,i,protocol['source_sha256']) for c in CELLS for i in range(c['n'])]
        for f in as_completed(fs):
            f.result();done+=1
            if done%20==0:
                write(OUT/'progress.json',dict(completed=done,total=600,elapsed_seconds=time.perf_counter()-start));print(f'{done}/600 worlds',flush=True)
    write(OUT/'COMPLETE.json',dict(complete=True,worlds=done,source_sha256=hashes(),elapsed_seconds=time.perf_counter()-start))
if __name__=='__main__':main()
