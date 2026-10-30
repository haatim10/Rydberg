"""Separate fixed-budget serial timing; launch only after campaign completes."""
import time,json,platform
import numpy as np
from p1_followup.run import OUT,write,select_mask,fit_selected
from codex_verification.worlds import make_world

def main():
    assert (OUT/'COMPLETE.json').exists()
    protocol=dict(worlds=12,seed_base=82_000_000,N=32,K=3,P=30,L=7,snr=5,rsr=12,iterations=100,selection_iterations=25,sweeps=[1,2,4,8],kinds=['hard','smooth'],thread_count=1,precision='complex128',order='Deterministically shuffled methods within each world; one call per method; no discarded warmup.',includes='Initialization, candidate selection, and all-pilot final fit; excludes data generation.',platform=platform.platform())
    dest=OUT/'timing';write(dest/'protocol.json',protocol)
    if (dest/'rows.json').exists():raise RuntimeError('Preserve existing timing results')
    rows=[]
    for i in range(12):
        w=make_world(82_000_000+i,N=32,K=3,P=30,L=7,snr=5.,rsr=12.)
        combos=[(k,q) for k in ('hard','smooth') for q in (1,2,4,8)]
        order=np.random.default_rng(82_100_000+i).permutation(8)
        for idx in order:
            kind,q=combos[idx];start=time.perf_counter()
            val,_=select_mask(w.S,w.Z,w.B,w.sigma2,np.arange(30),kind,q)
            middle=time.perf_counter();fit_selected(w.S,w.Z,w.B,w.sigma2,val,kind,q);end=time.perf_counter()
            rows.append(dict(trial=82_000_000+i,kind=kind,sweeps=q,seconds=end-start,selection_seconds=middle-start))
        print(f'Timing {i+1}/12',flush=True)
    write(dest/'rows.json',rows)
    summary=[dict(kind=k,sweeps=q,median_seconds=float(np.median([r['seconds'] for r in rows if r['kind']==k and r['sweeps']==q]))) for k in ('hard','smooth') for q in (1,2,4,8)]
    write(dest/'summary.json',summary);print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
