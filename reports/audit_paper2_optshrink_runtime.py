"""Post-run runtime boundary check; no tuning and no changes to frozen results.

The runner's timing pass precomputes the SNR lookup at this fixed operating
point. This separate pass includes that dispatch inside the measured boundary,
as required by the end-to-end protocol, and replays all three checkpoints.
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key]='1'
import json
import time
from pathlib import Path
import numpy as np
import torch
from sahure.train import load_model
from paper2_optshrink_loop.run import ROOT,CONFIG,SPECS,observations,configs_for,load,sha,write,sources
from paper2_optshrink_loop.estimator import estimate,hs_rician


@torch.no_grad()
def main():
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    if json.loads((ROOT/'status.json').read_text())['state']!='complete':
        raise RuntimeError('Do not time concurrently with the accuracy run')
    dest=ROOT/'timing_including_dispatch.json'
    if dest.exists(): raise RuntimeError('Completed timing audit already exists; preserving it')
    manifest=json.loads((ROOT/'manifest.json').read_text())
    if sources()!=manifest['sources']: raise RuntimeError('source mismatch')
    frozen=json.loads((ROOT/'frozen_selection.json').read_text())
    for entry in manifest['models'].values():
        if sha(entry['path'])!=entry['sha256']: raise RuntimeError('checkpoint mismatch')
    models={s:load_model(entry['path'])[0] for s,entry in manifest['models'].items()}
    replays=[]
    for spec in SPECS:
        obs=load(ROOT/'test_data'/spec['id']/'observations.npz')
        out=load(ROOT/'test_outputs'/spec['id']/'trial_000.npz')
        z,s,b,v=(obs[k][0] for k in ('z','s','b','variance'))
        for seed,model in models.items():
            actual=model.estimate(z,s,b,float(v)).numpy()
            saved=out['estimates'][out['names'].tolist().index(f'adaptive10_s{seed}')]
            np.testing.assert_allclose(actual,saved,rtol=1e-11,atol=1e-11)
            replays.append(dict(cell=spec['id'],seed=seed,max_absolute_error=float(np.max(abs(actual-saved)))))
    bank=[observations(CONFIG['timing_seed'],i,SPECS[3])[1] for i in range(CONFIG['timing_cases'])]
    names=list(configs_for(float(bank[0]['variance']),frozen))+['hs_rician']+[f'adaptive10_s{s}' for s in models]
    def call(name,row):
        z,s,b,v=(row[k] for k in ('z','s','b','variance'));v=float(v)
        if name=='hs_rician': return hs_rician(z,s,b,v)[0]
        if name.startswith('adaptive10_'): return models[name.split('_s')[1]].estimate(z,s,b,v).numpy()
        # Include the receiver-noise lookup and configuration dispatch.
        cfg=configs_for(v,frozen)[name]
        return estimate(z,s,b,v,cfg)
    for name in names: call(name,bank[0])
    times={n:[] for n in names}
    for repeat in range(CONFIG['timing_repeats']):
        for i,row in enumerate(bank):
            offset=(repeat*len(bank)+i)%len(names)
            for name in names[offset:]+names[:offset]:
                tick=time.perf_counter();g=call(name,row);elapsed=time.perf_counter()-tick
                if not np.isfinite(g).all(): raise FloatingPointError(name)
                times[name].append(elapsed)
        print(f'End-to-end timing repetition {repeat+1}/3 complete',flush=True)
    write(dest,dict(raw_seconds=times,summary={n:dict(mean_ms=1000*float(np.mean(v)),
        median_ms=1000*float(np.median(v)),p10_ms=1000*float(np.quantile(v,.1)),p90_ms=1000*float(np.quantile(v,.9)))
        for n,v in times.items()},checkpoint_replays=replays,
        note='Includes observation unpacking, SNR lookup/configuration dispatch, initialization, online rank search where used, and updates. Excludes model loading and file I/O.',
        distinction='Original timing.json precomputed the SNR lookup for this one fixed timing condition. This is the headline end-to-end timing pass; original file is preserved.',
        worlds=len(bank),repetitions=CONFIG['timing_repeats'],seed=CONFIG['timing_seed'],
        script_sha256=sha(__file__),frozen_selection_sha256=sha(ROOT/'frozen_selection.json')))
    print(json.dumps(json.loads(dest.read_text())['summary'],indent=2),flush=True)

if __name__=='__main__': main()
