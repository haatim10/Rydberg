"""Same-case, batch-one CPU timing after training and test evaluation stop."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key]='1'
import argparse
import json
from pathlib import Path
import platform
import time
import numpy as np
import torch
from sahure_expansion.protocol import write_json
from .common import ROOT,load,sha
from .models import load_model


@torch.no_grad()
def run(wait=False):
    torch.set_num_threads(1);torch.set_num_interop_threads(1);start=time.monotonic()
    test=ROOT/'training_test';training=ROOT/'training'
    while not (test/'summary.json').exists():
        if not wait:raise RuntimeError('fresh test must finish before uncontended timing')
        for p in (training/'status.json',test/'status.json'):
            if p.exists() and json.loads(p.read_text())['state'] in ('failed','partial'):raise RuntimeError('study stopped; timing not run')
        if time.monotonic()-start>7800:raise TimeoutError('waiting cap')
        time.sleep(15)
    path=ROOT/'training_timing.json'
    if path.exists():print('Timing already saved');return
    data=load(test/'data/cell_34_observations.npz') # RSR12, SNR5, random 3-7 paths
    models={p.parent.name:load_model(p)[0] for p in sorted(training.glob('*/selected.pt'))}
    raw={name:[] for name in models};rng=np.random.default_rng(202610086777)
    for m in models.values():m.estimate(data['z'][0],data['s'][0],data['b'][0],float(data['sigma2'][0]))
    for repeat in range(3):
        for i in range(32):
            for name in rng.permutation(list(models)):
                tick=time.perf_counter();models[name].estimate(data['z'][i],data['s'][i],data['b'][i],float(data['sigma2'][i]))
                raw[name].append(time.perf_counter()-tick)
    write_json(path,dict(raw_seconds=raw,summary={name:dict(mean_ms=1000*float(np.mean(v)),median_ms=1000*float(np.median(v)),
        parameters=sum(p.numel() for p in models[name].parameters())) for name,v in raw.items()},
        device=platform.platform(),torch=torch.__version__,python=platform.python_version(),
        protocol='CPU one thread, float64, batch1, 32 shared SNR5/RSR12 worlds, three repeats, shuffled model order; full estimation incl initialization',
        source_sha256=sha(__file__),data_sha256=sha(test/'data/cell_34_observations.npz'),
        checkpoint_hashes={str(p):sha(p) for p in sorted(training.glob('*/selected.pt'))}))
    print(path)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--wait-for-test',action='store_true');run(p.parse_args().wait_for_test)
