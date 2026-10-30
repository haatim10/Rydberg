"""Compact-bundle check: recalculate saved SNR scores and replay primary models.

Run from the extracted bundle root. This does not retrain or alter results.
Only load the supplied checkpoints if you trust their source.
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key]='1'
import json
from pathlib import Path
import numpy as np
import torch
from sahure.train import load_model
from sahure.data import Observations

torch.set_num_threads(1)
root=Path('results/sahure_expansion')
f=json.loads((root/'frozen_selection.json').read_text())
s=json.loads((root/'summary.json').read_text())
scores={}
for method in s['aggregate_snr']:
    names=[name for name,e in f['models'].items() if e['method']==method] or [method]
    cells=[]
    for p in f['axes']['snr']:
        seeds=[]
        for name in names:
            with np.load(root/'cells'/p['cell']/f'{name}.npz') as a:
                seeds.append(10*np.log10(a['error'].sum()/a['energy'].sum()))
        cells.append(float(np.mean(seeds)))
    scores[method]=float(np.mean(cells))
for method,score in scores.items():
    np.testing.assert_allclose(scores['hs_gs']-score,s['aggregate_snr'][method]['mean_per_snr_gain_db'],atol=1e-10,rtol=0)
print('All 19 main-study macro scores reproduced from saved per-case errors.')
with np.load('replay_subset.npz') as a:
    data=Observations(torch.tensor(a['z']),torch.tensor(a['s']),torch.tensor(a['b']),torch.tensor(a['sigma2']))
    with torch.no_grad():
        for seed in (9101,9102,9103):
            name=f'em_adaptive_s{seed}';model,_=load_model(f['models'][name]['path'])
            pred=model(*data.args()).numpy()
            np.testing.assert_allclose(pred,a[name],rtol=1e-8,atol=1e-9)
            print(f'Primary seed {seed}: eight-case replay passed.')
