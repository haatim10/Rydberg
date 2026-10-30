"""Independent preflight: historical invariants and one full baseline epoch."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key]='1'
import argparse
import hashlib
import json
import time
import torch
import numpy as np
from sahure.data import load_observations,split_observations
from sahure.model import AdaptiveHankelEstimator
from sahure.train import train_epoch
from sahure_expansion.train import validate
from .protocol import ROOT,DATA,CONFIG,sha,write

def baseline_replay():
    torch.set_num_threads(1)
    campaign=json.loads((DATA.parent/'campaign_manifest.json').read_text())
    for split in ('train','val'):
        if sha(DATA/f'{split}_observations.npz')!=campaign['observations_sha256'][split]:
            raise RuntimeError('historical observations changed')
    sources={};initial_hashes={};old_history={}
    for seed in CONFIG['seeds']:
        folder=DATA.parent/'checkpoints'/f'seed_{seed}'
        meta=json.loads((folder/'manifest.json').read_text())
        if meta['torch']!=torch.__version__:raise RuntimeError('historical PyTorch version changed')
        for path,digest in meta['sources'].items():
            if sha(path)!=digest:raise RuntimeError('historical training source changed: '+path)
            sources[path]=digest
        for key in ('epochs','batch_size','lr','stages','train_cases','val_cases','validation_mask_seed'):
            if meta['config'][key]!=CONFIG[key]:raise RuntimeError('historical protocol mismatch: '+key)
        state=torch.load(folder/'em_adaptive_latest.pt',map_location='cpu',weights_only=False)
        if [r['epoch'] for r in state['history']]!=list(range(1,21)):raise RuntimeError('historical incomplete history')
        old_history[str(seed)]=state['history']
        torch.manual_seed(seed);model=AdaptiveHankelEstimator()
        initial_hashes[str(seed)]=hashlib.sha256(b''.join(p.detach().numpy().tobytes() for p in model.parameters())).hexdigest()
    seed=CONFIG['seeds'][0];torch.manual_seed(seed);model=AdaptiveHankelEstimator()
    optimizer=torch.optim.Adam(model.parameters(),lr=CONFIG['lr'])
    train=load_observations(DATA/'train_observations.npz');val=load_observations(DATA/'val_observations.npz')
    a,b=split_observations(val,CONFIG['validation_mask_seed'])
    tick=time.monotonic();loss=train_epoch(model,optimizer,train,0,seed,CONFIG['batch_size'],time.monotonic()+900)
    if loss is None:raise TimeoutError('baseline replay exceeded15minutes')
    score,_=validate(model,a,b,np.load(DATA/'val_conditions.npy'))
    expected=old_history[str(seed)][0]
    # Same dependency versions and seed: require numerical replay, not merely a similar curve.
    np.testing.assert_allclose(loss,expected['train_nll'],rtol=1e-6,atol=1e-7)
    np.testing.assert_allclose(score,expected['val_nll'],rtol=1e-6,atol=1e-7)
    report=dict(passed=True,replay_seed=seed,replay_epoch=1,train_nll=loss,val_nll=score,
        expected_train_nll=expected['train_nll'],expected_val_nll=expected['val_nll'],
        seconds=time.monotonic()-tick,initial_parameters_sha256=initial_hashes,
        sources=sources,observation_sha256=campaign['observations_sha256'],
        note='In-memory replay only; no historical files or checkpoints written.')
    write(ROOT/'baseline_replay.json',report);print(json.dumps(report),flush=True)
    return report

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--baseline-replay',action='store_true')
    args=parser.parse_args()
    if args.baseline_replay:baseline_replay()
    else:parser.error('use --baseline-replay')
