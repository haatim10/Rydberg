"""Actual CPU forward/backward benchmark, without touching old training data."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key]='1'
from dataclasses import asdict
import json
from pathlib import Path
import platform
import time
import numpy as np
import torch
from sahure.data import Observations,split_observations
from sahure.train import heldout_loss
from sahure_expansion.data import world
from sahure_expansion.protocol import write_json
from .models import ARMS,ReviewEstimator


def run():
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    rows=[world(i,202610084101,N=32,K=3,P=30,snr=5,rsr=12,L=None)[0] for i in range(32)]
    data=Observations(torch.tensor(np.stack([w.Z for w in rows]),dtype=torch.float32),
        torch.tensor(np.stack([w.S for w in rows]),dtype=torch.complex64),
        torch.tensor(np.stack([w.B for w in rows]),dtype=torch.complex64),
        torch.tensor([w.sigma2 for w in rows],dtype=torch.float32))
    a,b=split_observations(data,202610084102)
    result={}
    for name,cfg in ARMS.items():
        torch.manual_seed(9101);model=ReviewEstimator(cfg)
        opt=torch.optim.Adam(model.parameters(),lr=.001)
        elapsed=[]
        for repeat in range(5):
            tick=time.perf_counter();opt.zero_grad(set_to_none=True)
            loss=heldout_loss(model,a,b);loss.backward()
            if not torch.isfinite(loss) or any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise FloatingPointError(name)
            opt.step()
            if repeat:elapsed.append(time.perf_counter()-tick)
        result[name]=dict(config=asdict(cfg),parameters=sum(p.numel() for p in model.parameters()),
            median_batch_seconds=float(np.median(elapsed)),raw_seconds=elapsed,
            estimated_train_only_seconds_2048x8=float(np.median(elapsed)*64*8),
            estimated_train_only_seconds_4096x12=float(np.median(elapsed)*128*12),
            estimated_train_only_seconds_16384x20=float(np.median(elapsed)*512*20))
        print(name,result[name],flush=True)
    write_json(Path('results/paper2_review_round2/preflight.json'),dict(arms=result,
        device='CPU, one thread',platform=platform.platform(),torch=torch.__version__,
        note='Training-only projection. Validation, generation, testing, packaging and contention add time.'))

if __name__=='__main__':run()
