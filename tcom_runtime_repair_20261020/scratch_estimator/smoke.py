"""Cheap equivalence smoke; no test outcome or parameter selection."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key]='1'
import numpy as np
import torch
from sahure.train import load_model
from tcom_retrained_bypass_20261018.study import model_paths
from tcom_followups_20261017.common import world_batch,REPO,sha,write_json
from .native import NativeEstimator


@torch.no_grad()
def main():
    torch.set_num_threads(1)
    conditions=((0.,5.,2),(10.,12.,4),(20.,20.,None),(40.,5.,None),(60.,12.,2))
    models={name:load_model(path,dtype=torch.float64)[0] for name,path in model_paths().items()}
    maxima={mode:{key:0. for key in ('g','beta','tau','correction','nll','scalar_batch')} for mode in ('solve','svd')}
    for snr,rsr,L in conditions:
        obs,_,_=world_batch(2026102000,range(2),snr,rsr,L)
        for name,model in models.items():
            expected,stages=model(*obs.args(),diagnostics=True)
            for mode in maxima:
                native=NativeEstimator(model,shrink_mode=mode)
                actual,trace=native(*obs.args(),diagnostics=True)
                for key in ('g','beta','tau','correction','nll'):
                    difference=max(float(torch.max(torch.abs(a[key]-b[key]))) for a,b in zip(stages,trace))
                    maxima[mode][key]=max(maxima[mode][key],difference)
                assert torch.max(torch.abs(expected-actual))<1e-10
                scalar=torch.cat([native(*obs.batch(slice(i,i+1)).args()) for i in range(2)])
                difference=float(torch.max(torch.abs(scalar-actual)))
                maxima[mode]['scalar_batch']=max(maxima[mode]['scalar_batch'],difference)
                assert difference<1e-10
    result=dict(passed=True,smoke_seed=2026102000,conditions=conditions,worlds_per_condition=2,models=3,
        dtype='float64/complex128',stages=10,weights_and_algorithm_unchanged=True,maximum_errors=maxima,
        source_sha256=sha(REPO/'tcom_runtime_repair_20261020/scratch_estimator/native.py'),
        checkpoint_hashes={str(p.relative_to(REPO)):sha(p) for p in model_paths().values()})
    write_json(REPO/'results/tcom_runtime_repair_20261020/scratch_estimator/smoke.json',result)
    print(result,flush=True)


if __name__=='__main__':main()
