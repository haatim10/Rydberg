"""Short isolated batch-one comparison; smoke data only, no policy tuning."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key]='1'
import time
import numpy as np
import torch
from sahure.train import load_model
from sahure.data import Observations
from tcom_pilot_em_20261018 import continuation
from tcom_retrained_bypass_20261018.study import model_paths
from tcom_followups_20261017.common import world_batch,REPO,sha,write_json
from .native import NativeEstimator


@torch.no_grad()
def main():
    torch.set_num_threads(1)
    models={name:load_model(path,dtype=torch.float64)[0] for name,path in model_paths().items()}
    arms={}
    for name,model in models.items():
        arms[f'torch_{name}']=model
        arms[f'native_solve_{name}']=NativeEstimator(model,shrink_mode='solve')
        arms[f'native_svd_{name}']=NativeEstimator(model,shrink_mode='svd')
    for name,model in list(arms.items()):
        def warm(*args,model=model):
            obs=Observations(*args)
            return continuation.estimate(obs,model(*args),100)
        arms[f'warm100_{name}']=warm
    rows=[]
    for ri,rsr in enumerate((5.,12.,20.)):
        for si,snr in enumerate((0.,35.,60.)):
            obs,_,_=world_batch(2026102000,[100],snr,rsr,(2,4,None)[(ri+si)%3])
            rows.append((dict(snr=snr,rsr=rsr),obs))
    for arm in arms.values():arm(*rows[0][1].args())
    rng=np.random.default_rng(20261020001);names=list(arms);raw={name:[] for name in names}
    for meta,obs in rows:
        for repeat in range(5):
            for ni in rng.permutation(len(names)):
                name=names[ni];start=time.perf_counter_ns();g=arms[name](*obs.args());elapsed=(time.perf_counter_ns()-start)/1e6
                assert torch.isfinite(g).all()
                raw[name].append(dict(**meta,repeat=repeat,ms=elapsed))
    kinds=('torch','native_solve','native_svd','warm100_torch','warm100_native_solve','warm100_native_svd')
    groups={kind:[r for name,values in raw.items() if name.startswith(kind+'_') for r in values] for kind in kinds}
    stats={kind:dict(median_ms=float(np.median([r['ms'] for r in values])),p90_ms=float(np.quantile([r['ms'] for r in values],.9)),calls=len(values)) for kind,values in groups.items()}
    result=dict(smoke_seed=2026102000,worlds=9,repeats=5,model_fits=3,threads=1,batch=1,stats=stats,raw_samples=raw,
        setup='Weights loaded and static shape constants warmed; all per-observation conversion, initializer and pilot setup inside each call.',
        source_sha256=sha(REPO/'tcom_runtime_repair_20261020/scratch_estimator/native.py'),benchmark_sha256=sha(__file__))
    write_json(REPO/'results/tcom_runtime_repair_20261020/scratch_estimator/benchmark.json',result)
    print(stats,flush=True)


if __name__=='__main__':main()
