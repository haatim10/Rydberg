"""Explicit, logged recovery of a finite-input LAPACK SVD nonconvergence.

Original estimator files and the frozen selection remain unchanged. This
standalone process retries the same HS-GS world: NumPy is still attempted
first, with SciPy's QR-based gesvd used only for a matrix that fails.
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key]='1'
import argparse
import json
import time
import numpy as np
from scipy.linalg import svd as scipy_svd
from .protocol import ROOT,sha,write_json
from .evaluate import freeze,observations_for,data_name,classical_estimate


def recover(cell,trial,verify=False):
    frozen=freeze();spec=next(s for s in frozen['cells'] if s['id']==cell)
    data=observations_for(spec)
    z,s,b,v=(x[trial].numpy() for x in data.args())
    path=ROOT/'cells'/cell/'classical.jsonl'
    prior=[json.loads(line) for line in path.read_text().splitlines()]
    existing=[r for r in prior if r['method']=='hs_gs' and r['trial']==trial]
    if existing and not verify: raise RuntimeError('trial already recorded; use --verify')
    original=np.linalg.svd;events=[];failures=[]

    def robust(a,*args,**kwargs):
        try: return original(a,*args,**kwargs)
        except np.linalg.LinAlgError as exc:
            if not np.isfinite(a).all(): raise
            if args or kwargs!={'full_matrices':False}:
                raise RuntimeError('unexpected SVD signature') from exc
            failures.append(dict(shape=list(a.shape),error=str(exc)))
            flat=a.reshape(-1,*a.shape[-2:]);pieces=[]
            for index,matrix in enumerate(flat):
                try: factor=original(matrix,full_matrices=False)
                except np.linalg.LinAlgError:
                    factor=scipy_svd(matrix,full_matrices=False,lapack_driver='gesvd',check_finite=True)
                    u,sv,vh=factor
                    residual=float(np.linalg.norm((u*sv)@vh-matrix)/np.linalg.norm(matrix))
                    orthogonality=float(max(np.linalg.norm(u.conj().T@u-np.eye(len(sv))),np.linalg.norm(vh@vh.conj().T-np.eye(len(sv)))))
                    if residual>1e-12 or orthogonality>1e-11:
                        raise RuntimeError('alternative SVD failed reconstruction/orthogonality check')
                    events.append(dict(batch_index=index,shape=list(matrix.shape),relative_reconstruction_error=residual,orthogonality_error=orthogonality,driver='scipy.linalg.svd/gesvd'))
                pieces.append(factor)
            return tuple(np.stack([item[j] for item in pieces]).reshape(*a.shape[:-2],*pieces[0][j].shape) for j in range(3))

    np.linalg.svd=robust
    try: estimated,info=classical_estimate('hs_gs',z,s,b,float(v),frozen['hard_rank'])
    finally: np.linalg.svd=original
    with np.load(ROOT/'data'/f'{data_name(spec)}_truth.npz') as f: truth=f['g'][trial]
    error=float(np.sum(abs(estimated-truth)**2));energy=float(np.sum(abs(truth)**2))
    if not np.isfinite(error): raise FloatingPointError('nonfinite recovered estimate')
    details=dict(cell=cell,trial=trial,method='hs_gs',error=error,energy=energy,
        original_failures=failures,alternative_decompositions=events,source_sha256=sha(__file__),
        observations_sha256=sha(ROOT/'data'/f'{data_name(spec)}_observations.npz'),
        explanation='Same inputs, initialization, ranks, iteration counts and equations; alternative SVD driver only after finite-input nonconvergence.')
    if verify:
        if len(existing)!=1: raise RuntimeError('missing/duplicate recovered record')
        np.testing.assert_allclose(error,existing[0]['error'],atol=1e-10,rtol=1e-10)
        stored=json.loads((ROOT/'numerical_recovery'/f'{cell}_trial_{trial}.json').read_text())
        if stored['source_sha256']!=sha(__file__): raise RuntimeError('recovery source changed')
    else:
        if not failures: raise RuntimeError('original failure did not reproduce; no recovery recorded')
        details['recorded_unix']=time.time()
        write_json(ROOT/'numerical_recovery'/f'{cell}_trial_{trial}.json',details)
        info['numerical_recovery']=dict(driver='gesvd after NumPy failure',decompositions=len(events),record=f'numerical_recovery/{cell}_trial_{trial}.json')
        with path.open('a') as stream:
            stream.write(json.dumps(dict(method='hs_gs',trial=trial,error=error,energy=energy,info=info))+'\n')
    print(json.dumps(details),flush=True)
    return details


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cell',required=True);p.add_argument('--trial',type=int,required=True);p.add_argument('--verify',action='store_true')
    a=p.parse_args();recover(a.cell,a.trial,a.verify)
