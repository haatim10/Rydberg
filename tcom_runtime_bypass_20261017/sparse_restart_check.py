"""Target the monotone-FISTA restart branch with a synthetic convex M-step.

This is a deterministic numerical unit check, not a performance experiment or
selection dataset. It intentionally uses an ill-conditioned pilot Gram.
"""
import json
from pathlib import Path
import numpy as np
import torch

from sahure.data import Observations
from tcom_followups_20261017.sparse import mstep as reference_mstep
from tcom_followups_20261017.common import sha, write_json
from . import sparse_fast as fast


def run():
    torch.set_num_threads(1)
    rng=np.random.default_rng(2026101749)
    n,d,k,p=32,64,3,30
    frame=fast.AngleFrame(n,d)
    dense=fast.AngleFrame(n,d,backend='matrix')
    base=rng.normal(size=(p,k))+1j*rng.normal(size=(p,k))
    q,_=np.linalg.qr(base)
    unitary,_=np.linalg.qr(rng.normal(size=(k,k))+1j*rng.normal(size=(k,k)))
    x0=(rng.normal(size=(1,d,k))+1j*rng.normal(size=(1,d,k)))*.6
    target_coefficients=(rng.normal(size=(1,d,k))+1j*rng.normal(size=(1,d,k)))*.1
    attempts=[]
    for condition in (100.,1000.,10000.):
        eigenvalues=np.array([30.,30./np.sqrt(condition),30./condition])
        s=(unitary@np.diag(np.sqrt(eigenvalues))@q.conj().T)[None]
        sh=s.conj().swapaxes(-1,-2);gram=s@sh
        target=frame.forward(target_coefficients)@s
        rhs=target@sh
        constant=.5*np.sum(np.abs(target)**2,axis=(-2,-1))
        step=1/((d/n)*np.linalg.eigvalsh(gram)[:,-1,None,None])
        for penalty in (.001,.01,.1):
            lam=np.array([penalty])
            x,g,info=fast._mstep(x0.copy(),frame.forward(x0),frame,gram,rhs,lam,step,600,
                               constant=constant,audit=True)
            attempts.append(dict(pilot_gram_condition=condition,penalty=penalty,
                                 inner_steps=600,restarts=info['restarts']))
            if info['restarts']==0:continue
            obs=Observations(torch.ones((1,n,p),dtype=torch.float64),torch.from_numpy(s),
                             torch.zeros((1,n,p),dtype=torch.complex128),torch.ones(1,dtype=torch.float64))
            ref,ref_info=reference_mstep(torch.from_numpy(x0.copy()),torch.from_numpy(dense.a),obs,
                torch.from_numpy(target),torch.from_numpy(lam),inner=600)
            ref=ref.numpy()
            coefficient_error=float(np.linalg.norm(x-ref)/max(1.,np.linalg.norm(ref)))
            channel_error=float(np.linalg.norm(g-dense.forward(ref))/max(1.,np.linalg.norm(dense.forward(ref))))
            old_restarts=int(ref_info['restarts'].sum())
            start=fast.quadratic(frame.forward(x0),x0,gram,rhs,lam)+constant
            final=fast.quadratic(g,x,gram,rhs,lam)+constant
            if coefficient_error>1e-8 or channel_error>1e-8 or old_restarts!=info['restarts']:
                raise AssertionError(dict(coefficient_error=coefficient_error,channel_error=channel_error,
                                          old_restarts=old_restarts,new_restarts=info['restarts']))
            return dict(passed=True,seed=2026101749,chosen=attempts[-1],attempts=attempts,
                reference_restarts=old_restarts,coefficient_relative_error=coefficient_error,
                channel_relative_error=channel_error,initial_surrogate=float(start[0]),
                final_surrogate=float(final[0]),surrogate_decrease=float((start-final)[0]),
                solver_source_sha256=sha(Path(fast.__file__)),
                note='Synthetic M-step only. Ill-conditioned 3x3 Gram, dense starting coefficients, strictly positive complex L1 penalty. Every inner restart/descent assertion active. No development/test outcomes used.')
    raise AssertionError('The deterministic cases did not exercise the restart branch')


if __name__=='__main__':
    result=run()
    path=Path(__file__).resolve().parents[1]/'results/tcom_runtime_bypass_20261017/sparse_restart_check.json'
    write_json(path,result)
    print(json.dumps(result,indent=2))
