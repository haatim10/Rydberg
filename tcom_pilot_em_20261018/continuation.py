"""Exact unregularized Rician EM updates starting from a supplied channel.

No initializer is called: the learned estimate is the initial iterate. Each
update has gamma=1 and solves the unregularized M-step exactly. Setup and data
conversion occur inside the call and are included in end-to-end timing.
"""
from __future__ import annotations
import numpy as np
from scipy.special import i0e,i1e
import torch
from tcom_runtime_bypass_20261017.sparse_fast import as_numpy


def posterior(z,mu,v):
    u=mu.real**2+mu.imag**2
    a=np.sqrt(np.maximum(u,1e-30));vv=v[:,None,None]
    x=2*z*a/vv;x2=4*z*z*u/(vv*vv)
    xs=np.minimum(x2,1e-4)
    small=z*z/vv*(1-xs/8+xs*xs/48)
    regular=z*(i1e(x)/i0e(x))/a
    return mu*np.where(x2<1e-4,small,regular)


def nll(z,mu,v):
    """Canonical stable observed NLL, excluding data-only constants."""
    u=mu.real**2+mu.imag**2
    a=np.sqrt(np.maximum(u,1e-30));vv=v[:,None,None]
    x=2*z*a/vv;x2=4*z*z*u/(vv*vv);xs=np.minimum(x2,1e-4)
    small=(z*z+u)/vv-(xs/4-xs*xs/64+xs**3/576)
    regular=(z-a)**2/vv-np.log(i0e(x))
    return np.where(x2<1e-4,small,regular)


def trajectory(obs,initial,budgets=(0,1,2,5),audit=False):
    """Return ({tail length: complex128 tensor}, audit diagnostics).

    The input initial estimate is never modified. Length zero is bitwise equal
    to that input. Auditing records the likelihood after every update and checks
    its expected nonincrease; production avoids those extra loss evaluations.
    """
    budgets=tuple(sorted(set(map(int,budgets))))
    if not budgets or budgets[0]<0:raise ValueError('nonnegative integer tail lengths required')
    if not torch.is_tensor(initial):initial=torch.as_tensor(initial,dtype=torch.complex128)
    if budgets[-1]==0:return {0:initial},dict(iterations=0,spectral_reinitialization=False)
    z,s,b,v=as_numpy(obs)
    g=np.asarray(initial.detach().cpu().numpy(),dtype=np.complex128).copy()
    if g.shape!=(len(z),z.shape[1],s.shape[1]) or not np.isfinite(g).all():
        raise ValueError('initial channel must be finite and have shape [batch,N,K]')
    sh=s.conj().swapaxes(-1,-2);gram=s@sh
    op=np.swapaxes(np.linalg.solve(np.swapaxes(gram,-1,-2),np.swapaxes(sh,-1,-2)),-1,-2)
    saved={0:initial} if 0 in budgets else {};history=[];maximum_increase=0.
    previous=nll(z,g@s+b,v).mean((-2,-1)) if audit else None
    if audit:history.append(previous.tolist())
    for step in range(1,budgets[-1]+1):
        g=(posterior(z,g@s+b,v)-b)@op
        if not np.isfinite(g).all():raise FloatingPointError('nonfinite EM continuation')
        if audit:
            current=nll(z,g@s+b,v).mean((-2,-1))
            maximum_increase=max(maximum_increase,float(np.max(current-previous)))
            if np.any(current>previous+1e-10*(1+np.abs(previous))):
                raise FloatingPointError('exact EM continuation increased observed NLL')
            history.append(current.tolist());previous=current
        if step in budgets:saved[step]=torch.from_numpy(g.copy())
    info=dict(iterations=budgets[-1],spectral_reinitialization=False)
    if audit:info.update(maximum_nll_increase=maximum_increase,nll_history=history)
    return saved,info


def estimate(obs,initial,iterations):
    iterations=int(iterations)
    return trajectory(obs,initial,(iterations,))[0][iterations]


def numerical_checks():
    from sahure.math import posterior_field,right_solve,nll as canonical_nll
    from tcom_followups_20261017.common import world_batch,load_learned_models,REPO,write_json,sha
    torch.set_num_threads(1)
    cases=[(0.,12.,2),(10.,12.,4),(20.,12.,None),(45.,5.,2),(60.,12.,None)]
    models=load_learned_models();maximum_parity=0.;maximum_scalar=0.;maximum_loss_error=0.;maximum_increase=0.
    with torch.no_grad():
        for snr,rsr,L in cases:
            obs,_,_=world_batch(2026101800,range(3),snr,rsr,L)
            for name,model in models.items():
                g=model(*obs.args());saved,info=trajectory(obs,g,audit=True)
                assert torch.equal(saved[0],g)
                op=right_solve(obs.s.mH,obs.s@obs.s.mH);reference=g.clone()
                for t in range(1,6):
                    reference=(posterior_field(obs.z,reference@obs.s+obs.b,obs.sigma2)-obs.b)@op
                    if t in saved:
                        difference=float(torch.max(torch.abs(reference-saved[t])));assert difference<1e-10
                        maximum_parity=max(maximum_parity,difference)
                scalar=torch.cat([estimate(obs.batch(slice(i,i+1)),g[i:i+1],5) for i in range(len(obs))])
                difference=float(torch.max(torch.abs(scalar-saved[5])));assert difference<1e-10
                maximum_scalar=max(maximum_scalar,difference);maximum_increase=max(maximum_increase,info['maximum_nll_increase'])
                z,s,b,v=as_numpy(obs);actual=nll(z,saved[5].numpy()@s+b,v)
                expected=canonical_nll(obs.z,saved[5]@obs.s+obs.b,obs.sigma2).numpy()
                error=float(np.max(np.abs(actual-expected)));assert error<1e-7
                maximum_loss_error=max(maximum_loss_error,error)
    z=np.array([[[0.,1e-8,.3,2.,3.]]]);mu=np.array([[[0.,1e-20+1e-20j,1e-7-1e-7j,.01j,2.+1j]]]);v=np.array([.3])
    q=posterior(z,mu,v);expected=posterior_field(torch.tensor(z),torch.tensor(mu),torch.tensor(v)).numpy()
    edge_error=float(np.max(abs(q-expected)));assert edge_error<1e-13
    result=dict(passed=True,smoke_seed=2026101800,cases=cases,models=3,worlds_per_case=3,
      tail_lengths=[0,1,2,5],gamma=1,spectral_reinitialization=False,zero_tail_bitwise_identity=True,
      maximum_canonical_channel_error=maximum_parity,maximum_scalar_batch_error=maximum_scalar,
      maximum_canonical_nll_error=maximum_loss_error,small_argument_posterior_error=edge_error,
      maximum_observed_nll_increase=maximum_increase,source_sha256=sha(__file__))
    write_json(REPO/'results/tcom_pilot_em_20261018/continuation_checks.json',result)
    print(result,flush=True)


if __name__=='__main__':numerical_checks()
