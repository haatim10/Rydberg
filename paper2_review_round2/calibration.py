"""Static nuisance-parameter EM. Unknown global phase is NOT estimated.

All functions consume observations only. Channel labels belong exclusively
to the evaluator. A positive global amplitude multiplier and a shared noise
variance are constrained to declared broad bounds in normalized coordinates.
"""
import numpy as np
from scipy.special import i0e
from codex_verification.solver import initial,ratio

VARIANCE_BOUNDS=(1e-6,1e3)
GAIN_BOUNDS=(.25,4.)


def nll(z,mu,v):
    # log(v) MUST remain when comparing different variance estimates.
    a=abs(mu);x=2*z*a/v
    return (z-a)**2/v-np.log(i0e(x))+np.log(v)


def fit(z,s,b,v,*,mode='joint',budgets=(5,20,100),g0=None):
    if mode not in ('none','noise','gain','joint'):raise ValueError(mode)
    g=initial(s,z,b) if g0 is None else np.array(g0,copy=True)
    gain=1.;v=float(v)
    sh=s.conj().T;op=np.linalg.solve((s@sh).T,sh.T).T
    residual_operator=np.eye(s.shape[1])-op@s
    br=b@residual_operator;den=float(np.sum(abs(br)**2))
    if mode in ('gain','joint') and den<1e-12*np.sum(abs(b)**2):
        raise ValueError('reference lies in pilot row space; amplitude is unidentifiable in this fit')
    saved={};history=[]
    for t in range(1,max(budgets)+1):
        mu=g@s+gain*b
        q=z*np.exp(1j*np.angle(mu))*ratio(2*z*abs(mu)/v)
        if mode in ('gain','joint'):
            gain=float(np.clip(np.real(np.vdot(br,q@residual_operator))/den,*GAIN_BOUNDS))
        g=(q-gain*b)@op
        mu=g@s+gain*b
        if mode in ('noise','joint'):
            v=float(np.clip(np.mean(z*z+abs(mu)**2-2*np.real(np.conj(mu)*q)),*VARIANCE_BOUNDS))
        score=float(np.mean(nll(z,mu,v)));history.append(score)
        if not np.isfinite(score) or not np.isfinite(g).all():raise FloatingPointError('calibration estimate')
        if t in budgets:saved[t]=dict(g=g.copy(),gain=gain,variance=v,training_nll=score)
    return saved,np.array(history)


def select_and_refit(z,s,b,v,modes=('none','noise','joint'),budgets=(5,20,100)):
    """Choose on 9 held-out pilots, then refit on all 30; count all work."""
    if s.shape[-1]!=30:raise ValueError('This calibration protocol uses exactly 30 observations')
    candidates=[]
    for mode in modes:
        bank,_=fit(z[:,:21],s[:,:21],b[:,:21],v,mode=mode,budgets=budgets)
        for t in budgets:
            row=bank[t]
            score=float(np.mean(nll(z[:,21:],row['g']@s[:,21:]+row['gain']*b[:,21:],row['variance'])))
            candidates.append(dict(mode=mode,iterations=t,heldout_nll=score))
    selected=min(candidates,key=lambda c:c['heldout_nll'])
    final,_=fit(z,s,b,v,mode=selected['mode'],budgets=(selected['iterations'],))
    return final[selected['iterations']],dict(selected=selected,candidates=candidates)


def amplitude_jacobian(g,s,b):
    mu=g@s+b;a=abs(mu)
    c=np.divide(np.conj(mu),a,out=np.zeros_like(mu),where=a>0)[:,None,:]*s[None,:,:]
    return np.concatenate([c.real,-c.imag],axis=1).transpose(0,2,1)


def conditioning(g,s,b):
    jac=amplitude_jacobian(g,s,b);sv=np.linalg.svd(jac,compute_uv=False)
    rank=np.sum(sv>1e-10*np.maximum(sv[:,:1],1e-30),axis=1)
    minimum=sv[:,-1] if jac.shape[1]>=jac.shape[2] else np.zeros(len(g))
    return dict(rank=rank,minimum=minimum,ratio=minimum/np.maximum(sv[:,0],1e-30))
