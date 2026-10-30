"""Fast angle-grid sparse generalized EM for exact referenced magnitudes.

Production uses NumPy/SciPy on CPU, with FFT application of the unit-column
oversampled DFT dictionary. The initializer, complex L1 penalty, exact Rician
posterior, and objective are the same as the earlier reference implementation.
FISTA restarts use its inexpensive quadratic surrogate in cached channel space.
The first two FISTA iterates are unaccelerated, so inner budgets one/two need
no objective evaluations; their Lipschitz-sized proximal steps guarantee
surrogate descent. All setup and observation conversions occur inside calls.

Observations only: no truth, path count, angle, or realized channel norm input.
"""
from __future__ import annotations

import numpy as np
import scipy.fft as fft
from scipy.special import i0e, i1e
import torch


def as_numpy(obs):
    """Convert observation tensors at entry; included in inference timing."""
    values = obs.args() if hasattr(obs, 'args') else obs
    arrays=[]
    for index,value in enumerate(values):
        if torch.is_tensor(value): value=value.detach().cpu().numpy()
        arrays.append(np.asarray(value,dtype=np.complex128 if index in (1,2) else np.float64))
    z,s,b,v=arrays
    if z.ndim!=3 or s.ndim!=3 or b.shape!=z.shape:
        raise ValueError('expect batched Z[B,N,P], S[B,K,P], B[B,N,P]')
    v=np.broadcast_to(v.reshape(-1),len(z))
    if np.any(v<=0):raise ValueError('positive total complex noise variance required')
    return z,s,b,v


class AngleFrame:
    """A[n,j]=exp(-i*n*(-pi+2*pi*j/D))/sqrt(N); A A^H=D/N I."""
    def __init__(self,n,d,backend='fft'):
        self.n,self.d=int(n),int(d)
        if self.d<self.n:raise ValueError('D must be at least N for this frame')
        if backend not in ('fft','matrix'):raise ValueError(backend)
        self.backend=backend
        self.sign=(1.-2.*(np.arange(n)%2))/np.sqrt(n)
        if backend=='matrix':
            freq=-np.pi+2*np.pi*np.arange(d)/d
            self.a=np.exp(-1j*np.arange(n)[:,None]*freq[None,:])/np.sqrt(n)
            self.ah=self.a.conj().T

    def forward(self,x):
        if self.backend=='matrix':return self.a@x
        return fft.fft(x,axis=-2,workers=1)[...,:self.n,:]*self.sign[:,None]

    def adjoint(self,g):
        if self.backend=='matrix':return self.ah@g
        return self.d*fft.ifft(g*self.sign[:,None],n=self.d,axis=-2,workers=1)


def spectral_initial(z,s,b):
    """Identical magnitude-weighted, reference-anchored spectral initializer."""
    batch,n,p=z.shape
    a=np.concatenate((np.broadcast_to(s.transpose(0,2,1)[:,None],(batch,n,p,s.shape[1])),b[...,None]),axis=-1)
    spec=a.conj().swapaxes(-1,-2)@(z[...,None]*a)
    spec=(spec+spec.conj().swapaxes(-1,-2))*.5
    _,vectors=np.linalg.eigh(spec)
    vec=vectors[...,-1]
    projection=np.abs((a@vec[...,None])[...,0])
    scale=np.sum(projection*z,axis=-1)/np.maximum(np.sum(projection**2,axis=-1),1e-30)
    anchored=vec*np.exp(-1j*np.angle(vec[...,-1]))[...,None]
    return (scale[...,None]*anchored)[...,:s.shape[1]]


def posterior_field(z,mu,v):
    """E[complex noisy field | magnitude, current mean], exact Rician law."""
    amplitude=np.abs(mu)
    argument=2*z*amplitude/v[:,None,None]
    ratio=i1e(argument)/i0e(argument)
    weight=np.divide(z*ratio,amplitude,out=np.zeros_like(amplitude),where=amplitude>0)
    return mu*weight


def observed_objective(z,mu,v,x,lam):
    amplitude=np.abs(mu)
    argument=2*z*amplitude/v[:,None,None]
    nll=(z-amplitude)**2/v[:,None,None]-np.log(i0e(argument))
    return .5*v*np.sum(nll,axis=(-2,-1))+lam*np.sum(np.abs(x),axis=(-2,-1))


def soft(x,threshold):
    magnitude=np.abs(x)
    return x*np.maximum(1-threshold/np.maximum(magnitude,1e-30),0)


def quadratic(g,x,gram,rhs,lam):
    """M-step surrogate without its constant .5||Q-B||^2."""
    return .5*np.sum((g.conj()*(g@gram)).real,axis=(-2,-1))-np.sum((g.conj()*rhs).real,axis=(-2,-1))+lam*np.sum(np.abs(x),axis=(-2,-1))


def _mstep(x,g,frame,gram,rhs,lam,step,inner,*,constant,mode='fista',audit=False):
    initial=quadratic(g,x,gram,rhs,lam) if audit else None
    restarts=0
    if inner<=2 or mode=='ista':
        # With FISTA's t0=1, its first two proposals are exactly ISTA.
        for _ in range(inner):
            gradient=frame.adjoint(g@gram-rhs)
            x=soft(x-step*gradient,step*lam[:,None,None])
            g=frame.forward(x)
    else:
        y=x.copy();gy=g.copy()
        momentum=np.ones(len(x))
        objective=quadratic(g,x,gram,rhs,lam)
        for _ in range(inner):
            gradient=frame.adjoint(gy@gram-rhs)
            new=soft(y-step*gradient,step*lam[:,None,None])
            gnew=frame.forward(new)
            value=quadratic(gnew,new,gram,rhs,lam)
            bad=value>objective+1e-10*(1+np.abs(objective+constant))
            if np.any(bad):
                gradient0=frame.adjoint(g[bad]@gram[bad]-rhs[bad])
                new[bad]=soft(x[bad]-step[bad]*gradient0,step[bad]*lam[bad,None,None])
                gnew[bad]=frame.forward(new[bad])
                value[bad]=quadratic(gnew[bad],new[bad],gram[bad],rhs[bad],lam[bad])
                momentum[bad]=1
                restarts+=int(np.sum(bad))
            if audit and np.any(value>objective+1e-8*(1+np.abs(objective+constant))):
                raise FloatingPointError('M-step objective increased')
            next_momentum=.5*(1+np.sqrt(1+4*momentum**2))
            acceleration=((momentum-1)/next_momentum)[:,None,None]
            y=new+acceleration*(new-x)
            # A is linear: avoid applying the FFT to y a second time.
            gy=gnew+acceleration*(gnew-g)
            x,g,momentum,objective=new,gnew,next_momentum,value
    info={'restarts':restarts}
    if audit:
        end=quadratic(g,x,gram,rhs,lam)
        if np.any(end>initial+1e-8*(1+np.abs(initial+constant))):
            raise FloatingPointError('finite M-step increased its starting objective')
        mapped=(x-soft(x-step*frame.adjoint(g@gram-rhs),step*lam[:,None,None]))/step
        info['surrogate_decrease']=initial-end
        info['relative_proximal_residual']=np.linalg.norm(mapped.reshape(len(x),-1),axis=1)/(1+np.linalg.norm(frame.adjoint(rhs).reshape(len(x),-1),axis=1))
    return x,g,info


def trajectory(obs,factor,*,d=128,inner=2,budgets=(2,4,8,12,20,40,80,100),audit=False,backend='fft',mode='fista'):
    """Return ({outer iteration: torch complex128 channels}, diagnostics).

    Penalty lambda=factor*sqrt(v*P_fit*log(D)) is fixed within a trajectory.
    Auditing is only for numerical tests; monotone-FISTA restart checks remain
    active in production. factor=0 uses the exact unregularized EM M-step.
    """
    if factor<0 or inner<1 or mode not in ('ista','fista'):raise ValueError('invalid sparse controls')
    budgets=tuple(sorted(set(map(int,budgets))))
    if not budgets or budgets[0]<1:raise ValueError('positive outer budgets required')
    z,s,b,v=as_numpy(obs)
    batch,n,p=z.shape
    sh=s.conj().swapaxes(-1,-2)
    gram=s@sh
    g=spectral_initial(z,s,b)
    if factor==0:
        frame=None;x=g;lam=np.zeros(batch);step=None
        op=np.swapaxes(np.linalg.solve(np.swapaxes(gram,-1,-2),np.swapaxes(sh,-1,-2)),-1,-2)
    else:
        frame=AngleFrame(n,d,backend=backend)
        x=(n/d)*frame.adjoint(g)
        lam=float(factor)*np.sqrt(v*p*np.log(d))
        step=1/((d/n)*np.linalg.eigvalsh(gram)[:,-1,None,None])
    saved={};history=[];max_increase=0.;restarts=0;last={}
    previous=observed_objective(z,g@s+b,v,x,lam) if audit else None
    for t in range(1,budgets[-1]+1):
        q=posterior_field(z,g@s+b,v)
        target=q-b
        if factor==0:
            g=target@op
            if audit:x=g
        else:
            rhs=target@sh
            constant=.5*np.sum(np.abs(target)**2,axis=(-2,-1)) if inner>2 or audit else None
            x,g,last=_mstep(x,g,frame,gram,rhs,lam,step,inner,constant=constant,mode=mode,audit=audit)
            restarts+=last['restarts']
        if audit:
            current=observed_objective(z,g@s+b,v,x,lam)
            max_increase=max(max_increase,float(np.max(current-previous)))
            if np.any(current>previous+1e-8*(1+np.abs(previous))):
                raise FloatingPointError('penalized observed Rician objective increased')
            history.append(current.tolist());previous=current
        if t in budgets:saved[t]=torch.from_numpy(g.copy())
    info=dict(backend=backend,mode=mode,inner_restarts=restarts)
    if audit:
        info.update(max_observed_objective_increase=max_increase,observed_objective_history=history,
            final_relative_proximal_residual=last.get('relative_proximal_residual',np.zeros(batch)).tolist())
    return saved,info


def estimate(obs,config):
    iterations=int(config['iterations'])
    return trajectory(obs,float(config['factor']),d=int(config.get('dictionary_size',128)),
        inner=int(config.get('inner_steps',2)),budgets=(iterations,),
        backend=config.get('backend','fft'),mode=config.get('mode','fista'))[0][iterations]
