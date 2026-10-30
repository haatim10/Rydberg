"""Classical comparisons on exact observations; double precision, no labels."""
import numpy as np
import time
from scipy.special import i0e
from codex_verification.solver import initial,ratio,project,fit,select


def likelihood(z,mu,sigma2):
    a=np.abs(mu);x=2*z*a/sigma2
    return (z-a)**2/sigma2-np.log(i0e(x))


def gradient_fit(z,s,b,sigma2, *, objective='rician',iterations=100,ranks=None,converge=False):
    """Gradient updates with ||S||² scaling; optional per-candidate backtracking.

    Rician fixed-budget step is precisely gamma=1 in the unfolded GD core.
    Hard rank uses one Cadzow sweep per update. For convergence reference,
    stop on scaled gradient norm, cap 1000, report failed convergence honestly.
    """
    ranklist=np.atleast_1d((z.shape[0]+1)//2 if ranks is None else ranks)
    g=np.broadcast_to(initial(s,z,b),(len(ranklist),z.shape[0],s.shape[0])).copy()
    sh=s.conj().T; norm2=np.linalg.norm(s,2)**2
    def value(x):
        mu=x@s+b
        return (likelihood(z,mu,sigma2) if objective=='rician' else (z-abs(mu))**2/sigma2).sum(axis=(-2,-1))
    iterations_used=0; converged=False
    for it in range(iterations):
        mu=g@s+b;a=abs(mu)
        if objective=='rician':
            # exp(i angle(0))*ratio(0) is zero, the continuous limit.
            q=z*np.exp(1j*np.angle(mu))*ratio(2*z*a/sigma2)
        else: q=z*np.exp(1j*np.angle(mu))
        direction=((mu-q)@sh)/norm2
        if converge and np.linalg.norm(direction)<=1e-7*max(1.,np.linalg.norm(g)):
            converged=True;break
        new=project(g-direction,ranklist,sweeps=1)
        if converge:
            old=value(g);steps=np.ones(len(g))
            for _ in range(30):
                bad=value(new)>old+1e-12
                if not bad.any(): break
                steps[bad]*=.5
                new=project(g-steps[:,None,None]*direction,ranklist,sweeps=1)
            if np.any(value(new)>old+1e-10): raise FloatingPointError('backtracking failed')
        g=new;iterations_used=it+1
    return g,dict(iterations=iterations_used,converged=converged)


def linearized_ls(z,s,b):
    """Xu 1-D amplitude linearization, solved by real least squares (no ridge).

    z-|b| ≈ Re(exp(-j angle(b)) * (g@s)). Observations remain exact.
    """
    out=np.empty((z.shape[0],s.shape[0]),complex)
    for n in range(z.shape[0]):
        design=np.exp(-1j*np.angle(b[n]))[:,None]*s.T
        a=np.concatenate((design.real,-design.imag),axis=1)
        result=np.linalg.lstsq(a,z[n]-abs(b[n]),rcond=None)[0]
        out[n]=result[:s.shape[0]]+1j*result[s.shape[0]:]
    return out


def classical_estimate(name,z,s,b,sigma2,hard_rank=None):
    if name.startswith('em_'):
        return fit(s,z,b,sigma2,iterations=int(name.split('_')[1]))[0],{}
    if name=='hs_gs':
        rank,scores=select(s,z,b,sigma2,iterations=25,sweeps=4)
        return fit(s,z,b,sigma2,iterations=100,ranks=[rank],sweeps=4)[0],dict(rank=rank,scores=scores.tolist())
    if name=='linearized_ls': return linearized_ls(z,s,b),{}
    if name=='rician_hard':
        g,info=gradient_fit(z,s,b,sigma2,iterations=100,ranks=[hard_rank]);return g[0],info
    objective='amplitude' if name.startswith('amplitude') else 'rician'
    converge=name.endswith('converged')
    g,info=gradient_fit(z,s,b,sigma2,objective=objective,iterations=1000 if converge else 100,converge=converge)
    return g[0],info


def select_global_hard_rank(context,heldout,deadline=float('inf')):
    cap=(context.z.shape[1]+1)//2
    scores=np.zeros(cap)
    for i in range(len(context)):
        if time.monotonic()>=deadline: raise TimeoutError('compute cap during validation rank selection')
        z,s,b,v=(x[i].numpy() for x in context.args())
        g,_=gradient_fit(z,s,b,float(v),iterations=25,ranks=np.arange(1,cap+1))
        zz,ss,bb,vv=(x[i].numpy() for x in heldout.args())
        scores+=likelihood(zz,g@ss+bb,float(vv)).mean(axis=(-2,-1))
    return int(1+np.argmin(scores)),(scores/len(context)).tolist()
