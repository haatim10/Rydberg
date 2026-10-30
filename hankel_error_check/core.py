"""Real-coordinate uncertainty diagnostics for row-separable Rician EM."""
from functools import lru_cache
import numpy as np
from numpy.polynomial.hermite import hermgauss
from scipy.special import i0e, i1e, ive
from codex_verification.solver import ratio

STAGES=(1,5,20,100)
STARTS=('spectral','zero','oracle')
MODELS=('missing_phase','frozen_update','full_sensitivity','isotropic','linear_ls')


def realify(g):return np.concatenate((g.real,g.imag),axis=-1)


def spectral(z,s,b):
    batch,n,p=z.shape;k=s.shape[0]
    a=np.concatenate((np.broadcast_to(s.T,(n,p,k)),b[...,None]),axis=-1)
    spec=np.einsum('npk,bnp,npl->bnkl',a.conj(),z,a,optimize=True)
    spec=(spec+spec.swapaxes(-1,-2).conj())/2
    _,vec=np.linalg.eigh(spec);vec=vec[...,-1]
    projected=abs(np.einsum('npk,bnk->bnp',a,vec,optimize=True))
    scale=(projected*z).sum(-1)/np.maximum((projected**2).sum(-1),1e-30)
    return (scale[...,None]*vec*np.exp(-1j*np.angle(vec[...,-1]))[...,None])[...,:k]


def operator(s):
    sh=s.conj().T
    return np.linalg.solve((s@sh).T,sh.T).T


def path(z,s,b,v,start='spectral',oracle=None,stages=STAGES):
    z=np.asarray(z,dtype=float)
    if z.ndim!=3 or np.any(z<0) or not np.isfinite(z).all() or v<=0:
        raise ValueError('invalid batched magnitude observations')
    if start=='spectral':g=spectral(z,s,b)
    elif start=='zero':g=np.zeros((len(z),len(b),len(s)),complex)
    elif start=='oracle' and oracle is not None:g=np.broadcast_to(oracle,(len(z),*oracle.shape)).copy()
    else:raise ValueError('unknown start or missing labelled oracle')
    op=operator(s);saved={}
    for t in range(1,max(stages)+1):
        previous=g
        mu=g@s+b
        g=(z*np.exp(1j*np.angle(mu))*ratio(2*z*abs(mu)/v)-b)@op
        if t in stages:saved[t]=(previous.copy(),g.copy())
    if not all(np.isfinite(g).all() for _,g in saved.values()):raise FloatingPointError('EM output')
    return saved


def latent_covariance(z,mu,v):
    """Conditional covariance of [Re X,Im X] given magnitude, not channel risk."""
    x=2*z*abs(mu)/v;r=i1e(x)/i0e(x);r2=ive(2,x)/i0e(x)
    radial=(1+r2)/2-r*r;tangential=(1-r2)/2
    high=x>1e4
    u=np.divide(1.,x,out=np.zeros_like(x),where=high)
    radial=np.where(high,u*u/2+u**3/4+3*u**4/8,radial)
    tangential=np.where(high,u-u*u/2-u**3/8-u**4/8,tangential)
    if min(float(radial.min()),float(tangential.min())) < -1e-12:
        raise FloatingPointError('negative phase variance')
    radial=np.maximum(radial,0)*z*z;tangential=np.maximum(tangential,0)*z*z
    phase=np.exp(1j*np.angle(mu));c=phase.real;s=phase.imag
    out=np.empty((*x.shape,2,2))
    out[...,0,0]=radial*c*c+tangential*s*s
    out[...,1,1]=radial*s*s+tangential*c*c
    out[...,0,1]=out[...,1,0]=(radial-tangential)*c*s
    return out


def rice_variance(a,v):
    rho=np.asarray(a)**2/v;x=rho/2
    mean=np.sqrt(v)*np.sqrt(np.pi)/2*((1+2*x)*i0e(x)+2*x*i1e(x))
    regular=a*a+v-mean*mean
    asymptotic=v/2-v/(8*np.maximum(rho,1))
    out=np.where(rho>1e4,asymptotic,regular)
    if np.any(out<=0) or not np.isfinite(out).all():raise FloatingPointError('Rician variance')
    return out


def integrated_mean_variance(r,order=64):
    """Var of posterior-mean radial amplitude for unit complex noise variance."""
    r=np.atleast_1d(r).astype(float);nodes,weights=hermgauss(order)
    weight=(weights[:,None]*weights[None,:]/np.pi).ravel()
    xx=np.broadcast_to(nodes[:,None],(order,order)).ravel(); yy=xx.reshape(order,order).T.ravel()
    z=np.sqrt((r[:,None]+xx)**2+yy**2)
    q=z*ratio(2*r[:,None]*z)
    # E q=r exactly under the matched model; centered integration avoids cancellation.
    return ((q-r[:,None])**2)@weight


@lru_cache(None)
def variance_table():
    grid=np.logspace(-6,3,361)
    return grid,integrated_mean_variance(grid)


def frozen_variance(a,v):
    r=np.asarray(a)/np.sqrt(v);grid,values=variance_table()
    safe=np.maximum(r,1e-300)
    out=np.exp(np.interp(np.log(safe),np.log(grid),np.log(values)))
    # q approximately r Z^2 at small r; large-r radial variance tends to 1/2.
    out=np.where(r<grid[0],r*r,out)
    out=np.where(r>grid[-1],.5,out)
    return v*out


def propagate_latent(cov,op):
    ar=np.concatenate((op.real,op.imag),axis=-1)
    ai=np.concatenate((-op.imag,op.real),axis=-1)
    return (np.einsum('...p,pi,pj->...ij',cov[...,0,0],ar,ar,optimize=True)
        +np.einsum('...p,pi,pj->...ij',cov[...,1,1],ai,ai,optimize=True)
        +np.einsum('...p,pi,pj->...ij',cov[...,0,1],ar,ai,optimize=True)
        +np.einsum('...p,pi,pj->...ij',cov[...,1,0],ai,ar,optimize=True))


def jacobians(z,s,b,v,start,oracle=None,step=1e-4,stages=STAGES):
    """Perturb all rows in one column; valid ONLY for this row-separable map."""
    a,n,p=z.shape;h=np.minimum(step*np.maximum(z,np.sqrt(v)),.1*z)
    if np.any(h<=0):raise ValueError('central differences require positive magnitudes')
    plus=np.broadcast_to(z[:,None],(a,p,n,p)).copy();minus=plus.copy()
    for j in range(p):
        plus[:,j,:,j]+=h[:,:,j];minus[:,j,:,j]-=h[:,:,j]
    zz=np.concatenate((plus,minus),axis=1).reshape(a*2*p,n,p)
    pp=path(zz,s,b,v,start,oracle,stages)
    result={}
    for t,(_,g) in pp.items():
        g=g.reshape(a,2*p,n,s.shape[0])
        diff=(g[:,:p]-g[:,p:])/(2*h.transpose(0,2,1)[...,None])
        result[t]=realify(diff).transpose(0,2,3,1)
    return result


def predictions(z,s,b,v,start,oracle=None,step=1e-4):
    pp=path(z,s,b,v,start,oracle);jj=jacobians(z,s,b,v,start,oracle,step)
    op=operator(s);q=2*len(s);output={}
    linear=propagate_latent(np.broadcast_to(v/2*np.eye(2),(len(z),len(b),s.shape[1],2,2)),op)
    for t,(previous,_) in pp.items():
        mu=previous@s+b
        missing=propagate_latent(latent_covariance(z,mu,v),op)
        direction=realify(np.exp(1j*np.angle(mu))[...,None]*op)
        frozen=np.einsum('...p,...pi,...pj->...ij',frozen_variance(abs(mu),v),direction,direction,optimize=True)
        full=np.einsum('...ip,...p,...jp->...ij',jj[t],rice_variance(abs(mu),v),jj[t],optimize=True)
        isotropic=np.trace(full,axis1=-2,axis2=-1)[...,None,None]/q*np.eye(q)
        output[t]=np.stack((missing,frozen,full,isotropic,linear))
    return output,pp,jj
