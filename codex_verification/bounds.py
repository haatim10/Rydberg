"""Paired geometric CCRB and Rician CRLB on the new Fig. 2(a) worlds."""
import json,time
from pathlib import Path
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.linalg import block_diag
from rydberg_sim.crlb import rician_fisher_scalar
from codex_verification.run import cells,world,OUT


def tangent(w):
    n,k=w.G.shape
    cols=[]
    for user,(psi,alpha) in enumerate(zip(w.psi,w.alpha)):
        for freq,gain in zip(psi,alpha):
            v=np.exp(-1j*np.arange(n)*freq)
            for d in (-1j*np.arange(n)*gain*v,v,1j*v):
                col=np.zeros((n,k),complex);col[:,user]=d
                cols.append(np.concatenate([col.real,col.imag],axis=1).ravel())
    return np.stack(cols,axis=1)


def fisher(w,beta):
    ph=np.exp(-1j*np.angle(w.G@w.S+w.B))
    c=ph[:,None,:]*w.S[None,:,:]
    grad=np.concatenate([c.real,-c.imag],axis=1)
    return (grad*(4*beta)[:,None,:])@grad.transpose(0,2,1)


def compute(w,beta):
    j=fisher(w,beta)
    unc=float(np.trace(np.linalg.inv(j),axis1=-2,axis2=-1).sum())
    d=tangent(w)
    u,s,_=np.linalg.svd(d,full_matrices=False)
    keep=s>s[0]*1e-9;u=u[:,keep].reshape(len(w.G),2*w.G.shape[1],-1)
    restricted=np.einsum("nai,nab,nbj->ij",u,j,u,optimize=True)
    con=float(np.trace(np.linalg.inv(restricted)))
    assert con>0 and con<=unc*(1+1e-8)
    return unc,con,int(keep.sum()),float(np.linalg.cond(restricted))


def validate(w):
    from scripts.constrained_crlb import jacobian
    d=tangent(w)
    np.testing.assert_allclose(d,jacobian(w.psi,w.alpha,*w.G.shape),atol=1e-13)
    # Finite differences for one path frequency, real gain, imaginary gain.
    eps=1e-6;n,k=w.G.shape;freq=w.psi[0][0];gain=w.alpha[0][0]
    def contribution(f,a):
        g=np.zeros((n,k),complex);g[:,0]=a*np.exp(-1j*np.arange(n)*f)
        return np.concatenate([g.real,g.imag],axis=1).ravel()
    fd=np.stack([(contribution(freq+eps,gain)-contribution(freq-eps,gain))/(2*eps),
                 (contribution(freq,gain+eps)-contribution(freq,gain-eps))/(2*eps),
                 (contribution(freq,gain+1j*eps)-contribution(freq,gain-1j*eps))/(2*eps)],axis=1)
    np.testing.assert_allclose(d[:,:3],fd,rtol=1e-6,atol=1e-8)
    return float(np.max(abs(d[:,:3]-fd)))


def main():
    from threadpoolctl import threadpool_limits
    threadpool_limits(1)
    out=OUT/"bounds";out.mkdir(exist_ok=True)
    for c in cells():
        if not(c["tag"].startswith("array_N32_P30")):continue
        path=out/(c["tag"]+".npz")
        if path.exists():continue
        t0=time.perf_counter();ws=[world(c,t) for t in range(c["n"])]
        sigma=ws[0].sigma2
        amps=np.concatenate([abs(w.G@w.S+w.B).ravel() for w in ws])
        grid=np.linspace(max(0,amps.min()*.99),amps.max()*1.01,800)
        vals=np.array([rician_fisher_scalar(float(a),sigma).beta for a in grid])
        interp=PchipInterpolator(grid,vals)
        probes=np.random.default_rng(100).choice(amps,200,replace=False)
        exact=np.array([rician_fisher_scalar(float(a),sigma).beta for a in probes])
        rel=float(np.max(abs(interp(probes)-exact)/np.maximum(exact,1e-30)))
        assert rel<1e-4,rel
        fd=validate(ws[0]);rows=[]
        for t,w in enumerate(ws):
            u,v,r,cond=compute(w,interp(abs(w.G@w.S+w.B)))
            rows.append([c["seed_base"]+t,u,v,float(np.sum(abs(w.G)**2)),r,cond])
        np.savez_compressed(path,rows=rows,columns=np.array(["trial","crlb","ccrb","den","rank","condition"]),
                            interpolation_max_relative_error=rel,finite_difference_max_error=fd)
        print(c["tag"],"seconds",time.perf_counter()-t0,"interp",rel,"fd",fd,flush=True)

if __name__=="__main__":main()
