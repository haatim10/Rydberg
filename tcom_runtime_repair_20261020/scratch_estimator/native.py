"""Exploratory float64 CPU implementation of the frozen learned estimator.

No training, iteration truncation, dtype reduction, or changes to observables.
This module is not yet an approved deployment implementation. Weight and
shape-only constants are prepared once, just as checkpoint loading is.
"""
import math
import numpy as np
from scipy.special import expit
import torch
from tcom_runtime_bypass_20261017.sparse_fast import as_numpy, spectral_initial
from tcom_pilot_em_20261018.continuation import posterior, nll


class NativeEstimator:
    def __init__(self, model, *, shrink_mode='solve'):
        cfg=model.config
        if cfg.core!='em' or cfg.structure!='adaptive' or cfg.final_only:
            raise ValueError('exploration supports the frozen adaptive EM model')
        if model.steps.dtype!=torch.float64:
            raise ValueError('retain the original float64 inference precision')
        if shrink_mode not in ('solve','svd'):
            raise ValueError(shrink_mode)
        self.config=cfg
        self.shrink_mode=shrink_mode
        self.gamma=(2*torch.sigmoid(model.steps.detach())).cpu().numpy().copy()
        self.w1=model.net[0].weight.detach().cpu().numpy().copy()
        self.b1=model.net[0].bias.detach().cpu().numpy().copy()
        self.w2=model.net[2].weight.detach().cpu().numpy().copy()
        self.b2=model.net[2].bias.detach().cpu().numpy().copy()
        self.shapes={}

    def shape_constants(self,n):
        if n not in self.shapes:
            cols=(n-1)//2+1;rows=n-cols+1
            idx=(np.arange(rows)[:,None]+np.arange(cols)[None,:]).reshape(-1)
            counts=np.bincount(idx,minlength=n)
            self.shapes[n]=(cols,rows,np.eye(cols,dtype=np.complex128),
                np.eye(n,dtype=np.float64)[idx]/counts[None,:])
        return self.shapes[n]

    def __call__(self,z,s,b,sigma2,*,diagnostics=False,gate_override=None):
        z,s,b,v=as_numpy((z,s,b,sigma2))
        batch,n,p=z.shape;k=s.shape[1]
        cols,rows,eye,unhankel=self.shape_constants(n)
        sh=s.conj().swapaxes(-1,-2);gram=s@sh
        # Exactly the same least-squares map; its pilot-only solve is shared
        # across the ten stages instead of repeated after each right product.
        op=np.linalg.solve(gram.swapaxes(-1,-2),sh.swapaxes(-1,-2)).swapaxes(-1,-2)
        inverse=np.linalg.solve(gram,np.broadcast_to(np.eye(k,dtype=np.complex128),gram.shape))
        noise_proxy=v*np.trace(inverse,axis1=-2,axis2=-1).real/k
        g=spectral_initial(z,s,b)
        trace=[];cap=min(rows,cols)
        for t,gamma in enumerate(self.gamma):
            q=posterior(z,g@s+b,v)
            dc=g+gamma*((q-b)@op-g)
            h=np.lib.stride_tricks.sliding_window_view(dc.swapaxes(-1,-2),cols,axis=-1)
            if self.shrink_mode=='svd':
                left,sv,right=np.linalg.svd(h,full_matrices=False)
            else:
                sv=np.linalg.svd(h,compute_uv=False)
            weights=sv/np.maximum(sv.sum(-1,keepdims=True),1e-12)
            ent=-np.sum(weights*np.log(np.maximum(weights,1e-12)),axis=-1)
            residual=np.log1p(np.mean((z-np.abs(dc@s+b))**2,axis=(-2,-1))/v)
            power=np.maximum(np.mean(np.abs(dc)**2,axis=-2),1e-12)
            features=np.stack((np.exp(ent)/cap,weights[...,0],weights[...,-((cap+1)//2):].sum(-1),
                np.broadcast_to(residual[:,None],(batch,k)),np.log1p(noise_proxy[:,None]/power),
                np.full((batch,k),p/(2*k)),np.full((batch,k),math.log(cap)),
                np.full((batch,k),(t+1)/len(self.gamma))),axis=-1)
            hidden=features@self.w1.T+self.b1
            hidden=hidden*expit(hidden)
            raw=hidden@self.w2.T+self.b2
            # Torch softplus uses an exact linear branch beyond threshold 20.
            relative_tau=.001+np.where(raw[...,0]>20,raw[...,0],np.logaddexp(0.,raw[...,0]))
            beta=expit(raw[...,1])
            if self.config.gate_one:beta=np.ones_like(beta)
            if gate_override is not None:beta=np.full_like(beta,gate_override)
            scale=np.sqrt(np.maximum(np.sum(np.abs(h)**2,axis=(-2,-1))/cap,1e-24))
            tau=relative_tau*scale
            if self.shrink_mode=='svd':
                mapped=sv**3/(sv**2+tau[...,None]**2)
                hs=(left*mapped[...,None,:])@right
            else:
                hn=h/scale[...,None,None];t2=relative_tau[...,None,None]**2
                hh=hn.conj().swapaxes(-1,-2)
                solved=np.linalg.solve(hh@hn+t2*eye,hh)
                hs=(hn-t2*solved.conj().swapaxes(-1,-2))*scale[...,None,None]
            structured=(hs.reshape(batch,k,-1)@unhankel).swapaxes(-1,-2)
            g=dc+beta[:,None,:]*(structured-dc)
            if diagnostics:
                trace.append(dict(g=torch.from_numpy(g.copy()),beta=torch.from_numpy(beta.copy()),
                    tau=torch.from_numpy(tau.copy()),gamma=torch.tensor(gamma),
                    correction=torch.from_numpy(np.linalg.norm(g-dc,axis=-2)/np.maximum(np.linalg.norm(dc,axis=-2),1e-12)),
                    nll=torch.from_numpy(nll(z,g@s+b,v).mean((-2,-1)))))
        result=torch.from_numpy(g)
        return (result,trace) if diagnostics else result

    def estimate(self,obs,**kwargs):
        return self(*obs.args(),**kwargs)
