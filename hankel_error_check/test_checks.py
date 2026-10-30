import unittest
import numpy as np
import torch
from scipy.special import i0e,i1e
from scipy.stats import rice
from sahure_expansion.data import world
from codex_verification.solver import fit
from sahure.math import spectral_initial,posterior_field,right_solve
from .core import (latent_covariance,rice_variance,integrated_mean_variance,
                   frozen_variance,path,jacobians,propagate_latent,operator,realify,predictions)


class Checks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.w,_=world(0,202610090901)

    def test_conditional_moments(self):
        angles=np.arange(32768)*2*np.pi/32768
        for kappa in (0.,.1,1.,5.,100.):
            z=2.;theta=.7;mu=kappa/4*np.exp(1j*theta)
            weights=np.exp(kappa*(np.cos(angles-theta)-1));weights/=weights.sum()
            x=z*np.stack((np.cos(angles),np.sin(angles)),axis=-1)
            mean=weights@x;err=x-mean
            cov=np.einsum('r,ri,rj->ij',weights,err,err)
            expected=z*i1e(kappa)/i0e(kappa)*np.array([np.cos(theta),np.sin(theta)])
            np.testing.assert_allclose(mean,expected,atol=1e-12)
            np.testing.assert_allclose(latent_covariance(np.array(z),np.array(mu),1.),cov,atol=1e-12)
        cov=latent_covariance(np.array([1.,1.]),np.array([1e-12,1e8]),1.)
        self.assertTrue(np.isfinite(cov).all())
        self.assertGreaterEqual(np.linalg.eigvalsh(cov).min(),-1e-13)

    def test_total_variance(self):
        rng=np.random.default_rng(202610090902)
        for a in (0.,.2,2.,20.):
            mu=a*np.exp(.7j);x=mu+(rng.normal(size=200000)+1j*rng.normal(size=200000))/np.sqrt(2)
            z=abs(x);k=2*z*a
            q=z*i1e(k)/i0e(k)*np.exp(1j*np.angle(mu))
            covq=np.cov(np.stack((q.real,q.imag)),bias=True)
            missing=latent_covariance(z,np.full(len(z),mu),1.).mean(0)
            np.testing.assert_allclose(covq+missing,.5*np.eye(2),atol=.006)

    def test_variance_numerics(self):
        r=np.array([0.,.02,.1,.5,1.,2.,10.,100.,1000.])
        x=integrated_mean_variance(r,64);xx=integrated_mean_variance(r,128)
        np.testing.assert_allclose(x,xx,rtol=2e-5,atol=1e-9)
        mid=np.array([.0001,.073,1.17,3.25,14.35,212.])
        np.testing.assert_allclose(frozen_variance(mid,1.),integrated_mean_variance(mid,128),rtol=.003)
        np.testing.assert_allclose(rice_variance(r[:7],1.),rice.var(np.sqrt(2)*r[:7],scale=1/np.sqrt(2)),rtol=1e-10,atol=1e-12)
        # Independently integrate centered magnitude errors without large subtraction.
        from numpy.polynomial.hermite import hermgauss
        nodes,weights=hermgauss(128);w=weights[:,None]*weights[None,:]/np.pi
        for a in (1.,10.,100.,10000.):
            z=np.sqrt((a+nodes[:,None])**2+nodes[None,:]**2)
            delta=(2*a*nodes[:,None]+nodes[:,None]**2+nodes[None,:]**2)/(z+a)
            vv=np.sum(w*delta*delta)-np.sum(w*delta)**2
            np.testing.assert_allclose(rice_variance(np.array(a),1.),vv,rtol=.001,atol=1e-8)

    def test_batched_replay(self):
        w=self.w
        pp=path(w.Z[None],w.S,w.B,w.sigma2)
        for t,(_,g) in pp.items():
            np.testing.assert_allclose(g[0],fit(w.S,w.Z,w.B,w.sigma2,iterations=t)[0],rtol=1e-10,atol=1e-10)

    def test_jacobian_autograd(self):
        w=self.w;s=w.S;b=w.B[:1];z=w.Z[None,:1];v=w.sigma2
        for start in ('spectral','zero'):
            jj=jacobians(z,s,b,v,start,stages=(1,5),step=1e-5)
            half=jacobians(z,s,b,v,start,stages=(1,5),step=5e-6)
            for t in (1,5):
                ss=torch.tensor(s[None]);bb=torch.tensor(b[None]);vv=torch.tensor([v],dtype=torch.float64)
                def fn(zz):
                    zz=zz.reshape(1,1,-1)
                    g=spectral_initial(zz,ss,bb) if start=='spectral' else torch.zeros(1,1,3,dtype=torch.complex128)
                    # spectral_initial is deliberately no_grad in production;
                    # recompute it with autograd here for the full-map derivative.
                    if start=='spectral':
                        aa=torch.cat((ss.mT[:,None],bb[...,None]),-1)
                        m=aa.mH@(zz[...,None]*aa);m=(m+m.mH)/2
                        _,ev=torch.linalg.eigh(m);ev=ev[...,-1]
                        proj=(aa@ev[...,None]).squeeze(-1).abs()
                        scale=(proj*zz).sum(-1)/proj.square().sum(-1)
                        g=(scale[...,None]*ev*torch.exp(-1j*torch.angle(ev[...,-1]))[...,None])[...,:3]
                    for _ in range(t):
                        q=posterior_field(zz,g@ss+bb,vv)
                        g=right_solve((q-bb)@ss.mH,ss@ss.mH)
                    return torch.cat((g.real,g.imag),-1).flatten()
                automatic=torch.autograd.functional.jacobian(fn,torch.tensor(z.flatten(),requires_grad=True)).numpy()
                np.testing.assert_allclose(jj[t][0,0],automatic,rtol=2e-5,atol=2e-7)
                np.testing.assert_allclose(jj[t],half[t],rtol=2e-5,atol=2e-7)

    def test_covariance_propagation_and_hankel(self):
        rng=np.random.default_rng(202610090903);s=self.w.S;op=operator(s)
        raw=rng.normal(size=(30,2,2));c=raw@raw.swapaxes(-1,-2)
        a=np.concatenate((op.real,op.imag),-1);b=np.concatenate((-op.imag,op.real),-1)
        expected=np.zeros((6,6))
        for p in range(30):
            m=np.stack((a[p],b[p]),axis=-1);expected+=m@c[p]@m.T
        np.testing.assert_allclose(propagate_latent(c,op),expected,atol=1e-13)
        n=8;idx=np.add.outer(np.arange(5),np.arange(4)).ravel();d=np.eye(n)[idx]
        count=np.bincount(idx);cv=np.diag(rng.uniform(.1,2,n));hc=d@cv@d.T
        self.assertEqual(np.linalg.matrix_rank(hc),n)
        np.testing.assert_allclose(np.trace(hc),count@np.diag(cv))
        np.testing.assert_allclose(np.linalg.norm(hc)**2,np.sum(count**2*np.diag(cv)**2))


if __name__=='__main__':unittest.main()
