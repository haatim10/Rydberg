"""Independent score check, old-selector parity, and reference-phase symmetry."""
import unittest
import numpy as np
from scipy.stats import rice
import torch
from codex_verification.solver import fit,select
from sahure.classical import likelihood
from sahure.train import load_model
from sahure_expansion.data import world
from paper2_review_checks.run import rank_scores,paired_classical

class Checks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.w,_=world(0,2026100999,N=8,K=2,P=10)

    def test_rician_score_against_scipy(self):
        z=np.array([.2,.8,2.,4.]);a=np.array([0.,.3,1.,5.]);v=.7
        omitted=likelihood(z,a,v)
        actual=-rice.logpdf(z,a/np.sqrt(v/2),scale=np.sqrt(v/2))
        np.testing.assert_allclose(omitted-np.log(2*z/v),actual,atol=1e-12)

    def test_residual_rank_exact_old_selector(self):
        w=self.w;nv=3;ranks=np.arange(1,5)
        gs=fit(w.S[:,:-nv],w.Z[:,:-nv],w.B[:,:-nv],w.sigma2,iterations=25,ranks=ranks,sweeps=4)
        residual,nll=rank_scores(w.Z,w.S,w.B,w.sigma2,gs)
        oldrank,oldscores=select(w.S,w.Z,w.B,w.sigma2)
        np.testing.assert_allclose(residual,oldscores,rtol=0,atol=0)
        self.assertEqual(int(ranks[residual.argmin()]),oldrank)
        self.assertTrue(np.isfinite(nll).all())

    def test_classical_reference_phase_equivariance(self):
        w=self.w;phase=np.exp(.17j)
        a,ra=paired_classical(w.Z,w.S,w.B,w.sigma2)
        b,rb=paired_classical(w.Z,w.S,w.B*phase,w.sigma2)
        self.assertEqual(ra['ranks'],rb['ranks'])
        for name in a:np.testing.assert_allclose(b[name],phase*a[name],atol=2e-10,rtol=2e-10)

    def test_learned_reference_phase_equivariance(self):
        w=self.w;phase=np.exp(.17j)
        model,_=load_model('results/sahure_expansion/checkpoints/seed_9101/em_adaptive_best.pt')
        with torch.no_grad():
            a=model.estimate(w.Z,w.S,w.B,w.sigma2).numpy()
            b=model.estimate(w.Z,w.S,w.B*phase,w.sigma2).numpy()
        np.testing.assert_allclose(b,phase*a,atol=2e-10,rtol=2e-10)

if __name__=='__main__':unittest.main()
