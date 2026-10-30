import unittest
import numpy as np
import torch
from sahure_expansion.data import world
from fair_optshrink.estimator import tensors, estimate
from paper2_optshrink_loop.estimator import trajectory, shrink, estimate as numpy_estimate
from sahure.classical import likelihood
from .estimator import fit_candidates, parameters, select_and_estimate
from .run import CONFIG, SPECS, frozen_parent


class Checks(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(1)

    def test_independent_rank_scores_selection_refit(self):
        for snr,rsr in ((-10,5),(5,12),(20,20)):
            w,_=world(0,202610519001,snr=snr,rsr=rsr,L=4)
            obs=tensors(dict(z=w.Z[None],s=w.S[None],b=w.B[None],sigma2=np.array([w.sigma2])))
            parent=frozen_parent();entry=parent['choices'][f'{snr}:{rsr}'];minima=[]
            for place in ('loop','final'):
                base=entry[place];t=base['iterations'];params=parameters(base['strength'])
                if place=='loop': gg=trajectory(w.Z[:,:21],w.S[:,:21],w.B[:,:21],w.sigma2,params,(t,))[t]
                else:
                    gg=trajectory(w.Z[:,:21],w.S[:,:21],w.B[:,:21],w.sigma2,[params[0]],(t,))[t]
                    gg=shrink(np.repeat(gg,16,0),params)
                ss=likelihood(w.Z[:,21:],gg@w.S[:,21:]+w.B[:,21:],w.sigma2).mean((-2,-1))
                cfg,scores=fit_candidates(obs,base)
                np.testing.assert_allclose(scores.numpy(),ss,rtol=2e-8,atol=2e-9)
                self.assertEqual(cfg['rank'],int(ss.argmin()));minima.append(float(ss.min()))
                actual,diagnostic=select_and_estimate(obs,parent,place)
                expected=numpy_estimate(w.Z,w.S,w.B,w.sigma2,cfg)
                np.testing.assert_allclose(actual.numpy()[0],expected,rtol=2e-8,atol=2e-9)
            _,diag=select_and_estimate(obs,parent,'joint')
            self.assertEqual(diag['config']['placement'],('loop','final')[int(minima[1]<minima[0])])

    def test_no_filter_and_truth_free_interface(self):
        self.assertEqual(parameters(.5)[0],dict(rank=0,strength=0.))
        import inspect
        self.assertEqual(list(inspect.signature(select_and_estimate).parameters),['obs','frozen','mode'])

    def test_disjoint_banks(self):
        seeds=[s+100*j for s in CONFIG['banks'] for j in range(len(SPECS))]
        self.assertEqual(len(seeds),len(set(seeds)))
        old={s+100*j for s in (202610210001,202610310001) for j in range(len(SPECS))}
        self.assertFalse(old.intersection(seeds))


if __name__=='__main__':unittest.main()
