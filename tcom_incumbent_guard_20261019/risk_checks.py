"""Artificial truth verifies a mathematical identity, never a policy choice."""
import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_key] = '1'
import json
import numpy as np
import torch
from sahure.data import Observations
from tcom_followups_20261017.common import REPO, sha, write_json
from .risk import paired_risk, paired_risk_numpy

OUT = REPO / 'results/tcom_incumbent_guard_20261019/risk_checks.json'


def cn(rng, shape):
    return (rng.standard_normal(shape) + 1j*rng.standard_normal(shape)) / np.sqrt(2.)


def run():
    rng = np.random.default_rng(2026101940)
    # Fixed toy channels and estimates, unrelated to any campaign channel.
    n, k, p, draws, chunk = 3, 2, 9, 65536, 2048
    g = cn(rng, (n, k))
    direction = cn(rng, (n, k)); direction /= np.linalg.norm(direction)
    other = cn(rng, (n, k)); other /= np.linalg.norm(other)
    configs = [(.8, .1, .2, .7), (3., .04, .7, .2), (10., .002, .12, .3),
               (2., 0., .3, -.3), (1.2, .5, 0., .65)]
    results = []; zero_checks = symmetry_checks = 0; maximum_parity = 0.
    for ci, (amplitude, variance, a, c) in enumerate(configs):
        left = g + a*direction + .07*other
        right = g + c*direction + .07*other
        if ci == 3:  # Exact equal-risk endpoints around truth; test zero expectation.
            left, right = g+a*direction, g+c*direction
        target = (np.abs(left-g)**2).sum() - (np.abs(right-g)**2).sum()
        target /= n*k
        reference = amplitude*np.exp(1j*np.arange(n)*.37)[:,None]
        scores = []; direct_scores = []
        for start in range(0, draws, chunk):
            count = min(chunk, draws-start)
            s = cn(rng, (count,k,p)); b = np.broadcast_to(reference,(count,n,p))
            noise = np.sqrt(variance)*cn(rng, (count,n,p))
            z = np.abs(g@s+b+noise); v = np.full(count,variance)
            aa = np.broadcast_to(left,(count,n,k)); cc = np.broadcast_to(right,(count,n,k))
            values = paired_risk_numpy(z,s,b,v,aa,cc)
            scores.append(values)
            if start == 0:
                np.testing.assert_array_equal(values,-paired_risk_numpy(z,s,b,v,cc,aa));symmetry_checks+=1
                np.testing.assert_array_equal(paired_risk_numpy(z,s,b,v,aa,aa),np.zeros(count));zero_checks+=1
                close = paired_risk_numpy(z,s,b,v,aa,cc,normalize=False)/(n*k)
                np.testing.assert_allclose(values,close,atol=0,rtol=0)
                obs = Observations(*(torch.from_numpy(x.copy()) for x in (z,s,b,v)))
                wrapped = paired_risk(obs,torch.from_numpy(aa.copy()),torch.from_numpy(cc.copy()))
                maximum_parity=max(maximum_parity,float(np.max(abs(values-wrapped))))
                np.testing.assert_array_equal(values,wrapped)
                before=[x.clone() for x in obs.args()]
                paired_risk(obs,aa,cc)
                for x,y in zip(before,obs.args()):assert torch.equal(x,y)
            # A mathematically independent scalar-form expression, with the
            # empirical pilot norm difference retained and cancelled explicitly.
            m=(aa+cc)/2@s
            pseudo=(z*z-np.abs(b+m)**2-v[:,None,None])/b.conj()+m
            direct=((np.abs(aa@s)**2-np.abs(cc@s)**2).sum((-2,-1))
                    -2*np.real(((aa-cc)@s).conj()*pseudo).sum((-2,-1)))/(p*n*k)
            np.testing.assert_allclose(values,direct,atol=2e-12,rtol=1e-12)
            direct_scores.append(direct)
        values=np.concatenate(scores);mean=float(values.mean());se=float(values.std(ddof=1)/np.sqrt(draws))
        if se == 0.:
            assert mean == target  # Noiseless exact-midpoint equal-risk identity.
            standardized = 0.
        else:
            standardized=(mean-target)/se
        assert abs(standardized)<5., (ci,mean,target,se,standardized)
        if abs(target)>5*se:assert np.sign(mean)==np.sign(target)
        results.append(dict(configuration=ci,reference_amplitude=amplitude,variance=variance,
            draws=draws,expected_raw_mse_difference_per_entry=float(target),estimated_mean=mean,
            monte_carlo_standard_error=se,uncertainty_standardized_error=float(standardized)))
    # Invalid observed inputs must fail explicitly.
    invalid_checks=0
    for zz,bb,vv in ((np.array([[[-1.]]]),np.ones((1,1,1),complex),np.zeros(1)),
                     (np.ones((1,1,1)),np.zeros((1,1,1),complex),np.zeros(1)),
                     (np.ones((1,1,1)),np.ones((1,1,1),complex),np.array([-1.])),
                     (np.array([[[np.nan]]]),np.ones((1,1,1),complex),np.zeros(1))):
        try:paired_risk_numpy(zz,np.ones((1,1,1),complex),bb,vv,np.zeros((1,1,1),complex),np.ones((1,1,1),complex))
        except ValueError:invalid_checks+=1
        else:raise AssertionError('invalid observations accepted')
    result=dict(passed=True,scope='Artificial fixed-channel Monte Carlo identity validation only; no campaign truth, model selection or policy tuning.',
        normalized_quantity='Raw squared coefficient error difference divided by N*K; not truth-normalized NMSE.',
        reference='Deterministic nonzero LOS-like row phase, repeated across columns.',
        cases=results,identity_zero_checks=zero_checks,antisymmetry_checks=symmetry_checks,
        invalid_input_checks=invalid_checks,maximum_tensor_wrapper_difference=maximum_parity,
        source_sha256=sha(__file__),risk_source_sha256=sha(REPO/'tcom_incumbent_guard_20261019/risk.py'))
    write_json(OUT,result);return result


if __name__=='__main__':
    torch.set_num_threads(1)
    print(json.dumps(run(),indent=2))
