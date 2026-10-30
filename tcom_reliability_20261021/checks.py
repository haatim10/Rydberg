"""Small algebraic and actual-dispatch checks; reserved smoke data only."""
import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[_key]='1'
from pathlib import Path
from unittest.mock import patch
import json
import time
import numpy as np
import torch
from sahure.data import Observations
from sahure.math import posterior_field,right_solve
from tcom_followups_20261017.common import REPO,sha,write_json
from tcom_incumbent_guard_20261019 import risk_data
from . import inference

SMOKE_SEED=2026102100
OUT=REPO/'results/tcom_reliability_20261021'


def obs(z,s,b,v):
    return Observations(*(torch.from_numpy(np.array(x,copy=True)) for x in (z,s,b,v)))


def concatenate(items):
    return Observations(*(torch.cat([o.args()[i] for o in items]) for i in range(4)))


def sources():
    paths=[Path(__file__),Path(__file__).with_name('inference.py'),
           REPO/'tcom_incumbent_guard_20261019/risk_data.py',REPO/'sahure/math.py']
    paths += [REPO/p for p in json.loads(inference.BASE_POLICY_PATH.read_text())['source_hashes']]
    return {str(p.relative_to(REPO)):sha(p) for p in sorted(set(paths))}


@torch.no_grad()
def main():
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    rng=np.random.default_rng(SMOKE_SEED)
    cn=lambda shape:(rng.standard_normal(shape)+1j*rng.standard_normal(shape))/np.sqrt(2.)
    s=cn((2,3,30));g=cn((2,32,3))
    b=np.repeat((10**.25*np.exp(1j*rng.uniform(-np.pi,np.pi,(2,32,1)))),30,axis=-1)
    mu=g@s+b
    noiseless=obs(np.abs(mu),s,b,np.zeros(2))
    initial,valid,info=inference.algebraic_start(noiseless)
    assert valid.all() and (info['rank']==15).all()
    algebraic_error=float(np.max(np.abs(initial.numpy()-g)))
    assert algebraic_error<1e-11
    # Recover the complete nuisance-parameter equation independently.
    maximum_design_error=0.
    for i in range(2):
        c=g[i]/b[i,:,0,None]
        coefficients=[c.real.T,c.imag.T,np.abs(c.T)**2]
        for left in range(3):
            for right in range(left+1,3):
                q=c[:,left]*c[:,right].conj()
                coefficients.extend((q.real[None],q.imag[None]))
        expected=(np.abs(mu[i])**2/np.abs(b[i])**2-1).T
        error=float(np.max(np.abs(inference.intensity_design(s[i])@np.concatenate(coefficients)-expected)))
        maximum_design_error=max(maximum_design_error,error)
        assert error<1e-11
    # Expected noisy intensity tests the variance subtraction without random error.
    variance=np.full(2,3e-4)
    mean_intensity=obs(np.sqrt(np.abs(mu)**2+variance[:,None,None]),s,b,variance)
    recovered,ok,_=inference.algebraic_start(mean_intensity)
    assert ok.all() and torch.allclose(initial,recovered,atol=1e-11,rtol=1e-11)
    scalar=torch.cat([inference.algebraic_start(mean_intensity.batch(slice(i,i+1)))[0] for i in range(2)])
    assert torch.equal(scalar,recovered)
    maximum_canonical=0.
    for tail in (10,20):
        alternative,ok,_=inference.fallback(mean_intensity,tail)
        assert ok.all()
        canonical=recovered.clone()
        for _ in range(tail):
            q=posterior_field(mean_intensity.z,canonical@mean_intensity.s+mean_intensity.b,mean_intensity.sigma2)
            canonical=right_solve((q-mean_intensity.b)@mean_intensity.s.mH,mean_intensity.s@mean_intensity.s.mH)
        canonical=inference.finish(mean_intensity,canonical)
        error=float((alternative-canonical).abs().max());maximum_canonical=max(maximum_canonical,error)
        assert torch.allclose(alternative,canonical,atol=1e-10,rtol=1e-10)
    invalid_cases={}
    real_s=s.real.astype(np.complex128)
    bad_reference=b.copy();bad_reference[0,0,1]*=np.exp(.001j)
    zero_reference=b.copy();zero_reference[0,0,:]=0
    ill_s=s.copy();ill_s[:,2]*=1e-5
    for label,ss,bb in [('rank',real_s,b),('varying_reference',s,bad_reference),
                        ('zero_reference',s,zero_reference),('condition',ill_s,b)]:
        test=obs(mean_intensity.z.numpy(),ss,bb,variance)
        _,ok,diag=inference.algebraic_start(test)
        assert not ok[0],label
        if label=='condition':assert diag['condition'][0]>inference.CONDITION_LIMIT
        invalid_cases[label]=dict(rank=diag['rank'].tolist(),condition=[float(x) if np.isfinite(x) else None for x in diag['condition']],
                                  reasons=diag['invalid_reason'].tolist())
    baseline_obs=obs(np.abs(g@real_s+b),real_s,b,variance)
    baseline,baseline_diag=inference.algebraic20(baseline_obs,diagnostics=True)
    assert baseline_diag['spectral_invalid_branch'].all() and torch.isfinite(baseline).all()
    spectral=torch.from_numpy(inference.spectral_initial(*[baseline_obs.args()[i].numpy() for i in (0,1,2)]))
    expected=inference.finish(baseline_obs,inference.continuation.estimate(baseline_obs,spectral,20))
    assert torch.equal(baseline,expected)
    finite_checks=0
    for field,value in [('z',np.nan),('z',-1.),('s',np.inf),('b',np.nan),('sigma2',-1.)]:
        arrays={key:x.numpy().copy() for key,x in zip(('z','s','b','sigma2'),mean_intensity.args())}
        arrays[field].flat[0]=value
        try:inference.algebraic_start(obs(*(arrays[key] for key in ('z','s','b','sigma2'))))
        except ValueError:finite_checks+=1
        else:raise AssertionError('invalid observations accepted')
    # Force every rule through accepted/rejected/invalid/untriggered paths.
    zero=torch.zeros_like(recovered);forced=0
    for rule in inference.RULES:
        result,diag=inference.from_base(mean_intensity,zero,rule,diagnostics=True)
        assert diag['triggered'].all() and diag['accepted'].all()
        assert np.all(diag['nll_after']<diag['nll_before']-inference.ACCEPTANCE_MARGIN)
        unchanged,none=inference.from_base(mean_intensity,zero,None,diagnostics=True)
        assert torch.equal(unchanged,zero) and not none['triggered'].any()
        info=dict(rank=np.full(2,15),condition=np.ones(2),invalid_reason=np.array(['valid']*2))
        with patch.object(inference,'fallback',return_value=(zero,np.ones(2,dtype=bool),info)):
            rejected,rd=inference.from_base(mean_intensity,zero,rule,diagnostics=True)
            assert rd['triggered'].all() and not rd['accepted'].any() and torch.equal(rejected,zero)
        with patch.object(inference,'fallback',return_value=(zero,np.zeros(2,dtype=bool),info)):
            invalid,idg=inference.from_base(mean_intensity,zero,rule,diagnostics=True)
            assert idg['rank_invalid'].all() and not idg['accepted'].any() and torch.equal(invalid,zero)
        with patch.object(inference,'fallback',side_effect=AssertionError('untriggered fallback called')):
            close,cd=inference.from_base(mean_intensity,recovered,rule,diagnostics=True)
            assert not cd['triggered'].any() and torch.equal(close,recovered)
        _,force_diag=inference.from_base(mean_intensity,recovered,dict(rule,id='inert_metadata'),
                                        diagnostics=True,force_fallback=True)
        assert force_diag['triggered'].all() and force_diag['forced_calibration']
        np.testing.assert_array_equal(force_diag['score'],cd['score'])
        forced+=1
    inference.previous.check_selection();est=inference.previous.engine()
    cases=((0.,5.,2),(20.,12.,4),(30.,5.,None),(32.5,5.,2),
           (40.,5.,4),(60.,5.,None),(40.,12.,2),(60.,20.,4))
    observations=[]
    for snr,ref,L in cases:
        anchor=risk_data.make_anchor(SMOKE_SEED,[10,11],L,ref,snrs=(snr,),holdout_seed=SMOKE_SEED,truth=False)
        assert not {'g','G','truth_g','unit_noise','paths'}.intersection(anchor)
        observations.append(risk_data.observed(anchor,snr))
    mixed=concatenate(observations);maximum_scalar=0.;maximum_sharing=0.;preserved=0
    counts=dict(triggered=0,accepted=0)
    for seed in (9101,9102,9103):
        base=inference.hybrid_inference.hybrid(mixed,seed,inference.BASE_POLICY,est)
        for rule in inference.RULES:
            actual,diag=inference.gated(mixed,seed,rule,est,diagnostics=True)
            shared=inference.from_base(mixed,base,rule)
            error=float((actual-shared).abs().max());maximum_sharing=max(maximum_sharing,error)
            assert torch.equal(actual,shared)
            metadata=inference.from_base(mixed,base,dict(rule,id='inert_metadata'))
            assert torch.equal(metadata,actual)
            scalars=torch.cat([inference.gated(mixed.batch(slice(i,i+1)),seed,rule,est) for i in range(len(mixed))])
            error=float((actual-scalars).abs().max());maximum_scalar=max(maximum_scalar,error)
            assert torch.allclose(actual,scalars,atol=1e-10,rtol=1e-10)
            mask=torch.from_numpy(~diag['accepted'])
            assert torch.equal(actual[mask],base[mask])
            assert not diag['triggered'][~diag['eligible']].any()
            preserved+=int((~diag['eligible']).sum())
            counts['triggered']+=int(diag['triggered'].sum());counts['accepted']+=int(diag['accepted'].sum())
            assert diag['pilots']==30
        force_rule=dict(inference.RULES[0],id='calibration_only')
        forced_full,forced_diag=inference.gated(mixed,seed,force_rule,est,diagnostics=True,force_fallback=True)
        forced_shared=inference.from_base(mixed,base,force_rule,force_fallback=True)
        assert torch.equal(forced_full,forced_shared)
        np.testing.assert_array_equal(forced_diag['triggered'],forced_diag['eligible'])
        assert torch.equal(forced_full[~forced_diag['eligible']],base[~forced_diag['eligible']])
        for item,(snr,ref,_) in zip(observations,cases):
            if snr<=20:
                expected=est['starts'].start('adaptive',item,seed=seed)
                with patch.object(inference,'fallback',side_effect=AssertionError('ordinary fallback called')):
                    actual=inference.gated(item,seed,inference.RULES[0],est)
                assert torch.equal(expected,actual)
    result=dict(passed=True,completed_unix=time.time(),smoke_seed=SMOKE_SEED,
        artificial_cases=2,physical_conditions=[list(c) for c in cases],physical_cases_per_condition=2,
        physical_local_indices=[10,11],fit_seeds=[9101,9102,9103],rules=len(inference.RULES),
        noiseless_recovery_max_abs_error=algebraic_error,design_identity_max_abs_error=maximum_design_error,
        canonical_fallback_max_abs_error=maximum_canonical,mixed_scalar_max_abs_error=maximum_scalar,
        standalone_shared_max_abs_error=maximum_sharing,preserved_output_checks=preserved,
        forced_all_rule_branch_checks=forced,finite_input_rejections=finite_checks,invalid_cases=invalid_cases,
        standalone_invalid_spectral20_exact=True,ordinary_raw_bitwise=True,no_extra_online_pilots=True,
        rule_metadata_ignored=True,forced_calibration_full_shared_exact=True,forced_gate_keeps_original_scores=True,
        diagnostic_counts=counts,source_hashes=sources(),base_protocol_sha256=sha(inference.BASE_POLICY_PATH),
        truth_used_only_for_artificial_identity=True,development_or_test_data_used=False)
    write_json(OUT/'checks.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='source_hashes'},indent=2),flush=True)


if __name__=='__main__':main()
