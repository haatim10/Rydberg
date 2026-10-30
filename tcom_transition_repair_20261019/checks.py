"""Independent smoke checks of observable dispatch and learned-initialized EM."""
import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[_key]='1'
import json
import numpy as np
import torch
from sahure.data import Observations
from sahure.math import posterior_field,right_solve
from tcom_followups_20261017.common import REPO,sha,write_json,world_batch,em_estimate
from tcom_runtime_bypass_20261017.bypass_audit import tagged_observations
from . import inference

ROOT=REPO/'results/tcom_transition_repair_20261019'
SMOKE_SEED=2026101920


def combine(rows):return Observations(*(torch.cat([r.args()[j] for r in rows]) for j in range(4)))


def table(choice='warm100'):
    return {f'{r:g}':{f'{s:g}':dict(choice=choice) for s in inference.SNRS} for r in inference.RSRS}


def branch_checks(est):
    policy=table();policy['5']['35']['choice']='warm5';policy['12']['40']['choice']='warm20'
    policy['20']['45']['choice']='spectral100'
    obs=tagged_observations([0,20,35,40,45,60],[5,12,5,12,20,5])
    choices=inference.choices_for(obs,policy)
    assert choices==['original','original','warm5','warm20','spectral100','warm100']
    altered=Observations(obs.z+700,obs.s+100+50j,obs.b,obs.sigma2)
    assert inference.choices_for(altered,policy)==choices
    saved=est.candidate;calls=[]
    def spy(subset,model,choice):
        ids=subset.z[:,0,0].to(torch.int64).tolist();calls.append(dict(choice=choice,local_ids=ids))
        return torch.full((len(subset),32,3),inference.MENU.index(choice)+1,dtype=torch.complex128)
    est.candidate=spy
    try:
        out=est.repair(obs,next(iter(est.models)),policy)
        expected=[dict(choice=c,local_ids=[i for i,x in enumerate(choices) if x==c]) for c in sorted(set(choices))]
        assert calls==expected and sorted(i for r in calls for i in r['local_ids'])==list(range(len(obs)))
        assert out[:,0,0].real.to(torch.int64).tolist()==[inference.MENU.index(c)+1 for c in choices]
    finally:est.candidate=saved
    boundary=tagged_observations([20-1e-8,20,20+1e-12,20+1e-8],[12]*4)
    assert inference.choices_for(boundary,table())==['original','original','original','warm100']
    for value in (0.,-1.,float('nan'),float('inf')):
        bad=Observations(obs.z,obs.s,obs.b,torch.full_like(obs.sigma2,value))
        try:inference.receiver_keys(bad)
        except ValueError:pass
        else:raise AssertionError('invalid variance accepted')
    return dict(one_candidate_per_world=calls,gate_invariant_to_magnitudes_and_pilot_values=True,ordinary_boundary_exact=True)


@torch.no_grad()
def run():
    from . import study
    torch.set_num_threads(1)
    try:torch.set_num_interop_threads(1)
    except RuntimeError:pass
    est=study.engine();dispatch=branch_checks(est)
    conditions=((0.,5.,2),(20.,12.,4),(35.,5.,2),(40.,5.,4),(45.,5.,None),(60.,12.,None))
    obs=combine([world_batch(SMOKE_SEED,[i],snr,rsr,L)[0] for i,(snr,rsr,L) in enumerate(conditions)])
    op=right_solve(obs.s.mH,obs.s@obs.s.mH);numerical={}
    spectral=em_estimate(obs,100)
    for name in est.models:
        initial=est.original(obs,name);keep=initial.clone();pred=est.candidate_predictions(obs,name)
        assert torch.equal(initial,pred['original'])
        reference=initial.clone();maximum=0.
        for step in range(1,101):
            reference=(posterior_field(obs.z,reference@obs.s+obs.b,obs.sigma2)-obs.b)@op
            if step in (5,20,100):
                torch.testing.assert_close(reference,pred[f'warm{step}'],atol=1e-10,rtol=1e-10)
                maximum=max(maximum,float((reference-pred[f'warm{step}']).abs().max()))
        torch.testing.assert_close(pred['spectral100'],spectral,atol=1e-10,rtol=1e-10)
        assert torch.equal(initial,keep)
        ordinary=obs.batch(slice(0,2))
        assert torch.equal(est.repair(ordinary,name,table()),est.original(ordinary,name))
        policy=table('original')
        for (snr,ref,_),choice in zip(conditions,('warm100','spectral100','warm5','warm20','warm100','spectral100')):
            policy[f'{ref:g}'][f'{snr:g}']['choice']=choice
        batched=est.repair(obs,name,policy)
        scalar=torch.cat([est.repair(obs.batch(slice(i,i+1)),name,policy) for i in range(len(obs))])
        torch.testing.assert_close(batched,scalar,atol=1e-10,rtol=1e-10)
        # Warm calls must initialize once from the learned output and never invoke spectral EM.
        original,em=est.original,est.em100;counts=dict(original=0,em100=0)
        def count_original(o,n):counts['original']+=1;return original(o,n)
        def count_em(o):counts['em100']+=1;return em(o)
        est.original=count_original;est.em100=count_em
        try:
            est.candidate(obs,name,'warm5');assert counts==dict(original=1,em100=0)
            counts=dict(original=0,em100=0);est.candidate(obs,name,'spectral100');assert counts==dict(original=0,em100=1)
        finally:est.original=original;est.em100=em
        numerical[name]=dict(maximum_canonical_warm_error=maximum,
            maximum_canonical_spectral_error=float((pred['spectral100']-spectral).abs().max()),
            maximum_grouped_vs_scalar_error=float((batched-scalar).abs().max()),
            ordinary_bitwise_identity=True,warm_starts_learned_once_without_spectral=True,spectral_skips_learned=True)
    # Actual new generator and new fitting split, not an earlier study's split.
    dev=study.make_anchor(SMOKE_SEED,[80,81],None,12.,truth=False)
    truth=study.make_anchor(SMOKE_SEED,[80,81],None,12.,truth=True)
    assert not {'g','G','mu','paths','unit_noise','truth_g'}.intersection(dev)
    for snr in study.SNRS:
        bank=study.observed(dev,snr,P=39);reference=study.observed(truth,snr,P=39)
        for a,b in zip(bank.args(),reference.args()):torch.testing.assert_close(a,b,atol=0,rtol=0)
        fitting,held=study.fit_holdout(bank,'random3to7',12.)
        order=torch.rand(2,39,generator=torch.Generator().manual_seed(2026101925+200+1)).argsort(-1)
        assert torch.equal(torch.cat((order[:,:30],order[:,30:]),1).sort(1).values,torch.arange(39).expand(2,39))
        for a,b in zip(fitting.args(),bank.columns(order[:,:30]).args()):torch.testing.assert_close(a,b,atol=0,rtol=0)
        for a,b in zip(held.args(),bank.columns(order[:,30:]).args()):torch.testing.assert_close(a,b,atol=0,rtol=0)
        assert torch.equal(study.observed(truth,snr,P=30).z,bank.z[...,:30])
    result=dict(passed=True,smoke_seed=SMOKE_SEED,dispatch=dispatch,numerical=numerical,
        observable_only_development=True,disjoint_30fit_9heldout=True,all17_coupled_SNRs_and_deployment_slice_verified=True,
        source_sha256=sha(__file__),runner_sha256=sha(study.__file__),inference_sha256=sha(inference.__file__),
        interpretation='Implementation checks on synthetic policies and a distinct smoke bank only; no calibration, selection or test tuning.')
    write_json(ROOT/'checks.json',result);return result


if __name__=='__main__':print(json.dumps(run(),indent=2))
