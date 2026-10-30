"""Independent smoke checks; no policy selection or test outcomes are used."""
import os
for _name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[_name]='1'
import json
from pathlib import Path
import numpy as np
import torch
from sahure.data import Observations
from tcom_followups_20261017.common import REPO,sha,write_json,em_estimate
from tcom_pilot_em_20261018.study import generated,first30
from tcom_pilot_em_20261018.training_data import paired_split
from tcom_runtime_bypass_20261017.bypass_audit import call_count_checks,tagged_observations

ROOT=REPO/'results/tcom_retrained_bypass_20261018'
SMOKE_SEED=2026101820


def combine(rows):
    return Observations(*(torch.cat([row.args()[j] for row in rows]) for j in range(4)))


def smoke(snr,ref,index=0):
    return first30(generated(SMOKE_SEED,[index],snr,ref,None))


@torch.no_grad()
def run():
    from . import study
    torch.set_num_threads(1)
    try:torch.set_num_interop_threads(1)
    except RuntimeError:pass
    wrapper=study.engine();policy={'5':20.,'12':25.,'20':40.}
    calls=call_count_checks(wrapper,policy)
    tagged=tagged_observations([20,21,25,26,40,41],[5,5,12,12,20,20])
    mask=wrapper.masks(tagged,policy)['bypass']
    assert mask.tolist()==[False,True,False,True,False,True]
    altered=Observations(tagged.z+1000,tagged.s+20+10j,tagged.b,tagged.sigma2)
    assert torch.equal(mask,wrapper.masks(altered,policy)['bypass'])
    boundary=tagged_observations([20-1e-8,20,20+1e-12,20+1e-8,60],[12]*5)
    assert wrapper.masks(boundary,{'5':20.,'12':20.,'20':20.})['bypass'].tolist()==[False,False,False,True,True]
    for invalid in (0.,-1.,float('nan'),float('inf')):
        bad=Observations(tagged.z,tagged.s,tagged.b,torch.full_like(tagged.sigma2,invalid))
        try:wrapper.masks(bad,policy)
        except ValueError:pass
        else:raise AssertionError('invalid supplied noise variance accepted')
    try:wrapper.masks(tagged,{'5':19.99,'12':25.,'20':40.})
    except ValueError:pass
    else:raise AssertionError('cutoff below ordinary preservation boundary accepted')
    ordinary=combine([smoke(s,r,i) for i,(s,r) in enumerate([(0,5),(5,12),(10,20),(15,5),(20,12),(20,20)])])
    mixed=combine([smoke(s,r,20+i) for i,(s,r) in enumerate([(20,5),(30,5),(25,12),(30,12),(40,20),(60,20)])])
    numerical={}
    for name in wrapper.models:
        a=wrapper.models[name].state_dict();b=wrapper.unfiltered_models[name].state_dict()
        assert a.keys()==b.keys() and all(torch.equal(a[k],b[k]) for k in a)
        original=wrapper.original(ordinary,name);row={}
        off=wrapper.unfiltered(mixed,name)
        gate0=wrapper.models[name](*mixed.args(),gate_override=0.)
        torch.testing.assert_close(off,gate0,atol=1e-12,rtol=1e-12)
        row['filter_off_vs_zero_gate_max_abs_error']=float((off-gate0).abs().max())
        for fallback in ('same_steps','em100'):
            assert torch.equal(wrapper.estimate(ordinary,name,policy,fallback=fallback),original)
            grouped=wrapper.estimate(mixed,name,policy,fallback=fallback)
            scalar=torch.cat([wrapper.estimate(mixed.batch(slice(i,i+1)),name,policy,fallback=fallback) for i in range(len(mixed))])
            torch.testing.assert_close(grouped,scalar,atol=1e-11,rtol=1e-11)
            row[f'{fallback}_ordinary_bitwise_equal']=True
            row[f'{fallback}_grouped_vs_scalar_max_abs_error']=float((grouped-scalar).abs().max())
        numerical[name]=dict(**row,complete_checkpoint_state_unchanged=True)
    high=combine([smoke(60,r,50+i) for i,r in enumerate((5,12,20))])
    fast=wrapper.em100(high);canonical=em_estimate(high,100)
    torch.testing.assert_close(fast,canonical,atol=1e-10,rtol=1e-10)
    # Check the shared 39-column generator/split without labels, independently of development banks.
    bank=generated(SMOKE_SEED,[80,81],20.,12.,None)
    context21,context30,heldout=paired_split(bank,2026101820)
    assert context30.z.shape[-1]==30 and heldout.z.shape[-1]==9
    torch.testing.assert_close(context21.z,context30.z[:,:,:21],atol=0,rtol=0)
    from tcom_pilot_em_20261018.training_data import split_indices
    _,ids,hold=split_indices(len(bank),2026101820)
    assert torch.equal(torch.cat((ids,hold),1).sort(1).values,torch.arange(39).expand(len(bank),39))
    torch.testing.assert_close(context30.z,bank.columns(ids).z,atol=0,rtol=0)
    changed_holdout=Observations(heldout.z+999,heldout.s,heldout.b,heldout.sigma2)
    assert not torch.equal(changed_holdout.z,heldout.z)
    assert torch.equal(context30.z,bank.columns(ids).z)
    dev=study.make_anchor(SMOKE_SEED,[90,91],None,12.,truth=False)
    test=study.make_anchor(SMOKE_SEED,[90,91],None,12.,truth=True)
    assert not {'g','unit_noise','paths','mu'}.intersection(dev)
    for snr in study.SNRS:
        observed=study.observed(dev,snr,P=39)
        observed_truth=study.observed(test,snr,P=39)
        for a,b in zip(observed.args(),observed_truth.args()):torch.testing.assert_close(a,b,atol=0,rtol=0)
        fitting,held=study.fit_holdout(observed,'random3to7',12.)
        order=torch.rand(2,39,generator=torch.Generator().manual_seed(2026101825+200+1)).argsort(-1)
        for a,b in zip(fitting.args(),observed.columns(order[:,:30]).args()):torch.testing.assert_close(a,b,atol=0,rtol=0)
        for a,b in zip(held.args(),observed.columns(order[:,30:]).args()):torch.testing.assert_close(a,b,atol=0,rtol=0)
        assert torch.equal(torch.cat((order[:,:30],order[:,30:]),1).sort(1).values,torch.arange(39).expand(2,39))
        assert torch.equal(study.observed(test,snr,P=30).z,observed.z[...,:30])
    return dict(passed=True,smoke_seed=SMOKE_SEED,synthetic_policy=policy,
        dispatch_invariant_to_magnitudes_and_pilot_values=True,strict_boundaries_pass=True,
        invalid_variance_and_sub20_cutoffs_rejected=True,exactly_one_branch_per_world=calls,
        numerical=numerical,high_snr_em100_max_abs_error=float((fast-canonical).abs().max()),
        disjoint_30fit_9heldout=True,heldout_observations_not_in_context=True,
        new_runner_observable_only_development_and_coupled39_to30_parity=True,
        source_sha256=sha(__file__),runner_sha256=sha(study.__file__),
        interpretation='Synthetic smoke policies test implementation only. Gate uses supplied noise, K, and known reference. These checks do not choose cutoffs or establish robustness.')


if __name__=='__main__':
    result=run();write_json(ROOT/'checks.json',result)
    print(json.dumps(result,indent=2))
