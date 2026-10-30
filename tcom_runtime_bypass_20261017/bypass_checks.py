"""Numerical boundary, mixed-batch, and fallback parity checks on smoke worlds."""
import json
from pathlib import Path
import torch

from sahure.data import Observations
from tcom_followups_20261017.common import world_batch,em_estimate,write_json,sha
from . import bypass


def combine(items):
    return Observations(*(torch.cat([item.args()[j] for item in items]) for j in range(4)))


def run():
    torch.set_num_threads(1)
    wrapper=bypass.BypassEstimator()
    policy={'5':20.,'12':25.,'20':40.}
    ordinary=combine([world_batch(2026101760,[0],snr=snr,rsr=rsr,L=None)[0]
        for snr,rsr in ((-5.,5.),(0.,12.),(10.,20.),(20.,5.),(20.,12.),(20.,20.))])
    high=combine([world_batch(2026101760,[1],snr=snr,rsr=rsr,L=2)[0]
        for snr,rsr in ((25.,5.),(30.,12.),(45.,20.),(60.,5.),(60.,12.),(60.,20.))])
    mixed=combine([ordinary,high])
    results={}
    with torch.no_grad():
        for name in wrapper.models:
            original=wrapper.original(ordinary,name)
            off=wrapper.unfiltered(mixed,name)
            gate_zero=wrapper.models[name](*mixed.args(),gate_override=0.)
            zero_error=float((off-gate_zero).abs().max())
            same=wrapper.estimate(ordinary,name,policy,fallback='same_steps')
            em=wrapper.estimate(ordinary,name,policy,fallback='em100')
            ordinary_bitwise=torch.equal(same,original) and torch.equal(em,original)
            mixed_same=wrapper.estimate(mixed,name,policy,fallback='same_steps')
            mixed_em=wrapper.estimate(mixed,name,policy,fallback='em100')
            indices=wrapper.masks(mixed,policy)['bypass']
            expected_same=torch.empty_like(mixed_same);expected_em=torch.empty_like(mixed_em)
            expected_same[~indices]=wrapper.original(mixed.batch(~indices),name)
            expected_same[indices]=wrapper.unfiltered(mixed.batch(indices),name)
            expected_em[~indices]=expected_same[~indices]
            expected_em[indices]=wrapper.em100(mixed.batch(indices))
            row=dict(ordinary_branch_bitwise_equal=ordinary_bitwise,
                filter_off_vs_gate_zero_max_abs_error=zero_error,
                mixed_same_steps_bitwise_equal=torch.equal(mixed_same,expected_same),
                mixed_em100_bitwise_equal=torch.equal(mixed_em,expected_em))
            if not ordinary_bitwise or zero_error>1e-12 or not row['mixed_same_steps_bitwise_equal'] or not row['mixed_em100_bitwise_equal']:
                raise AssertionError(row)
            results[name]=row
        fast_em=wrapper.em100(high);canonical_em=em_estimate(high,100)
        em_error=float((fast_em-canonical_em).abs().max())
        em_relative=float(torch.linalg.vector_norm(fast_em-canonical_em)/torch.linalg.vector_norm(canonical_em))
        if em_error>1e-10 or em_relative>1e-11:raise AssertionError(dict(em_error=em_error,em_relative=em_relative))
    # Explicit near-boundary variances exercise tolerance independently of RNG.
    base=world_batch(2026101760,[2],snr=20.,rsr=12.,L=None)[0]
    snrs=torch.tensor([20.-1e-8,20.,20.+1e-12,20.+1e-8,25.,25.+1e-8,60.],dtype=torch.float64)
    count=len(snrs)
    boundary=Observations(base.z.expand(count,-1,-1),base.s.expand(count,-1,-1),
        base.b.expand(count,-1,-1),3*torch.pow(10.,-snrs/10.))
    boundary_mask=wrapper.masks(boundary,{'5':20.,'12':20.,'20':20.})['bypass'].tolist()
    expected=[False,False,False,True,True,True,True]
    if boundary_mask!=expected:raise AssertionError(boundary_mask)
    rejected=False
    try:wrapper.masks(base,{'5':19.999,'12':25.,'20':40.})
    except ValueError:rejected=True
    if not rejected:raise AssertionError('below-20 cutoff accepted')
    return dict(passed=True,smoke_master_seed=2026101760,model_checks=results,
        high_snr_em100_max_abs_error=em_error,high_snr_em100_relative_error=em_relative,
        near_20_boundary_decisions=boundary_mask,below_20_cutoff_rejected=rejected,
        mixed_expected_bypass=wrapper.masks(mixed,policy)['bypass'].tolist(),
        source_sha256=sha(Path(bypass.__file__)),
        notes='All estimator calls observation-only. Smoke worlds do not select thresholds. Dispatch policy in this test is synthetic; study calibration remains separate.')


if __name__=='__main__':
    result=run()
    path=Path(__file__).resolve().parents[1]/'results/tcom_runtime_bypass_20261017/bypass_checks.json'
    write_json(path,result)
    print(json.dumps(result,indent=2))
