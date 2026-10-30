"""Pre-freeze implementation checks on an independent synthetic smoke bank."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[key]='1'
import numpy as np
import torch
from sahure.math import posterior_field,right_solve
from sahure.data import Observations
from tcom_followups_20261017.common import sha,write_json
from tcom_runtime_bypass_20261017.bypass_audit import tagged_observations
from . import study,inference,shrink,risk,risk_data,ber,sensitivity,historical_replay

@torch.no_grad()
def run():
    torch.set_num_threads(1)
    est=study.engine();a=study.make_anchor(study.SMOKE_SEED,[0,1],None,20.,truth=False)
    obs=study.observed(a,35.);c,h=study.fit_holdout(study.observed(a,35.,P=39))
    assert all(torch.equal(x,y) for x,y in zip(obs.args(),c.args()))
    assert not {'g','G','truth_g','paths','unit_noise'}.intersection(a)
    maximum=0.;comparisons=0
    for seed in study.SEEDS:
        name=f'learned_{seed}';values=est.candidate_predictions(obs,name);initial=est.original(obs,name)
        assert torch.equal(values['original'],initial)
        op=right_solve(obs.s.mH,obs.s@obs.s.mH);g=initial.clone()
        for length in range(1,101):
            g=(posterior_field(obs.z,g@obs.s+obs.b,obs.sigma2)-obs.b)@op
            if length in inference.LENGTHS:
                error=float((g-values[f'warm{length}']).abs().max());maximum=max(maximum,error);assert error<1e-10
        current=est.repair(obs,name,study.CURRENT_REPAIR['repair_choices'])
        for choice,cfg in inference.CONFIGS.items():
            expected=values[choice]
            if cfg['kind']=='convex_channel_blend':
                torch.testing.assert_close(expected,(1-cfg['weight_second'])*values[cfg['left']]+cfg['weight_second']*values[cfg['right']],atol=0,rtol=0)
            if cfg['kind']=='noise_scaled_hankel':
                torch.testing.assert_close(expected,shrink.apply(obs,current,cfg['c'],cfg['beta']),atol=0,rtol=0)
            single=est.candidate(obs,name,choice);error=float((single-expected).abs().max());maximum=max(maximum,error);assert error<1e-10;comparisons+=1
        assert np.all(risk.paired_risk(h,initial,initial)==0)
    # All cells use actual full dispatch; shrinking different incumbent branches
    # in one batch must match separately dispatched worlds.
    rows=[study.observed(study.make_anchor(study.SMOKE_SEED,[10+i],None,r,truth=False),s)
          for i,(r,s) in enumerate(((5.,35.),(12.,32.5),(20.,35.),(20.,37.5),(12.,20.),(20.,60.)))]
    mixed=Observations(*(torch.cat([r.args()[j] for r in rows]) for j in range(4)))
    table={f'{r:g}':{f'{s:g}':dict(choice='shrink_c2_beta1') if r in study.STRONG_RSRS and s in study.TARGET_SNRS else study.fixed_choice(r,s) for s in study.SNRS} for r in study.RSRS}
    for seed in study.SEEDS:
        batched=est.repair(mixed,f'learned_{seed}',table)
        scalar=torch.cat([est.repair(row,f'learned_{seed}',table) for row in rows])
        torch.testing.assert_close(batched,scalar,atol=1e-10,rtol=1e-10)
    identities=0
    for ref in study.RSRS:
        bank=study.make_anchor(study.SMOKE_SEED,[51],None,ref,truth=False)
        for snr in study.SNRS:
            if ref in study.STRONG_RSRS and snr in study.TARGET_SNRS:continue
            one=study.observed(bank,snr)
            for seed in study.SEEDS:
                output=study.estimate_method(one,f'repair_{seed}',dict(repair_choices=table),est)
                expected=study.estimate_method(one,f'current_repair_{seed}',dict(repair_choices=table),est)
                assert torch.equal(output,expected);identities+=1
    probe=tagged_observations([30.9,31.6,38.6,38.9],[12]*4)
    assert inference.choices_for(probe,table)==['original','shrink_c2_beta1','shrink_c2_beta1','spectral100']
    for value in (0.,-1.,float('nan'),float('inf')):
        bad=Observations(obs.z,obs.s,obs.b,torch.full_like(obs.sigma2,value))
        try:inference.receiver_keys(bad)
        except ValueError:pass
        else:raise AssertionError('invalid variance accepted')
    assert len(inference.MENU)==36 and len(study.NAMES)==20 and len(sensitivity.NAMES)==17 and len(historical_replay.conditions())==18
    ber.protocol_spec()
    result=dict(passed=True,source_sha256=sha(__file__),runner_sha256=sha(study.__file__),inference_sha256=sha(inference.__file__),
        candidate_scalar_batch_comparisons=comparisons,maximum_canonical_error=maximum,fixed_domain_identities=identities,
        mixed_dispatch_matches_scalar=True,invalid_variance_rejected=True,observable_only_development=True,
        independent_context30_holdout9=True,protocol_schema_checks=True)
    write_json(study.OUT/'checks.json',result);print(result)

if __name__=='__main__':run()
