"""Independent preflight for the fixed high-SNR filtering extension.

Only two-case smoke observations are used. No campaign development or test
anchors are generated. Prior sources and their completed checks stay immutable.
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key]='1'
import json
import numpy as np
import torch
from sahure.data import Observations
from tcom_incumbent_guard_20261019 import shrink,risk_data
from tcom_followups_20261017.common import REPO,sha,write_json
from . import study

SMOKE_SEED=2026101950


def concatenate(observations):
    return Observations(*(torch.cat([obs.args()[i] for obs in observations]) for i in range(4)))


def perturb_variance(obs,nominal_delta_db):
    """Same physical observations; only the supplied receiver variance changes."""
    return Observations(obs.z,obs.s,obs.b,obs.sigma2*10.**(-nominal_delta_db/10.))


@torch.no_grad()
def run():
    torch.set_num_threads(1)
    table=study.candidate_table();selection=dict(repair_choices=table)
    est=study.engine();conditions=0;outside=0;early=0;weak=0;ordinary=0;changed=0
    maximum_formula=0.;maximum_scalar=0.;maximum_identity=0.
    for ref in study.RSRS:
        for regime,L in study.REGIMES:
            bank=study.make_anchor(SMOKE_SEED,[0,1],L,ref,truth=False)
            assert not {'g','G','truth_g','paths','unit_noise'}.intersection(bank)
            for snr in study.SNRS:
                obs=study.observed(bank,snr);target=ref in (12.,20.) and snr>=40.
                actual_choice=study.inference.choices_for(obs,table)
                assert len(set(actual_choice))==1
                latest_choice=study.inference.choices_for(obs,study.CURRENT_REPAIR['repair_choices'])
                if target:
                    assert actual_choice==['shrink_c4_beta1']*2
                    assert latest_choice==['spectral100']*2
                    direct=shrink.apply(obs,est.em100(obs),4.,1.)
                else:assert actual_choice==latest_choice
                for seed in study.SEEDS:
                    actual=study.estimate_method(obs,f'repair_{seed}',selection,est)
                    if target:
                        error=float((actual-direct).abs().max());maximum_formula=max(maximum_formula,error)
                        torch.testing.assert_close(actual,direct,atol=1e-12,rtol=0)
                    else:
                        expected=study.estimate_method(obs,f'current_repair_{seed}',selection,est)
                        error=float((actual-expected).abs().max());maximum_identity=max(maximum_identity,error)
                        assert torch.equal(actual,expected)
                conditions+=1
                if target:changed+=1
                else:outside+=1
                if ref in (12.,20.) and snr in (32.5,35.,37.5):early+=1
                if ref==5.:weak+=1
                if snr<=20.:ordinary+=1
    assert (conditions,outside,changed,early,weak,ordinary)==(153,123,30,18,51,45)
    # Heterogeneous actual dispatch: unchanged neural/warm branches and new
    # filtered spectral branches must give the same outputs when called alone.
    rows=[]
    for i,(ref,snr,L) in enumerate(((5.,35.,2),(12.,32.5,4),(20.,37.5,None),
                                  (12.,40.,2),(20.,60.,4),(12.,20.,None))):
        bank=study.make_anchor(SMOKE_SEED,[20+i*2,21+i*2],L,ref,truth=False)
        obs=study.observed(bank,snr)
        rows.extend(obs.batch(slice(j,j+1)) for j in range(2))
    mixed=concatenate(rows)
    for seed in study.SEEDS:
        actual=study.estimate_method(mixed,f'repair_{seed}',selection,est)
        separate=torch.cat([study.estimate_method(obs,f'repair_{seed}',selection,est) for obs in rows])
        error=float((actual-separate).abs().max());maximum_scalar=max(maximum_scalar,error)
        torch.testing.assert_close(actual,separate,atol=1e-10,rtol=1e-10)
    # The previous nearest-bin switch occurred at38.75dB. Both sides now have
    # the identical spectral100+filter mapping, evaluated using actual variance.
    # This checks local continuity for fixed observations, not monotone NMSE.
    boundaries=[]
    for ref in (12.,20.):
        bank=risk_data.make_anchor(SMOKE_SEED,[80,81],None,ref,snrs=(38.75,),truth=False)
        base=risk_data.observed(bank,38.75)
        difference_by_epsilon={}
        for epsilon in (1e-4,1e-6):
            sides=[];latest=[];routes=[]
            for sign in (-1.,1.):
                obs=perturb_variance(base,sign*epsilon)
                for a,b in zip(obs.args()[:3],base.args()[:3]):assert torch.equal(a,b)
                routes.append(study.inference.receiver_keys(obs)[0][:2])
                assert study.inference.choices_for(obs,table)==['shrink_c4_beta1']*2
                expected=shrink.apply(obs,est.em100(obs),4.,1.)
                for seed in study.SEEDS:
                    actual=study.estimate_method(obs,f'repair_{seed}',selection,est)
                    error=float((actual-expected).abs().max());maximum_formula=max(maximum_formula,error)
                    torch.testing.assert_close(actual,expected,atol=1e-12,rtol=0)
                sides.append(expected)
                latest.append(study.estimate_method(obs,f'current_repair_{study.SEEDS[0]}',selection,est))
            assert routes==[(f'{ref:g}','37.5'),(f'{ref:g}','40')]
            absolute=float((sides[1]-sides[0]).abs().max())
            scale=max(float(sides[0].abs().max()),1.)
            difference_by_epsilon[epsilon]=absolute/scale
            boundaries.append(dict(reference=ref,epsilon_db=epsilon,lookup_sides=routes,
                maximum_new_neighbor_change=absolute,relative_new_neighbor_change=absolute/scale,
                maximum_latest_neighbor_change=float((latest[1]-latest[0]).abs().max())))
        assert difference_by_epsilon[1e-6] <= .025*difference_by_epsilon[1e-4]+1e-12
        assert difference_by_epsilon[1e-6] < 1e-6
    # Reuse the already independent SVD mathematical check, including rank-one
    # and zero estimates at positive variances1e-12..1. Do not alter/rerun it.
    tiny_path=REPO/'results/tcom_incumbent_guard_20261019/shrink_checks.json'
    tiny=json.loads(tiny_path.read_text())
    assert tiny['passed'] and tiny['source_sha256']==sha(shrink.__file__)
    assert tiny['checks_source_sha256']==sha(REPO/'tcom_incumbent_guard_20261019/shrink_checks.py')
    assert max(max(errors) for errors in tiny['svd_max_errors'].values())<1e-9
    result=dict(passed=True,smoke_seed=SMOKE_SEED,cases_per_condition=2,
        conditions=conditions,model_fits=len(study.SEEDS),changed_conditions=changed,
        exact_current_conditions=outside,exact_earlier_transition_conditions=early,
        exact_weak_reference_conditions=weak,exact_ordinary_conditions=ordinary,
        maximum_exact_identity_error=maximum_identity,maximum_explicit_formula_error=maximum_formula,
        mixed_batch_worlds=len(rows),maximum_actual_scalar_batch_error=maximum_scalar,
        boundary_checks=boundaries,local_continuity_claim='Fixed observations with supplied variance perturbed across38.75dB; no global NMSE monotonicity claim.',
        reused_tiny_variance_check_sha256=sha(tiny_path),source_sha256=sha(__file__),
        runner_sha256=sha(study.__file__),inference_sha256=sha(study.inference.__file__),shrink_source_sha256=sha(shrink.__file__))
    write_json(study.OUT/'checks.json',result);print(json.dumps(result,indent=2))
    return result


if __name__=='__main__':run()
