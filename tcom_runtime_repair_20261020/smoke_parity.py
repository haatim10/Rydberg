"""Full-grid smoke parity for a backend-only implementation change.

This is numerical verification on an explicitly separate smoke seed. It neither
uses channel truth for decisions nor evaluates a fresh confirmatory test bank.
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key]='1'
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from sahure.data import Observations
from tcom_followups_20261017.common import world_batch,REPO,sha,write_json
from . import inference

SMOKE_SEED=2026102000
REGIMES=(('L2',2),('L4',4),('random3to7',None))
OUT=REPO/'results/tcom_runtime_repair_20261020'


def maximum_error(a,b):
    assert a.shape==b.shape and a.dtype==b.dtype==torch.complex128
    assert torch.isfinite(a).all() and torch.isfinite(b).all()
    difference=float(torch.max(torch.abs(a-b)))
    assert difference<1e-10,difference
    return difference


def concatenate(observations):
    return Observations(*(torch.cat([getattr(o,name) for o in observations]) for name in ('z','s','b','sigma2')))


@torch.no_grad()
def main():
    torch.set_num_threads(1)
    canonical=inference.canonical_engine()
    optimized=inference.optimized_engine(canonical.models)
    checkpoint_hashes={str(p.relative_to(REPO)):sha(p) for p in inference.model_paths().values()}
    errors=dict(original=0.,pipeline=0.,scalar_batch=0.,mixed_scalar=0.,near_zero=0.)
    stage_errors={key:0. for key in ('g','beta','tau','correction','nll')}
    routes=Counter();mixed=[];spectral_identical=0;cells=0
    for rsr in inference.RSRS:
        for regime,L in REGIMES:
            for snr in inference.SNRS:
                obs,_,_=world_batch(SMOKE_SEED,range(2),snr,rsr,L)
                choice=inference.choices_for(obs,inference.LATEST_POLICY['repair_choices'])
                assert len(set(choice))==1
                routes[choice[0]]+=1
                for model_name,model in canonical.models.items():
                    expected_original,reference_trace=model(*obs.args(),diagnostics=True)
                    actual_original,trace=optimized.native_models[model_name](*obs.args(),diagnostics=True)
                    errors['original']=max(errors['original'],maximum_error(expected_original,actual_original))
                    for key in stage_errors:
                        diff=max(float(torch.max(torch.abs(a[key]-b[key]))) for a,b in zip(reference_trace,trace))
                        stage_errors[key]=max(stage_errors[key],diff)
                    expected=inference.latest_estimate(obs,model_name,canonical)
                    actual=inference.latest_estimate(obs,model_name,optimized)
                    errors['pipeline']=max(errors['pipeline'],maximum_error(expected,actual))
                    scalar=torch.cat([inference.latest_estimate(obs.batch(slice(i,i+1)),model_name,optimized) for i in range(len(obs))])
                    errors['scalar_batch']=max(errors['scalar_batch'],maximum_error(actual,scalar))
                    # Some earlier final filters sit on a learned warm start;
                    # only filters whose inherited base is spectral are exact.
                    base_choice=inference.choices_for(obs,inference.canonical.CURRENT_POLICY['repair_choices'])[0]
                    spectral_only=choice[0]=='spectral100' or (choice[0].startswith('shrink_') and base_choice=='spectral100')
                    if spectral_only:
                        assert torch.equal(expected,actual)
                        spectral_identical+=1
                if regime=='random3to7' and snr in (0.,20.,35.,37.5,40.,60.):mixed.append(obs.batch(slice(0,1)))
                cells+=1
            print('Smoke grid',rsr,regime,'cells',cells,flush=True)
    # Mixed-world grouping must retain the exact old choice for every input.
    obs=concatenate(mixed)
    for model_name in canonical.models:
        expected=inference.latest_estimate(obs,model_name,canonical)
        actual=inference.latest_estimate(obs,model_name,optimized)
        errors['pipeline']=max(errors['pipeline'],maximum_error(expected,actual))
        scalar=torch.cat([inference.latest_estimate(obs.batch(slice(i,i+1)),model_name,optimized) for i in range(len(obs))])
        errors['mixed_scalar']=max(errors['mixed_scalar'],maximum_error(actual,scalar))
    # Near-zero measured amplitudes exercise the stable posterior series and
    # Hankel/feature floors, not an approximation to the forward model.
    template=mixed[0]
    for factor in (0.,1e-12):
        edge=Observations(template.z*factor,template.s,template.b,template.sigma2)
        for model_name in canonical.models:
            expected=canonical.original(edge,model_name);actual=optimized.original(edge,model_name)
            errors['near_zero']=max(errors['near_zero'],maximum_error(expected,actual))
    for bad in (0.,-1.,float('nan'),float('inf')):
        invalid=Observations(template.z,template.s,template.b,torch.full_like(template.sigma2,bad))
        try:optimized.original(invalid,'learned_9101')
        except ValueError:pass
        else:raise AssertionError('invalid supplied variance accepted')
    assert checkpoint_hashes=={str(p.relative_to(REPO)):sha(p) for p in inference.model_paths().values()}
    result=dict(passed=True,smoke_seed=SMOKE_SEED,cells=cells,worlds_per_cell=2,model_fits=3,
        dtype='float64/complex128',learned_stages=10,warm_iterations_preserved=True,latest_policy_preserved=True,
        routes=dict(routes),maximum_errors=errors,maximum_stage_errors=stage_errors,
        bitwise_identical_spectral_only_fit_cells=spectral_identical,mixed_worlds=len(mixed),
        near_zero_amplitude_factors=[0.,1e-12],positive_finite_variance_validation=True,
        checkpoint_hashes=checkpoint_hashes,latest_policy_sha256=sha(inference.LATEST_POLICY_PATH),
        source_hashes={str(p.relative_to(REPO)):sha(p) for p in (Path(__file__),Path(inference.__file__),Path(__file__).with_name('native.py'))},
        caveat='Roundoff equivalence is verified on smoke cases; learned branches are not declared bitwise identical. This is not accuracy or latency confirmation.')
    write_json(OUT/'smoke_parity.json',result)
    print(result,flush=True)


if __name__=='__main__':main()
