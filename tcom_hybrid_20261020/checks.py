"""Small prefreeze checks for the actual hybrid wrapper; no accuracy selection."""
import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_key] = '1'
from pathlib import Path
from unittest.mock import patch
import json
import time
import numpy as np
import torch
from sahure.data import Observations
from sahure.math import posterior_field, right_solve
from tcom_followups_20261017.common import REPO, sha, write_json
from tcom_start_budget_20261020 import study as previous
from tcom_incumbent_guard_20261019 import risk_data
from . import inference

OUT = REPO / 'results/tcom_hybrid_20261020'
SMOKE_SEED = 2026102050
HOLDOUT_SEED = 2026102055


def family_at(snr, crossover):
    if snr <= 20. + 1e-10:
        return 'adaptive'
    return 'global' if snr >= 40. - 1e-10 or snr > crossover + 1e-10 else 'adaptive'


def synthetic_policy(crossover):
    return dict(crossover_db=float(crossover),
        tails={f'{ref:g}': {f'{snr:g}': dict(family=family_at(snr, crossover),
                 steps=0 if snr <= 20 else (10,20,40)[index % 3])
             for index, snr in enumerate(previous.SNRS)} for ref in previous.RSRS})


def concatenate(items):
    return Observations(*(torch.cat([o.args()[i] for o in items]) for i in range(4)))


def with_proxy(obs, snr_db):
    value = obs.s.shape[1] * 10.**(-float(snr_db)/10.)
    return Observations(obs.z, obs.s, obs.b, torch.full_like(obs.sigma2, value))


def sources():
    paths = [Path(__file__), Path(__file__).with_name('inference.py')]
    immutable = json.loads((previous.OUT/'protocol.json').read_text())['source_hashes']
    paths += [REPO/name for name in immutable]
    # Explicit dependencies used by the independent checks and dispatch.
    paths += [REPO/'tcom_incumbent_guard_20261019/risk_data.py',
              REPO/'tcom_incumbent_guard_20261019/shrink.py',
              REPO/'sahure/data.py', REPO/'sahure/math.py']
    return {str(p.relative_to(REPO)):sha(p) for p in sorted(set(paths))}


@torch.no_grad()
def main():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    # Verify the entire immutable predecessor chain without editing its files.
    previous.check_selection()
    est = previous.engine()
    condition_specs = ((0.,5.,2), (20.,12.,4), (22.5,5.,None),
        (25.,12.,2), (27.5,20.,4), (30.,5.,None), (32.5,12.,4),
        (35.,20.,2), (37.5,5.,None), (40.,12.,4), (50.,20.,2), (60.,12.,None))
    observations = []
    for snr, ref, paths in condition_specs:
        anchor = risk_data.make_anchor(SMOKE_SEED, [0,1], paths, ref,
            snrs=(snr,), holdout_seed=HOLDOUT_SEED, truth=False)
        assert not {'g','G','truth_g','unit_noise','paths'}.intersection(anchor)
        observations.append(risk_data.observed(anchor, snr))
    mixed = concatenate(observations)
    policy = synthetic_policy(30.)
    rows = inference.plans(mixed, policy)
    assert all(o.z.shape[-1] == 30 for o in observations)
    maximum_scalar = maximum_canonical = maximum_zero = 0.
    canonical_lengths = set()
    ordinary_worlds = 0
    for seed in previous.SEEDS:
        actual = inference.hybrid(mixed, seed, policy, est)
        scalar = torch.cat([inference.hybrid(mixed.batch(slice(i,i+1)), seed, policy, est)
                            for i in range(len(mixed))])
        error = float((actual-scalar).abs().max())
        assert torch.allclose(actual, scalar, atol=1e-10, rtol=1e-10)
        maximum_scalar = max(maximum_scalar, error)
        without_tail = inference.hybrid(mixed, seed, policy, est, zero_tail=True)
        for index, obs in enumerate(observations):
            row = rows[2*index]
            assert rows[2*index+1] == row
            initial = est['starts'].start(row['family'], obs, seed=seed)
            if row['ordinary']:
                assert torch.equal(actual[2*index:2*index+2], initial)
                assert torch.equal(without_tail[2*index:2*index+2], initial)
                with patch.object(inference.continuation, 'estimate', side_effect=AssertionError('ordinary EM called')), \
                     patch.object(inference.previous, 'finish', side_effect=AssertionError('ordinary filter called')):
                    assert torch.equal(inference.hybrid(obs, seed, policy, est), initial)
                ordinary_worlds += len(obs)
                continue
            expected_zero = previous.finish(obs, initial)
            zero_error = float((expected_zero-without_tail[2*index:2*index+2]).abs().max())
            assert torch.allclose(expected_zero, without_tail[2*index:2*index+2], atol=1e-10, rtol=1e-10)
            maximum_zero = max(maximum_zero, zero_error)
            canonical = initial.clone()
            for _ in range(row['steps']):
                q = posterior_field(obs.z, canonical@obs.s+obs.b, obs.sigma2)
                canonical = right_solve((q-obs.b)@obs.s.mH, obs.s@obs.s.mH)
            canonical = previous.finish(obs, canonical)
            error = float((canonical-actual[2*index:2*index+2]).abs().max())
            assert torch.allclose(canonical, actual[2*index:2*index+2], atol=1e-10, rtol=1e-10)
            maximum_canonical = max(maximum_canonical, error)
            canonical_lengths.add(row['steps'])
    assert canonical_lengths == {10,20,40}

    # Count actual wrapper calls, including grouped dispatch, without timing.
    calls = []
    class CountingStarts:
        def start(self, name, obs, seed=None, config=None):
            calls.append(dict(family=name, worlds=len(obs), seed=seed))
            return est['starts'].start(name, obs, seed=seed, config=config)
    tail_calls = []
    original_tail = inference.continuation.estimate
    def counting_tail(obs, initial, steps):
        tail_calls.append(dict(worlds=len(obs), steps=steps))
        return original_tail(obs, initial, steps)
    with patch.object(inference.continuation, 'estimate', side_effect=counting_tail):
        inference.hybrid(mixed, previous.SEEDS[0], policy, dict(est, starts=CountingStarts()))
    assert sum(x['worlds'] for x in calls) == len(mixed)
    assert sum(x['worlds'] for x in tail_calls) == sum(not r['ordinary'] and r['steps']>0 for r in rows)
    for family in ('adaptive','global'):
        assert sum(x['worlds'] for x in calls if x['family']==family) == sum(r['family']==family for r in rows)

    # Routing-only probes alter supplied variance while preserving observations.
    base = observations[6].batch(slice(0,1))
    routing = []; mismatch_outputs = []
    for crossover in inference.CROSSOVERS:
        current_policy = synthetic_policy(crossover)
        probes = sorted(set((20.-1e-6,20.,20.+5e-11,20.+1e-6,
            40.-1e-6,40.-5e-11,40.,40.+1e-6,
            crossover-1e-6,crossover,crossover+5e-11,crossover+1e-6)))
        for proxy in probes:
            obs = with_proxy(base, proxy)
            row = inference.plans(obs, current_policy)[0]
            expected_family = family_at(row['snr_db'], crossover)
            nearest = f'{min(previous.SNRS,key=lambda x:abs(x-row["snr_db"])):g}'
            assert row['family']==expected_family and row['snr_key']==nearest
            assert row['ordinary']==(row['snr_db'] <= 20.+1e-10)
            expected_mismatch = (not row['ordinary'] and
                current_policy['tails'][row['reference_key']][nearest]['family'] != expected_family)
            assert row['mismatch_fallback']==expected_mismatch
            if expected_mismatch:
                assert row['steps']==0
            if row['snr_db'] >= 40.-1e-10:
                assert row['family']=='global'
            routing.append(dict(crossover=crossover,requested_proxy_db=proxy,**row))
        mismatch_obs = with_proxy(base, crossover+1e-6)
        mismatch = inference.plans(mismatch_obs,current_policy)[0]
        assert mismatch['mismatch_fallback'] and mismatch['steps']==0
        raw = est['starts'].start('global',mismatch_obs,seed=previous.SEEDS[0])
        expected = previous.finish(mismatch_obs,raw)
        with patch.object(inference.continuation,'estimate',side_effect=AssertionError('mismatch fallback ran EM')):
            actual = inference.hybrid(mismatch_obs,previous.SEEDS[0],current_policy,est)
        assert torch.equal(actual,expected)
        mismatch_outputs.append(dict(crossover=crossover,bitwise_expected=True))

    mismatch_routes = []
    for obs in (observations[1],observations[6],observations[-1]):
        reference = inference.plans(obs, policy)
        for offset in (-3.,0.,3.):
            changed = Observations(obs.z,obs.s,obs.b,obs.sigma2*10.**(offset/10.))
            assert all(torch.equal(a,b) for a,b in zip(obs.args()[:3],changed.args()[:3]))
            rows_changed = inference.plans(changed,policy)
            for before,after in zip(reference,rows_changed):
                assert abs(after['snr_db']-(before['snr_db']-offset))<1e-10
                assert after['family']==family_at(after['snr_db'],policy['crossover_db'])
            actual = inference.hybrid(changed,previous.SEEDS[0],policy,est)
            assert torch.isfinite(actual).all()
            mismatch_routes.append(dict(actual_proxy_db=reference[0]['snr_db'],variance_offset_db=offset,
                                        supplied_route=rows_changed[0]))

    result = dict(passed=True,completed_unix=time.time(),smoke_seed=SMOKE_SEED,holdout_seed=HOLDOUT_SEED,
        condition_count=len(condition_specs),worlds_per_condition=2,fits=list(previous.SEEDS),
        condition_specs=condition_specs,canonical_tail_lengths=sorted(canonical_lengths),
        maximum_scalar_batch_abs_error=maximum_scalar,maximum_canonical_pipeline_abs_error=maximum_canonical,
        maximum_zero_tail_abs_error=maximum_zero,ordinary_bitwise_checks=ordinary_worlds,
        ordinary_no_EM_no_final_filter=True,one_start_per_world=True,one_selected_tail_per_nonordinary_world=True,
        actual_start_calls=calls,actual_tail_calls=tail_calls,continuous_routing_probes=routing,
        mismatch_fallback_exact_checks=mismatch_outputs,supplied_variance_dispatch_checks=mismatch_routes,
        variance_check_scope='Routing/finite-output smoke only; no performance or robustness claim.',
        independent_observable_inputs=True,only30_deployment_pilots=True,
        previous_protocol_sha256=sha(previous.OUT/'protocol.json'),
        previous_policy_sha256=sha(previous.OUT/'policy.json'),source_hashes=sources())
    write_json(OUT/'checks.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in
        ('source_hashes','continuous_routing_probes','supplied_variance_dispatch_checks')},indent=2))


if __name__=='__main__':
    main()
