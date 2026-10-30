"""Prefreeze parity of shared accuracy predictions and standalone inference.

Only independent smoke observations are generated. This check performs no
development selection, scoring against truth, or runtime measurement.
"""
import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_key] = '1'
from pathlib import Path
import json
import time
import torch
from tcom_followups_20261017.common import REPO, sha, write_json
from tcom_incumbent_guard_20261019 import risk_data
from tcom_pilot_em_20261018 import continuation
from tcom_start_budget_20261020 import study as previous
from . import study

SMOKE_SEED = 2026102050
LOCAL_INDICES = (10, 11)


def sources():
    local = [Path(__file__), Path(__file__).with_name('study.py'),
             Path(__file__).with_name('inference.py')]
    inherited = json.loads((previous.OUT/'protocol.json').read_text())['source_hashes']
    paths = local + [REPO/name for name in inherited]
    paths += [REPO/'tcom_incumbent_guard_20261019/risk_data.py']
    return {str(p.relative_to(REPO)):sha(p) for p in sorted(set(paths))}


@torch.no_grad()
def main():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    baseline_policy, baseline_digest = previous.check_selection()
    est = study.engine()
    # Ordinary zero; adaptive transition10; strong global40; weak global20.
    specifications = (
        dict(snr=10., ref=12., L=2, family='adaptive', steps=0, ordinary=True),
        dict(snr=25., ref=12., L=4, family='adaptive', steps=10, ordinary=False),
        dict(snr=40., ref=20., L=2, family='global', steps=40, ordinary=False),
        dict(snr=50., ref=5., L=None, family='global', steps=20, ordinary=False),
    )
    tails = {f'{ref:g}': {f'{snr:g}': dict(
                family='adaptive' if snr<=30. else 'global', steps=0)
                for snr in study.SNRS} for ref in study.RSRS}
    for case in specifications:
        tails[f'{case["ref"]:g}'][f'{case["snr"]:g}']['steps'] = case['steps']
    policy = dict(crossover_db=30., tails=tails, baseline_policy=baseline_policy)
    assert len(study.RAW_NAMES)==6 and len(study.NAMES)==19
    maximum_raw = {name:0. for name in study.RAW_NAMES}
    maximum_final = {name:0. for name in study.NAMES}
    maximum_direct40 = {name:0. for name in study.NAMES if name.startswith(('adaptive40_','global40_'))}
    maximum_classical = 0.
    ordinary_checks = raw_checks = final_checks = direct40_checks = 0
    rows = []
    for case in specifications:
        anchor = risk_data.make_anchor(SMOKE_SEED, LOCAL_INDICES, case['L'], case['ref'],
            snrs=(case['snr'],), holdout_seed=SMOKE_SEED, truth=False)
        assert not {'g','G','truth_g','unit_noise','paths'}.intersection(anchor)
        obs = risk_data.observed(anchor, case['snr'])
        assert obs.z.shape[-1]==30
        plan = study.inference.plans(obs, policy)
        assert all(r['family']==case['family'] and r['steps']==case['steps'] and
                   r['ordinary']==case['ordinary'] and not r['mismatch_fallback'] for r in plan)
        raw, output = study.predictions(obs, policy, est)
        assert raw.shape==(6,2,32,3) and output.shape==(19,2,32,3)
        raw_index = {name:index for index,name in enumerate(study.RAW_NAMES)}
        final_index = {name:index for index,name in enumerate(study.NAMES)}
        for name,index in raw_index.items():
            family,seed = name.rsplit('_',1)
            direct = est['starts'].start(family, obs, seed=int(seed))
            error = float((direct-raw[index]).abs().max())
            maximum_raw[name] = max(maximum_raw[name], error)
            assert torch.equal(direct, raw[index]), (case,name,error)
            raw_checks += 1
        for name,index in final_index.items():
            standalone = study.estimate_method(obs, name, policy, est)
            error = float((standalone-output[index]).abs().max())
            maximum_final[name] = max(maximum_final[name], error)
            assert torch.allclose(standalone, output[index], atol=1e-10, rtol=1e-10), (case,name,error)
            final_checks += 1
        for family in ('adaptive','global'):
            for seed in study.SEEDS:
                name = f'{family}40_{seed}'
                initial = raw[raw_index[f'{family}_{seed}']]
                direct = previous.finish(obs, continuation.estimate(obs, initial, 40))
                error = float((direct-output[final_index[name]]).abs().max())
                maximum_direct40[name] = max(maximum_direct40[name], error)
                assert torch.allclose(direct, output[final_index[name]], atol=1e-10, rtol=1e-10)
                direct40_checks += 1
        classical = est['starts'].start('classical', obs, config=baseline_policy['classical_config'])
        classical = previous.finish(obs, continuation.estimate(obs, classical, 40))
        error = float((classical-output[final_index['classical40']]).abs().max())
        maximum_classical = max(maximum_classical, error)
        assert torch.allclose(classical, output[final_index['classical40']], atol=1e-10, rtol=1e-10)
        if case['ordinary']:
            for seed in study.SEEDS:
                expected = raw[raw_index[f'adaptive_{seed}']]
                for family in ('hybrid','hybrid0','latest'):
                    assert torch.equal(output[final_index[f'{family}_{seed}']], expected)
                    ordinary_checks += 1
        rows.append(dict(**case,local_indices=list(LOCAL_INDICES),
                         raw_names=list(study.RAW_NAMES),method_names=list(study.NAMES),
                         supplied_plans=plan))
    result = dict(passed=True,completed_unix=time.time(),smoke_seed=SMOKE_SEED,
        holdout_seed=SMOKE_SEED,local_indices=list(LOCAL_INDICES),
        conditions=rows,worlds_per_condition=2,fits=list(study.SEEDS),
        raw_tensor_checks=raw_checks,standalone_method_tensor_checks=final_checks,
        direct40_tensor_checks=direct40_checks,ordinary_raw_latest_bitwise_checks=ordinary_checks,
        selected_tail_lengths=sorted({x['steps'] for x in specifications}),
        raw_maximum_abs_errors=maximum_raw,final_maximum_abs_errors=maximum_final,
        direct40_maximum_abs_errors=maximum_direct40,classical40_direct_maximum_abs_error=maximum_classical,
        tolerance=dict(atol=1e-10,rtol=1e-10),ordinary_raw_reuse_bitwise=True,
        only_smoke_observations=True,channel_truth_loaded=False,
        development_or_test_world_seeds_used=False,
        baseline_policy_sha256=baseline_digest,previous_protocol_sha256=sha(previous.OUT/'protocol.json'),
        source_hashes=sources())
    write_json(study.OUT/'pipeline_checks.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('source_hashes','conditions')},indent=2))


if __name__=='__main__':
    main()
