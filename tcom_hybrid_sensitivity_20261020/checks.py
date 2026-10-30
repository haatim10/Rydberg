"""Small implementation checks for the exact B-proposal-only sensitivity."""
import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[_key] = '1'
import copy
import json
from pathlib import Path
import time
import torch
from tcom_followups_20261017.common import REPO, sha, write_json
from tcom_incumbent_guard_20261019 import risk_data
from tcom_pilot_em_20261018 import continuation
from . import study


def sources():
    paths = [Path(__file__),Path(__file__).with_name('study.py'),Path(__file__).with_name('__init__.py')]
    for name in ('protocol.json','ber/protocol.json'):
        paths += [REPO/p for p in json.loads((study.SOURCE/name).read_text())['source_hashes']]
    return {str(p.relative_to(REPO)):sha(p) for p in sorted(set(paths))}


@torch.no_grad()
def main():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    parent, digest = study.PARENT.check_selection()
    before = sha(study.SOURCE/'policy.json')
    proposed = study.build_proposal_policy(parent)
    b = json.loads((study.SOURCE/'tail_proposals.json').read_text())['proposals']
    c = json.loads((study.SOURCE/'tail_confirmation.json').read_text())['confirmed']
    independent = copy.deepcopy(parent)
    changed = [];not_best = []
    for ref in study.RSRS:
        for snr in study.SNRS:
            key, cell = f'{ref:g}', f'{snr:g}'
            if snr <= 20:
                assert proposed['tails'][key][cell] == parent['tails'][key][cell] == dict(family='adaptive',steps=0)
                continue
            record = b[key][cell]
            assert record['family'] == independent['tails'][key][cell]['family']
            independent['tails'][key][cell]['steps'] = record['proposed_steps']
            assert proposed['tails'][key][cell]['steps'] == record['proposed_steps']
            assert c[key][cell]['steps'] == parent['tails'][key][cell]['steps'] == 0
            if record['proposed_steps']:
                assert not c[key][cell]['accepted']
                changed.append(dict(reference_key=key,snr_key=cell,steps=record['proposed_steps'],family=record['family']))
            if record['best_steps'] != record['proposed_steps']:
                not_best.append(dict(reference_key=key,snr_key=cell,best_steps=record['best_steps'],
                                     proposed_steps=record['proposed_steps']))
    assert proposed == independent and len(changed)==21
    assert parent == json.loads((study.SOURCE/'policy.json').read_text())
    assert study.failed_confirmation()
    wrapper=dict(proposal_policy=proposed,main_policy=parent,protocol_sha256='SMOKE_ONLY',
                 parent_selection_sha256=digest)
    est=study.PARENT.engine()
    # Ordinary identity plus one smoke condition for every B-selected length.
    representatives = [(10.,12.,2)]
    seen = set()
    for ref in study.RSRS:
        for snr in study.SNRS:
            if snr <= 20:continue
            value=proposed['tails'][f'{ref:g}'][f'{snr:g}']['steps']
            if value not in seen:
                representatives.append((snr,ref,(2,4,None)[len(seen)%3]))
                seen.add(value)
    representatives += [(40.,5.,2),(60.,20.,None)]
    maximum=0.;ordinary=0;checked=[]
    for snr,ref,L in representatives:
        anchor=risk_data.make_anchor(study.SMOKE_SEED,[20,21],L,ref,
            snrs=(snr,),holdout_seed=study.SMOKE_SEED,truth=False)
        assert not {'g','G','truth_g','unit_noise','paths'}.intersection(anchor)
        obs=risk_data.observed(anchor,snr)
        # This independent manual dispatch uses only the frozen B records.
        family='adaptive' if snr<=20.+1e-10 or (snr<40.-1e-10 and snr<=parent['crossover_db']+1e-10) else 'global'
        step=0 if snr<=20 else b[f'{ref:g}'][f'{snr:g}']['proposed_steps']
        for seed in study.SEEDS:
            initial=est['starts'].start(family,obs,seed=seed)
            expected=initial
            if snr>20:
                expected=study.PARENT.finish(obs,continuation.estimate(obs,initial,step))
            actual=study.estimate_method(obs,f'proposal_{seed}',wrapper,est)
            error=float((actual-expected).abs().max())
            assert torch.equal(actual,expected),(snr,ref,seed,error)
            maximum=max(maximum,error)
            if snr<=20:
                confirmed=study.PARENT.inference.hybrid(obs,seed,parent,est)
                assert torch.equal(actual,confirmed) and torch.equal(actual,initial)
                ordinary+=1
            elif step==0:
                assert torch.equal(actual,study.PARENT.inference.hybrid(obs,seed,parent,est))
        checked.append(dict(snr=snr,rsr=ref,L=L,family=family,steps=step))
    assert before==sha(study.SOURCE/'policy.json')
    assert not list((study.SOURCE/'test/anchors').glob('*.npz'))
    result=dict(passed=True,completed_unix=time.time(),smoke_seed=study.SMOKE_SEED,
        local_indices=[20,21],worlds_per_condition=2,fits=list(study.SEEDS),conditions=checked,
        exact_B_nonzero_proposals=changed,C_failed_nonzero_count=21,steps_from_proposed_not_best=True,
        best_different_from_proposed_cases=not_best,only_tail_steps_changed=True,parent_policy_unchanged=True,
        ordinary_bitwise_checks=ordinary,manual_dispatch_max_abs_error=maximum,
        only_smoke_observations=True,channel_truth_loaded=False,development_or_test_worlds_generated=False,
        parent_selection_sha256=digest,source_hashes=sources())
    write_json(study.OUT/'checks.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('source_hashes','exact_B_nonzero_proposals')},indent=2))


if __name__=='__main__':main()
