"""Attached B-proposal sensitivity with no selection or parent mutations."""
import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[_key] = '1'
import argparse
import copy
import json
from pathlib import Path
import platform
import time
import numpy as np
import scipy
import torch
from sahure.data import Observations
from tcom_followups_20261017.common import REPO, sha, write_json
from tcom_hybrid_20261020 import study as PARENT

OUT = REPO/'results/tcom_hybrid_sensitivity_20261020'
SOURCE = PARENT_OUT = PARENT.OUT
SEEDS, SNRS, RSRS, REGIMES = PARENT.SEEDS, PARENT.SNRS, PARENT.RSRS, PARENT.REGIMES
TEST_COUNT = COUNT = PARENT.TEST_COUNT
TEST_SEED, DATA_SEED, BOOT_SEED = PARENT.TEST_SEED, PARENT.DATA_SEED, PARENT.BOOT_SEED
TIMING_SEED, SMOKE_SEED, CAP_MS = PARENT.TIMING_SEED, PARENT.SMOKE_SEED, PARENT.CAP_MS
NAMES = tuple(f'proposal_{seed}' for seed in SEEDS)
GROUPS = {'proposal_mean':NAMES}
TIMING_NAMES = NAMES + tuple(f'hybrid_{s}' for s in SEEDS) + tuple(f'latest_{s}' for s in SEEDS) + ('compact_sparse',)
TIMING_GROUPS = dict(GROUPS, hybrid_mean=tuple(f'hybrid_{s}' for s in SEEDS),
                    latest_mean=tuple(f'latest_{s}' for s in SEEDS))
TIMING_DOMAIN = 0x4250524F
load, save, cellname, stats = PARENT.load, PARENT.save, PARENT.cellname, PARENT.stats
parent_selection_sha256 = sha(SOURCE/'policy.json')


def build_proposal_policy(main_policy=None):
    """Change only tail steps to the already selected B proposed_steps."""
    if main_policy is None:
        main_policy, _ = PARENT.check_selection()
    proposed = copy.deepcopy(main_policy)
    evidence = json.loads((SOURCE/'tail_proposals.json').read_text())['proposals']
    for ref in RSRS:
        for snr in SNRS:
            if snr <= 20:
                continue
            key, cell = f'{ref:g}', f'{snr:g}'
            item = evidence[key][cell]
            assert proposed['tails'][key][cell]['family'] == item['family']
            assert item['proposed_steps'] in PARENT.TAILS
            proposed['tails'][key][cell]['steps'] = item['proposed_steps']
    return proposed


def failed_confirmation():
    b = json.loads((SOURCE/'tail_proposals.json').read_text())['proposals']
    c = json.loads((SOURCE/'tail_confirmation.json').read_text())['confirmed']
    records = []
    for ref in RSRS:
        for snr in SNRS:
            if snr <= 20:
                continue
            key, cell = f'{ref:g}', f'{snr:g}'
            row, confirmation = b[key][cell], c[key][cell]
            assert confirmation['proposed_steps'] == row['proposed_steps']
            if row['proposed_steps']:
                records.append(dict(reference_key=key,snr_key=cell,family=row['family'],
                    proposed_steps=row['proposed_steps'],C_accepted=confirmation['accepted'],
                    C_selected_steps=confirmation['steps'],C_guards=confirmation['guards']))
    assert len(records)==21 and all(not r['C_accepted'] and r['C_selected_steps']==0 for r in records)
    return records


def source_paths():
    here = Path(__file__).parent
    local = [here/name for name in ('__init__.py','study.py','checks.py','ber.py','analysis.py','audit.py','execution.py')]
    parent_protocol = json.loads((SOURCE/'protocol.json').read_text())
    inherited = [REPO/name for name in parent_protocol['source_hashes']]
    parent_ber = json.loads((SOURCE/'ber/protocol.json').read_text())
    inherited += [REPO/name for name in parent_ber['source_hashes']]
    return sorted(set(local+inherited))


def parent_inputs():
    names = ('protocol.json','policy.json','runtime_calibration.json','crossover_selection.json',
             'tail_proposals.json','tail_confirmation.json','checks.json','pipeline_checks.json','ber/protocol.json')
    return {name:sha(SOURCE/name) for name in names}


def freeze_protocol():
    from . import ber
    main, main_digest = PARENT.check_selection()
    derived = build_proposal_policy(main)
    checks = json.loads((OUT/'checks.json').read_text())
    assert checks['passed']
    for name,digest in checks['source_hashes'].items():
        assert sha(REPO/name)==digest,name
    original = json.loads((SOURCE/'protocol.json').read_text())
    spec = dict(name='Exploratory B-proposal sensitivity attached to unchanged confirmed hybrid',
        inference='Exact B proposed_steps only; never B best_steps, no runner-up, no gate change and no new validation search.',
        status='Exploratory only. All21 nonzero B proposals failed the prespecified independent C confirmation.',
        primary_scope='No primary tests added or changed; parent C-gated hybrid remains unchanged.',
        proposal_policy=derived,main_policy_sha256=main_digest,parent_input_hashes=parent_inputs(),
        failed_confirmation=failed_confirmation(),names=NAMES,groups=GROUPS,
        N=32,K=3,P=30,test_count=TEST_COUNT,snrs=SNRS,rsrs=RSRS,regimes=REGIMES,seeds=SEEDS,
        test_seed=TEST_SEED,data_seed=DATA_SEED,bootstrap_seed=BOOT_SEED,
        paired_data='Exact same parent fresh test observations/channels and saved parent BER data; these are paired comparisons, not another independent sample.',
        storage='Only three new channel estimates/errors plus metadata and source hashes; parent observations/truth/baseline arrays are referenced, not copied.',
        timing=dict(cap_ms=CAP_MS,windows=2,indices=[[500,501,502],[600,601,602]],names=TIMING_NAMES,
            groups=TIMING_GROUPS,worlds_per_window=459,calls_per_window=4590,
            order_seed=[TIMING_SEED,TIMING_DOMAIN,'window'],batch_one=True,threads=1,
            warmup='Every method on the first ordinary row and the first RSR12/SNR40 row, before timing.',
            ordering='Window1 before any parent test anchor; window2 after all main/secondary numerical jobs are idle.',
            rule='One standalone start/selected tail/filter per call. No retries until passing; report all overruns. Loading and shape-only constants are excluded.'),
        analysis='Descriptive paired NMSE/BER intervals and adverse cells only; no adjusted confirmatory claims, no deployment or manuscript change selected from this arm.',
        ber=ber.protocol_spec(),checks_sha256=sha(OUT/'checks.json'),
        checkpoint_hashes=original['checkpoint_hashes'],
        source_hashes={str(p.relative_to(REPO)):sha(p) for p in source_paths()})
    path = OUT/'protocol.json'
    if path.exists():
        old = json.loads(path.read_text());old.pop('frozen_unix')
        assert old==json.loads(json.dumps(spec))
    else:
        assert not list((SOURCE/'test/anchors').glob('*.npz')), 'secondary must freeze before any parent test anchor'
        assert not list((SOURCE/'test/cells').glob('*.npz'))
        assert not list((OUT/'test/cells').glob('*.npz'))
        write_json(path,dict(**spec,frozen_unix=time.time()))
    print('Secondary protocol frozen',sha(path),flush=True)


def verify_protocol():
    path = OUT/'protocol.json';spec = json.loads(path.read_text())
    _,digest = PARENT.check_selection()
    assert digest==spec['main_policy_sha256']==parent_selection_sha256
    for name,value in spec['parent_input_hashes'].items():
        assert sha(SOURCE/name)==value,name
    for key in ('source_hashes','checkpoint_hashes'):
        for name,value in spec[key].items():
            assert sha(REPO/name)==value,name
    assert sha(OUT/'checks.json')==spec['checks_sha256']
    assert spec['proposal_policy']==build_proposal_policy()
    return sha(path)


def check_selection():
    digest = verify_protocol()
    spec = json.loads((OUT/'protocol.json').read_text())
    main = json.loads((SOURCE/'policy.json').read_text())
    return dict(proposal_policy=copy.deepcopy(spec['proposal_policy']),main_policy=main,
                protocol_sha256=digest,parent_selection_sha256=spec['main_policy_sha256']),digest


@torch.no_grad()
def estimate_method(obs, name, policy, est):
    if name in NAMES:
        seed = int(name.rsplit('_',1)[1])
        return PARENT.inference.hybrid(obs,seed,policy['proposal_policy'],est)
    if name in TIMING_NAMES:
        return PARENT.estimate_method(obs,name,policy['main_policy'],est)
    raise KeyError(name)


def observations(parent_arrays):
    return Observations(*(torch.as_tensor(parent_arrays[key],
        dtype=torch.complex128 if key in ('s','b') else torch.float64)
        for key in ('z','s','b','sigma2')))


@torch.no_grad()
def test(reference=None):
    policy,digest = check_selection();est=PARENT.engine()
    assert (OUT/'timing_window1.json').exists(),'secondary window1 must precede test'
    assert (SOURCE/'timing_window1.json').exists()
    for ref in RSRS:
        if reference is not None and reference!=ref:continue
        for regime,_ in REGIMES:
            for snr in SNRS:
                label=cellname(regime,ref,snr)
                parent_path=SOURCE/'test/cells'/f'{label}.npz'
                ber_path=SOURCE/'ber/cells'/f'{label}.npz'
                data_path=SOURCE/'ber/data'/f'{label}.npz'
                assert all(p.exists() for p in (parent_path,ber_path,data_path)),'parent prediction and BER cell/data must exist'
                bindings=dict(parent_source_sha256=sha(parent_path),parent_ber_cell_sha256=sha(ber_path),
                              parent_ber_data_sha256=sha(data_path))
                destination=OUT/'test/cells'/f'{label}.npz'
                if destination.exists():
                    old=load(destination)
                    assert str(old['selection_sha256'])==digest
                    for key,value in bindings.items():assert str(old[key])==value,key
                    continue
                parent=load(parent_path)
                assert str(parent['selection_sha256'])==policy['parent_selection_sha256']
                assert int(parent['seed'])==TEST_SEED
                assert tuple(map(str,parent['method_names']))==PARENT.NAMES
                ber=load(ber_path)
                assert str(ber['source_sha256'])==bindings['parent_source_sha256']
                assert str(ber['data_sha256'])==bindings['parent_ber_data_sha256']
                obs=observations(parent)
                estimates=torch.stack([estimate_method(obs,name,policy,est) for name in NAMES]).numpy()
                assert estimates.shape==(3,TEST_COUNT,32,3) and np.isfinite(estimates).all()
                energy=np.sum(abs(parent['truth_g'])**2,axis=(-2,-1))
                np.testing.assert_array_equal(energy,parent['truth_energy'])
                errors=np.sum(abs(estimates-parent['truth_g'][None])**2,axis=(-2,-1))
                maximum=0.
                for index,name in enumerate(NAMES):
                    scalar=estimate_method(obs.batch(slice(0,1)),name,policy,est).numpy()
                    difference=float(np.max(abs(scalar-estimates[index,0:1])))
                    assert difference<1e-10
                    maximum=max(maximum,difference)
                save(destination,method_names=np.array(NAMES),estimates=estimates,errors=errors,
                    truth_energy=energy,nmse=errors/energy[None],indices=parent['indices'],local_indices=parent['local_indices'],
                    snr=np.array(snr),rsr=np.array(ref),regime=np.array(regime),seed=np.array(TEST_SEED),
                    selection_sha256=np.array(digest),protocol_sha256=np.array(digest),
                    parent_selection_sha256=np.array(policy['parent_selection_sha256']),
                    parent_protocol_sha256=parent['protocol_sha256'],
                    actual_dispatch_max_abs_error=np.array(maximum),
                    **{k:np.array(v) for k,v in bindings.items()})
                print('Secondary test complete',label,flush=True)


@torch.no_grad()
def timing(window):
    policy,digest=check_selection();destination=OUT/f'timing_window{window}.json'
    assert window in (1,2) and not destination.exists(),'one prespecified timing window only'
    if window==1:
        assert not list((SOURCE/'test/anchors').glob('*.npz'))
    indices=(500,501,502) if window==1 else (600,601,602)
    rows=PARENT.timing_rows(indices);est=PARENT.engine()
    assert len(rows)==459
    samples={name:[] for name in TIMING_NAMES}
    high = next(obs for meta,obs in rows if meta['rsr']==12 and meta['snr']==40)
    for obs in (rows[0][1],high):
        for name in TIMING_NAMES:estimate_method(obs,name,policy,est)
    rng=np.random.default_rng(np.random.SeedSequence([TIMING_SEED,TIMING_DOMAIN,window]))
    started=time.time()
    for completed,wi in enumerate(rng.permutation(len(rows)),1):
        meta,obs=rows[wi]
        for ni in rng.permutation(len(TIMING_NAMES)):
            name=TIMING_NAMES[ni];tick=time.perf_counter_ns()
            result=estimate_method(obs,name,policy,est)
            elapsed=(time.perf_counter_ns()-tick)/1e6
            assert torch.isfinite(result).all()
            samples[name].append(dict(ms=elapsed,**meta))
        if completed%25==0:print('Secondary timing',window,completed,'/',len(rows),flush=True)
    values=dict(samples)
    values.update({group:[r for name in members for r in samples[name]] for group,members in TIMING_GROUPS.items()})
    methods={}
    for name,records in values.items():
        methods[name]=dict(**stats([r['ms'] for r in records]),
            weak_warm100=stats([r['ms'] for r in records if r['rsr']==5 and r['snr']>=37.5]),
            by_condition={f'rsr{ref:g}_snr{snr:g}':stats([r['ms'] for r in records if r['rsr']==ref and r['snr']==snr])
                          for ref in RSRS for snr in SNRS})
    result=dict(window=window,selection_sha256=digest,protocol_sha256=digest,
        parent_selection_sha256=policy['parent_selection_sha256'],started_unix=started,completed_unix=time.time(),
        indices=indices,worlds=len(rows),calls=sum(len(x) for x in samples.values()),
        raw_samples=samples,methods=methods,order_seed=[TIMING_SEED,TIMING_DOMAIN,window],
        cap_verification={name:dict(p90_ms=row['p90_ms'],passed=row['p90_ms']<=CAP_MS) for name,row in methods.items()},
        environment=dict(platform=platform.platform(),python=platform.python_version(),numpy=np.__version__,
                         scipy=scipy.__version__,torch=torch.__version__,torch_threads=torch.get_num_threads()))
    write_json(destination,result)
    print('Secondary timing complete',window,result['cap_verification']['proposal_mean'],flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('phase',choices=('freeze','test','timing1','timing2'))
    parser.add_argument('--reference',type=float,choices=RSRS)
    args=parser.parse_args();torch.set_num_threads(1)
    if args.phase=='freeze':freeze_protocol()
    elif args.phase=='test':test(args.reference)
    else:timing(int(args.phase[-1]))


if __name__=='__main__':main()
