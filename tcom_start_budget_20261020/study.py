"""Frozen comparison of starting estimators under common postprocessing."""
import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[_key]='1'
import argparse
import json
from pathlib import Path
import platform
import time
import numpy as np
import scipy
import torch
from tcom_followups_20261017.common import REPO,sha,write_json,frozen_optshrink
from tcom_runtime_repair_20261020 import study as previous
from tcom_runtime_repair_20261020 import inference as latest
from tcom_pilot_em_20261018 import continuation
from tcom_incumbent_guard_20261019 import shrink,risk_data
from tcom_runtime_bypass_20261017 import sparse_fast

OUT=REPO/'results/tcom_start_budget_20261020'
TRAIN_OUT=REPO/'results/tcom_start_ablation_20261020'
TRAINING_OUT=TRAIN_OUT
TEST_SEED=2026102012
DATA_SEED=2026102013
TIMING_SEED=2026102014
MASK_SEED=2026102015
BOOT_SEED=2026102019
SMOKE_SEED=2026102010
TEST_COUNT=128
SEEDS=previous.SEEDS
SNRS,RSRS,REGIMES=previous.SNRS,previous.RSRS,previous.REGIMES
CAP_MS=10.74
CALIBRATION_CAP_MS=10.
COMMON_STEPS=40
BUDGET_GRID=(40,50,75,100)
RAW_NAMES=tuple([f'adaptive_{s}' for s in SEEDS]+[f'global_{s}' for s in SEEDS]+['classical','spectral','em10'])
NAMES=RAW_NAMES+tuple('budget_'+n for n in RAW_NAMES)+tuple(f'latest_{s}' for s in SEEDS)+('em100','compact_sparse','optshrink')
GROUPS={**{f'{name}_mean':[f'{name}_{s}' for s in SEEDS] for name in ('adaptive','global','latest')},
        **{f'budget_{name}_mean':[f'budget_{name}_{s}' for s in SEEDS] for name in ('adaptive','global')}}
CONTRASTS={'adaptive_vs_global':('global_mean','adaptive_mean'),
           'adaptive_vs_classical':('classical','adaptive_mean'),
           'adaptive_vs_spectral':('spectral','adaptive_mean'),
           'adaptive_vs_em10':('em10','adaptive_mean'),
           'budget_adaptive_vs_global':('budget_global_mean','budget_adaptive_mean'),
           'budget_adaptive_vs_classical':('budget_classical','budget_adaptive_mean'),
           'adaptive_vs_latest':('latest_mean','adaptive_mean'),
           'adaptive_vs_compact':('compact_sparse','adaptive_mean'),
           'adaptive_vs_em100':('em100','adaptive_mean'),
           'adaptive_vs_optshrink':('optshrink','adaptive_mean'),
           'budget_adaptive_vs_adaptive':('adaptive_mean','budget_adaptive_mean')}
CLASSICAL_GRID=tuple(
    [dict(kind='relative',relative_tau=t,beta=b) for t in (.01,.03,.1,.3,1.) for b in (.1,.5,1.)]+
    [dict(kind='noise',c=c,beta=b) for c in (.5,1.,2.,4.,8.) for b in (.1,.5,1.)])
save,load,cellname,stats=previous.save,previous.load,previous.cellname,previous.stats
observed=risk_data.observed


def make_anchor(*args,**kwargs):
    kwargs.setdefault('holdout_seed',MASK_SEED)
    kwargs.setdefault('snrs',SNRS)
    return risk_data.make_anchor(*args,**kwargs)


def filter_config(rsr,snr):
    """The inherited final filter is common to every experimental start."""
    choice=latest.LATEST_POLICY['repair_choices'][f'{rsr:g}'][f'{snr:g}']['choice']
    if choice not in shrink.CONFIGS:
        return dict(c=0.,beta=0.)
    return dict(shrink.CONFIGS[choice])


def filter_for(obs):
    keys=latest.receiver_keys(obs)
    configs=[filter_config(float(r),float(s)) for r,s,_ in keys]
    # Main batches are homogeneous in known reference and variance.
    if any(c!=configs[0] for c in configs):
        raise ValueError('heterogeneous filter batch; dispatch individual conditions')
    return configs[0]


def finish(obs,g):
    cfg=filter_for(obs)
    return shrink.apply(obs,g,c=cfg['c'],beta=cfg['beta'])


def engine(require_global=True):
    from tcom_start_ablation_20261020.starts import load_starts
    return dict(starts=load_starts(require_global=require_global),latest=latest.optimized_engine())


def start_estimate(obs,name,selection,est):
    if name.startswith(('adaptive_','global_')):
        family,seed=name.rsplit('_',1)
        return est['starts'].start(family,obs,seed=int(seed))
    return est['starts'].start(name,obs,config=selection['classical_config'] if name=='classical' else None)


def family(name):
    return name.rsplit('_',1)[0] if name.startswith(('adaptive_','global_')) else name


@torch.no_grad()
def estimate_method(obs,name,selection,est):
    raw=name.removeprefix('budget_')
    if raw in RAW_NAMES:
        steps=selection['budget_steps'][family(raw)] if name.startswith('budget_') else COMMON_STEPS
        return finish(obs,continuation.estimate(obs,start_estimate(obs,raw,selection,est),steps))
    if name.startswith('latest_'):
        return est['latest'].repair(obs,'learned_'+name.rsplit('_',1)[1],latest.LATEST_POLICY['repair_choices'])
    if name=='em100':
        return est['latest'].em100(obs)
    if name=='compact_sparse':
        return sparse_fast.estimate(obs,previous.compact_config(obs,previous.CURRENT_REPAIR))
    if name=='optshrink':
        return frozen_optshrink(obs,previous.OPT_LOOKUP)
    raise KeyError(name)


def source_paths():
    here=Path(__file__).parent
    paths=[here/n for n in ('study.py','analysis.py','execution.py','checks.py')]
    paths += [REPO/'tcom_start_ablation_20261020'/n for n in ('starts.py','training.py')]
    paths += [REPO/p for p in json.loads((previous.OUT/'protocol.json').read_text())['source_hashes']]
    return sorted(set(paths))


def freeze_protocol():
    from . import ber
    previous.check_selection()
    checks=json.loads((OUT/'checks.json').read_text())
    assert checks['passed']
    for name,digest in checks['source_hashes'].items():
        assert sha(REPO/name)==digest,name
    training=json.loads((TRAIN_OUT/'training_summary.json').read_text())
    assert training['passed'] and training['protocol_sha256']==sha(TRAIN_OUT/'training_protocol.json')
    assert json.loads((TRAIN_OUT/'final_global_parity.json').read_text())['passed']
    ber_hash=ber.freeze()
    paths=latest.model_paths()
    paths.update({f'global_{s}':TRAIN_OUT/f'checkpoints/global/seed_{s}/em_global_best.pt' for s in SEEDS})
    spec=dict(name='Adaptive, nonadaptive learned and classical structured starts with common refinement',
        N=32,K=3,P=30,test_cases_per_reference_regime=TEST_COUNT,snrs=SNRS,rsrs=RSRS,regimes=REGIMES,seeds=SEEDS,
        test_seed=TEST_SEED,data_seed=DATA_SEED,timing_seed=TIMING_SEED,holdout_seed=MASK_SEED,bootstrap_seed=BOOT_SEED,
        names=NAMES,raw_names=RAW_NAMES,groups=GROUPS,contrasts=CONTRASTS,
        primary=dict(contrasts=['adaptive_vs_global','adaptive_vs_classical'],regions=['all'],
                     outcomes=['NMSE','BER'],adjustment='Bonferroni4, 20,000 paired channel-index bootstrap draws'),
        main=dict(start='Every start is used everywhere; no spectral bypass',continuation_steps=COMMON_STEPS,
                  final_filter='Same inherited observable reference/noise filter rule for all starts',
                  raw_start_scores='Descriptive NMSE, before common continuation and final filtering'),
        fixed_control='Thirty parameters trained on identical immutable context30 data/masks/orders/updates/heldout-NLL selection; no input-dependent control. Relative tau retains amplitude normalization.',
        classical=dict(grid=CLASSICAL_GRID,criterion='Minimum equal-stratum heldout Rician NLL of raw start on identical 2048 validation bank; one global configuration; all candidates reported; no channel truth'),
        budget=dict(grid=BUDGET_GRID,calibration_index=401,calibration_cap_ms=CALIBRATION_CAP_MS,
                    rule='Largest tail count satisfying full-grid and weak-reference warm-region pooled p90 <=10.0 ms in one prespecified calibration sweep; same count across three fits. If none qualify retain40 and flag failure. Timing only; no test accuracy selection.',
                    interpretation='Secondary computation-budget experiment; unequal tail counts do not isolate initialization alone. More EM is not guaranteed to improve generalization.'),
        timing=dict(cap_ms=CAP_MS,windows=2,indices=[[500,501,502],[600,601,602]],
                    methods=NAMES,order='Before accuracy generation, then after all accuracy/BER workers complete',
                    rule='No retries until passing; report all methods including overruns; full batch-one calls, single CPU thread, fixed loading excluded',
                    acceptance='Both full-grid pooled p90 values <=10.74 ms per reported method family'),
        previous_protocol_sha256=previous.verify_protocol(),previous_policy_sha256=sha(previous.CURRENT_REPAIR_PATH),
        training_protocol_sha256=sha(TRAIN_OUT/'training_protocol.json'),
        training_summary_sha256=sha(TRAIN_OUT/'training_summary.json'),checks_sha256=sha(OUT/'checks.json'),
        pretest_revision=dict(previous_protocol_sha256=sha(TRAIN_OUT/'protocol.json'),previous_policy_sha256=sha(TRAIN_OUT/'policy.json'),previous_calibration_sha256=sha(TRAIN_OUT/'budget_selection.json'),previous_classical_sha256=sha(TRAIN_OUT/'classical_selection.json'),reason='Every100-step pipeline exceeded the timing allowance before any fresh test data were generated. Main refinement reduced uniformly to40; one prespecified40/50/75/100 budget calibration. Previous sources/results preserved; test seeds untouched.'),
        final_global_parity_sha256=sha(TRAIN_OUT/'final_global_parity.json'),
        source_hashes={str(p.relative_to(REPO)):sha(p) for p in source_paths()},
        checkpoint_hashes={str(p.relative_to(REPO)):sha(p) for p in paths.values()},
        ber_protocol_sha256=ber_hash)
    path=OUT/'protocol.json'
    if path.exists():
        old=json.loads(path.read_text());old.pop('frozen_unix')
        assert old==json.loads(json.dumps(spec))
    else:
        write_json(path,dict(**spec,frozen_unix=time.time()))
    print('Protocol frozen',sha(path),flush=True)


def verify_protocol():
    path=OUT/'protocol.json';spec=json.loads(path.read_text())
    for key in ('source_hashes','checkpoint_hashes'):
        for name,digest in spec[key].items():
            assert sha(REPO/name)==digest,name
    assert previous.verify_protocol()==spec['previous_protocol_sha256']
    assert sha(previous.CURRENT_REPAIR_PATH)==spec['previous_policy_sha256']
    assert sha(TRAIN_OUT/'training_protocol.json')==spec['training_protocol_sha256']
    assert sha(TRAIN_OUT/'training_summary.json')==spec['training_summary_sha256']
    assert sha(OUT/'checks.json')==spec['checks_sha256']
    assert sha(TRAIN_OUT/'final_global_parity.json')==spec['final_global_parity_sha256']
    revision=spec['pretest_revision']
    assert sha(TRAIN_OUT/'protocol.json')==revision['previous_protocol_sha256']
    assert sha(TRAIN_OUT/'policy.json')==revision['previous_policy_sha256']
    assert sha(TRAIN_OUT/'budget_selection.json')==revision['previous_calibration_sha256']
    assert sha(TRAIN_OUT/'classical_selection.json')==revision['previous_classical_sha256']
    return sha(path)


def check_selection():
    digest=verify_protocol();path=OUT/'policy.json';p=json.loads(path.read_text())
    assert p['protocol_sha256']==digest
    assert p['ber_protocol_sha256']==sha(OUT/'ber/protocol.json')
    for name,d in p['ber_source_hashes'].items():
        assert sha(REPO/name)==d,name
    for name,d in p['selection_evidence'].items():
        assert sha(OUT/name)==d,name
    return p,sha(path)


@torch.no_grad()
def select_classical():
    verify_protocol()
    destination=OUT/'classical_selection.json'
    if destination.exists():
        return json.loads(destination.read_text())
    original=TRAIN_OUT/'classical_selection.json'
    inherited=json.loads(original.read_text())
    assert [r['config'] for r in inherited['candidates']]==list(CLASSICAL_GRID)
    assert inherited['config']==min(inherited['candidates'],key=lambda r:r['score'])['config']
    result=dict(inherited)
    result.update(protocol_sha256=sha(OUT/'protocol.json'),inherited_from_sha256=sha(original),
                  original_completed_unix=inherited['completed_unix'],completed_unix=time.time(),
                  reuse='Identical raw-start candidates, weights, observations and criterion; reuse unchanged calibration, no new selection search.')
    write_json(destination,result)
    print('Classical calibration inherited unchanged',result['config'],flush=True)
    return result


def timing_rows(indices):
    rows=[]
    for index in indices:
        for rsr in RSRS:
            for regime,L in REGIMES:
                a=make_anchor(TIMING_SEED,[index],L,rsr,truth=False)
                for snr in SNRS:
                    rows.append((dict(snr=snr,rsr=rsr,regime=regime,index=index),observed(a,snr)))
    return rows


@torch.no_grad()
def select_budgets():
    verify_protocol()
    destination=OUT/'budget_selection.json'
    if destination.exists():
        return json.loads(destination.read_text())
    classical=select_classical()
    selection=dict(classical_config=classical['config'])
    rows=timing_rows([401]);est=engine();samples={n:{str(c):[] for c in BUDGET_GRID} for n in RAW_NAMES}
    rng=np.random.default_rng(np.random.SeedSequence([TIMING_SEED,401]))
    for name in RAW_NAMES:
        obs=rows[0][1]
        finish(obs,continuation.estimate(obs,start_estimate(obs,name,selection,est),COMMON_STEPS))
    for completed,wi in enumerate(rng.permutation(len(rows)),1):
        meta,obs=rows[wi]
        candidates=[(n,c) for n in RAW_NAMES for c in BUDGET_GRID]
        for ci in rng.permutation(len(candidates)):
            name,count=candidates[ci]
            tick=time.perf_counter_ns()
            g=finish(obs,continuation.estimate(obs,start_estimate(obs,name,selection,est),count))
            elapsed=(time.perf_counter_ns()-tick)/1e6
            assert torch.isfinite(g).all()
            samples[name][str(count)].append(dict(ms=elapsed,**meta))
        if completed%15==0:
            print('Budget calibration world',completed,'/',len(rows),flush=True)
    choices={};scores={};fallback={}
    for group in ('adaptive','global','classical','spectral','em10'):
        members=[n for n in RAW_NAMES if family(n)==group];scores[group]={};eligible=[]
        for count in BUDGET_GRID:
            records=[v for n in members for v in samples[n][str(count)]]
            allp=float(np.quantile([v['ms'] for v in records],.9))
            weakp=float(np.quantile([v['ms'] for v in records if v['rsr']==5 and v['snr']>=37.5],.9))
            ok=allp<=CALIBRATION_CAP_MS and weakp<=CALIBRATION_CAP_MS
            scores[group][str(count)]=dict(p90_ms=allp,weak_p90_ms=weakp,eligible=ok)
            if ok:eligible.append(count)
        choices[group]=max(eligible) if eligible else COMMON_STEPS
        fallback[group]=not bool(eligible)
    result=dict(budget_steps=choices,fallback_to_40=fallback,scores=scores,raw_samples=samples,
                protocol_sha256=sha(OUT/'protocol.json'),completed_unix=time.time())
    write_json(destination,result);return result


def freeze_selection():
    from . import ber
    verify_protocol()
    classical=select_classical();budget=select_budgets()
    path=OUT/'policy.json'
    spec=dict(protocol_sha256=sha(OUT/'protocol.json'),classical_config=classical['config'],budget_steps=budget['budget_steps'],
              budget_fallback_to_40=budget['fallback_to_40'],
              ber_protocol_sha256=sha(OUT/'ber/protocol.json'),ber_source_hashes=ber.protocol_spec()['source_hashes'],
              selection_evidence={n:sha(OUT/n) for n in ('classical_selection.json','budget_selection.json')})
    if path.exists():
        old=json.loads(path.read_text());old.pop('frozen_unix');assert old==spec
    else:write_json(path,dict(**spec,frozen_unix=time.time()))
    print('Selection frozen',sha(path),spec['classical_config'],spec['budget_steps'],flush=True)


@torch.no_grad()
def predictions(obs,selection,est):
    raw=torch.stack([start_estimate(obs,n,selection,est) for n in RAW_NAMES])
    common=[];budget=[]
    for name,g in zip(RAW_NAMES,raw):
        count=selection['budget_steps'][family(name)]
        values,_=continuation.trajectory(obs,g,(COMMON_STEPS,count))
        common.append(finish(obs,values[COMMON_STEPS]))
        budget.append(finish(obs,values[count]) if count!=COMMON_STEPS else common[-1])
    references=[estimate_method(obs,n,selection,est) for n in NAMES[len(RAW_NAMES)*2:]]
    return raw,torch.stack(common+budget+references)


@torch.no_grad()
def test(reference=None):
    selection,digest=check_selection();est=engine()
    for rsr in RSRS:
        if reference is not None and reference!=rsr:continue
        for regime,L in REGIMES:
            anchor_path=OUT/'test/anchors'/f'{regime}_rsr{rsr:g}.npz'
            if anchor_path.exists():a=load(anchor_path)
            else:
                a=make_anchor(TEST_SEED,range(TEST_COUNT),L,rsr,truth=True)
                a.update(selection_sha256=np.array(digest),protocol_sha256=np.array(selection['protocol_sha256']),regime=np.array(regime))
                save(anchor_path,**a)
            assert str(a['selection_sha256'])==digest
            for snr in SNRS:
                path=OUT/'test/cells'/f'{cellname(regime,rsr,snr)}.npz'
                if path.exists():
                    assert str(load(path)['selection_sha256'])==digest
                    continue
                obs=observed(a,snr);raw,pred=predictions(obs,selection,est)
                arr=pred.numpy();starts=raw.numpy();assert np.isfinite(arr).all() and np.isfinite(starts).all()
                energy=np.sum(abs(a['g'])**2,axis=(-2,-1))
                errors=np.sum(abs(arr-a['g'][None])**2,axis=(-2,-1))
                raw_errors=np.sum(abs(starts-a['g'][None])**2,axis=(-2,-1))
                maximum=0.
                for i,name in enumerate(NAMES):
                    actual=estimate_method(obs.batch(slice(0,1)),name,selection,est)
                    difference=float((actual-pred[i,0:1]).abs().max())
                    assert difference<1e-10,(name,difference)
                    maximum=max(maximum,difference)
                save(path,**{k:v.numpy() for k,v in zip(('z','s','b','sigma2'),obs.args())},
                     truth_g=a['g'],paths=a['paths'],indices=a['indices'],local_indices=a['local_indices'],
                     method_names=np.array(NAMES),estimates=arr,errors=errors,truth_energy=energy,nmse=errors/energy[None],
                     raw_names=np.array(RAW_NAMES),raw_estimates=starts,raw_errors=raw_errors,
                     snr=np.array(snr),rsr=np.array(rsr),regime=np.array(regime),seed=np.array(TEST_SEED),
                     selection_sha256=np.array(digest),protocol_sha256=np.array(selection['protocol_sha256']),
                     actual_dispatch_max_abs_error=np.array(maximum))
                print('Test complete',path.stem,flush=True)


@torch.no_grad()
def timing(window):
    selection,digest=check_selection();destination=OUT/f'timing_window{window}.json'
    assert not destination.exists(),'No timing retries until passing'
    indices=(500,501,502) if window==1 else (600,601,602)
    assert window in (1,2)
    est=engine();rows=timing_rows(indices);samples={n:[] for n in NAMES}
    for name in NAMES:estimate_method(rows[0][1],name,selection,est)
    rng=np.random.default_rng(np.random.SeedSequence([TIMING_SEED,window]));started=time.time()
    for wi in rng.permutation(len(rows)):
        meta,obs=rows[wi]
        for ni in rng.permutation(len(NAMES)):
            name=NAMES[ni];tick=time.perf_counter_ns()
            result=estimate_method(obs,name,selection,est)
            elapsed=(time.perf_counter_ns()-tick)/1e6;assert torch.isfinite(result).all()
            samples[name].append(dict(ms=elapsed,**meta))
        if len(samples[NAMES[0]])%25==0:print('Timing',window,len(samples[NAMES[0]]),'/',len(rows),flush=True)
    values=dict(samples)
    values.update({group:[v for n in members for v in samples[n]] for group,members in GROUPS.items()})
    methods={}
    for name,records in values.items():
        methods[name]=dict(**stats([r['ms'] for r in records]),
             weak_warm100=stats([r['ms'] for r in records if r['rsr']==5 and r['snr']>=37.5]),
             by_condition={f'rsr{r:g}_snr{s:g}':stats([v['ms'] for v in records if v['rsr']==r and v['snr']==s]) for r in RSRS for s in SNRS})
    result=dict(window=window,selection_sha256=digest,protocol_sha256=selection['protocol_sha256'],
                started_unix=started,completed_unix=time.time(),indices=indices,worlds=len(rows),methods=methods,raw_samples=samples,
                cap_verification={n:dict(p90_ms=m['p90_ms'],passed=m['p90_ms']<=CAP_MS) for n,m in methods.items()},
                environment=dict(platform=platform.platform(),python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,torch=torch.__version__,torch_threads=torch.get_num_threads()))
    write_json(destination,result)
    print('Timing complete',window,{n:result['cap_verification'][n] for n in ('adaptive_mean','global_mean','classical','budget_adaptive_mean','budget_global_mean','budget_classical')},flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('phase',choices=('freeze','select-classical','select-budget','select','test','timing1','timing2'))
    p.add_argument('--reference',type=float,choices=RSRS);args=p.parse_args()
    torch.set_num_threads(1)
    {'freeze':freeze_protocol,'select-classical':select_classical,'select-budget':select_budgets,
     'select':freeze_selection,'test':lambda:test(args.reference),'timing1':lambda:timing(1),'timing2':lambda:timing(2)}[args.phase]()


if __name__=='__main__':main()
