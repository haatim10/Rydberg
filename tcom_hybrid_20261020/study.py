"""Observable-only hybrid selection followed by a separate paired test."""
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
from scipy.stats import t as student_t
import torch
from tcom_followups_20261017.common import REPO,sha,write_json
from tcom_start_budget_20261020 import study as previous
from tcom_pilot_em_20261018 import continuation
from tcom_incumbent_guard_20261019 import risk_data
from . import inference

OUT=REPO/'results/tcom_hybrid_20261020'
TRAIN_OUT=previous.TRAIN_OUT
SMOKE_SEED=2026102050
DEV_SEEDS={'A':2026102051,'B':2026102052,'C':2026102053}
TEST_SEED=2026102054
DATA_SEED=2026102055
TIMING_SEED=2026102056
MASK_SEED=2026102057
BOOT_SEED=2026102058
SNRS,RSRS,REGIMES,SEEDS=previous.SNRS,previous.RSRS,previous.REGIMES,previous.SEEDS
TEST_COUNT=128
DEV_COUNT=32
CAP_MS=10.74
CALIBRATION_CAP_MS=10.
CROSSOVERS=(20.,25.,30.,35.)
TAILS=(0,10,20,40)
FAMILIES=('adaptive','global')
PRIMARY=('hybrid_vs_latest','hybrid_vs_compact')
NAMES=tuple(f'{family}_{seed}' for family in ('hybrid','hybrid0','latest','adaptive40','global40') for seed in SEEDS)+('classical40','em100','compact_sparse','optshrink')
RAW_NAMES=tuple(f'{family}_{seed}' for family in FAMILIES for seed in SEEDS)
GROUPS={f'{family}_mean':[f'{family}_{seed}' for seed in SEEDS] for family in ('hybrid','hybrid0','latest','adaptive40','global40','adaptive','global')}
CONTRASTS={
    'hybrid_vs_latest':('latest_mean','hybrid_mean'),
    'hybrid_vs_compact':('compact_sparse','hybrid_mean'),
    'hybrid_vs_adaptive40':('adaptive40_mean','hybrid_mean'),
    'hybrid_vs_global40':('global40_mean','hybrid_mean'),
    'hybrid_vs_classical40':('classical40','hybrid_mean'),
    'hybrid_vs_em100':('em100','hybrid_mean'),
    'hybrid_vs_optshrink':('optshrink','hybrid_mean'),
    'hybrid_vs_noem':('hybrid0_mean','hybrid_mean'),
}
save,load,cellname,stats=previous.save,previous.load,previous.cellname,previous.stats
observed=risk_data.observed
finish=previous.finish


def make_anchor(*args,**kwargs):
    kwargs.setdefault('holdout_seed',MASK_SEED)
    kwargs.setdefault('snrs',SNRS)
    return risk_data.make_anchor(*args,**kwargs)


def engine():
    return previous.engine()


def chosen_family(snr,crossover):
    return 'adaptive' if snr<=20.+1e-10 or (snr<40.-1e-10 and snr<=crossover+1e-10) else 'global'


def start_estimate(obs,name,selection,est):
    family,seed=name.rsplit('_',1)
    assert family in FAMILIES
    return est['starts'].start(family,obs,seed=int(seed))


@torch.no_grad()
def candidate(obs,family,seed,steps,est):
    initial=est['starts'].start(family,obs,seed=seed)
    return finish(obs,continuation.estimate(obs,initial,steps))


@torch.no_grad()
def estimate_method(obs,name,selection,est):
    if name.startswith(('hybrid_','hybrid0_')):
        family,seed=name.rsplit('_',1)
        return inference.hybrid(obs,int(seed),selection,est,zero_tail=family=='hybrid0')
    if name.startswith(('adaptive40_','global40_')):
        family,seed=name.rsplit('_',1)
        return candidate(obs,family.removesuffix('40'),int(seed),40,est)
    old_name='classical' if name=='classical40' else name
    return previous.estimate_method(obs,old_name,selection['baseline_policy'],est)


def source_paths():
    here=Path(__file__).parent
    local=[here/n for n in ('study.py','inference.py','checks.py','pipeline_checks.py','analysis.py','execution.py')]
    inherited=[REPO/p for p in json.loads((previous.OUT/'protocol.json').read_text())['source_hashes']]
    return sorted(set(local+inherited+[REPO/'tcom_incumbent_guard_20261019/risk_data.py']))


def freeze_protocol():
    from . import ber
    old_policy,old_digest=previous.check_selection()
    checks=json.loads((OUT/'checks.json').read_text())
    assert checks['passed']
    for name,digest in checks['source_hashes'].items():
        assert sha(REPO/name)==digest,name
    pipeline_checks=json.loads((OUT/'pipeline_checks.json').read_text())
    assert pipeline_checks['passed']
    for name,digest in pipeline_checks['source_hashes'].items():
        assert sha(REPO/name)==digest,name
    ber_hash=ber.freeze()
    old_protocol=json.loads((previous.OUT/'protocol.json').read_text())
    spec=dict(name='Validation-selected adaptive/fixed hybrid with selective EM',
        N=32,K=3,P=30,test_cases_per_reference_regime=TEST_COUNT,dev_cases_per_reference_regime=DEV_COUNT,
        snrs=SNRS,rsrs=RSRS,regimes=REGIMES,seeds=SEEDS,names=NAMES,raw_names=RAW_NAMES,groups=GROUPS,contrasts=CONTRASTS,
        test_seed=TEST_SEED,data_seed=DATA_SEED,timing_seed=TIMING_SEED,holdout_seed=MASK_SEED,bootstrap_seed=BOOT_SEED,
        dev_seeds=DEV_SEEDS,smoke_seed=SMOKE_SEED,
        primary=dict(contrasts=PRIMARY,regions=['all'],outcomes=['NMSE','BER'],
                     adjustment='Bonferroni4; 20,000 paired channel-index bootstrap draws'),
        main=dict(ordinary='Continuous nominal SNR proxy <=20: exact adaptive raw output; no continuation or extra filter',
                  high='Continuous proxy >=40: fixed global schedule',
                  transition='One shared crossover from20/25/30/35; global above crossover, adaptive otherwise',
                  tails=TAILS,final_filter='Same frozen reference/noise filter as the preceding start study, outside ordinary branch',
                  hybrid0='Same chosen family and final filter, with zero additional EM',
                  offgrid='Nearest reference/SNR lookup for tails/filter; continuous family decision. If lookup family differs, use zero EM. No continuity or variance-mismatch guarantee.'),
        validation=dict(context_pilots=30,independent_heldout_pilots=9,truth_exported=False,
            A='One crossover minimizes equal-cell, equal-fit mean heldout Rician NLL of complete zero-tail/filter outputs on63 transition cells. Ties within1e-12 choose smaller cutoff.',
            B='Independent bank: each reference/SNR selects lowest mean NLL among runtime-eligible0/10/20/40. Nonzero requires positive paired lower bound versus0 in EACH path regime, averaging fit losses within each channel. Alpha=.05/3. Failure proposes0.',
            C='Third independent bank: only the single B proposal is compared against0. All3 path regimes require positive paired lower bounds at alpha=.05/108. Failure retains0; never test a runner-up.',
            guard='Approximate one-sided Student-t screening bound with n=32 per-path channel anchors, ddof=1, critical t(31), and mean improvement>1e-12. Nominal Bonferroni correction; no finite-sample risk guarantee, no guarantee of NMSE/BER improvement.',
            independence='Separate world seeds for A/B/C and test. Three fit losses averaged per channel, not treated as additional independent channels.'),
        timing=dict(cap_ms=CAP_MS,calibration_cap_ms=CALIBRATION_CAP_MS,calibration_index=401,
            eligibility='Before development scores: full-grid and weak-region pooled p90<=10.0 for family/tail. Zero always remains an explicit fallback; any zero-tail calibration failure is retained.',
            windows=2,indices=[[500,501,502],[600,601,602]],methods=NAMES,
            order='Window1 after policy freeze and before test; window2 after all accuracy/BER workers and analyses are idle.',
            acceptance='Both full-grid pooled batch-one p90<=10.74 ms; report weak and condition bins separately. No retiming until passing.',
            includes='One selected start, one selected tail and final filter, conversion and routing; model loading and shape constants preloaded. One CPU thread.'),
        previous_protocol_sha256=previous.verify_protocol(),previous_policy_sha256=old_digest,
        previous_completion_sha256=sha(previous.OUT/'completion.json'),
        checks_sha256=sha(OUT/'checks.json'),pipeline_checks_sha256=sha(OUT/'pipeline_checks.json'),
        source_hashes={str(p.relative_to(REPO)):sha(p) for p in source_paths()},
        checkpoint_hashes=old_protocol['checkpoint_hashes'],ber_protocol_sha256=ber_hash)
    path=OUT/'protocol.json'
    if path.exists():
        old=json.loads(path.read_text());old.pop('frozen_unix')
        assert old==json.loads(json.dumps(spec))
    else:write_json(path,dict(**spec,frozen_unix=time.time()))
    print('Protocol frozen',sha(path),flush=True)


def verify_protocol():
    p=json.loads((OUT/'protocol.json').read_text())
    for key in ('source_hashes','checkpoint_hashes'):
        for name,digest in p[key].items():assert sha(REPO/name)==digest,name
    assert previous.verify_protocol()==p['previous_protocol_sha256']
    assert sha(previous.OUT/'policy.json')==p['previous_policy_sha256']
    assert sha(previous.OUT/'completion.json')==p['previous_completion_sha256']
    assert sha(OUT/'checks.json')==p['checks_sha256']
    assert sha(OUT/'pipeline_checks.json')==p['pipeline_checks_sha256']
    assert sha(OUT/'ber/protocol.json')==p['ber_protocol_sha256']
    return sha(OUT/'protocol.json')


def check_selection():
    digest=verify_protocol();p=json.loads((OUT/'policy.json').read_text())
    assert p['protocol_sha256']==digest
    assert p['ber_protocol_sha256']==sha(OUT/'ber/protocol.json')
    for name,d in p['ber_source_hashes'].items():assert sha(REPO/name)==d,name
    for name,d in p['selection_evidence'].items():assert sha(OUT/name)==d,name
    assert p['baseline_policy']==json.loads((previous.OUT/'policy.json').read_text())
    return p,sha(OUT/'policy.json')


def timing_rows(indices):
    rows=[]
    for ref in RSRS:
        for regime,L in REGIMES:
            a=make_anchor(TIMING_SEED,indices,L,ref,truth=False)
            for index in range(len(indices)):
                for snr in SNRS:
                    rows.append((dict(regime=regime,rsr=ref,snr=snr,index=indices[index]),observed(a,snr).batch(slice(index,index+1))))
    return rows


@torch.no_grad()
def calibrate_runtime():
    verify_protocol();path=OUT/'runtime_calibration.json'
    assert not path.exists(),'One prespecified calibration only'
    est=engine();rows=timing_rows((401,))
    menu=[(family,seed,tail) for family in FAMILIES for seed in SEEDS for tail in TAILS]
    for family,seed,tail in menu:candidate(rows[0][1],family,seed,tail,est)
    records=[];rng=np.random.default_rng(np.random.SeedSequence([TIMING_SEED,0]));started=time.time()
    for wi in rng.permutation(len(rows)):
        meta,obs=rows[wi]
        for ci in rng.permutation(len(menu)):
            family,seed,tail=menu[ci];tick=time.perf_counter_ns()
            g=candidate(obs,family,seed,tail,est)
            elapsed=(time.perf_counter_ns()-tick)/1e6
            assert torch.isfinite(g).all()
            records.append(dict(**meta,family=family,seed=seed,steps=tail,ms=elapsed))
        if (len(records)//len(menu))%25==0:print('Calibration',len(records)//len(menu),'/',len(rows),flush=True)
    scores={}
    for family in FAMILIES:
        scores[family]={}
        for tail in TAILS:
            selected=[r for r in records if r['family']==family and r['steps']==tail]
            all_stats=stats([r['ms'] for r in selected])
            weak=stats([r['ms'] for r in selected if r['rsr']==5 and r['snr']>=37.5])
            scores[family][str(tail)]=dict(**all_stats,weak_p90_ms=weak['p90_ms'],
                eligible=all_stats['p90_ms']<=CALIBRATION_CAP_MS and weak['p90_ms']<=CALIBRATION_CAP_MS)
    write_json(path,dict(protocol_sha256=verify_protocol(),started_unix=started,completed_unix=time.time(),scores=scores,records=records))
    print('Runtime eligibility',scores,flush=True)


def dev_anchor(bank,regime,L,ref):
    path=OUT/'validation'/bank/'anchors'/f'{regime}_rsr{ref:g}.npz'
    if path.exists():a=load(path)
    else:
        a=make_anchor(DEV_SEEDS[bank],range(DEV_COUNT),L,ref,truth=False)
        a.update(protocol_sha256=np.array(verify_protocol()),regime=np.array(regime))
        save(path,**a)
    assert str(a['protocol_sha256'])==sha(OUT/'protocol.json')
    assert not {'g','G','truth_g','unit_noise','paths','mu'}.intersection(a)
    return a


def predictive_loss(heldout,g):
    z,s,b,v=(x.numpy() for x in heldout.args())
    return continuation.nll(z,g.numpy()@s+b,v).mean((-2,-1))


def tie_argmin(values,keys):
    best=min(float(values[k]) for k in keys)
    return min(k for k in keys if float(values[k])<=best+1e-12)


def guard(delta,alpha):
    delta=np.asarray(delta,dtype=np.float64)
    assert delta.shape==(DEV_COUNT,) and np.isfinite(delta).all()
    mean=float(delta.mean());se=float(delta.std(ddof=1)/np.sqrt(DEV_COUNT))
    critical=float(student_t.ppf(1-alpha,DEV_COUNT-1))
    lower=mean-critical*se
    return dict(mean_improvement=mean,standard_error=se,critical=critical,alpha=alpha,
                lower_bound=lower,passed=mean>1e-12 and lower>0.)


@torch.no_grad()
def select_crossover():
    verify_protocol();assert (OUT/'runtime_calibration.json').exists()
    destination=OUT/'crossover_selection.json'
    if destination.exists():return json.loads(destination.read_text())
    est=engine();rows=[];hashes={};started=time.time()
    for ref in RSRS:
        for regime,L in REGIMES:
            a=dev_anchor('A',regime,L,ref)
            for snr in SNRS:
                if not 20<snr<40:continue
                path=OUT/'validation/A/cells'/f'{cellname(regime,ref,snr)}.npz'
                if path.exists():data=load(path)
                else:
                    obs,held=risk_data.fit_holdout(observed(a,snr,P=39))
                    losses=np.stack([np.stack([predictive_loss(held,candidate(obs,family,seed,0,est)) for seed in SEEDS]) for family in FAMILIES])
                    save(path,losses=losses,families=np.array(FAMILIES),snr=np.array(snr),rsr=np.array(ref),regime=np.array(regime),protocol_sha256=np.array(sha(OUT/'protocol.json')))
                    data=load(path)
                assert data['losses'].shape==(2,3,DEV_COUNT)
                rows.append(dict(snr=snr,means=data['losses'].mean((1,2))))
                hashes[str(path.relative_to(OUT))]=sha(path)
            print('Crossover validation',regime,ref,flush=True)
    scores={cut:float(np.mean([r['means'][FAMILIES.index(chosen_family(r['snr'],cut))] for r in rows])) for cut in CROSSOVERS}
    chosen=tie_argmin(scores,CROSSOVERS)
    for p in (OUT/'validation/A/anchors').glob('*.npz'):hashes[str(p.relative_to(OUT))]=sha(p)
    result=dict(protocol_sha256=verify_protocol(),started_unix=started,completed_unix=time.time(),crossover_db=chosen,
                scores={f'{k:g}':v for k,v in scores.items()},validation_hashes=hashes,cells=len(rows))
    write_json(destination,result);print('Crossover selected',chosen,scores,flush=True)
    return result


@torch.no_grad()
def propose_tails():
    verify_protocol();cut=select_crossover()['crossover_db']
    destination=OUT/'tail_proposals.json'
    if destination.exists():return json.loads(destination.read_text())
    calibration=json.loads((OUT/'runtime_calibration.json').read_text())
    est=engine();hashes={};tables={};started=time.time()
    for ref in RSRS:
        data_by_snr={snr:[] for snr in SNRS if snr>20}
        for regime,L in REGIMES:
            a=dev_anchor('B',regime,L,ref)
            for snr in data_by_snr:
                family=chosen_family(snr,cut)
                path=OUT/'validation/B/cells'/f'{cellname(regime,ref,snr)}.npz'
                if path.exists():data=load(path)
                else:
                    obs,held=risk_data.fit_holdout(observed(a,snr,P=39))
                    by_seed=[]
                    for seed in SEEDS:
                        initial=est['starts'].start(family,obs,seed=seed)
                        trajectory,_=continuation.trajectory(obs,initial,TAILS)
                        by_seed.append(np.stack([predictive_loss(held,finish(obs,trajectory[step])) for step in TAILS]))
                    losses=np.stack(by_seed,axis=1)
                    save(path,losses=losses,steps=np.array(TAILS),family=np.array(family),snr=np.array(snr),rsr=np.array(ref),regime=np.array(regime),protocol_sha256=np.array(sha(OUT/'protocol.json')))
                    data=load(path)
                assert data['losses'].shape==(4,3,DEV_COUNT)
                data_by_snr[snr].append(data['losses'].mean(1))
                hashes[str(path.relative_to(OUT))]=sha(path)
            print('Tail proposal validation',regime,ref,flush=True)
        tables[f'{ref:g}']={}
        for snr,regime_losses in data_by_snr.items():
            family=chosen_family(snr,cut)
            eligible=[step for step in TAILS if calibration['scores'][family][str(step)]['eligible']]
            if 0 not in eligible:eligible=[0]+eligible
            scores={step:float(np.mean([losses[TAILS.index(step)] for losses in regime_losses])) for step in TAILS}
            best=tie_argmin(scores,eligible)
            guards={regime:guard(losses[0]-losses[TAILS.index(best)],.05/3) for (regime,_),losses in zip(REGIMES,regime_losses)}
            proposed=best if best>0 and all(g['passed'] for g in guards.values()) else 0
            tables[f'{ref:g}'][f'{snr:g}']=dict(family=family,best_steps=best,proposed_steps=proposed,
                eligible_steps=eligible,scores={str(k):v for k,v in scores.items()},guards=guards)
    for p in (OUT/'validation/B/anchors').glob('*.npz'):hashes[str(p.relative_to(OUT))]=sha(p)
    result=dict(protocol_sha256=verify_protocol(),crossover_sha256=sha(OUT/'crossover_selection.json'),
                runtime_calibration_sha256=sha(OUT/'runtime_calibration.json'),started_unix=started,completed_unix=time.time(),proposals=tables,validation_hashes=hashes)
    write_json(destination,result);print('Tail proposals frozen',flush=True)
    return result


@torch.no_grad()
def confirm_tails():
    verify_protocol();proposal=propose_tails()
    destination=OUT/'tail_confirmation.json'
    if destination.exists():return json.loads(destination.read_text())
    est=engine();hashes={};tables={};started=time.time()
    for ref in RSRS:
        data_by_snr={snr:[] for snr in SNRS if snr>20}
        for regime,L in REGIMES:
            a=dev_anchor('C',regime,L,ref)
            for snr in data_by_snr:
                proposed=proposal['proposals'][f'{ref:g}'][f'{snr:g}']
                family,step=proposed['family'],proposed['proposed_steps']
                path=OUT/'validation/C/cells'/f'{cellname(regime,ref,snr)}.npz'
                if path.exists():data=load(path)
                else:
                    obs,held=risk_data.fit_holdout(observed(a,snr,P=39));by_seed=[]
                    for seed in SEEDS:
                        initial=est['starts'].start(family,obs,seed=seed)
                        trajectory,_=continuation.trajectory(obs,initial,(0,step))
                        by_seed.append(np.stack([predictive_loss(held,finish(obs,trajectory[t])) for t in (0,step)]))
                    losses=np.stack(by_seed,axis=1)
                    save(path,losses=losses,steps=np.array((0,step)),family=np.array(family),snr=np.array(snr),rsr=np.array(ref),regime=np.array(regime),protocol_sha256=np.array(sha(OUT/'protocol.json')))
                    data=load(path)
                assert data['losses'].shape==(2,3,DEV_COUNT)
                data_by_snr[snr].append(data['losses'].mean(1))
                hashes[str(path.relative_to(OUT))]=sha(path)
            print('Tail confirmation',regime,ref,flush=True)
        tables[f'{ref:g}']={}
        for snr,regime_losses in data_by_snr.items():
            proposed=proposal['proposals'][f'{ref:g}'][f'{snr:g}']
            guards={regime:guard(losses[0]-losses[1],.05/108) for (regime,_),losses in zip(REGIMES,regime_losses)}
            accepted=proposed['proposed_steps']>0 and all(g['passed'] for g in guards.values())
            tables[f'{ref:g}'][f'{snr:g}']=dict(family=proposed['family'],proposed_steps=proposed['proposed_steps'],
                steps=proposed['proposed_steps'] if accepted else 0,accepted=accepted,guards=guards)
    for p in (OUT/'validation/C/anchors').glob('*.npz'):hashes[str(p.relative_to(OUT))]=sha(p)
    result=dict(protocol_sha256=verify_protocol(),proposal_sha256=sha(OUT/'tail_proposals.json'),started_unix=started,
                completed_unix=time.time(),confirmed=tables,validation_hashes=hashes)
    write_json(destination,result);print('Tail confirmation complete',flush=True)
    return result


def freeze_selection():
    verify_protocol();confirmation=confirm_tails()
    crossover=json.loads((OUT/'crossover_selection.json').read_text())
    tails={}
    for ref in RSRS:
        tails[f'{ref:g}']={}
        for snr in SNRS:
            row=dict(family='adaptive',steps=0) if snr<=20 else {k:confirmation['confirmed'][f'{ref:g}'][f'{snr:g}'][k] for k in ('family','steps')}
            tails[f'{ref:g}'][f'{snr:g}']=row
    from . import ber
    spec=dict(protocol_sha256=verify_protocol(),crossover_db=crossover['crossover_db'],tails=tails,
        baseline_policy=json.loads((previous.OUT/'policy.json').read_text()),
        ber_protocol_sha256=sha(OUT/'ber/protocol.json'),
        ber_source_hashes=json.loads((OUT/'ber/protocol.json').read_text())['source_hashes'],
        selection_evidence={name:sha(OUT/name) for name in ('runtime_calibration.json','crossover_selection.json','tail_proposals.json','tail_confirmation.json')})
    path=OUT/'policy.json'
    if path.exists():
        old=json.loads(path.read_text());old.pop('frozen_unix');assert old==spec
    else:write_json(path,dict(**spec,frozen_unix=time.time()))
    print('Hybrid selection frozen',sha(path),spec['crossover_db'],tails,flush=True)


@torch.no_grad()
def predictions(obs,selection,est):
    raw={name:start_estimate(obs,name,selection,est) for name in RAW_NAMES}
    # One homogeneous main-grid cell. Share starts/trajectory prefixes solely
    # for accuracy; runtime always executes estimate_method independently.
    plans=inference.plans(obs,selection)
    assert all((p['family'],p['steps'],p['ordinary'])==(plans[0]['family'],plans[0]['steps'],plans[0]['ordinary']) for p in plans)
    plan=plans[0];outputs={}
    for seed in SEEDS:
        values={}
        for family in FAMILIES:
            needed=(0,40,plan['steps']) if family==plan['family'] else (0,40)
            values[family]=continuation.trajectory(obs,raw[f'{family}_{seed}'],needed)[0]
            outputs[f'{family}40_{seed}']=finish(obs,values[family][40])
        chosen=values[plan['family']]
        outputs[f'hybrid_{seed}']=chosen[0] if plan['ordinary'] else finish(obs,chosen[plan['steps']])
        outputs[f'hybrid0_{seed}']=chosen[0] if plan['ordinary'] else finish(obs,chosen[0])
        outputs[f'latest_{seed}']=estimate_method(obs,f'latest_{seed}',selection,est)
    for name in ('classical40','em100','compact_sparse','optshrink'):
        outputs[name]=estimate_method(obs,name,selection,est)
    return torch.stack([raw[name] for name in RAW_NAMES]),torch.stack([outputs[name] for name in NAMES])


@torch.no_grad()
def test(reference=None):
    selection,digest=check_selection();est=engine()
    assert (OUT/'timing_window1.json').exists(),'First timing window precedes test'
    for ref in RSRS:
        if reference is not None and reference!=ref:continue
        for regime,L in REGIMES:
            anchor_path=OUT/'test/anchors'/f'{regime}_rsr{ref:g}.npz'
            if anchor_path.exists():a=load(anchor_path)
            else:
                a=make_anchor(TEST_SEED,range(TEST_COUNT),L,ref,truth=True)
                a.update(selection_sha256=np.array(digest),protocol_sha256=np.array(selection['protocol_sha256']),regime=np.array(regime))
                save(anchor_path,**a)
            assert str(a['selection_sha256'])==digest
            for snr in SNRS:
                path=OUT/'test/cells'/f'{cellname(regime,ref,snr)}.npz'
                if path.exists():
                    assert str(load(path)['selection_sha256'])==digest
                    continue
                obs=observed(a,snr);raw,pred=predictions(obs,selection,est)
                arr,starts=pred.numpy(),raw.numpy();assert np.isfinite(arr).all() and np.isfinite(starts).all()
                energy=np.sum(abs(a['g'])**2,axis=(-2,-1))
                errors=np.sum(abs(arr-a['g'][None])**2,axis=(-2,-1));raw_errors=np.sum(abs(starts-a['g'][None])**2,axis=(-2,-1))
                maximum=0.
                for i,name in enumerate(NAMES):
                    actual=estimate_method(obs.batch(slice(0,1)),name,selection,est)
                    difference=float((actual-pred[i,0:1]).abs().max())
                    assert difference<1e-10,(name,difference);maximum=max(maximum,difference)
                save(path,**{k:v.numpy() for k,v in zip(('z','s','b','sigma2'),obs.args())},truth_g=a['g'],paths=a['paths'],
                    indices=a['indices'],local_indices=a['local_indices'],method_names=np.array(NAMES),estimates=arr,errors=errors,
                    truth_energy=energy,nmse=errors/energy[None],raw_names=np.array(RAW_NAMES),raw_estimates=starts,raw_errors=raw_errors,
                    snr=np.array(snr),rsr=np.array(ref),regime=np.array(regime),seed=np.array(TEST_SEED),
                    selection_sha256=np.array(digest),protocol_sha256=np.array(selection['protocol_sha256']),
                    actual_dispatch_max_abs_error=np.array(maximum))
                print('Test complete',path.stem,flush=True)


@torch.no_grad()
def timing(window):
    selection,digest=check_selection();destination=OUT/f'timing_window{window}.json'
    assert not destination.exists(),'No timing retries until passing'
    assert window in (1,2)
    indices=(500,501,502) if window==1 else (600,601,602)
    est=engine();rows=timing_rows(indices);samples={n:[] for n in NAMES}
    for name in NAMES:estimate_method(rows[0][1],name,selection,est)
    rng=np.random.default_rng(np.random.SeedSequence([TIMING_SEED,window]));started=time.time()
    for wi in rng.permutation(len(rows)):
        meta,obs=rows[wi]
        for ni in rng.permutation(len(NAMES)):
            name=NAMES[ni];tick=time.perf_counter_ns()
            result=estimate_method(obs,name,selection,est)
            elapsed=(time.perf_counter_ns()-tick)/1e6
            assert torch.isfinite(result).all();samples[name].append(dict(ms=elapsed,**meta))
        if len(samples[NAMES[0]])%25==0:print('Timing',window,len(samples[NAMES[0]]),'/',len(rows),flush=True)
    values=dict(samples)
    values.update({group:[v for n in members for v in samples[n]] for group,members in GROUPS.items() if all(n in samples for n in members)})
    methods={}
    for name,records in values.items():
        methods[name]=dict(**stats([r['ms'] for r in records]),
            weak_warm100=stats([r['ms'] for r in records if r['rsr']==5 and r['snr']>=37.5]),
            by_condition={f'rsr{r:g}_snr{s:g}':stats([v['ms'] for v in records if v['rsr']==r and v['snr']==s]) for r in RSRS for s in SNRS})
    result=dict(window=window,selection_sha256=digest,protocol_sha256=selection['protocol_sha256'],started_unix=started,
        completed_unix=time.time(),indices=indices,worlds=len(rows),methods=methods,raw_samples=samples,
        cap_verification={n:dict(p90_ms=m['p90_ms'],passed=m['p90_ms']<=CAP_MS) for n,m in methods.items()},
        environment=dict(platform=platform.platform(),python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,torch=torch.__version__,torch_threads=torch.get_num_threads()))
    write_json(destination,result)
    print('Timing complete',window,result['cap_verification']['hybrid_mean'],flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('phase',choices=('freeze','calibrate','crossover','propose','confirm','select','test','timing1','timing2'))
    p.add_argument('--reference',type=float,choices=RSRS);args=p.parse_args();torch.set_num_threads(1)
    {'freeze':freeze_protocol,'calibrate':calibrate_runtime,'crossover':select_crossover,'propose':propose_tails,
     'confirm':confirm_tails,'select':freeze_selection,'test':lambda:test(args.reference),
     'timing1':lambda:timing(1),'timing2':lambda:timing(2)}[args.phase]()


if __name__=='__main__':main()
