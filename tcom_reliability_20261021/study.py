"""Fresh, observable-calibrated reference-aware fallback experiment."""
import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[_key]='1'
import argparse
import json
import platform
import time
from pathlib import Path
import numpy as np
import scipy
import torch
from tcom_followups_20261017.common import REPO,sha,write_json
from tcom_followups_20261017.ber_detector import hypotheses,detect
from tcom_hybrid_20261020 import study as parent
from tcom_incumbent_guard_20261019 import risk_data
from tcom_pilot_em_20261018 import continuation
from . import inference

OUT=REPO/'results/tcom_reliability_20261021'
SOURCE=REPO/'results/tcom_hybrid_sensitivity_20261020'
BASE_SPEC=json.loads((SOURCE/'protocol.json').read_text())
BASE_POLICY=BASE_SPEC['proposal_policy']
SEEDS=parent.SEEDS
SNRS,RSRS,REGIMES=parent.SNRS,parent.RSRS,parent.REGIMES
WEAK_SNRS=tuple(s for s in SNRS if s>30.)
SMOKE_SEED=2026102100
DEV_SEEDS={'A':2026102101,'C':2026102102}
CALIBRATION_SEED=2026102103
TEST_SEED=2026102104
DATA_SEED=2026102105
TIMING_SEED=2026102106
MASK_SEED=2026102107
BOOT_SEED=2026102108
PROBE_SEED=2026102109
TEST_COUNT=128
DEV_COUNT=256
CAP_MS=10.74
CALIBRATION_CAP_MS=10.
RULES=tuple(dict(id=f'{metric}_t{threshold}_e{tail}',metric=metric,threshold=float(threshold),tail=tail)
            for metric in ('mean','max') for threshold in (2,4,8) for tail in (10,20))
RULE_NAMES=('incumbent',)+tuple(r['id'] for r in RULES)
NAMES=tuple(f'{family}_{seed}' for family in ('receiver','prior','latest','adaptive40','global40') for seed in SEEDS)+('classical40','em100','compact_sparse','algebraic20')
GROUPS={f'{family}_mean':[f'{family}_{s}' for s in SEEDS] for family in ('receiver','prior','latest','adaptive40','global40')}
PRIMARY=('receiver_vs_prior','receiver_vs_latest')
save,load,cellname,stats=parent.save,parent.load,parent.cellname,parent.stats
finish,engine,observed=parent.finish,parent.engine,risk_data.observed


def physical_seed(seed,ref):
    if float(ref) not in RSRS:raise ValueError('reference outside frozen grid')
    return int(seed)*1000+int(round(float(ref)*10))


def make_anchor(seed,ids,L,ref,*,truth=False,snrs=SNRS):
    derived=physical_seed(seed,ref)
    result=risk_data.make_anchor(derived,ids,L,ref,truth=truth,snrs=snrs,holdout_seed=MASK_SEED)
    result.update(seed=np.array(seed),physical_seed=np.array(derived))
    return result


def source_paths():
    here=Path(__file__).parent
    own=[here/n for n in ('study.py','inference.py','checks.py','pipeline_checks.py','execution.py','ber.py','analysis.py')]
    inherited=[REPO/p for p in BASE_SPEC['source_hashes']]
    return sorted(set(own+inherited))


def freeze():
    from . import ber
    parent.check_selection()
    checks=json.loads((OUT/'checks.json').read_text());assert checks['passed']
    for path,digest in checks['source_hashes'].items():assert sha(REPO/path)==digest
    pipeline=json.loads((OUT/'pipeline_checks.json').read_text());assert pipeline['passed']
    for path,digest in pipeline['source_hashes'].items():assert sha(REPO/path)==digest
    assert not list((OUT/'validation').glob('*/cells/*.npz'))
    assert not list((OUT/'test/cells').glob('*.npz'))
    spec=dict(name='Reference-aware observable residual gate with algebraic fallback',
        N=32,K=3,P=30,snrs=SNRS,rsrs=RSRS,regimes=REGIMES,seeds=SEEDS,names=NAMES,groups=GROUPS,
        test_cases_per_reference_regime=TEST_COUNT,development_cases_per_regime=DEV_COUNT,
        seeds_by_role=dict(smoke=SMOKE_SEED,development=DEV_SEEDS,calibration=CALIBRATION_SEED,test=TEST_SEED,
            data=DATA_SEED,timing=TIMING_SEED,holdout=MASK_SEED,bootstrap=BOOT_SEED,development_probes=PROBE_SEED),
        rules=RULES,base_policy=BASE_POLICY,base_protocol_sha256=sha(SOURCE/'protocol.json'),
        base_completion_sha256=sha(SOURCE/'completion.json'),
        base_history='Frozen B-proposal estimator; all21 nonzero tails failed its prior C confirmation. Explicit incumbent for this follow-up, not retroactively confirmed.',
        gate=dict(eligible='Nearest reference key5 and continuous nominal SNR>30+1e-10; elsewhere exact incumbent.',
            score='Per-antenna 2*mean_p((z-abs(GS+B))**2)/supplied_variance; mean or maximum across antennas. Heuristic fitted residual, not calibrated failure probability.',
            initializer='Real intensity LS with6 linear and9 Hermitian quadratic nuisance columns; correct variance subtraction, rank15, condition<=1e8; repeated nonzero reference.',
            acceptance='Only finite final filtered fallback with mean fitting Rician NLL strictly lower than final incumbent by>1e-12; otherwise incumbent bitwise.',
            tails=[10,20],invalid='Gated invalid LS retains incumbent. Standalone algebraic20 uses spectral+20 on invalid LS.',
            no_extra_online_pilots=True,ordinary_bitwise_preserved=True),
        development=dict(banks=['A','C'],reference=5,snrs=WEAK_SNRS,regimes=REGIMES,anchors_per_regime=DEV_COUNT,
            context=30,heldout_gaussian_pilots=9,known_symbol_probes=64,truth_exported=False,
            selection='A: runtime-eligible complete rule must improve mean heldout NLL>1e-12 in EACH path regime and not increase known-probe total BER in EACH regime. Incumbent always admissible. Minimize equal-fit/SNR/regime NLL.',
            tie='Within1e-12 of minimum NLL: smaller trigger fraction; smaller EM tail; larger threshold; mean before max. Incumbent wins a numerical tie.',
            confirmation='C: evaluate only A choice versus incumbent. Positive pooled weak-scope paired-bootstrap one-sided95 NLL lower bound and empirical probe BER nonincrease in EACH path regime. Regime NLL intervals descriptive; no formal BER noninferiority claim.',
            failure='No candidate replacement/search on C. Operational rule remains incumbent if confirmation fails. Fresh receiver arm always names frozen A proposal and is labeled unconfirmed when applicable.'),
        test=dict(cells=153,physical_anchors=1152,heldout_used=False,primary_contrasts=PRIMARY,
            reference_seed_domain='physical_seed=1000*role_seed+round(10*reference_dB); independently separates random path counts, channels, fitting/heldout pilots and noise across references.',
            outcomes=['NMSE','BER'],adjustment='Bonferroni4 on all153 only',
            bootstrap='20,000 draws: independently resample128 indices within each of9(regime,reference) blocks; shared across SNR/method/fit within block.1152 physical anchor clusters.',
            omitted_baseline='OptShrink omitted prospectively from budget-focused follow-up; previous measured p90~79-81ms exceeded10.74. No historical number presented as fresh.'),
        timing=dict(cap_ms=CAP_MS,calibration_cap_ms=CALIBRATION_CAP_MS,calibration_indices=[401],
            calibration='One complete12-rule randomized full-grid calibration before A, pooled full and weak_target p90<=10ms. Forced-fallback timing separately reported.',
            windows=2,indices=[[500,501,502],[600,601,602]],worlds_per_window=459,
            acceptance='Both windows require full-grid AND weak-target p90<=10.74ms; weak-warm, triggered branch and51 bins separately disclosed. A p90 cap is not a hard per-call deadline.',
            complete_pipeline=True,one_cpu_thread=True,warmup='Every method on ordinary and reference5/SNR60 observations; model loading/shape constants excluded.',
            order='Window1 after policy freeze before fresh test; window2 after all numerical accuracy jobs are idle. No retries until passing.'),
        ber=ber.protocol_spec(),checks_sha256=sha(OUT/'checks.json'),pipeline_checks_sha256=sha(OUT/'pipeline_checks.json'),
        source_hashes={str(p.relative_to(REPO)):sha(p) for p in source_paths()},
        checkpoint_hashes=BASE_SPEC['checkpoint_hashes'])
    path=OUT/'protocol.json';assert not path.exists(),'Protocol already frozen'
    write_json(path,dict(**spec,frozen_unix=time.time()));print('Protocol frozen',sha(path),flush=True)


def verify_protocol():
    spec=json.loads((OUT/'protocol.json').read_text())
    for field in ('source_hashes','checkpoint_hashes'):
        for name,digest in spec[field].items():assert sha(REPO/name)==digest,name
    assert sha(SOURCE/'protocol.json')==spec['base_protocol_sha256']
    assert sha(SOURCE/'completion.json')==spec['base_completion_sha256']
    assert sha(OUT/'checks.json')==spec['checks_sha256']
    assert sha(OUT/'pipeline_checks.json')==spec['pipeline_checks_sha256']
    return sha(OUT/'protocol.json')


def check_selection():
    digest=verify_protocol();p=json.loads((OUT/'policy.json').read_text())
    assert p['protocol_sha256']==digest
    for name,h in p['selection_evidence'].items():assert sha(OUT/name)==h,name
    return p,sha(OUT/'policy.json')


def timing_rows(indices,seed=TIMING_SEED):
    rows=[]
    for ref in RSRS:
        for regime,L in REGIMES:
            a=make_anchor(seed,indices,L,ref)
            for j,index in enumerate(indices):
                for snr in SNRS:rows.append((dict(regime=regime,rsr=ref,snr=snr,index=index),observed(a,snr).batch(slice(j,j+1))))
    return rows


def calibrate():
    verify_protocol();path=OUT/'runtime_calibration.json';assert not path.exists()
    est=engine();rows=timing_rows((401,),CALIBRATION_SEED);rng=np.random.default_rng(CALIBRATION_SEED)
    warm=[rows[0][1],next(o for m,o in rows if m['rsr']==5 and m['snr']==60)]
    for obs in warm:
        for rule in RULES:
            for seed in SEEDS:inference.gated(obs,seed,rule,est,BASE_POLICY)
    menu=[(r,s) for r in RULES for s in SEEDS];records=[];started=time.time()
    for count,wi in enumerate(rng.permutation(len(rows)),1):
        meta,obs=rows[wi]
        for ci in rng.permutation(len(menu)):
            rule,seed=menu[ci];tick=time.perf_counter_ns()
            g,diag=inference.gated(obs,seed,rule,est,BASE_POLICY,diagnostics=True)
            ms=(time.perf_counter_ns()-tick)/1e6;assert torch.isfinite(g).all()
            records.append(dict(**meta,rule=rule['id'],seed=seed,ms=ms,triggered=bool(diag['triggered'][0]),accepted=bool(diag['accepted'][0])))
        if count%25==0:print('Calibration',count,'/',len(rows),flush=True)
    scores={}
    for rule in RULES:
        selected=[r for r in records if r['rule']==rule['id']]
        full=stats([r['ms'] for r in selected]);weak=stats([r['ms'] for r in selected if r['rsr']==5 and r['snr']>30])
        scores[rule['id']]=dict(**full,weak_target=weak,eligible=full['p90_ms']<=10 and weak['p90_ms']<=10)
    # A synthetic threshold forces the whole fallback path, without selecting a receiver.
    forced=[]
    for meta,obs in rows:
        if meta['rsr']!=5 or meta['snr']<=30:continue
        for tail in (10,20):
            for seed in SEEDS:
                rule=dict(id='forced',metric='mean',threshold=2.,tail=tail);tick=time.perf_counter_ns()
                _,diag=inference.gated(obs,seed,rule,est,BASE_POLICY,diagnostics=True,force_fallback=True)
                forced.append(dict(**meta,tail=tail,seed=seed,ms=(time.perf_counter_ns()-tick)/1e6,triggered=bool(diag['triggered'][0])))
    write_json(path,dict(protocol_sha256=verify_protocol(),started_unix=started,completed_unix=time.time(),records=records,scores=scores,forced_records=forced))
    print('Calibration complete',scores,flush=True)


def development_anchor(bank,regime,L):
    path=OUT/'validation'/bank/'anchors'/f'{regime}.npz'
    if path.exists():return load(path)
    seed=DEV_SEEDS[bank];a=make_anchor(seed,range(DEV_COUNT),L,5.,truth=True,snrs=WEAK_SNRS)
    symbols,bits=hypotheses(3);probe=[]
    noise=[]
    for j,index in enumerate(a['indices']):
        rng=np.random.default_rng(np.random.SeedSequence([PROBE_SEED,seed,int(index),0x50524f42]))
        x=rng.standard_normal((32,64,2));noise.append((x[:,:,0]+1j*x[:,:,1])/np.sqrt(2.))
    noise=np.stack(noise);field=a['g']@symbols.T+a['b'][...,:1]
    for snr in WEAK_SNRS:
        variance=a['anchor_variance']*10.**((float(a['anchor_snr_db'])-snr)/10.)
        probe.append(abs(field+np.sqrt(variance)[:,None,None]*noise))
    public={k:v for k,v in a.items() if k not in ('g','unit_noise','paths')}
    public.update(probe_z_by_snr=np.stack(probe),probe_symbols=symbols,probe_bits=bits,
        bank=np.array(bank),regime=np.array(regime),protocol_sha256=np.array(verify_protocol()))
    assert not {'g','truth_g','unit_noise','paths'}.intersection(public)
    save(path,**public);return public


def rule_arrays(cell,rule):
    base=cell['heldout_nll_base'];counts=cell['probe_errors_base']
    if rule is None:return base,counts,np.zeros(base.shape,bool),np.zeros(base.shape,bool)
    j=(10,20).index(rule['tail']);trigger=cell['score_'+rule['metric']]>rule['threshold']
    accept=trigger & cell['fallback_valid'][j][None] & (cell['fit_nll_fallback'][j][None] < cell['fit_nll_base']-1e-12)
    losses=np.where(accept,cell['heldout_nll_fallback'][j][None],base)
    errors=np.where(accept[:,:,None],cell['probe_errors_fallback'][j][None],counts)
    return losses,errors,trigger,accept


@torch.no_grad()
def validation(bank,rule=None,regime_filter=None):
    verify_protocol();est=engine();count=0
    for regime,L in REGIMES:
        if regime_filter is not None and regime!=regime_filter:continue
        anchor=development_anchor(bank,regime,L)
        for si,snr in enumerate(WEAK_SNRS):
            path=OUT/'validation'/bank/'cells'/f'{cellname(regime,5.,snr)}.npz'
            if path.exists():continue
            full=observed(anchor,snr,P=39);obs,hold=risk_data.fit_holdout(full)
            bases=torch.stack([parent.inference.hybrid(obs,seed,BASE_POLICY,est) for seed in SEEDS]).numpy()
            tails=(10,20) if bank=='A' else (() if rule is None else (rule['tail'],))
            fallback=np.stack([bases[0].copy(),bases[0].copy()]);valid=np.zeros((2,DEV_COUNT),bool)
            conditions=np.full((2,DEV_COUNT),np.inf)
            for tail in tails:
                g,v,diag=inference.fallback(obs,tail);j=(10,20).index(tail)
                fallback[j]=g.numpy();valid[j]=v;conditions[j]=diag['condition']
            z,s,b,v=(x.numpy() for x in obs.args());hz,hs,hb,hv=(x.numpy() for x in hold.args())
            def loss(g,zz,ss,bb,vv):return continuation.nll(zz,g@ss+bb,vv).mean((-2,-1))
            fitbase=np.stack([loss(g,z,s,b,v) for g in bases]);fitfb=np.stack([loss(g,z,s,b,v) for g in fallback])
            holdbase=np.stack([loss(g,hz,hs,hb,hv) for g in bases]);holdfb=np.stack([loss(g,hz,hs,hb,hv) for g in fallback])
            residual=2*np.mean((z[None]-abs(bases@s[None]+b[None]))**2,axis=-1)/v[None,:,None]
            probe=anchor['probe_z_by_snr'][si];bits=anchor['probe_bits']
            eb=np.zeros((3,DEV_COUNT,3),np.int64);ef=np.zeros((2,DEV_COUNT,3),np.int64)
            for k,g in enumerate(bases):
                for j in range(DEV_COUNT):eb[k,j]=(detect(probe[j],g[j],b[j,:,:1],float(v[j]))!=bits).sum((0,2))
            for tail in tails:
                k=(10,20).index(tail)
                for j in range(DEV_COUNT):
                    if valid[k,j]:ef[k,j]=(detect(probe[j],fallback[k,j],b[j,:,:1],float(v[j]))!=bits).sum((0,2))
            save(path,base_g=bases,fallback_g=fallback,fallback_valid=valid,fallback_condition=conditions,
                score_mean=residual.mean(-1),score_max=residual.max(-1),fit_nll_base=fitbase,fit_nll_fallback=fitfb,
                heldout_nll_base=holdbase,heldout_nll_fallback=holdfb,probe_errors_base=eb,probe_errors_fallback=ef,
                bank=np.array(bank),snr=np.array(snr),rsr=np.array(5.),regime=np.array(regime),
                anchor_sha256=np.array(sha(OUT/'validation'/bank/'anchors'/f'{regime}.npz')),protocol_sha256=np.array(verify_protocol()))
            count+=1;print('Validation',bank,path.stem,flush=True)


def development_scores(bank,rule):
    results={};hashes={};all_trigger=[];all_accept=[]
    for regime,_ in REGIMES:
        losses=[];errors=[];deltas=[];base_errors=[]
        for snr in WEAK_SNRS:
            p=OUT/'validation'/bank/'cells'/f'{cellname(regime,5.,snr)}.npz';a=load(p)
            loss,err,trigger,accept=rule_arrays(a,rule)
            losses.append(loss);errors.append(err);deltas.append(a['heldout_nll_base']-loss);base_errors.append(a['probe_errors_base'])
            all_trigger.append(trigger);all_accept.append(accept);hashes[str(p.relative_to(OUT))]=sha(p)
        delta=np.stack(deltas).mean((0,1))
        results[regime]=dict(mean_nll=float(np.mean(losses)),mean_improvement=float(delta.mean()),
            candidate_errors=int(np.sum(errors)),base_errors=int(np.sum(base_errors)),anchor_improvements=delta.tolist(),
            bits_per_fit=len(WEAK_SNRS)*DEV_COUNT*384)
    return dict(regimes=results,mean_nll=float(np.mean([r['mean_nll'] for r in results.values()])),
        trigger_fraction=float(np.mean(all_trigger)),accept_fraction=float(np.mean(all_accept)),cell_hashes=hashes)


def select():
    verify_protocol();assert not (OUT/'selection_A.json').exists()
    calibration=json.loads((OUT/'runtime_calibration.json').read_text());validation('A')
    results={};baseline=development_scores('A',None);results['incumbent']=dict(rule=None,eligible=True,**baseline)
    for rule in RULES:
        score=development_scores('A',rule)
        qualifies=calibration['scores'][rule['id']]['eligible'] and all(r['mean_improvement']>1e-12 and r['candidate_errors']<=r['base_errors'] for r in score['regimes'].values())
        results[rule['id']]=dict(rule=rule,eligible=bool(qualifies),**score)
    eligible=[r for r in results.values() if r['eligible']];minimum=min(r['mean_nll'] for r in eligible)
    tied=[r for r in eligible if r['mean_nll']<=minimum+1e-12]
    def key(r):
        rule=r['rule']
        return (-1,0,0,0) if rule is None else (r['trigger_fraction'],rule['tail'],-rule['threshold'],int(rule['metric']=='max'))
    chosen=min(tied,key=key)
    write_json(OUT/'selection_A.json',dict(protocol_sha256=verify_protocol(),runtime_calibration_sha256=sha(OUT/'runtime_calibration.json'),
        selected_rule=chosen['rule'],scores=results,completed_unix=time.time()))
    print('A selected',chosen['rule'],flush=True)


def confirm():
    verify_protocol();assert not (OUT/'confirmation_C.json').exists()
    choice=json.loads((OUT/'selection_A.json').read_text())['selected_rule'];validation('C',choice)
    score=development_scores('C',choice);rng=np.random.default_rng(np.random.SeedSequence([BOOT_SEED,0x434f4e46]))
    boot=[];regime_ci={}
    for regime,_ in REGIMES:
        d=np.array(score['regimes'][regime]['anchor_improvements'])
        weights=rng.multinomial(DEV_COUNT,np.full(DEV_COUNT,1/DEV_COUNT),size=20000)
        means=weights@d/DEV_COUNT;boot.append(means);regime_ci[regime]=np.quantile(means,[.025,.975]).tolist()
    pooled=np.mean(boot,axis=0);lower=float(np.quantile(pooled,.05))
    ber_ok=all(r['candidate_errors']<=r['base_errors'] for r in score['regimes'].values())
    passed=choice is not None and lower>0 and ber_ok
    result=dict(protocol_sha256=verify_protocol(),selection_A_sha256=sha(OUT/'selection_A.json'),candidate_rule=choice,
        passed=bool(passed),status='incumbent_selected' if choice is None else ('confirmed' if passed else 'unconfirmed'),
        pooled_nll_lower_one_sided95=lower,pooled_nll_ci95=np.quantile(pooled,[.025,.975]).tolist(),
        regime_nll_ci95_descriptive=regime_ci,empirical_ber_nonincrease_every_regime=bool(ber_ok),
        score=score,completed_unix=time.time())
    write_json(OUT/'confirmation_C.json',result)
    policy=dict(protocol_sha256=verify_protocol(),base_policy=BASE_POLICY,base_policy_source_sha256=sha(SOURCE/'protocol.json'),
        candidate_rule=choice,operational_rule=choice if passed else None,confirmation=result,
        selection_evidence={name:sha(OUT/name) for name in ('runtime_calibration.json','selection_A.json','confirmation_C.json')},frozen_unix=time.time())
    assert not (OUT/'policy.json').exists();write_json(OUT/'policy.json',policy)
    print('Policy frozen',choice,result['status'],sha(OUT/'policy.json'),flush=True)


@torch.no_grad()
def estimate_method(obs,name,selection,est,diagnostics=False):
    if name.startswith('receiver_'):
        return inference.gated(obs,int(name.rsplit('_',1)[1]),selection['candidate_rule'],est,BASE_POLICY,diagnostics=diagnostics)
    if name.startswith('prior_'):return parent.inference.hybrid(obs,int(name.rsplit('_',1)[1]),BASE_POLICY,est)
    if name=='algebraic20':return inference.algebraic20(obs,diagnostics=diagnostics)
    return parent.estimate_method(obs,name,BASE_POLICY,est)


@torch.no_grad()
def predictions(obs,selection,est):
    plan=parent.inference.plans(obs,BASE_POLICY)[0];outputs={};diags=[]
    for seed in SEEDS:
        values={}
        for family in ('adaptive','global'):
            initial=est['starts'].start(family,obs,seed=seed)
            steps=(0,40,plan['steps']) if family==plan['family'] else (0,40)
            values[family]=continuation.trajectory(obs,initial,steps)[0]
            outputs[f'{family}40_{seed}']=finish(obs,values[family][40])
        prior=values[plan['family']][0] if plan['ordinary'] else finish(obs,values[plan['family']][plan['steps']])
        outputs[f'prior_{seed}']=prior
        outputs[f'receiver_{seed}'],diag=inference.from_base(obs,prior,selection['candidate_rule'],diagnostics=True);diags.append(diag)
        outputs[f'latest_{seed}']=estimate_method(obs,f'latest_{seed}',selection,est)
    for name in ('classical40','em100','compact_sparse','algebraic20'):outputs[name]=estimate_method(obs,name,selection,est)
    return torch.stack([outputs[n] for n in NAMES]),diags


@torch.no_grad()
def test(reference=None):
    selection,digest=check_selection();assert (OUT/'timing_window1.json').exists();est=engine()
    for ref in RSRS:
        if reference is not None and ref!=reference:continue
        for regime,L in REGIMES:
            ap=OUT/'test/anchors'/f'{regime}_rsr{ref:g}.npz'
            if ap.exists():a=load(ap)
            else:
                a=make_anchor(TEST_SEED,range(TEST_COUNT),L,ref,truth=True)
                a.update(selection_sha256=np.array(digest),protocol_sha256=np.array(selection['protocol_sha256']));save(ap,**a)
            for snr in SNRS:
                p=OUT/'test/cells'/f'{cellname(regime,ref,snr)}.npz'
                if p.exists():assert str(load(p)['selection_sha256'])==digest;continue
                obs=observed(a,snr);pred,diags=predictions(obs,selection,est);arr=pred.numpy();assert np.isfinite(arr).all()
                maximum=0.
                for i,name in enumerate(NAMES):
                    actual=estimate_method(obs.batch(slice(0,1)),name,selection,est);diff=float((actual-pred[i,:1]).abs().max())
                    assert diff<1e-10,(name,diff);maximum=max(maximum,diff)
                energy=np.sum(abs(a['g'])**2,axis=(-2,-1));errors=np.sum(abs(arr-a['g'][None])**2,axis=(-2,-1))
                save(p,**{k:v.numpy() for k,v in zip(('z','s','b','sigma2'),obs.args())},truth_g=a['g'],paths=a['paths'],
                    indices=a['indices'],local_indices=a['local_indices'],method_names=np.array(NAMES),estimates=arr,errors=errors,
                    truth_energy=energy,nmse=errors/energy[None],gate_scores=np.stack([d['score'] for d in diags]),
                    triggers=np.stack([d['triggered'] for d in diags]),accepted=np.stack([d['accepted'] for d in diags]),
                    rank_invalid=np.stack([d['rank_invalid'] for d in diags]),snr=np.array(snr),rsr=np.array(ref),regime=np.array(regime),
                    seed=np.array(TEST_SEED),selection_sha256=np.array(digest),protocol_sha256=np.array(selection['protocol_sha256']),actual_dispatch_max_abs_error=np.array(maximum))
                print('Test complete',p.stem,flush=True)


@torch.no_grad()
def timing(window):
    selection,digest=check_selection();path=OUT/f'timing_window{window}.json';assert not path.exists()
    if window==1:assert not list((OUT/'test/anchors').glob('*.npz'))
    indices=(500,501,502) if window==1 else (600,601,602)
    rows=timing_rows(indices);est=engine();samples={n:[] for n in NAMES}
    for obs in (rows[0][1],next(o for m,o in rows if m['rsr']==5 and m['snr']==60)):
        for name in NAMES:estimate_method(obs,name,selection,est)
    rng=np.random.default_rng(np.random.SeedSequence([TIMING_SEED,window]));started=time.time()
    for count,wi in enumerate(rng.permutation(len(rows)),1):
        meta,obs=rows[wi]
        for ni in rng.permutation(len(NAMES)):
            name=NAMES[ni];tick=time.perf_counter_ns()
            if name.startswith('receiver_'):
                g,diag=estimate_method(obs,name,selection,est,diagnostics=True)
            else:g=estimate_method(obs,name,selection,est);diag=None
            elapsed=(time.perf_counter_ns()-tick)/1e6;assert torch.isfinite(g).all()
            extra={} if diag is None else dict(gate_triggered=bool(diag['triggered'][0]),fallback_accepted=bool(diag['accepted'][0]),rank_invalid=bool(diag['rank_invalid'][0]))
            samples[name].append(dict(**meta,ms=elapsed,**extra))
        if count%25==0:print('Timing',window,count,'/',len(rows),flush=True)
    pooled=dict(samples);pooled.update({g:[r for n in members for r in samples[n]] for g,members in GROUPS.items()})
    methods={}
    for name,records in pooled.items():
        triggered=[r['ms'] for r in records if r.get('gate_triggered')]
        methods[name]=dict(**stats([r['ms'] for r in records]),
            weak_warm100=stats([r['ms'] for r in records if r['rsr']==5 and r['snr']>=37.5]),
            weak_target=stats([r['ms'] for r in records if r['rsr']==5 and r['snr']>30]),
            triggered_branch=stats(triggered) if triggered else None,
            trigger_count=len(triggered),accepted_count=sum(r.get('fallback_accepted',False) for r in records),
            by_condition={f'rsr{ref:g}_snr{snr:g}':stats([r['ms'] for r in records if r['rsr']==ref and r['snr']==snr]) for ref in RSRS for snr in SNRS})
    write_json(path,dict(window=window,selection_sha256=digest,protocol_sha256=selection['protocol_sha256'],
        started_unix=started,completed_unix=time.time(),indices=indices,worlds=len(rows),raw_samples=samples,methods=methods,
        calls=sum(len(v) for v in samples.values()),cap_verification={n:dict(p90_ms=v['p90_ms'],weak_target_p90_ms=v['weak_target']['p90_ms'],passed=v['p90_ms']<=CAP_MS and v['weak_target']['p90_ms']<=CAP_MS) for n,v in methods.items()},
        environment=dict(platform=platform.platform(),python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,torch=torch.__version__,torch_threads=torch.get_num_threads())))
    print('Timing complete',window,methods['receiver_mean']['p90_ms'],flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=('freeze','calibrate','validateA','validateC','select','confirm','test','timing1','timing2'))
    parser.add_argument('--reference',type=float,choices=RSRS);parser.add_argument('--regime',choices=[r for r,_ in REGIMES]);args=parser.parse_args();torch.set_num_threads(1)
    {'freeze':freeze,'calibrate':calibrate,'select':select,'confirm':confirm,'test':lambda:test(args.reference),
        'validateA':lambda:validation('A',regime_filter=args.regime),
        'validateC':lambda:validation('C',json.loads((OUT/'selection_A.json').read_text())['selected_rule'],args.regime),
        'timing1':lambda:timing(1),'timing2':lambda:timing(2)}[args.phase]()


if __name__=='__main__':main()
