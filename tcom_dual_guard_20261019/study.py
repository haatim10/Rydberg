"""Fresh dual-reference guard study; prior sources and results stay immutable."""
import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[_key]='1'
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from sahure.data import Observations
from sahure.math import nll
from tcom_transition_repair_20261019 import study as previous
from tcom_retrained_bypass_20261018 import study as base
from tcom_runtime_bypass_20261017 import sparse_fast
from tcom_pilot_em_20261018 import continuation
from tcom_followups_20261017.common import REPO,sha,write_json,frozen_optshrink
from . import inference
from .inference import SNRS,RSRS,SEEDS,MENU,PREFERENCE_ORDER,OLD_POLICY,model_paths

OUT=REPO/'results/tcom_dual_guard_20261019'
SMOKE_SEED=2026101930;DEV_SEED=2026101931;TEST_SEED=2026101932;DATA_SEED=2026101933;TIMING_SEED=2026101934
MASK_SEED=2026101935;GUARD_BOOT_SEED=2026101937;BOOT_SEED=2026101939
GUARD_BOOT_COUNT=4000;DEV_COUNT=384;DEV_BASE_COUNT=96;TEST_COUNT=128;PREFERENCE_BAND=0.001
TARGET_SNRS=(32.5,35.,37.5);STRONG_RSRS=(12.,20.)
REGIMES=(('L2',2),('L4',4),('random3to7',None))
NAMES=tuple([f'{group}_{s}' for group in ('original','old_bypass','prior_repair','repair') for s in SEEDS]+
    ['em100','optshrink','prior_compact','compact_sparse','prior_sparse_full'])
GROUPS={f'{group}_mean':[f'{group}_{s}' for s in SEEDS] for group in ('original','old_bypass','prior_repair','repair')}
CONTRASTS=dict(repair_vs_old=('old_bypass_mean','repair_mean'),repair_vs_prior=('prior_repair_mean','repair_mean'),
    repair_vs_original=('original_mean','repair_mean'),repair_vs_em100=('em100','repair_mean'),repair_vs_compact=('compact_sparse','repair_mean'))
PRIOR_REPAIR_PATH=previous.OUT/'policy.json';PRIOR_REPAIR=json.loads(PRIOR_REPAIR_PATH.read_text())
PRIOR_COMPACT_PATH=base.OUT/'policy.json';PRIOR_COMPACT=json.loads(PRIOR_COMPACT_PATH.read_text())
OPT_LOOKUP=previous.OPT_LOOKUP
save=previous.save;load=previous.load;cellname=previous.cellname;make_anchor=previous.make_anchor;observed=previous.observed


def engine():return inference.RepairEstimator()


def development_count(rsr,snr):
    return DEV_COUNT if float(rsr) in STRONG_RSRS and float(snr) in TARGET_SNRS else DEV_BASE_COUNT


def fit_holdout(obs,regime,rsr):
    assert obs.z.shape[-1]==39
    pi=[r for r,L in REGIMES].index(regime);ri=list(RSRS).index(float(rsr))
    order=torch.rand(len(obs),39,generator=torch.Generator().manual_seed(MASK_SEED+100*pi+ri)).argsort(-1)
    return obs.columns(order[:,:30]),obs.columns(order[:,30:])


def observed_key(obs):
    keys=inference.receiver_keys(obs)
    if len(set((r,s) for r,s,_ in keys))!=1:raise ValueError('expected homogeneous receiver lookup')
    return keys[0][:2]


def selected_repair_choice(obs,selection):
    choices=inference.choices_for(obs,selection['repair_choices'])
    if len(set(choices))!=1:raise ValueError('expected homogeneous repair choice')
    return choices[0]


def compact_config(obs,selection):
    ref,snr=observed_key(obs);return selection['compact_choices'][ref][snr]['config']


@torch.no_grad()
def estimate_method(obs,name,selection,est):
    if name=='em100':return est.em100(obs)
    if name=='optshrink':return frozen_optshrink(obs,OPT_LOOKUP)
    if name=='prior_sparse_full':return sparse_fast.estimate(obs,base.FULL)
    if name=='prior_compact':return sparse_fast.estimate(obs,compact_config(obs,PRIOR_COMPACT))
    if name=='compact_sparse':return sparse_fast.estimate(obs,compact_config(obs,selection))
    seed=int(name.rsplit('_',1)[1]);model=f'learned_{seed}'
    if name.startswith('original_'):return est.original(obs,model)
    if name.startswith('old_bypass_'):return est.estimate(obs,model,OLD_POLICY,fallback='em100')
    if name.startswith('prior_repair_'):return est.repair(obs,model,PRIOR_REPAIR['repair_choices'])
    if name.startswith('repair_'):return est.repair(obs,model,selection['repair_choices'])
    raise KeyError(name)


def freeze_protocol():
    previous_digest=previous.verify_protocol();previous.check_selection()
    sources=[Path(__file__),Path(inference.__file__),Path(continuation.__file__),
        REPO/'tcom_dual_guard_20261019/sensitivity.py',REPO/'tcom_dual_guard_20261019/historical_replay.py']
    spec=dict(version=1,name='Short EM continuation with strict original/old-bypass development guards',
        smoke_seed=SMOKE_SEED,development_seed=DEV_SEED,test_seed=TEST_SEED,data_seed=DATA_SEED,timing_seed=TIMING_SEED,mask_seed=MASK_SEED,
        N=32,K=3,generated_P=39,context_P=30,heldout_P=9,deployment_P=30,snrs=SNRS,rsrs=RSRS,regimes=REGIMES,model_seeds=SEEDS,
        development_counts={f'{r:g}':{f'{s:g}':development_count(r,s) for s in SNRS} for r in RSRS},
        development_target_count=DEV_COUNT,development_elsewhere_count=DEV_BASE_COUNT,test_cases_per_reference_regime=TEST_COUNT,
        menu=MENU,preference_order=PREFERENCE_ORDER,preference_band_mean_nll=PREFERENCE_BAND,
        preference='After strict dual admissibility only, retain candidates within0.001 mean heldout NLL of the admissible optimum and choose the fixed preference order. This is not measured speed and never relaxes a guard.',
        routing='Use nearest declared reference and SNR bins, ties lower. Reference5 exactly retains previous frozen repair at every bin. Strong references12/20 use original through bin30, calibrate only32.5/35/37.5, and canonical spectral100 from bin40. Ordinary proxy<=20 remains original.',
        model='Same three frozen context30 checkpoints. No neural training. Exact gamma1 Rician EM continuation starts from learned estimate; no spectral reset.',
        candidate_design='Expanded short-tail menu motivated only by the previous observable development screen. Prior exposed test regressions motivate the question; they are not reused as a fresh test.',
        data='Fresh observable-only development anchors;39 columns with fixed30/9 masks. Test and timing use first30. The10dB anchor retains G/S/B/standardized noise across SNR. Bootstrap shares local indices across reference/SNR within each regime.',
        masks='Torchseed2026101935+100*regime_index+reference_index. Target384 and elsewhere96 banks share the first96 anchor/mask rows. Every method in a bin receives exactly the same saved context/heldout bank.',
        repair_guard=dict(samples=GUARD_BOOT_COUNT,seed=GUARD_BOOT_SEED,
            rule='Reuse one4000x384 multinomial count matrix per regime across all six calibrated bins. Paired heldout losses average the three fixed fits per anchor. Candidate admissible only if bootstrap-mean upper95<=0 versus BOTH original and old spectral100 bypass in EACH of three regimes. Original is not automatically admissible.',
            fallback='If the feasible set is empty, use spectral100 with selection_status=fallback_old, dual_admissible=false, and explicit failed constraints. This is a declared priority fallback, not a successful dual guard.',
            caveat='Empirical development evidence, not a population noninferiority guarantee after adaptive selection.'),
        calibration=dict(local_index=100,snrs=SNRS,rsrs=RSRS,regimes=REGIMES,repeats=3,
            rule='Freeze repair table first. With other CPU jobs idle, time actual selected repair plus all16 compact candidates. Pooled three-fit repair p90 defines the global eligibility cap; include lookup, initialization, setup and conversions.'),
        compact=dict(grid=base.candidates(),selection='Same saved384-case context/heldout banks in six target bins and96-case banks elsewhere; select eligible minimum equal-regime heldout NLL per reference/SNR. This is a limited compact family, not a global optimum.'),
        methods=NAMES,groups=GROUPS,contrasts=CONTRASTS,
        primary='Strong references12/20 at SNR32.5,35,37.5 across three regimes:18 cells. Repair versus old bypass and versus previous repair, with NMSE and BER as four prespecified co-outcomes.',
        success_review='Report all primary cells and path subgroups, every adverse cell and tail, all-grid outcomes, and exact weak-reference preservation. Do not require or promise zero negative sample point estimates.',
        bootstrap=dict(samples=20000,seed=BOOT_SEED,unit='Multinomial128 local-index weights per regime shared across references/SNR/methods/fits',
            scope='Conditional fixed fits/policies. Pointwise95 plus Bonferroni4 individual98.75% primary intervals. Other comparisons descriptive.'),
        sensitivity='Frozen45-condition supplied-variance diagnostic, offsets-3/0/+3dB, same fresh test anchors; no recalibration and no BER diagnostic.',
        historical_replay='After policy freeze only:22 exposed old cells, the union of18 strong-transition cells and the10 previously flagged cells. Diagnostic repair check, never independent confirmatory evidence or a basis for policy revision.',
        final_timing=dict(local_index=200,snrs=SNRS,rsrs=RSRS,regimes=REGIMES,repeats=3,
            scope='Fresh isolated full153-grid timing of all17 methods. Report pooled and conditional latency and every cap exceedance; no reselection.'),
        freeze='Scientific sources/checkpoints before development; correction before calibration; calibration and compact/BER bindings before every fresh test anchor or historical replay.',
        source_hashes={str(p.relative_to(REPO)):sha(p) for p in sources},checkpoint_hashes={str(p.relative_to(REPO)):sha(p) for p in model_paths().values()},
        previous_protocol_sha256=previous_digest,previous_repair_policy_sha256=sha(PRIOR_REPAIR_PATH),previous_compact_policy_sha256=sha(PRIOR_COMPACT_PATH),
        thread_environment={k:os.environ[k] for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS')})
    path=OUT/'protocol.json'
    if path.exists():
        old=json.loads(path.read_text());old.pop('frozen_unix',None);assert old==json.loads(json.dumps(spec)),'protocol changed'
    else:write_json(path,dict(**spec,frozen_unix=time.time()))
    return sha(path)


def verify_protocol():
    path=OUT/'protocol.json';record=json.loads(path.read_text())
    for key in ('source_hashes','checkpoint_hashes'):
        for name,digest in record[key].items():assert sha(REPO/name)==digest,name
    assert previous.verify_protocol()==record['previous_protocol_sha256']
    assert sha(PRIOR_REPAIR_PATH)==record['previous_repair_policy_sha256']
    assert sha(PRIOR_COMPACT_PATH)==record['previous_compact_policy_sha256']
    return sha(path)


def anchors(split,regime,L,rsr,digest):
    path=OUT/split/'anchors'/f'{regime}_rsr{rsr:g}.npz';freeze=check_selection()[1] if split=='test' else 'development'
    if path.exists():
        a=load(path);assert str(a['protocol_sha256'])==digest and str(a['selection_sha256'])==freeze;return a
    seed=TEST_SEED if split=='test' else DEV_SEED
    count=TEST_COUNT if split=='test' else DEV_COUNT if float(rsr) in STRONG_RSRS else DEV_BASE_COUNT
    a=make_anchor(seed,range(count),L,rsr,truth=split=='test')
    a.update(protocol_sha256=np.array(digest),selection_sha256=np.array(freeze),regime=np.array(regime));save(path,**a);return a


def development(rsr,snr,digest):
    count=development_count(rsr,snr);path=OUT/'development/banks'/f'rsr{rsr:g}_snr{snr:g}.npz'
    if path.exists():
        a=load(path);assert str(a['protocol_sha256'])==digest and int(a['cases_per_regime'])==count
        return tuple(Observations(*(torch.from_numpy(a[f'{prefix}_{name}']) for name in ('z','s','b','sigma2'))) for prefix in ('context','heldout'))
    split=[]
    for regime,L in REGIMES:
        obs=observed(anchors('development',regime,L,rsr,digest),snr,P=39).batch(slice(0,count))
        split.append(fit_holdout(obs,regime,rsr))
    pairs=tuple(Observations(*(torch.cat([pair[k].args()[j] for pair in split]) for j in range(4))) for k in (0,1))
    arrays={f'{prefix}_{name}':value.numpy() for prefix,obs in zip(('context','heldout'),pairs) for name,value in zip(('z','s','b','sigma2'),obs.args())}
    save(path,**arrays,cases_per_regime=np.array(count),snr=np.array(snr),rsr=np.array(rsr),protocol_sha256=np.array(digest))
    return pairs


def fixed_choice(rsr,snr):
    if rsr==5.:return dict(choice=PRIOR_REPAIR['repair_choices'][f'{rsr:g}'][f'{snr:g}']['choice'],selection_status='fixed_prior_repair',dual_admissible=None,failed_constraints=[])
    if snr<=30.:return dict(choice='original',selection_status='fixed_original',dual_admissible=None,failed_constraints=[])
    if snr>=40.:return dict(choice='spectral100',selection_status='fixed_spectral100',dual_admissible=None,failed_constraints=[])
    raise ValueError('calibrated bin has no fixed choice')


@torch.no_grad()
def develop_repair():
    digest=verify_protocol();est=engine()
    for rsr in STRONG_RSRS:
        for snr in TARGET_SNRS:
            path=OUT/'development/repair_losses'/f'rsr{rsr:g}_snr{snr:g}.npz'
            if path.exists():assert str(load(path)['protocol_sha256'])==digest;continue
            c,h=development(rsr,snr,digest);spectral=est.em100(c);fits=[];holds=[]
            for seed in SEEDS:
                values=est.candidate_predictions(c,f'learned_{seed}',spectral=spectral)
                fit=torch.stack([nll(c.z,values[m]@c.s+c.b,c.sigma2).mean((-2,-1)) for m in MENU])
                hold=torch.stack([nll(h.z,values[m]@h.s+h.b,h.sigma2).mean((-2,-1)) for m in MENU])
                assert torch.isfinite(fit).all() and torch.isfinite(hold).all()
                assert torch.all(fit[1:-1]<=fit[0]+1e-10*(1+fit[0].abs()))
                fits.append(fit.numpy().reshape(len(MENU),3,DEV_COUNT));holds.append(hold.numpy().reshape(len(MENU),3,DEV_COUNT))
            save(path,names=np.array(MENU),seeds=np.array(SEEDS),regimes=np.array([r for r,L in REGIMES]),fit_nll=np.stack(fits),heldout_nll=np.stack(holds),
                cases_per_regime=np.array(DEV_COUNT),snr=np.array(snr),rsr=np.array(rsr),protocol_sha256=np.array(digest))
            print('Repair development',path.stem,flush=True)
    rng=np.random.default_rng(GUARD_BOOT_SEED)
    weights={r:rng.multinomial(DEV_COUNT,np.full(DEV_COUNT,1/DEV_COUNT),size=GUARD_BOOT_COUNT) for r,L in REGIMES}
    weightpath=OUT/'development/repair_bootstrap_weights.npz'
    if weightpath.exists():
        saved=load(weightpath)
        for r,w in weights.items():np.testing.assert_array_equal(saved[r],w)
    else:save(weightpath,**weights)
    choices={};tables={}
    for rsr in RSRS:
        choices[f'{rsr:g}']={};tables[f'{rsr:g}']={}
        for snr in SNRS:
            if rsr not in STRONG_RSRS or snr not in TARGET_SNRS:
                choices[f'{rsr:g}'][f'{snr:g}']=fixed_choice(rsr,snr);tables[f'{rsr:g}'][f'{snr:g}']=[];continue
            a=load(OUT/'development/repair_losses'/f'rsr{rsr:g}_snr{snr:g}.npz');loss=a['heldout_nll'];scores=[]
            for mi,name in enumerate(MENU):
                references={};failed=[]
                for reference,refindex in (('original',MENU.index('original')),('old_bypass',MENU.index('spectral100'))):
                    delta=(loss[:,mi]-loss[:,refindex]).mean(0);regimes={}
                    for pi,(regime,L) in enumerate(REGIMES):
                        upper=float(np.quantile((weights[regime]@delta[pi])/DEV_COUNT,.95))
                        regimes[regime]=dict(mean_delta=float(delta[pi].mean()),upper95=upper)
                        if upper>0.:failed.append(f'{reference}:{regime}')
                    references[reference]=dict(mean_delta=float(delta.mean()),by_regime=regimes)
                scores.append(dict(choice=name,mean_nll=float(loss[:,mi].mean()),references=references,dual_admissible=not failed,
                    failed_constraints=failed,preference_order=PREFERENCE_ORDER.index(name)))
            admissible=[x for x in scores if x['dual_admissible']]
            if admissible:
                optimum=min(x['mean_nll'] for x in admissible)
                selected=dict(min((x for x in admissible if x['mean_nll']<=optimum+PREFERENCE_BAND),key=lambda x:x['preference_order']),
                    selection_status='dual_admissible',admissible_optimum_mean_nll=optimum,preference_band=PREFERENCE_BAND)
            else:
                selected=dict(scores[MENU.index('spectral100')],selection_status='fallback_old',admissible_optimum_mean_nll=None,preference_band=PREFERENCE_BAND)
                assert not selected['dual_admissible']
            choices[f'{rsr:g}'][f'{snr:g}']=selected;tables[f'{rsr:g}'][f'{snr:g}']=scores
    record=dict(protocol_sha256=digest,repair_choices=choices,candidate_scores=tables,bootstrap_weights_sha256=sha(weightpath),
        development_counts={f'{r:g}':{f'{s:g}':development_count(r,s) for s in SNRS} for r in RSRS},
        development_anchor_hashes={f'{regime}_rsr{rsr:g}.npz':sha(OUT/'development/anchors'/f'{regime}_rsr{rsr:g}.npz') for rsr in STRONG_RSRS for regime,L in REGIMES},
        development_bank_hashes={f'rsr{rsr:g}_snr{snr:g}.npz':sha(OUT/'development/banks'/f'rsr{rsr:g}_snr{snr:g}.npz') for rsr in STRONG_RSRS for snr in TARGET_SNRS},
        loss_table_hashes={f'rsr{rsr:g}_snr{snr:g}.npz':sha(OUT/'development/repair_losses'/f'rsr{rsr:g}_snr{snr:g}.npz') for rsr in STRONG_RSRS for snr in TARGET_SNRS})
    path=OUT/'repair_selection.json'
    if path.exists():
        old=json.loads(path.read_text());old.pop('frozen_unix',None);assert old==record
    else:
        assert not (OUT/'calibration.json').exists() and not list((OUT/'test').glob('**/*.npz'));write_json(path,dict(**record,frozen_unix=time.time()))
    print('Repair choices frozen',{r:{s:(c['choice'],c['selection_status']) for s,c in v.items()} for r,v in choices.items()},flush=True)


def check_repair():
    digest=verify_protocol();p=json.loads((OUT/'repair_selection.json').read_text());assert p['protocol_sha256']==digest
    assert p['bootstrap_weights_sha256']==sha(OUT/'development/repair_bootstrap_weights.npz')
    for key,folder in (('development_anchor_hashes','anchors'),('development_bank_hashes','banks'),('loss_table_hashes','repair_losses')):
        for name,h in p[key].items():assert sha(OUT/'development'/folder/name)==h,name
    return p


def stats(values):return dict(p90_ms=float(np.quantile(values,.9)),median_ms=float(np.median(values)),mean_ms=float(np.mean(values)),calls=len(values))


def timing_rows(index):
    rows=[]
    for rsr in RSRS:
        for regime,L in REGIMES:
            a=make_anchor(TIMING_SEED,[index],L,rsr,truth=False)
            for snr in SNRS:rows.append((dict(snr=snr,rsr=rsr,regime=regime,index=index),observed(a,snr)))
    return rows


@torch.no_grad()
def calibrate():
    selection=check_repair();assert not (OUT/'calibration.json').exists(),'calibration already frozen'
    est=engine();rows=timing_rows(100);configs=base.candidates()
    samples={f'repair_{s}':[] for s in SEEDS};samples.update({f'compact_{key}':[] for key in configs})
    tables={key:dict(compact_choices={f'{r:g}':{f'{s:g}':dict(config=cfg) for s in SNRS} for r in RSRS}) for key,cfg in configs.items()}
    def call(name,obs):
        return estimate_method(obs,name,selection,est) if name.startswith('repair_') else estimate_method(obs,'compact_sparse',tables[name[8:]],est)
    for name in samples:call(name,rows[0][1])
    rng=np.random.default_rng(20261019340);names=list(samples)
    for wi,(meta,obs) in enumerate(rows):
        choice=selected_repair_choice(obs,selection)
        for repeat in range(3):
            for ni in rng.permutation(len(names)):
                name=names[ni];start=time.perf_counter_ns();g=call(name,obs);ms=(time.perf_counter_ns()-start)/1e6
                assert torch.isfinite(g).all();samples[name].append(dict(ms=ms,repeat=repeat,repair_choice=choice,**meta))
        print('Calibration world',wi+1,'/',len(rows),flush=True)
    summary={n:stats([x['ms'] for x in v]) for n,v in samples.items()};cap=stats([x['ms'] for s in SEEDS for x in samples[f'repair_{s}']])
    eligible=[key for key in configs if summary[f'compact_{key}']['p90_ms']<=cap['p90_ms']];assert eligible
    write_json(OUT/'calibration.json',dict(protocol_sha256=selection['protocol_sha256'],repair_selection_sha256=sha(OUT/'repair_selection.json'),cap_ms=cap['p90_ms'],
        repair_timing=cap,candidates={k:dict(config=v,**summary[f'compact_{k}']) for k,v in configs.items()},eligible=eligible,methods=summary,raw_samples=samples,completed_unix=time.time()))
    print('Calibration complete',cap,'eligible',len(eligible),flush=True)


def check_calibration():
    repair=check_repair();p=json.loads((OUT/'calibration.json').read_text())
    assert p['repair_selection_sha256']==sha(OUT/'repair_selection.json') and p['protocol_sha256']==repair['protocol_sha256'];return p


@torch.no_grad()
def develop_compact():
    calibration=check_calibration();repair=check_repair();configs=base.candidates();digest=repair['protocol_sha256'];tables={};choices={}
    for rsr in RSRS:
        tables[f'{rsr:g}']={};choices[f'{rsr:g}']={}
        for snr in SNRS:
            count=development_count(rsr,snr);path=OUT/'development/compact_losses'/f'rsr{rsr:g}_snr{snr:g}.npz'
            if path.exists():
                a=load(path);assert str(a['calibration_sha256'])==sha(OUT/'calibration.json') and int(a['cases_per_regime'])==count
            else:
                c,h=development(rsr,snr,digest);losses=[]
                for key in calibration['eligible']:
                    g=sparse_fast.estimate(c,configs[key]);loss=nll(h.z,g@h.s+h.b,h.sigma2).mean((-2,-1)).numpy().reshape(3,count);losses.append(loss)
                save(path,names=np.array(calibration['eligible']),losses=np.stack(losses),cases_per_regime=np.array(count),snr=np.array(snr),rsr=np.array(rsr),calibration_sha256=np.array(sha(OUT/'calibration.json')))
                a=load(path)
            scores=[dict(candidate=str(key),config=configs[str(key)],mean_nll=float(a['losses'][i].mean()),by_regime=a['losses'][i].mean(-1).tolist(),
                p90_ms=calibration['candidates'][str(key)]['p90_ms']) for i,key in enumerate(a['names'])]
            choices[f'{rsr:g}'][f'{snr:g}']=min(scores,key=lambda x:(x['mean_nll'],x['p90_ms'],x['candidate']));tables[f'{rsr:g}'][f'{snr:g}']=scores
            print('Compact development',path.stem,flush=True)
    berpath=OUT/'ber/protocol.json';ber=json.loads(berpath.read_text());assert ber['source_methods']==list(NAMES) and ber['snrs']==list(SNRS)
    for name,h in ber['source_hashes'].items():assert sha(REPO/name)==h,name
    record=dict(protocol_sha256=digest,repair_selection_sha256=sha(OUT/'repair_selection.json'),calibration_sha256=sha(OUT/'calibration.json'),
        repair_choices=repair['repair_choices'],compact_choices=choices,compact_tables=tables,development_counts=repair['development_counts'],
        development_anchor_hashes={p.name:sha(p) for p in sorted((OUT/'development/anchors').glob('*.npz'))},
        development_bank_hashes={p.name:sha(p) for p in sorted((OUT/'development/banks').glob('*.npz'))},
        compact_loss_hashes={p.name:sha(p) for p in sorted((OUT/'development/compact_losses').glob('*.npz'))},ber_protocol_sha256=sha(berpath),ber_source_hashes=ber['source_hashes'])
    path=OUT/'policy.json'
    if path.exists():
        old=json.loads(path.read_text());old.pop('frozen_unix',None);assert old==record
    else:
        assert not list((OUT/'test').glob('**/*.npz'));write_json(path,dict(**record,frozen_unix=time.time()))
    print('All policies and comparators frozen',flush=True)


def check_selection():
    calibration=check_calibration();repair=check_repair();path=OUT/'policy.json';p=json.loads(path.read_text())
    assert p['protocol_sha256']==repair['protocol_sha256'] and p['repair_selection_sha256']==sha(OUT/'repair_selection.json')
    assert p['calibration_sha256']==sha(OUT/'calibration.json') and p['repair_choices']==repair['repair_choices']
    assert p['ber_protocol_sha256']==sha(OUT/'ber/protocol.json')
    for name,h in p['ber_source_hashes'].items():assert sha(REPO/name)==h,name
    for key,folder in (('development_anchor_hashes','anchors'),('development_bank_hashes','banks'),('compact_loss_hashes','compact_losses')):
        for name,h in p[key].items():assert sha(OUT/'development'/folder/name)==h,name
    return p,sha(path)


@torch.no_grad()
def predictions(est,obs,selection):
    original=torch.stack([est.original(obs,f'learned_{s}') for s in SEEDS]);em=est.em100(obs)
    oldmask=est.masks(obs,OLD_POLICY)['bypass'];old=torch.where(oldmask[None,:,None,None],em[None],original)
    priorchoice=selected_repair_choice(obs,PRIOR_REPAIR);choice=selected_repair_choice(obs,selection)
    cache={'original':original,'spectral100':em[None].expand(len(SEEDS),-1,-1,-1)}
    for which in (priorchoice,choice):
        if which not in cache:cache[which]=torch.stack([continuation.estimate(obs,g,int(which[4:])) for g in original])
    outputs=[*original,*old,*cache[priorchoice],*cache[choice],em,frozen_optshrink(obs,OPT_LOOKUP)]
    sparse_cache={json.dumps(base.EM_CONFIG,sort_keys=True):em}
    for name in ('prior_compact','compact_sparse','prior_sparse_full'):
        cfg=base.FULL if name=='prior_sparse_full' else compact_config(obs,PRIOR_COMPACT if name=='prior_compact' else selection)
        key=json.dumps(cfg,sort_keys=True)
        if key not in sparse_cache:sparse_cache[key]=sparse_fast.estimate(obs,cfg)
        outputs.append(sparse_cache[key])
    return torch.stack(outputs),oldmask,priorchoice,choice


@torch.no_grad()
def test():
    selection,freeze=check_selection();digest=selection['protocol_sha256'];est=engine()
    for rsr in RSRS:
        for regime,L in REGIMES:
            a=anchors('test',regime,L,rsr,digest)
            for snr in SNRS:
                path=OUT/'test/cells'/f'{cellname(regime,rsr,snr)}.npz'
                if path.exists():assert str(load(path)['selection_sha256'])==freeze;continue
                obs=observed(a,snr);pred,oldmask,priorchoice,choice=predictions(est,obs,selection)
                if snr<=20:
                    assert all(torch.equal(pred[start:start+3],pred[:3]) for start in (3,6,9)) and not oldmask.any()
                if rsr==5:assert torch.equal(pred[9:12],pred[6:9]) and choice==priorchoice
                if rsr in STRONG_RSRS and snr<=30:assert torch.equal(pred[9:12],pred[:3]) and choice=='original'
                if rsr in STRONG_RSRS and snr>=40:assert torch.equal(pred[9:12],pred[12:13].expand(3,-1,-1,-1)) and choice=='spectral100'
                maximum=0.
                for i,name in enumerate(NAMES):
                    if not any(name.startswith(prefix) for prefix in ('old_bypass_','prior_repair_','repair_')):continue
                    actual=estimate_method(obs.batch(slice(0,1)),name,selection,est);error=float((actual-pred[i,0:1]).abs().max());assert error<1e-10;maximum=max(maximum,error)
                arr=pred.numpy();assert np.isfinite(arr).all();errors=(np.abs(arr-a['g'][None])**2).sum((-2,-1));energy=(np.abs(a['g'])**2).sum((-2,-1))
                save(path,**{k:x.numpy() for k,x in zip(('z','s','b','sigma2'),obs.args())},truth_g=a['g'],paths=a['paths'],indices=a['indices'],local_indices=a['local_indices'],
                    method_names=np.array(NAMES),estimates=arr,errors=errors,truth_energy=energy,nmse=errors/energy[None],old_bypass=oldmask.numpy(),
                    prior_repair_choice=np.array(priorchoice),repair_choice=np.array(choice),snr=np.array(snr),rsr=np.array(rsr),regime=np.array(regime),seed=np.array(TEST_SEED),
                    selection_sha256=np.array(freeze),protocol_sha256=np.array(digest),actual_dispatch_max_abs_error=np.array(maximum),
                    ordinary_range_exact_identity=np.array(snr<=20),weak_reference_exact_identity=np.array(rsr==5))
                print('Test complete',path.stem,flush=True);write_json(OUT/'status.json',dict(phase='testing',last_cell=path.stem,completed_cells=len(list((OUT/'test/cells').glob('*.npz')))))
    write_json(OUT/'status.json',dict(phase='test_complete',cells=len(SNRS)*len(RSRS)*len(REGIMES),selection_sha256=freeze))


@torch.no_grad()
def timing():
    selection,freeze=check_selection();assert not (OUT/'timing.json').exists(),'timing already recorded';est=engine();rows=timing_rows(200)
    for name in NAMES:estimate_method(rows[0][1],name,selection,est)
    samples={n:[] for n in NAMES};rng=np.random.default_rng(20261019341)
    for wi,(meta,obs) in enumerate(rows):
        choice=selected_repair_choice(obs,selection);priorchoice=selected_repair_choice(obs,PRIOR_REPAIR);oldmask=bool(est.masks(obs,OLD_POLICY)['bypass'][0])
        for repeat in range(3):
            for ni in rng.permutation(len(NAMES)):
                name=NAMES[ni];start=time.perf_counter_ns();g=estimate_method(obs,name,selection,est);ms=(time.perf_counter_ns()-start)/1e6;assert torch.isfinite(g).all()
                branch=choice if name.startswith('repair_') else priorchoice if name.startswith('prior_repair_') else ('spectral100' if oldmask else 'original') if name.startswith('old_bypass_') else 'original' if name.startswith('original_') else name
                samples[name].append(dict(ms=ms,repeat=repeat,branch=branch,repair_choice=choice,prior_repair_choice=priorchoice,**meta))
        print('Timing world',wi+1,'/',len(rows),flush=True)
    values=dict(samples)
    for name,parts in GROUPS.items():values[name]=[x for part in parts for x in samples[part]]
    methods={n:dict(**stats([x['ms'] for x in v]),by_branch={b:stats([x['ms'] for x in v if x['branch']==b]) for b in sorted(set(x['branch'] for x in v))},
        by_condition={f'rsr{r:g}_snr{s:g}':stats([x['ms'] for x in v if x['rsr']==r and x['snr']==s]) for r in RSRS for s in SNRS}) for n,v in values.items()}
    cap=check_calibration()['cap_ms'];verification=dict(calibrated_cap_ms=cap,repair_p90_ms=methods['repair_mean']['p90_ms'],compact_p90_ms=methods['compact_sparse']['p90_ms'],
        repair_exceeds_calibrated_cap=methods['repair_mean']['p90_ms']>cap,compact_exceeds_calibrated_cap=methods['compact_sparse']['p90_ms']>cap,
        compact_exceeds_fresh_repair=methods['compact_sparse']['p90_ms']>methods['repair_mean']['p90_ms'])
    write_json(OUT/'timing.json',dict(selection_sha256=freeze,completed_unix=time.time(),worlds=len(rows),repeats=3,methods=methods,raw_samples=samples,cap_verification=verification))
    print('Timing complete',verification,flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['prepare','repair','calibrate','compact','test','timing']);args=parser.parse_args()
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    {'prepare':freeze_protocol,'repair':develop_repair,'calibrate':calibrate,'compact':develop_compact,'test':test,'timing':timing}[args.phase]()


if __name__=='__main__':main()
