"""Fresh observable-only calibration of retrained SAHURE high-SNR bypass.

Every previous experiment remains immutable. Development, runtime calibration,
test generation and final timing are explicit separate phases.
"""
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
from sahure.train import load_model
from sahure_expansion.data import world
from tcom_followups_20261017.common import REPO,sha,write_json,effective_trial_id,frozen_inputs,frozen_optshrink,OPT_SELECTION
from tcom_runtime_bypass_20261017.bypass import BypassEstimator
from tcom_runtime_bypass_20261017 import sparse_fast
from tcom_runtime_bypass_20261017.runtime_study import FULL

OUT=REPO/'results/tcom_retrained_bypass_20261018'
DEV_SEED=2026101821;TEST_SEED=2026101822;DATA_SEED=2026101823;TIMING_SEED=2026101824;SMOKE_SEED=2026101820
DEV_COUNT=48;TEST_COUNT=64
SNRS=(0.,5.,10.,15.,20.,22.5,25.,27.5,30.,32.5,35.,37.5,40.,42.5,45.,50.,60.)
RSRS=(5.,12.,20.);REGIMES=(('L2',2),('L4',4),('random3to7',None));SEEDS=(9101,9102,9103)
CUTOFFS=tuple(s for s in SNRS if s>=20)
FACTORS=(.03,.1,.3,1.,3.)
STRUCTURES=((128,8,12),(256,8,12),(256,2,40))
EM_CONFIG=dict(dictionary_size=128,inner_steps=1,iterations=100,factor=0.)
NAMES=tuple([f'original_{s}' for s in SEEDS]+[f'off_{s}' for s in SEEDS]+[f'fast_bypass_{s}' for s in SEEDS]+[f'em100_bypass_{s}' for s in SEEDS]+
    ['em100','optshrink','prior_sparse_matched','prior_sparse_50ms','prior_sparse_full','compact_sparse'])
GROUPS=dict(original_mean=[f'original_{s}' for s in SEEDS],off_mean=[f'off_{s}' for s in SEEDS],
    fast_bypass_mean=[f'fast_bypass_{s}' for s in SEEDS],em100_bypass_mean=[f'em100_bypass_{s}' for s in SEEDS])
CONTRASTS=dict(bypass_vs_original=('original_mean','em100_bypass_mean'),bypass_vs_em100=('em100','em100_bypass_mean'),
    bypass_vs_compact=('compact_sparse','em100_bypass_mean'))
MODEL_ROOT=REPO/'results/tcom_pilot_em_20261018'
OLD_SELECTION=REPO/'results/tcom_runtime_bypass_20261017/runtime/validation_selection.json'
COMPACT_ORIGIN=MODEL_ROOT/'evaluation/selection.json'
OLD_LOOKUP=json.loads(OLD_SELECTION.read_text());OPT_LOOKUP=json.loads(OPT_SELECTION.read_text())


def save(path,**arrays):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix('.tmp.npz')
    np.savez_compressed(temp,**arrays);temp.replace(path)


def load(path):
    with np.load(path,allow_pickle=False) as arrays:return {k:arrays[k] for k in arrays.files}


def cellname(regime,rsr,snr):return f'{regime}_rsr{rsr:g}_snr{snr:g}'
def model_paths():return {f'learned_{s}':MODEL_ROOT/f'checkpoints/context30/seed_{s}/em_adaptive_best.pt' for s in SEEDS}
def engine():return BypassEstimator({name:load_model(path,dtype=torch.float64)[0] for name,path in model_paths().items()})


def candidates():
    result={f'd{d}_i{inner}_o{outer}_f{factor:g}':dict(dictionary_size=d,inner_steps=inner,iterations=outer,factor=factor)
        for d,inner,outer in STRUCTURES for factor in FACTORS}
    result['em100']=EM_CONFIG.copy();return result


def freeze_protocol():
    sources=[Path(__file__),REPO/'tcom_retrained_bypass_20261018/sensitivity.py']
    sources += [REPO/name for name in ('tcom_runtime_bypass_20261017/bypass.py','tcom_runtime_bypass_20261017/sparse_fast.py',
        'tcom_runtime_bypass_20261017/runtime_study.py','tcom_followups_20261017/common.py','sahure/data.py','sahure/train.py')]
    prior=frozen_inputs()
    for key in ('development_seed','test_seed','data_seed'):prior.pop(key,None)
    original_compact=json.loads(COMPACT_ORIGIN.read_text())
    actual={(int(r['context30']['config']['dictionary_size']),int(r['context30']['config']['inner_steps']),int(r['context30']['config']['iterations'])) for r in original_compact['sparse_choices'].values()}
    assert actual==set(STRUCTURES),'compact structures must be the frozen prior context30 union'
    spec=dict(version=1,name='Retrained30-pilot SAHURE with validation-calibrated high-SNR bypass',
        development_seed=DEV_SEED,test_seed=TEST_SEED,data_seed=DATA_SEED,timing_seed=TIMING_SEED,smoke_seed=SMOKE_SEED,
        N=32,K=3,generated_P=39,context_P=30,heldout_P=9,deployment_P=30,snrs=SNRS,rsrs=RSRS,regimes=REGIMES,
        development_anchors_per_reference_regime=DEV_COUNT,test_anchors_per_reference_regime=TEST_COUNT,
        model_seeds=SEEDS,model='Three frozen context30 checkpoints from the completed paired retraining. No neural training or EM continuation added.',
        cutoffs=CUTOFFS,dispatch='Known supplied noise and nominal K normalization define10log10(K/v); known reference defines nearest5/12/20dB bin. Strict proxy>cutoff+1e-10 and proxy>20+1e-10. One cutoff per reference shared across fits/path regimes; exactly one branch runs.',
        fallback='Primary optimized exact EM100 with spectral initialization. Secondary same frozen weights and ten learned steps with only Hankel filtering/controller skipped. No warm-start continuation.',
        selection='Separate fallback policies minimize equal-cell heldout RicianNLL, averaged over all declared SNR/path cells and three individual fit losses. Use centered excess over original to avoid cancellation; ties prefer larger cutoff. Cutoff60 means no bypass on calibrated0–60 domain; behavior above60 is unvalidated.',
        development='Generate fresh39-column coupled observable banks. Initialize, compute features and fit using only30 context pilots. Remaining9 are used only for scoring. Development storage contains noG/noisefreefield/unitnoise/pathtruth. Truth never enters policy or compact sparse selection.',
        masks='Per-reference/regime48-case random30/9 split with torch seed2026101825+100*regime_index+reference_index; exactly the same split acrossSNR, branches and candidates.',
        coupling='At each reference/regime, generate10dB39-pilot anchor and rescale the same standardized noise across SNR while preserving G/S/B. Test uses first30columns. Reference draws differ but randompathcounts share localcase index; bootstrap shares indices across all SNR/reference withinregime.',
        compact_grid=dict(structures=STRUCTURES,factors=FACTORS,zero_penalty_EM100=True,origin=str(COMPACT_ORIGIN.relative_to(REPO)),
            scope='Compact preregistered family, not globally optimized sparse baseline. Independently select on all SNR/reference cells using balancedpath30fit/9heldoutNLL under frozen latencycap.'),
        calibration=dict(snrs=[0.,20.,40.,60.],local_index=100,regimes=REGIMES,rsrs=RSRS,repeats=3,
            cap='Before development, isolated actual wrapper original-branch pooled3fitp90 and EM100-branch pooled3fitp90; cap=max(branchp90s). Each of16candidate timings includes observed lookup and full estimation; eligibility bycandidatep90<=cap. Models/lookups preloaded; initialization/setup/conversions included.',
            original_branch='Use cutoff60 so every0–60 observation stays original. EM branch uses cutoff20 and only40/60dB observations.',
            interpretation='Calibrated latency quantile, not hard deadline. Final policy timing and any cap overshoot reported without reselection.'),
        baselines='Direct optimizedEM100; frozen offlineOptShrink; old sparse matched/50ms/full; fresh compact sparse. Old sparse lookup clamps nominalSNR to0–20 and was calibrated onlyRSR12. OptShrink clamps to its prior lookupgrid. These are frozen extrapolation controls, not highSNRoptimizedcontrols.',
        freeze='Freeze sources/checkpoints/priorselectionhashes before timing/development. Freeze calibration, full loss tables, policies and compact choices before generating any test anchor.',
        methods=NAMES,groups=GROUPS,primary_contrasts=CONTRASTS,
        primary='SNR>20; three EM100-bypass contrasts times two co-outcomes(NMSE andBER). Same-step bypass diagnostic only.',
        summaries='Report ordinary<=20, transition20<SNR<=45, stress50/60, high>20 and all. Equal-cell dB ratio-of-sumsNMSE, pooledlinearNMSE, allcells/worstcells/p95/p99/severeNMSE>1 and fit-specificresults. Fitgroups average individualmetrics, never predictions.',
        bootstrap=dict(samples=20000,seed=2026101829,unit='Multinomial64casecounts perregime, shared acrossreference/SNR/method/fit',
            inference='Conditional on fittedweights and selectedpolicies. Pointwise95 and Bonferroni6-outcome individual99.1666667% intervals. Per-cell statements descriptive.'),
        ber='All153cells,64testchannels,all64jointQPSK vectors once/channel with independent data-noise law shared acrossmethods/SNR; dedicated BER protocol freezes exact construction before evaluation.',
        sensitivity=dict(regime='random3to7',rsrs=RSRS,actual_snrs='20,60,EMcutoff-2.5,EMcutoff,EMcutoff+2.5, clipped to0–60 and deduplicated',
            variance_offsets_db=[-3.,0.,3.],methods='original, same-stepbypass, EM100bypass, directEM100, compact sparse',
            scope='Fixedpolicy descriptive diagnostic on samefresh testanchors; no retuning and no BER. Actualmeasurements unchanged; supplied variance consistently changes inference and dispatch.'),
        final_timing=dict(snrs=SNRS,local_index=200,repeats=3,rsrs=RSRS,regimes=REGIMES,
            scope='Separate isolated freshbatch1 window after all prediction/BER/audit jobs idle. Everyarm includes lookup and fullinference. Report pooled/bycondition/bybranchmedian andp90 andcapovershoots.'),
        source_hashes={str(p.relative_to(REPO)):sha(p) for p in sources},checkpoint_hashes={str(p.relative_to(REPO)):sha(p) for p in model_paths().values()},
        prior_hashes={str(p.relative_to(REPO)):sha(p) for p in (OLD_SELECTION,COMPACT_ORIGIN,OPT_SELECTION,MODEL_ROOT/'training_summary.json')},frozen_prior_inputs=prior,
        thread_environment={k:os.environ[k] for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS')})
    path=OUT/'protocol.json'
    if path.exists():
        old=json.loads(path.read_text());old.pop('frozen_unix',None);assert old==json.loads(json.dumps(spec)),'frozen protocol changed'
    else:write_json(path,dict(**spec,frozen_unix=time.time()))
    return sha(path)


def verify_protocol():
    path=OUT/'protocol.json';record=json.loads(path.read_text())
    for key in ('source_hashes','checkpoint_hashes','prior_hashes'):
        for name,digest in record[key].items():assert sha(REPO/name)==digest,name
    for name,digest in record['frozen_prior_inputs']['hashes'].items():assert sha(REPO/name)==digest,name
    return sha(path)


def make_anchor(seed,ids,L,rsr,truth=True):
    """Simulator boundary: development exports observables only, no hidden field."""
    ids=list(ids);rows=[world(effective_trial_id(i,L),seed,N=32,K=3,P=39,snr=10.,rsr=rsr,L=L) for i in ids]
    g=np.stack([w.G for w,p in rows]);s=np.stack([w.S for w,p in rows]);b=np.stack([w.B for w,p in rows])
    v=np.array([w.sigma2 for w,p in rows]);unit=np.stack([w.W/np.sqrt(w.sigma2) for w,p in rows])
    mu=g@s+b
    np.testing.assert_allclose(np.abs(mu+unit*np.sqrt(v)[:,None,None]),np.stack([w.Z for w,p in rows]),rtol=0,atol=1e-13)
    a=dict(s=s,b=b,anchor_variance=v,indices=np.array([effective_trial_id(i,L) for i in ids]),local_indices=np.array(ids),seed=np.array(seed),rsr=np.array(rsr))
    if truth:a.update(g=g,unit_noise=unit,paths=np.array([p for w,p in rows]))
    else:
        a.update(snrs=np.array(SNRS),z_by_snr=np.stack([np.abs(mu+unit*np.sqrt(v*10**((10-snr)/10))[:,None,None]) for snr in SNRS]))
        assert not {'g','unit_noise','paths','mu'}.intersection(a)
    return a


def observed(a,snr,variance_offset_db=0.,P=30):
    v=a['anchor_variance']*10**((10-float(snr))/10)
    if 'g' in a:z=np.abs(a['g']@a['s']+a['b']+a['unit_noise']*np.sqrt(v)[:,None,None])
    else:
        ids=np.flatnonzero(np.isclose(a['snrs'],snr,rtol=0,atol=1e-12));assert len(ids)==1
        z=a['z_by_snr'][int(ids[0])]
    return Observations(torch.tensor(z[...,:P],dtype=torch.float64),torch.tensor(a['s'][...,:P],dtype=torch.complex128),
        torch.tensor(a['b'][...,:P],dtype=torch.complex128),torch.tensor(v*10**(variance_offset_db/10),dtype=torch.float64))


def fit_holdout(obs,regime,rsr):
    assert obs.z.shape[-1]==39
    pi=[r for r,L in REGIMES].index(regime);ri=list(RSRS).index(float(rsr));seed=2026101825+100*pi+ri
    order=torch.rand(len(obs),39,generator=torch.Generator().manual_seed(seed)).argsort(-1)
    return obs.columns(order[:,:30]),obs.columns(order[:,30:])


def anchors(split,regime,L,rsr,digest):
    path=OUT/split/'anchors'/f'{regime}_rsr{rsr:g}.npz'
    freeze=check_selection()[1] if split=='test' else 'development'
    if path.exists():
        a=load(path);assert str(a['protocol_sha256'])==digest and str(a['selection_sha256'])==freeze;return a
    seed,count=(TEST_SEED,TEST_COUNT) if split=='test' else (DEV_SEED,DEV_COUNT)
    a=make_anchor(seed,range(count),L,rsr,truth=split=='test')
    a.update(protocol_sha256=np.array(digest),selection_sha256=np.array(freeze),regime=np.array(regime));save(path,**a);return a


def observed_key(obs):
    """Homogeneous nominal-noise/reference batch key, no true signal power."""
    snr=10*torch.log10(obs.s.shape[1]/obs.sigma2);rsr=10*torch.log10(obs.b.abs().square().mean((-2,-1)))
    assert torch.max(snr)-torch.min(snr)<1e-8 and torch.max(rsr)-torch.min(rsr)<1e-8
    return f'{min(RSRS,key=lambda r:abs(r-float(rsr[0]))):g}',f'{min(SNRS,key=lambda s:abs(s-float(snr[0]))):g}'


def prior_config(obs,name):
    if name=='prior_sparse_full':return FULL
    rsr,snr=observed_key(obs);key=f'{min((0.,5.,10.,15.,20.),key=lambda s:abs(s-float(snr))):g}'
    return OLD_LOOKUP['selections'][key]['sparse_'+name[len('prior_sparse_'):]]['config']


def compact_config(obs,selection):
    rsr,snr=observed_key(obs);return selection['compact_choices'][rsr][snr]['config']


@torch.no_grad()
def estimate_method(obs,name,selection,est):
    if name=='em100':return est.em100(obs)
    if name=='optshrink':return frozen_optshrink(obs,OPT_LOOKUP)
    if name.startswith('prior_sparse_'):return sparse_fast.estimate(obs,prior_config(obs,name))
    if name=='compact_sparse':return sparse_fast.estimate(obs,compact_config(obs,selection))
    seed=int(name.rsplit('_',1)[1]);model=f'learned_{seed}'
    if name.startswith('original_'):return est.original(obs,model)
    if name.startswith('off_'):return est.unfiltered(obs,model)
    kind='same_steps' if name.startswith('fast_bypass_') else 'em100'
    return est.estimate(obs,model,selection['policies'][kind],fallback=kind)


def stats(values):
    values=np.asarray(values);return dict(p90_ms=float(np.quantile(values,.9)),median_ms=float(np.median(values)),mean_ms=float(np.mean(values)),calls=len(values))


@torch.no_grad()
def calibrate():
    digest=verify_protocol();assert not (OUT/'calibration.json').exists(),'calibration already frozen'
    assert not list((OUT/'development').glob('**/*.npz')),'development must follow isolated calibration'
    est=engine();rows=[]
    for rsr in RSRS:
        for regime,L in REGIMES:
            a=make_anchor(TIMING_SEED,[100],L,rsr,truth=False)
            for snr in (0.,20.,40.,60.):rows.append((dict(snr=snr,rsr=rsr,regime=regime,index=100),observed(a,snr)))
    policy_orig={f'{r:g}':60. for r in RSRS};policy_em={f'{r:g}':20. for r in RSRS}
    configs=candidates();samples={f'compact_{key}':[] for key in configs}
    samples.update({f'{branch}_{seed}':[] for branch in ('filtered','em100branch') for seed in SEEDS})
    tables={key:dict(compact_choices={f'{r:g}':{f'{s:g}':dict(config=config) for s in SNRS} for r in RSRS}) for key,config in configs.items()}
    def call(name,obs):
        if name.startswith('compact_'):return estimate_method(obs,'compact_sparse',tables[name[8:]],est)
        branch,seed=name.split('_');return est.estimate(obs,f'learned_{seed}',policy_orig if branch=='filtered' else policy_em,fallback='em100')
    for name in samples:call(name,rows[-1][1])
    rng=np.random.default_rng(20261018240)
    for wi,(meta,obs) in enumerate(rows):
        names=[n for n in samples if not n.startswith('em100branch_') or meta['snr']>20]
        for repeat in range(3):
            for i in rng.permutation(len(names)):
                name=names[i];start=time.perf_counter_ns();g=call(name,obs);ms=(time.perf_counter_ns()-start)/1e6
                assert torch.isfinite(g).all();samples[name].append(dict(ms=ms,repeat=repeat,**meta))
        print('Calibration world',wi+1,'/',len(rows),flush=True)
    summary={n:stats([x['ms'] for x in v]) for n,v in samples.items()}
    branch={b:stats([x['ms'] for s in SEEDS for x in samples[f'{b}_{s}']]) for b in ('filtered','em100branch')}
    cap=max(v['p90_ms'] for v in branch.values());eligible=[key for key in configs if summary[f'compact_{key}']['p90_ms']<=cap]
    assert eligible,'no compact candidate met calibrated cap'
    write_json(OUT/'calibration.json',dict(protocol_sha256=digest,cap_ms=cap,branch_timings=branch,candidates={key:dict(config=cfg,**summary[f'compact_{key}']) for key,cfg in configs.items()},
        eligible=eligible,methods=summary,raw_samples=samples,completed_unix=time.time()))
    print('Calibration complete',cap,'ms;',len(eligible),'eligible candidates',flush=True)


def check_calibration():
    digest=verify_protocol();record=json.loads((OUT/'calibration.json').read_text());assert record['protocol_sha256']==digest
    return record


@torch.no_grad()
def develop():
    calibration=check_calibration();digest=calibration['protocol_sha256'];est=engine();configs=candidates()
    loss_names=[f'original_{s}' for s in SEEDS]+[f'off_{s}' for s in SEEDS]+['em100']+[f'compact_{key}' for key in calibration['eligible']]
    for rsr in RSRS:
        banks=[anchors('development',regime,L,rsr,digest) for regime,L in REGIMES]
        for snr in SNRS:
            path=OUT/'development'/'losses'/f'rsr{rsr:g}_snr{snr:g}.npz'
            if path.exists():
                a=load(path);assert str(a['protocol_sha256'])==digest and str(a['calibration_sha256'])==sha(OUT/'calibration.json');continue
            splits=[fit_holdout(observed(a,snr,P=39),regime,rsr) for a,(regime,L) in zip(banks,REGIMES)]
            context=Observations(*(torch.cat([p[0].args()[j] for p in splits]) for j in range(4)))
            heldout=Observations(*(torch.cat([p[1].args()[j] for p in splits]) for j in range(4)))
            pred=[est.original(context,f'learned_{s}') for s in SEEDS]+[est.unfiltered(context,f'learned_{s}') for s in SEEDS]
            em=est.em100(context);pred.append(em)
            for key in calibration['eligible']:pred.append(em if key=='em100' else sparse_fast.estimate(context,configs[key]))
            losses=np.stack([nll(heldout.z,g@heldout.s+heldout.b,heldout.sigma2).mean((-2,-1)).numpy().reshape(len(REGIMES),DEV_COUNT) for g in pred])
            assert np.isfinite(losses).all()
            save(path,names=np.array(loss_names),losses=losses,regimes=np.array([r for r,L in REGIMES]),snr=np.array(snr),rsr=np.array(rsr),
                protocol_sha256=np.array(digest),calibration_sha256=np.array(sha(OUT/'calibration.json')))
            print('Development complete',path.stem,flush=True)
    policies={kind:{} for kind in ('same_steps','em100')};tables={kind:{} for kind in policies};compact={};compact_tables={}
    for rsr in RSRS:
        cells=[load(OUT/'development'/'losses'/f'rsr{rsr:g}_snr{snr:g}.npz') for snr in SNRS]
        for kind in policies:
            scores=[]
            for cutoff in CUTOFFS:
                rows=[]
                for a in cells:
                    original=a['losses'][:3];fallback=a['losses'][3:6] if kind=='same_steps' else np.broadcast_to(a['losses'][6:7],original.shape)
                    use=float(a['snr'])>cutoff+1e-10 and float(a['snr'])>20+1e-10
                    selected=fallback if use else original;delta=selected-original
                    rows.append(dict(snr=float(a['snr']),mean_nll=float(selected.mean()),excess_nll=float(delta.mean()),by_regime=[float(x) for x in delta.mean((0,2))]))
                scores.append(dict(cutoff_db=cutoff,mean_nll=float(np.mean([x['mean_nll'] for x in rows])),excess_nll=float(np.mean([x['excess_nll'] for x in rows])),cells=rows))
            best=min(scores,key=lambda x:(x['excess_nll'],-x['cutoff_db']));policies[kind][f'{rsr:g}']=best['cutoff_db'];tables[kind][f'{rsr:g}']=scores
        compact[f'{rsr:g}']={};compact_tables[f'{rsr:g}']={}
        for a in cells:
            scores=[]
            for key in calibration['eligible']:
                idx=list(map(str,a['names'])).index(f'compact_{key}');loss=a['losses'][idx]
                scores.append(dict(candidate=key,config=configs[key],mean_nll=float(loss.mean()),by_regime=loss.mean(-1).tolist(),p90_ms=calibration['candidates'][key]['p90_ms']))
            best=min(scores,key=lambda x:(x['mean_nll'],x['p90_ms'],x['candidate']));compact[f'{rsr:g}'][f'{float(a["snr"]):g}']=best
            compact_tables[f'{rsr:g}'][f'{float(a["snr"]):g}']=scores
    lossfiles=sorted((OUT/'development'/'losses').glob('*.npz'));assert len(lossfiles)==len(RSRS)*len(SNRS)
    ber_path=OUT/'ber/protocol.json';ber_record=json.loads(ber_path.read_text())
    assert ber_record['source_methods']==list(NAMES) and ber_record['snrs']==list(SNRS)
    for name,h in ber_record['source_hashes'].items():assert sha(REPO/name)==h,name
    record=dict(protocol_sha256=digest,calibration_sha256=sha(OUT/'calibration.json'),ber_protocol_sha256=sha(ber_path),ber_source_hashes=ber_record['source_hashes'],
        policies=policies,validation_tables=tables,compact_choices=compact,compact_tables=compact_tables,
        development_anchor_hashes={p.name:sha(p) for p in sorted((OUT/'development'/'anchors').glob('*.npz'))},loss_table_hashes={p.name:sha(p) for p in lossfiles})
    path=OUT/'policy.json'
    if path.exists():
        old=json.loads(path.read_text());old.pop('frozen_unix',None);assert old==record
    else:
        assert not list((OUT/'test').glob('**/*.npz')),'test existed before policy freeze';write_json(path,dict(**record,frozen_unix=time.time()))
    write_json(OUT/'status.json',dict(phase='development_complete',policies=policies,selection_sha256=sha(path)))
    print('Policies frozen',policies,flush=True)


def check_selection():
    calibration=check_calibration();path=OUT/'policy.json';record=json.loads(path.read_text())
    assert record['protocol_sha256']==calibration['protocol_sha256'] and record['calibration_sha256']==sha(OUT/'calibration.json')
    assert record['ber_protocol_sha256']==sha(OUT/'ber/protocol.json')
    for name,digest in record['ber_source_hashes'].items():assert sha(REPO/name)==digest,name
    for name,digest in record['development_anchor_hashes'].items():assert sha(OUT/'development'/'anchors'/name)==digest,name
    for name,digest in record['loss_table_hashes'].items():assert sha(OUT/'development'/'losses'/name)==digest,name
    return record,sha(path)


@torch.no_grad()
def predictions(est,obs,selection):
    original=torch.stack([est.original(obs,f'learned_{s}') for s in SEEDS]);off=torch.stack([est.unfiltered(obs,f'learned_{s}') for s in SEEDS]);em=est.em100(obs)
    mf=est.masks(obs,selection['policies']['same_steps'])['bypass'];me=est.masks(obs,selection['policies']['em100'])['bypass']
    fast=torch.where(mf[None,:,None,None],off,original);slow=torch.where(me[None,:,None,None],em[None],original)
    outputs=[*original,*off,*fast,*slow,em,frozen_optshrink(obs,OPT_LOOKUP)]
    cache={json.dumps(EM_CONFIG,sort_keys=True):em}
    for name in ('prior_sparse_matched','prior_sparse_50ms','prior_sparse_full','compact_sparse'):
        cfg=compact_config(obs,selection) if name=='compact_sparse' else prior_config(obs,name);key=json.dumps(cfg,sort_keys=True)
        if key not in cache:cache[key]=sparse_fast.estimate(obs,cfg)
        outputs.append(cache[key])
    return torch.stack(outputs),mf,me


@torch.no_grad()
def test():
    selection,freeze=check_selection();digest=selection['protocol_sha256'];est=engine()
    for rsr in RSRS:
        for regime,L in REGIMES:
            a=anchors('test',regime,L,rsr,digest)
            for snr in SNRS:
                path=OUT/'test'/'cells'/f'{cellname(regime,rsr,snr)}.npz'
                if path.exists():assert str(load(path)['selection_sha256'])==freeze;continue
                obs=observed(a,snr);pred,mf,me=predictions(est,obs,selection)
                if snr<=20:assert torch.equal(pred[6:9],pred[:3]) and torch.equal(pred[9:12],pred[:3]) and not mf.any() and not me.any()
                dispatch_max=0.
                for i,name in enumerate(NAMES):
                    if not (name.startswith('fast_bypass_') or name.startswith('em100_bypass_')):continue
                    actual=estimate_method(obs.batch(slice(0,1)),name,selection,est)
                    error=float(torch.max(torch.abs(actual-pred[i,0:1])));assert error<1e-10;dispatch_max=max(dispatch_max,error)
                arr=pred.numpy();assert np.isfinite(arr).all();errors=np.sum(abs(arr-a['g'][None])**2,axis=(-2,-1));energy=np.sum(abs(a['g'])**2,axis=(-2,-1))
                save(path,**{k:x.numpy() for k,x in zip(('z','s','b','sigma2'),obs.args())},truth_g=a['g'],paths=a['paths'],indices=a['indices'],local_indices=a['local_indices'],
                    method_names=np.array(NAMES),estimates=arr,errors=errors,truth_energy=energy,nmse=errors/energy[None],bypass_fast=mf.numpy(),bypass_em100=me.numpy(),
                    snr=np.array(snr),rsr=np.array(rsr),regime=np.array(regime),seed=np.array(TEST_SEED),selection_sha256=np.array(freeze),protocol_sha256=np.array(digest),
                    actual_dispatch_max_abs_error=np.array(dispatch_max),ordinary_range_exact_identity=np.array(snr<=20))
                print('Test complete',path.stem,flush=True)
                write_json(OUT/'status.json',dict(phase='testing',last_cell=path.stem,completed_cells=len(list((OUT/'test'/'cells').glob('*.npz')))))
    write_json(OUT/'status.json',dict(phase='test_complete',cells=len(SNRS)*len(RSRS)*len(REGIMES),selection_sha256=freeze))


@torch.no_grad()
def timing():
    selection,freeze=check_selection();assert not (OUT/'timing.json').exists(),'final timing already recorded';est=engine();rows=[]
    for rsr in RSRS:
        for regime,L in REGIMES:
            a=make_anchor(TIMING_SEED,[200],L,rsr,truth=False)
            for snr in SNRS:rows.append((dict(snr=snr,rsr=rsr,regime=regime,index=200),observed(a,snr)))
    for name in NAMES:estimate_method(rows[0][1],name,selection,est)
    samples={name:[] for name in NAMES};rng=np.random.default_rng(20261018241)
    for wi,(meta,obs) in enumerate(rows):
        mf=bool(est.masks(obs,selection['policies']['same_steps'])['bypass'][0]);me=bool(est.masks(obs,selection['policies']['em100'])['bypass'][0])
        for repeat in range(3):
            for ni in rng.permutation(len(NAMES)):
                name=NAMES[ni];start=time.perf_counter_ns();g=estimate_method(obs,name,selection,est);ms=(time.perf_counter_ns()-start)/1e6
                assert torch.isfinite(g).all();branch=name
                if name.startswith('original_') or name.startswith('fast_bypass_') and not mf or name.startswith('em100_bypass_') and not me:branch='original'
                if name.startswith('off_') or name.startswith('fast_bypass_') and mf:branch='off'
                if name=='em100' or name.startswith('em100_bypass_') and me:branch='em100'
                samples[name].append(dict(ms=ms,repeat=repeat,branch=branch,**meta))
        print('Timing world',wi+1,'/',len(rows),flush=True)
    values=dict(samples)
    for name,parts in GROUPS.items():values[name]=[x for part in parts for x in samples[part]]
    methods={name:dict(**stats([x['ms'] for x in values]),by_branch={b:stats([x['ms'] for x in values if x['branch']==b]) for b in sorted(set(x['branch'] for x in values))},
        by_condition={f'rsr{r:g}_snr{s:g}':stats([x['ms'] for x in values if x['rsr']==r and x['snr']==s]) for r in RSRS for s in SNRS}) for name,values in values.items()}
    cap=check_calibration()['cap_ms'];verification=dict(calibrated_cap_ms=cap,compact_p90_ms=methods['compact_sparse']['p90_ms'],primary_pipeline_p90_ms=methods['em100_bypass_mean']['p90_ms'],
        compact_exceeds_calibrated_cap=methods['compact_sparse']['p90_ms']>cap,compact_exceeds_fresh_pipeline=methods['compact_sparse']['p90_ms']>methods['em100_bypass_mean']['p90_ms'],
        primary_pipeline_exceeds_calibrated_cap=methods['em100_bypass_mean']['p90_ms']>cap)
    write_json(OUT/'timing.json',dict(selection_sha256=freeze,completed_unix=time.time(),worlds=len(rows),repeats=3,methods=methods,raw_samples=samples,cap_verification=verification))
    print('Timing complete',verification,flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['prepare','calibrate','develop','test','timing']);args=parser.parse_args()
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    {'prepare':freeze_protocol,'calibrate':calibrate,'develop':develop,'test':test,'timing':timing}[args.phase]()

if __name__=='__main__':main()
