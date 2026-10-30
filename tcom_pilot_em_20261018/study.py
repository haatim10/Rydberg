"""Frozen pilot-matched retraining, exact EM tails, and fresh latency controls."""
import os
for _k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[_k]='1'
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from sahure.data import Observations
from sahure.math import nll
from sahure.train import load_model
from rydberg_sim.track_b_drivers import draw_L_k,track_b_spec
from rydberg_sim.monte_carlo import generate_channel_estimation_trial
from tcom_followups_20261017.common import REPO,sha,write_json,frozen_inputs,effective_trial_id,checkpoint_path,frozen_optshrink,OPT_SELECTION
from tcom_runtime_bypass_20261017 import sparse_fast
from tcom_runtime_bypass_20261017.runtime_study import configs,family_key,cfg_with_factor,FULL
from . import continuation
from .training_data import simulate_observation,paired_split

ROOT=REPO/'results/tcom_pilot_em_20261018';OUT=ROOT/'evaluation'
DEV_SEED=2026101803;TEST_SEED=2026101804;DATA_SEED=2026101805;TIMING_SEED=2026101806
SNRS=(0.,5.,10.,15.,20.);REGIMES=(('L2',2),('L4',4),('random3to7',None))
FAMILIES=('old','context21','context30');SEEDS=(9101,9102,9103);TAILS=(0,1,2,5)
DEV_COUNT=24;TEST_COUNT=128;GUARD_COUNT=64;FACTORS=(.03,.1,.3,1.,3.)
GUARDS=(dict(name='weak_reference',snr=45.,rsr=5.,regime='L2',L=2),dict(name='high_snr',snr=60.,rsr=12.,regime='random3to7',L=None))
OLD_SELECTION=REPO/'results/tcom_runtime_bypass_20261017/runtime/validation_selection.json'
OLD_LOOKUP=json.loads(OLD_SELECTION.read_text());OPT_LOOKUP=json.loads(OPT_SELECTION.read_text())
PRIOR_NAMES=('prior_sparse_matched','prior_sparse_10ms','prior_sparse_20ms','prior_sparse_50ms','prior_sparse_full')
CONTRASTS=dict(pilot_count=('context21_tail0_mean','context30_tail0_mean'),
 old_continuation=('old_tail0_mean','old_selected_mean'),matched_continuation=('context30_tail0_mean','context30_selected_mean'),
 matched_vs_sparse=('sparse_fresh_context30','context30_selected_mean'))


def save(path,**arrays):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp.npz')
    np.savez_compressed(tmp,**arrays);tmp.replace(path)


def load(path):
    with np.load(path,allow_pickle=False) as a:return {k:a[k] for k in a.files}


def model_paths():
    return {f'{family}_{seed}':checkpoint_path(seed) if family=='old' else ROOT/f'checkpoints/{family}/seed_{seed}/em_adaptive_best.pt' for family in FAMILIES for seed in SEEDS}


def freeze_protocol():
    originals=frozen_inputs()
    for k in ('development_seed','test_seed','data_seed'):originals.pop(k,None)
    paths=[Path(__file__),Path(continuation.__file__),REPO/'tcom_pilot_em_20261018/training_data.py',REPO/'tcom_pilot_em_20261018/training_protocol.py',
      REPO/'tcom_runtime_bypass_20261017/sparse_fast.py',REPO/'tcom_runtime_bypass_20261017/runtime_study.py',REPO/'tcom_followups_20261017/common.py',REPO/'sahure/train.py',REPO/'sahure/data.py',OLD_SELECTION,OLD_SELECTION.parent/'calibration.json']
    spec=dict(version=1,name='Paired context-count retraining and exact EM continuation with fresh runtime-matched controls',
      development_seed=DEV_SEED,test_seed=TEST_SEED,data_seed=DATA_SEED,timing_seed=TIMING_SEED,N=32,K=3,deployment_P=30,generated_bank_P=39,
      snrs=SNRS,rsr=12.,regimes=REGIMES,development_cases_per_cell=DEV_COUNT,test_cases_per_cell=TEST_COUNT,
      guards=GUARDS,guard_count=GUARD_COUNT,guard_local_indices=list(range(1000,1000+GUARD_COUNT)),
      families=FAMILIES,model_seeds=SEEDS,tail_lengths=TAILS,
      models='Original checkpoints plus fresh paired context21/context30 retraining; imported training protocol determines both arms. No old source/checkpoint/result changes.',
      continuation='Exact gamma1 unregularized Rician EM starting at model output, never spectral reinitialization. Length0 included. NLL nonincrease and canonical/scalar parity checked before experiment.',
      data='Fresh39-column banks from same physical generator. Development gives each method30 fitting observations and9 withheld columns. Test/timing use only first30 columns from independent39-column banks; unused9 are discarded. Only test scoring returns G.',
      masks='2026101830+int(SNR), same30/9 split across all learned tails and sparse candidates at each SNR; balanced24cases for eachpathregime.',
      tail_selection='For each model family and observed nominal SNR, choose one length shared across three fitted seeds minimizing equal-path heldout RicianNLL; average losses not predictions. Ties choose shorter tail.',
      runtime_matching='After tail selection and training idle, isolated batch1 end-to-end selected pipeline p90 across three fits defines a separate cap for eachfamily. Fresh timing of all152 sparse structural families at factors.03 and3; maximum factorp90 determines eligibility. Zero penalty exact EM timed separately. Fresh30fit9heldout sparse selection perSNR under eachfamilycap. No test outcomes used.',
      optional_cache='Before freshcalibration, may cache observation-only development losses for structuralfamilies whose oldcalibrationp90<=10ms. This is only computation scheduling: final eligibility uses exclusivelyfresh timing, and all newlyeligiblefamilies are completed beforeselection.',
      sparse_grid=dict(dictionary_sizes=[64,128,256],inner_steps=[1,2,4,8,16,25],outer_steps=[2,4,8,12,20,40,80,100],factors=FACTORS,zero_penalty=True),
      calibration=dict(snrs=[0.,10.,20.],local_index=0,regimes=REGIMES,repeats=3,order_seed=20261018060,init_and_setup_included=True),
      final_timing=dict(snrs=SNRS,local_indices=[100,101],regimes=REGIMES,repeats=3,order_seed=20261018061,policy='Freshbatch1 verification of everyfixedtail and selectedpipeline pluscontrols; report overshoots without reselection.'),
      baselines='Frozen old sparse matched/10/20/50/full controls, fresh selected sparse under each ofthree actual selectedpipelinecaps, optimized exact EM100, frozen offlineOptShrink. Outside0–20 lookup clamps to nearest20; guards descriptive, no new guard-specific choices.',
      freeze='Freeze protocol/source before development. All six training workers arms must complete before checkpoint freeze. Freeze checkpoints, tail choices, timings and sparse choices before generating any fresh test/guard world.',
      primary_contrasts=CONTRASTS,quality_cooutcomes=['equal15-cell mean dB ratio-of-sums NMSE','aggregate BER'],
      uncertainty='20000 pairedcase multinomial bootstrap replicates, seed2026101894, shared localindices acrossSNR within regime and shared acrossall methods/fits. Pointwise95 plus Bonferroni8-outcome99.375% individual intervals for4contrasts×2metrics. Conditional on checkpoints and selectedcontrols. Guards and individualcurves descriptive.',
      diagnostics='Fulltailgrid, seed-specific scores, pooledlinearNMSE, tailp95/p99, severeNMSE>1 and worstcells; learnedgroup averagesfitmetrics, neverensembles estimates.',
      threads=dict(environment={k:os.environ[k] for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS')},torch=1,interop=1,scipy_fft_workers=1,blas='environmentlimitsrequested,threadpoolctlunavailable'),
      source_hashes={str(p.relative_to(REPO)):sha(p) for p in paths},original_inputs=originals,
      training_protocol_sha256=sha(ROOT/'training_protocol.json'))
    path=OUT/'protocol.json'
    if path.exists():
        old=json.loads(path.read_text());old.pop('frozen_unix',None);assert old==json.loads(json.dumps(spec)),'protocolchanged'
    else:write_json(path,dict(**spec,frozen_unix=time.time()))
    return sha(path)


def verify_protocol():
    path=OUT/'protocol.json';p=json.loads(path.read_text())
    for name,h in p['source_hashes'].items():assert sha(REPO/name)==h,name
    for name,h in p['original_inputs']['hashes'].items():assert sha(REPO/name)==h,name
    assert sha(ROOT/'training_protocol.json')==p['training_protocol_sha256']
    return sha(path)


def generated(seed,ids,snr,rsr,L,truth=False):
    values={k:[] for k in ('z','s','b','sigma2')};gs=[];paths=[]
    for local in ids:
        i=effective_trial_id(local,L)
        if truth:
            lp=draw_L_k(i,3,master_seed=seed) if L is None else (L,)*3
            spec=track_b_spec(P=39,n_trials=i+1,N=32,K=3,L=lp,master_seed=seed,experiment='tcom_pilot_em_20261018')
            w=generate_channel_estimation_trial(spec,i,snr,rsr);arrays=(w.Z,w.S,w.B,w.sigma2);gs.append(w.G);paths.append(lp)
        else:arrays=simulate_observation(i,seed,dict(snr=snr,rsr=rsr,L=L))
        for k,v in zip(values,arrays):values[k].append(v)
    obs=Observations(*(torch.tensor(np.stack(values[k]),dtype=torch.complex128 if k in ('s','b') else torch.float64) for k in values))
    if truth:return obs,np.stack(gs),np.stack(paths)
    return obs


def combine(rows):return Observations(*(torch.cat([r.args()[j] for r in rows]) for j in range(4)))
def first30(obs):return Observations(obs.z[...,:30],obs.s[...,:30],obs.b[...,:30],obs.sigma2)


def development(snr):
    path=OUT/'development'/f'observations_snr{snr:g}.npz'
    if path.exists():
        a=load(path);obs=Observations(*(torch.tensor(a[k],dtype=torch.complex128 if k in ('s','b') else torch.float64) for k in ('z','s','b','sigma2')))
    else:
        obs=combine([generated(DEV_SEED,range(DEV_COUNT),snr,12.,L) for regime,L in REGIMES])
        save(path,**{k:x.numpy() for k,x in zip(('z','s','b','sigma2'),obs.args())},seed=np.array(DEV_SEED),snr=np.array(snr))
    _,context,heldout=paired_split(obs,2026101830+int(snr))
    assert context.z.shape[-1]==30 and heldout.z.shape[-1]==9
    return context,heldout


def prepare():
    freeze_protocol()
    for snr in SNRS:development(snr)
    print('Protocol and observation-only development banks ready',flush=True)


@torch.no_grad()
def old_development():
    verify_protocol();models={f'old_{s}':load_model(checkpoint_path(s),dtype=torch.float64)[0] for s in SEEDS}
    for snr in SNRS:
        path=OUT/'development'/f'old_tails_snr{snr:g}.npz'
        if path.exists():continue
        context,heldout=development(snr);losses=[];names=[]
        for seed in SEEDS:
            name=f'old_{seed}';g=models[name](*context.args());pred,_=continuation.trajectory(context,g,TAILS,audit=True)
            for t in TAILS:
                names.append(f'{name}_tail{t}');losses.append(nll(heldout.z,pred[t]@heldout.s+heldout.b,heldout.sigma2).mean((-2,-1)).numpy())
        save(path,names=np.array(names),losses=np.stack(losses),protocol_sha256=np.array(verify_protocol()))
        print('Old model observation-only tail cache',snr,flush=True)


def sparse_cache_path(snr,d,inner,factor):return OUT/'development'/'sparse_cache'/f'snr{snr:g}_d{d}_i{inner}_f{factor:g}.npz'


@torch.no_grad()
def cache_sparse(snr,d,inner,factor,budgets,context,heldout):
    path=sparse_cache_path(snr,d,inner,factor);budgets=tuple(sorted(set(budgets)))
    if path.exists():
        a=load(path);assert str(a['protocol_sha256'])==verify_protocol()
        existing=set(map(int,a['budgets']))
        if set(budgets)<=existing:return a
        budgets=tuple(sorted(set(budgets)|existing))
    pred,_=sparse_fast.trajectory(context,factor,d=d,inner=max(inner,1),budgets=budgets)
    losses=np.stack([nll(heldout.z,pred[t]@heldout.s+heldout.b,heldout.sigma2).mean((-2,-1)).numpy() for t in budgets])
    assert np.isfinite(losses).all()
    save(path,budgets=np.array(budgets),losses=losses,protocol_sha256=np.array(verify_protocol()))
    return load(path)


@torch.no_grad()
def precompute_sparse():
    verify_protocol();old=json.loads((OLD_SELECTION.parent/'calibration.json').read_text());families={}
    for x in old['families'].values():
        if x['p90_ms']>10.:continue
        c=x['family'];families.setdefault((c['dictionary_size'],c['inner_steps']),[]).append(c['iterations'])
    for snr in SNRS:
        context,heldout=development(snr)
        for (d,inner),budgets in families.items():
            for factor in ((0.,) if inner==0 else FACTORS):cache_sparse(snr,d,inner,factor,budgets,context,heldout)
            print('Scheduling-only sparse cache',snr,d,inner,flush=True)


def freeze_checkpoints():
    digest=verify_protocol();statuses={}
    for seed in SEEDS:
        path=ROOT/f'training_workers/seed_{seed}/status.json';s=json.loads(path.read_text());assert s['state']=='complete',s
        for arm in ('context21','context30'):
            assert s['outputs'][arm]['epochs']==20
            assert sha(model_paths()[f'{arm}_{seed}'])==s['outputs'][arm]['best_sha256']
        statuses[str(seed)]=sha(path)
    spec=dict(protocol_sha256=digest,checkpoints={name:dict(path=str(path.relative_to(REPO)),sha256=sha(path)) for name,path in model_paths().items()},training_status_hashes=statuses)
    path=OUT/'checkpoints.json'
    if path.exists():
        old=json.loads(path.read_text());old.pop('frozen_unix',None);assert old==spec
    else:write_json(path,dict(**spec,frozen_unix=time.time()))
    return sha(path)


def checkpoints():
    verify_protocol();p=json.loads((OUT/'checkpoints.json').read_text())
    for name,x in p['checkpoints'].items():assert sha(REPO/x['path'])==x['sha256'],name
    return {name:load_model(REPO/x['path'],dtype=torch.float64)[0] for name,x in p['checkpoints'].items()}


def noise_key(obs):
    snr=10*torch.log10(3/obs.sigma2);assert float(snr.max()-snr.min())<1e-8,'study expects homogeneous noise batches'
    return f'{min(SNRS,key=lambda s:abs(s-float(snr.mean()))):g}'


@torch.no_grad()
def tails():
    checkpoint_hash=freeze_checkpoints();models=checkpoints();choices={f:{} for f in FAMILIES};hashes={}
    for snr in SNRS:
        path=OUT/'development'/f'tails_snr{snr:g}.npz'
        if not path.exists():
            context,heldout=development(snr);losses=[];names=[];auditmax=0.
            for family in FAMILIES:
                for seed in SEEDS:
                    name=f'{family}_{seed}'
                    cached=OUT/'development'/f'old_tails_snr{snr:g}.npz'
                    if family=='old' and cached.exists():
                        a=load(cached);assert str(a['protocol_sha256'])==verify_protocol();cn=list(map(str,a['names']))
                        for t in TAILS:
                            label=f'{name}_tail{t}';names.append(label);losses.append(a['losses'][cn.index(label)])
                        continue
                    g=models[name](*context.args());trajectory,info=continuation.trajectory(context,g,TAILS,audit=True)
                    auditmax=max(auditmax,info['maximum_nll_increase'])
                    for t in TAILS:
                        names.append(f'{name}_tail{t}');losses.append(nll(heldout.z,trajectory[t]@heldout.s+heldout.b,heldout.sigma2).mean((-2,-1)).numpy())
            save(path,names=np.array(names),losses=np.stack(losses),snr=np.array(snr),checkpoint_sha256=np.array(checkpoint_hash),maximum_nll_increase=np.array(auditmax))
        a=load(path);assert str(a['checkpoint_sha256'])==checkpoint_hash;names=list(map(str,a['names']))
        for family in FAMILIES:
            table=[dict(tail=t,mean_nll=float(a['losses'][[names.index(f'{family}_{seed}_tail{t}') for seed in SEEDS]].mean())) for t in TAILS]
            best=min(table,key=lambda x:(x['mean_nll'],x['tail']));choices[family][f'{snr:g}']=dict(selected_tail=best['tail'],scores=table)
        hashes[path.name]=sha(path);print('Tail development complete',snr,flush=True)
    record=dict(protocol_sha256=verify_protocol(),checkpoint_sha256=checkpoint_hash,choices=choices,loss_hashes=hashes)
    path=OUT/'tails.json'
    if path.exists():
        old=json.loads(path.read_text());old.pop('frozen_unix',None);assert old==record
    else:write_json(path,dict(**record,frozen_unix=time.time()))
    print('Tail choices frozen',choices,flush=True)


def check_tails():
    verify_protocol();p=json.loads((OUT/'tails.json').read_text());assert sha(OUT/'checkpoints.json')==p['checkpoint_sha256']
    for name,h in p['loss_hashes'].items():assert sha(OUT/'development'/name)==h,name
    return p


def selected_tail(record,family,obs):return record['choices'][family][noise_key(obs)]['selected_tail']


def timing_rows(final=False):
    result=[]
    for snr in (SNRS if final else (0.,10.,20.)):
        for regime,L in REGIMES:
            for i in ((100,101) if final else (0,)):
                obs=first30(generated(TIMING_SEED,[i],snr,12.,L));result.append((dict(snr=snr,regime=regime,index=i),obs))
    return result


@torch.no_grad()
def calibrate():
    if (OUT/'calibration.json').exists():raise RuntimeError('calibration already frozen; no implicit rerun')
    tails_record=check_tails();models=checkpoints();rows=timing_rows();probes=[]
    for fam in configs():
        for factor in ((0.,) if fam['inner_steps']==0 else (.03,3.)):
            probes.append((family_key(fam)+f'_f{factor:g}',lambda obs,c=cfg_with_factor(fam,factor):sparse_fast.estimate(obs,c),dict(family=fam,factor=factor)))
    for family in FAMILIES:
        for seed in SEEDS:
            name=f'{family}_{seed}'
            probes.append((name,lambda obs,f=family,n=name:continuation.estimate(obs,models[n](*obs.args()),selected_tail(tails_record,f,obs)),dict(model_family=family)))
    for name,fn,meta in probes:fn(rows[0][1])
    samples={name:[] for name,fn,meta in probes};rng=np.random.default_rng(20261018060)
    for i,(meta,obs) in enumerate(rows):
        for repeat in range(3):
            for j in rng.permutation(len(probes)):
                name,fn,_=probes[j];start=time.perf_counter_ns();g=fn(obs);ms=(time.perf_counter_ns()-start)/1e6
                assert torch.isfinite(g).all();samples[name].append(dict(ms=ms,repeat=repeat,**meta))
        print('Calibration case',i+1,'/',len(rows),flush=True)
    summaries={name:dict(meta=meta,p90_ms=float(np.quantile([x['ms'] for x in samples[name]],.9)),median_ms=float(np.median([x['ms'] for x in samples[name]])),samples=samples[name]) for name,fn,meta in probes}
    caps={family:float(np.quantile([x['ms'] for seed in SEEDS for x in samples[f'{family}_{seed}']],.9)) for family in FAMILIES}
    families={family_key(fam):dict(config=fam,p90_ms=max(summaries[family_key(fam)+f'_f{f:g}']['p90_ms'] for f in ((0.,) if fam['inner_steps']==0 else (.03,3.)))) for fam in configs()}
    eligible={family:[k for k,x in families.items() if x['p90_ms']<=caps[family]] for family in FAMILIES}
    assert all(eligible.values())
    path=OUT/'calibration.json';assert not path.exists(),'calibration already frozen'
    write_json(path,dict(protocol_sha256=verify_protocol(),tails_sha256=sha(OUT/'tails.json'),caps_ms=caps,families=families,eligibility=eligible,probes=summaries,completed_unix=time.time()))
    print('Calibration complete',caps,{k:len(v) for k,v in eligible.items()},flush=True)


@torch.no_grad()
def sparse_development():
    check_tails();cal=json.loads((OUT/'calibration.json').read_text());assert cal['tails_sha256']==sha(OUT/'tails.json')
    union=set(k for e in cal['eligibility'].values() for k in e);families={}
    for key in sorted(union):
        c=cal['families'][key]['config'];families.setdefault((c['dictionary_size'],c['inner_steps']),[]).append(c['iterations'])
    selections={};hashes={}
    for snr in SNRS:
        path=OUT/'development'/f'sparse_snr{snr:g}.json'
        if path.exists():record=json.loads(path.read_text());assert record['calibration_sha256']==sha(OUT/'calibration.json')
        else:
            context,heldout=development(snr);scores=[]
            for (d,inner),budgets in families.items():
                for factor in ((0.,) if inner==0 else FACTORS):
                    cached=cache_sparse(snr,d,inner,factor,budgets,context,heldout)
                    for t in sorted(set(budgets)):
                        fam=dict(dictionary_size=d,inner_steps=inner,iterations=t);cfg=cfg_with_factor(fam,factor)
                        losses=cached['losses'][list(map(int,cached['budgets'])).index(t)]
                        scores.append(dict(config=cfg,family_key=family_key(fam),mean_nll=float(losses.mean()),per_world_nll=losses.tolist(),p90_ms=cal['families'][family_key(fam)]['p90_ms']))
                print('Sparse development',snr,d,inner,flush=True)
            chosen={}
            for family in FAMILIES:
                eligible=[x for x in scores if x['family_key'] in cal['eligibility'][family]]
                chosen[family]=min(eligible,key=lambda x:(x['mean_nll'],x['p90_ms'],x['config']['iterations'],x['config']['inner_steps'],x['config']['dictionary_size'],x['config']['factor']))
            record=dict(calibration_sha256=sha(OUT/'calibration.json'),snr=snr,scores=scores,selected=chosen);write_json(path,record)
        selections[f'{snr:g}']=record['selected'];hashes[path.name]=sha(path)
    record=dict(protocol_sha256=verify_protocol(),checkpoint_sha256=sha(OUT/'checkpoints.json'),tails_sha256=sha(OUT/'tails.json'),calibration_sha256=sha(OUT/'calibration.json'),sparse_choices=selections,development_hashes=hashes,
      observation_hashes={p.name:sha(p) for p in sorted((OUT/'development').glob('observations*.npz'))},sparse_cache_hashes={p.name:sha(p) for p in sorted((OUT/'development'/'sparse_cache').glob('*.npz'))})
    path=OUT/'selection.json'
    if path.exists():
        old=json.loads(path.read_text());old.pop('frozen_unix',None);assert old==record
    else:
        assert not list((OUT/'test').glob('*.npz')) and not list((OUT/'guards').glob('*.npz'))
        write_json(path,dict(**record,frozen_unix=time.time()))
    print('All test choices frozen',flush=True)


def check_selection():
    tails_record=check_tails();p=json.loads((OUT/'selection.json').read_text())
    assert p['protocol_sha256']==verify_protocol()
    for file,key in [('checkpoints.json','checkpoint_sha256'),('tails.json','tails_sha256'),('calibration.json','calibration_sha256')]:assert sha(OUT/file)==p[key]
    for name,h in p['development_hashes'].items():assert sha(OUT/'development'/name)==h,name
    for name,h in p['observation_hashes'].items():assert sha(OUT/'development'/name)==h,name
    for name,h in p['sparse_cache_hashes'].items():assert sha(OUT/'development'/'sparse_cache'/name)==h,name
    return p,tails_record,sha(OUT/'selection.json')


def prior_config(obs,name):
    if name=='prior_sparse_full':return FULL
    key=noise_key(obs);cap=name.replace('prior_sparse_','sparse_')
    return OLD_LOOKUP['selections'][key][cap]['config']


@torch.no_grad()
def predictions(models,obs,selection,tails_record):
    outputs={};maximum_increase=0.
    for family in FAMILIES:
        for seed in SEEDS:
            base=f'{family}_{seed}';g=models[base](*obs.args());tail,info=continuation.trajectory(obs,g,TAILS,audit=True)
            maximum_increase=max(maximum_increase,info['maximum_nll_increase'])
            for t in TAILS:outputs[f'{base}_tail{t}']=tail[t].numpy()
            outputs[f'{base}_selected']=tail[selected_tail(tails_record,family,obs)].numpy()
    for name in PRIOR_NAMES:outputs[name]=sparse_fast.estimate(obs,prior_config(obs,name)).numpy()
    outputs['em100']=sparse_fast.estimate(obs,dict(factor=0.,dictionary_size=128,inner_steps=1,iterations=100)).numpy()
    outputs['optshrink']=frozen_optshrink(obs,OPT_LOOKUP).numpy()
    for family in FAMILIES:outputs[f'sparse_fresh_{family}']=sparse_fast.estimate(obs,selection['sparse_choices'][noise_key(obs)][family]['config']).numpy()
    return outputs,maximum_increase


@torch.no_grad()
def test():
    selection,tails_record,freeze=check_selection();models=checkpoints()
    settings=[dict(name=f'{regime}_snr{snr:g}',snr=snr,rsr=12.,regime=regime,L=L,guard=False) for snr in SNRS for regime,L in REGIMES]
    settings += [dict(x,guard=True) for x in GUARDS]
    for c in settings:
        path=OUT/('guards' if c['guard'] else 'test')/(c['name']+'.npz')
        if path.exists():assert str(load(path)['selection_sha256'])==freeze;continue
        ids=range(1000,1000+GUARD_COUNT) if c['guard'] else range(TEST_COUNT)
        full,g,paths=generated(TEST_SEED,ids,c['snr'],c['rsr'],c['L'],truth=True);obs=first30(full)
        outputs,maximum_increase=predictions(models,obs,selection,tails_record);arr=np.stack(list(outputs.values()));assert np.isfinite(arr).all()
        error=np.sum(abs(arr-g[None])**2,axis=(-2,-1));energy=np.sum(abs(g)**2,axis=(-2,-1))
        save(path,**{k:x.numpy() for k,x in zip(('z','s','b','sigma2'),obs.args())},truth_g=g,paths=paths,method_names=np.array(list(outputs)),estimates=arr,errors=error,truth_energy=energy,nmse=error/energy[None],
          seed=np.array(TEST_SEED),snr=np.array(c['snr']),rsr=np.array(c['rsr']),regime=np.array(c['regime']),guard=np.array(c['guard']),indices=np.array([effective_trial_id(i,c['L']) for i in ids]),
          selection_sha256=np.array(freeze),maximum_continuation_nll_increase=np.array(maximum_increase))
        print('Test complete',c['name'],flush=True)
    write_json(OUT/'status.json',dict(phase='test_complete',main_cells=15,guard_cells=2,selection_sha256=freeze))


@torch.no_grad()
def final_timing():
    selection,tails_record,freeze=check_selection();models=checkpoints();rows=timing_rows(final=True)
    names=[f'{f}_{s}_tail{t}' for f in FAMILIES for s in SEEDS for t in TAILS]+[f'{f}_{s}_selected' for f in FAMILIES for s in SEEDS]
    names+=list(PRIOR_NAMES)+['em100','optshrink']+[f'sparse_fresh_{f}' for f in FAMILIES]
    def call(name,obs):
        if name in PRIOR_NAMES:return sparse_fast.estimate(obs,prior_config(obs,name))
        if name=='em100':return sparse_fast.estimate(obs,dict(factor=0.,dictionary_size=128,inner_steps=1,iterations=100))
        if name=='optshrink':return frozen_optshrink(obs,OPT_LOOKUP)
        if name.startswith('sparse_fresh_'):
            family=name[len('sparse_fresh_'):];return sparse_fast.estimate(obs,selection['sparse_choices'][noise_key(obs)][family]['config'])
        family,seed,kind=name.split('_');base=f'{family}_{seed}'
        t=selected_tail(tails_record,family,obs) if kind=='selected' else int(kind[4:])
        return continuation.estimate(obs,models[base](*obs.args()),t)
    for name in names:call(name,rows[0][1])
    samples={name:[] for name in names};rng=np.random.default_rng(20261018061)
    for wi,(meta,obs) in enumerate(rows):
        for repeat in range(3):
            for i in rng.permutation(len(names)):
                name=names[i];start=time.perf_counter_ns();g=call(name,obs);ms=(time.perf_counter_ns()-start)/1e6
                assert torch.isfinite(g).all();samples[name].append(dict(ms=ms,repeat=repeat,**meta))
        print('Final timing world',wi+1,'/',len(rows),flush=True)
    def stats(v):return dict(p90_ms=float(np.quantile(v,.9)),median_ms=float(np.median(v)),mean_ms=float(np.mean(v)))
    groups={f'{f}_{kind}_mean':[f'{f}_{s}_{kind}' for s in SEEDS] for f in FAMILIES for kind in ['tail0','tail1','tail2','tail5','selected']}
    values=dict(samples)
    for name,parts in groups.items():values[name]=[x for p in parts for x in samples[p]]
    methods={name:dict(**stats([x['ms'] for x in v]),per_snr={f'{snr:g}':stats([x['ms'] for x in v if x['snr']==snr]) for snr in SNRS}) for name,v in values.items()}
    cal=json.loads((OUT/'calibration.json').read_text());overshoot={f:dict(cap_ms=cal['caps_ms'][f],selected_pipeline_p90_ms=methods[f'{f}_selected_mean']['p90_ms'],sparse_p90_ms=methods[f'sparse_fresh_{f}']['p90_ms'],
      sparse_exceeds_calibrated_cap=methods[f'sparse_fresh_{f}']['p90_ms']>cal['caps_ms'][f],sparse_exceeds_fresh_pipeline=methods[f'sparse_fresh_{f}']['p90_ms']>methods[f'{f}_selected_mean']['p90_ms']) for f in FAMILIES}
    path=OUT/'final_timing.json';assert not path.exists(),'final timing already recorded'
    write_json(path,dict(selection_sha256=freeze,methods=methods,raw_samples=samples,cap_verification=overshoot,worlds=len(rows),repeats=3,completed_unix=time.time()))
    print('Final timing complete',flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','old-dev','precompute-sparse','tails','calibrate','sparse','test','final-timing']);args=p.parse_args()
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    {'prepare':prepare,'old-dev':old_development,'precompute-sparse':precompute_sparse,'tails':tails,'calibrate':calibrate,'sparse':sparse_development,'test':test,'final-timing':final_timing}[args.phase]()

if __name__=='__main__':main()
