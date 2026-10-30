"""Fresh validation-selected sparse comparison under measured runtime budgets.

Timing, development, test, and final timing are separate explicit phases so
other computations can be paused during batch-one CPU timing.
"""
import os
for _k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[_k]='1'
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from sahure.data import Observations, split_observations
from sahure.math import nll
from tcom_followups_20261017.common import (REPO, sha, write_json, frozen_inputs,
    effective_trial_id, world_batch, load_learned_models, em_estimate, frozen_optshrink, OPT_SELECTION)

OUT=REPO/'results/tcom_runtime_bypass_20261017/runtime'
DEV_SEED=2026101741; TEST_SEED=2026101742; DATA_SEED=2026101743; TIMING_SEED=2026101744
SNRS=(0.,5.,10.,15.,20.)
REGIMES=(('L2',2),('L4',4),('random3to7',None))
FACTORS=(.03,.1,.3,1.,3.)
BUDGETS=(2,4,8,12,20,40,80,100)
DS=(64,128,256); INNERS=(1,2,4,8,16,25)
DEV_COUNT=12; TEST_COUNT=64
CAP_NAMES=('sparse_matched','sparse_10ms','sparse_20ms','sparse_50ms')
FULL=dict(factor=.3,dictionary_size=128,inner_steps=25,iterations=100)
OPT_LOOKUP=json.loads(OPT_SELECTION.read_text())


def solver():
    from . import sparse_fast
    return sparse_fast


def save(path,**arrays):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp.npz');np.savez_compressed(temp,**arrays);temp.replace(path)


def load(path):
    with np.load(path,allow_pickle=False) as a:return {k:a[k] for k in a.files}


def configs():
    families=[dict(dictionary_size=d,inner_steps=inner,iterations=t) for d in DS for inner in INNERS for t in BUDGETS]
    return families+[dict(dictionary_size=64,inner_steps=0,iterations=t) for t in BUDGETS]


def family_key(c):return f'd{c["dictionary_size"]}_i{c["inner_steps"]}_t{c["iterations"]}'
def config_key(c):return family_key(c)+f'_f{c["factor"]:g}'

def cfg_with_factor(fam,factor):
    c=dict(fam,factor=float(factor))
    if c['inner_steps']==0:c.update(inner_steps=1,factor=0.)
    return c


def thread_setup():
    torch.set_num_threads(1);torch.set_num_interop_threads(1)


def freeze_protocol():
    originals=frozen_inputs()
    for k in ('development_seed','test_seed','data_seed'):originals.pop(k,None)
    spec=dict(version=1,name='Validation-selected optimized sparse Rician EM under runtime budgets',
      development_seed=DEV_SEED,test_seed=TEST_SEED,data_seed=DATA_SEED,timing_seed=TIMING_SEED,
      N=32,K=3,P=30,RSR=12.,snrs=list(SNRS),regimes=REGIMES,development_cases_per_cell=DEV_COUNT,test_cases_per_cell=TEST_COUNT,
      dictionary_sizes=DS,inner_steps=INNERS,outer_steps=BUDGETS,factors=FACTORS,zero_penalty_exact_em=True,
      requested_thread_environment={k:os.environ[k] for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS')},
      numerical_backend='NumPy/SciPy Apple Accelerate; environment requests one thread; threadpoolctl unavailable. Torch threads/interops=1 and SciPy FFT workers=1.',
      algorithm='Accelerated monotone FISTA inner solver; inner1/2 use equivalent first FISTA iterates; all transforms fixed before timing. Positive lambda=factor*sqrt(v*P_fit*log(D)). Lambda0 uses exact unconstrained EM.',
      eligibility='Batch-one isolated CPU p90 timing; all30pilots and initialization included. Each positive-penalty structural family measured at factors.03and3, conservative maximump90 used for all factors in family. Zero-penaltyEM timed separately.',
      calibration=dict(snrs=[0.,10.,20.],regimes=REGIMES,local_indices=[0],repeats=3,
        warmup_calls_per_probe=1,randomized_order_seed=20261017440,probe_factors=[.03,3.]),
      caps='Firstcap is pooled p90 of three original learned models on same calibration cases; others fixed10/20/50ms. If families unavailable report no candidate rather than silently exceedcap.',
      selection='For each observedSNR only, equal average heldoutRicianNLL over12worlds per each ofthree pathregimes. Fixedrandom21/9split. Fitting initialization/features/updates seeonly21pilots. Tiebreak lower familytiming thenfewerouter,inner,D,factor. No truechannel or actualpathcount selects configuration.',
      masks='seed20261017410+int(SNR); same masks for all candidates at eachSNR',
      trajectory_sharing='Prefix reuse only within fixedD,inner,factor; all eligible outerbudgets saved from one trajectory. Identical mathematics to separate runs.',
      test='Freeze all selected controls and source/checkpoint hashes before generating960fresh testworlds. Refit on all30pilots. Same observations across every method.',
      baselines=['three original learned fits','EM100','frozen offlineOptShrink','fast_full(.3,D128,inner25,outer100) diagnostic, outside capselection'],
      final_timing=dict(seed=TIMING_SEED,local_indices=[100,101],snrs=SNRS,regimes=REGIMES,repeats=3,
        policy='Fresh timingcases, isolated batch1; report per-method/perSNR median,p90 and capovershoot without reselection.'),
      uncertainty='Conditional paired whole-channel bootstrap; fixed checkpoints and validation selections. Seedmodels averaged as scores, never ensembled. Sharedpathdraws acrossSNRdo not become independent training replications.',
      source_hashes={str(p.relative_to(REPO)):sha(p) for p in [Path(__file__),REPO/'tcom_runtime_bypass_20261017/sparse_fast.py']},
      frozen_original_inputs=originals)
    target=OUT/'protocol.json'
    if target.exists():
        old=json.loads(target.read_text());old.pop('frozen_unix',None)
        if old!=json.loads(json.dumps(spec)):raise RuntimeError('frozen protocol changed')
    else:write_json(target,dict(**spec,frozen_unix=time.time()))
    return sha(target)


def verify_protocol():
    p=OUT/'protocol.json';a=json.loads(p.read_text())
    for name,digest in a['source_hashes'].items():assert sha(REPO/name)==digest,name
    for name,digest in a['frozen_original_inputs']['hashes'].items():assert sha(REPO/name)==digest,name
    return sha(p)


def combine(items):return Observations(*(torch.cat([a.args()[j] for a in items]) for j in range(4)))


def timing_observations(final=False):
    rows=[]
    for snr in (SNRS if final else (0.,10.,20.)):
        for regime,L in REGIMES:
            for i in ((100,101) if final else (0,)):
                obs,_,_=world_batch(TIMING_SEED,[i],snr,L=L)
                rows.append((dict(snr=snr,regime=regime,index=i,effective_trial_id=effective_trial_id(i,L)),obs))
    return rows


def measured(call):
    start=time.perf_counter_ns();out=call();elapsed=(time.perf_counter_ns()-start)/1e6
    return elapsed,out


@torch.no_grad()
def calibrate():
    digest=freeze_protocol();target=OUT/'calibration.json'
    if target.exists():print('Calibration already complete; keeping frozen timings',flush=True);return
    fast=solver();models=load_learned_models();obslist=timing_observations()
    probes=[]
    for fam in configs():
        for factor in ((0.,) if fam['inner_steps']==0 else (.03,3.)):
            probes.append((family_key(fam)+f'_probe{factor:g}',lambda obs,c=cfg_with_factor(fam,factor):fast.estimate(obs,c),dict(family=fam,factor=factor)))
    for name,m in models.items():probes.append((name,lambda obs,m=m:m(*obs.args()),dict(learned=True)))
    probes.append(('fast_full',lambda obs:fast.estimate(obs,FULL),dict(diagnostic=True,config=FULL)))
    probes.append(('em100',lambda obs:em_estimate(obs),dict(diagnostic=True)))
    probes.append(('optshrink',lambda obs:frozen_optshrink(obs,OPT_LOOKUP),dict(diagnostic=True)))
    for name,fn,meta in probes:fn(obslist[0][1])
    values={name:[] for name,_,_ in probes};rng=np.random.default_rng(20261017440)
    for oi,(worldmeta,obs) in enumerate(obslist):
        for repeat in range(3):
            for index in rng.permutation(len(probes)):
                name,fn,meta=probes[index];ms,g=measured(lambda:fn(obs));assert torch.isfinite(g).all(),name
                values[name].append(dict(ms=ms,repeat=repeat,**worldmeta))
        write_json(OUT/'status.json',dict(phase='calibration',world=oi+1,worlds=len(obslist)))
        print('Calibration case',oi+1,'/',len(obslist),flush=True)
    probe_summary={name:dict(meta=meta,median_ms=float(np.median([x['ms'] for x in values[name]])),p90_ms=float(np.quantile([x['ms'] for x in values[name]],.9)),samples=values[name]) for name,fn,meta in probes}
    family_times={}
    for fam in configs():
        vals=[probe_summary[family_key(fam)+f'_probe{f:g}']['p90_ms'] for f in ((0.,) if fam['inner_steps']==0 else (.03,3.))]
        family_times[family_key(fam)]=dict(family=fam,p90_ms=max(vals))
    learnedtimes=[x['ms'] for name in models for x in values[name]]
    cap=float(np.quantile(learnedtimes,.9));caps=dict(zip(CAP_NAMES,[cap,10.,20.,50.]))
    eligibility={name:[key for key,val in family_times.items() if val['p90_ms']<=budget] for name,budget in caps.items()}
    if any(not x for x in eligibility.values()):raise RuntimeError('runtimecap has no eligible candidates')
    write_json(target,dict(protocol_sha256=digest,completed_unix=time.time(),caps_ms=caps,families=family_times,
      eligibility=eligibility,probes=probe_summary,learned_pooled_p90_ms=cap,learned_pooled_median_ms=float(np.median(learnedtimes))))
    print('Calibration complete',caps,{k:len(v) for k,v in eligibility.items()},flush=True)


def development_data(snr):
    path=OUT/'development'/f'snr{snr:g}.npz'
    if path.exists():
        a=load(path);obs=Observations(*(torch.tensor(a[k],dtype=torch.complex128 if k in ('s','b') else torch.float64) for k in ('z','s','b','sigma2')))
        return obs
    blocks=[]
    for regime,L in REGIMES:
        obs,_,_=world_batch(DEV_SEED,range(DEV_COUNT),snr,L=L);blocks.append(obs)
    allobs=combine(blocks)
    save(path,**{k:x.numpy() for k,x in zip(('z','s','b','sigma2'),allobs.args())},
      snr=np.array(snr),seed=np.array(DEV_SEED),path_strata=np.repeat([r[0] for r in REGIMES],DEV_COUNT))
    return allobs


@torch.no_grad()
def develop():
    digest=verify_protocol();cal=json.loads((OUT/'calibration.json').read_text());fast=solver()
    selected={};all_scores={};eligible=set(x for a in cal['eligibility'].values() for x in a)
    for snr in SNRS:
        path=OUT/'development'/f'scores_snr{snr:g}.json'
        if path.exists():record=json.loads(path.read_text());all_scores[f'{snr:g}']=record['scores'];selected[f'{snr:g}']=record['selected'];continue
        obs=development_data(snr);context,heldout=split_observations(obs,20261017410+int(snr));scores=[]
        families={}
        for key in sorted(eligible):
            fam=cal['families'][key]['family'];families.setdefault((fam['dictionary_size'],fam['inner_steps']),[]).append(fam['iterations'])
        for (d,inner),budgets in families.items():
            for factor in ((0.,) if inner==0 else FACTORS):
                pathout,diag=fast.trajectory(context,factor,d=d,inner=max(inner,1),budgets=tuple(sorted(budgets)))
                for t,g in pathout.items():
                    fam=dict(dictionary_size=d,inner_steps=inner,iterations=int(t));cfg=cfg_with_factor(fam,factor)
                    losses=nll(heldout.z,g@heldout.s+heldout.b,heldout.sigma2).mean((-2,-1)).numpy()
                    if not np.isfinite(losses).all():raise FloatingPointError(config_key(cfg))
                    scores.append(dict(config=cfg,family_key=family_key(fam),mean_nll=float(losses.mean()),
                      per_world_nll=losses.tolist(),per_regime_nll=[float(x.mean()) for x in np.split(losses,3)],
                      eligibility_p90_ms=cal['families'][family_key(fam)]['p90_ms']))
            print('Development SNR',snr,'D',d,'inner',inner,'done',flush=True)
        choices={}
        for cap in CAP_NAMES:
            options=[x for x in scores if x['family_key'] in cal['eligibility'][cap]]
            best=min(options,key=lambda x:(x['mean_nll'],x['eligibility_p90_ms'],x['config']['iterations'],x['config']['inner_steps'],x['config']['dictionary_size'],x['config']['factor']))
            choices[cap]=dict(**best,cap_ms=cal['caps_ms'][cap],candidate_count=len(options))
        record=dict(protocol_sha256=digest,snr=snr,selected=choices,scores=scores);write_json(path,record)
        selected[f'{snr:g}']=choices;all_scores[f'{snr:g}']=scores
    freeze=OUT/'validation_selection.json'
    record=dict(protocol_sha256=digest,calibration_sha256=sha(OUT/'calibration.json'),caps_ms=cal['caps_ms'],
      selections=selected,development_scores_sha256={p.name:sha(p) for p in sorted((OUT/'development').glob('scores*.json'))},
      source_hashes=json.loads((OUT/'protocol.json').read_text())['source_hashes'])
    if freeze.exists():
        old=json.loads(freeze.read_text());old.pop('frozen_unix',None);assert old==record
    else:
        if (OUT/'test').exists() and list((OUT/'test').glob('*.npz')):raise RuntimeError('test generated before selectionfreeze')
        write_json(freeze,dict(**record,frozen_unix=time.time()))
    write_json(OUT/'status.json',dict(phase='development_complete',selection_sha256=sha(freeze)))
    print('Development selection frozen',flush=True)


def check_selection():
    digest=verify_protocol();path=OUT/'validation_selection.json';a=json.loads(path.read_text())
    assert a['protocol_sha256']==digest
    assert a['calibration_sha256']==sha(OUT/'calibration.json')
    for name,h in a['development_scores_sha256'].items():assert sha(OUT/'development'/name)==h
    return a,sha(path)


def observed_snr(obs):return float((10*torch.log10(3/obs.sigma2)).mean())

def selected_config(selection,obs,cap):
    # Receiver-visible nominal variance, not truth, SNR labels or actual path count.
    snr=observed_snr(obs);key=min(selection['selections'],key=lambda x:abs(float(x)-snr))
    return selection['selections'][key][cap]['config']


@torch.no_grad()
def test():
    selection,freeze=check_selection();fast=solver();models=load_learned_models()
    for snr in SNRS:
        for regime,L in REGIMES:
            path=OUT/'test'/f'{regime}_snr{snr:g}.npz'
            if path.exists():assert str(load(path)['selection_sha256'])==freeze;continue
            obs,g,paths=world_batch(TEST_SEED,range(TEST_COUNT),snr,L=L)
            outputs={name:m(*obs.args()).numpy() for name,m in models.items()}
            for cap in CAP_NAMES:outputs[cap]=fast.estimate(obs,selected_config(selection,obs,cap)).numpy()
            outputs['em100']=em_estimate(obs).numpy();outputs['optshrink']=frozen_optshrink(obs,OPT_LOOKUP).numpy();outputs['fast_full']=fast.estimate(obs,FULL).numpy()
            arr=np.stack(list(outputs.values()));assert np.isfinite(arr).all()
            errors=np.sum(abs(arr-g[None])**2,axis=(-2,-1));energy=np.sum(abs(g)**2,axis=(-2,-1))
            save(path,z=obs.z.numpy(),s=obs.s.numpy(),b=obs.b.numpy(),sigma2=obs.sigma2.numpy(),truth_g=g,paths=paths,
              effective_trial_ids=np.array([effective_trial_id(i,L) for i in range(TEST_COUNT)]),seed=np.array(TEST_SEED),
              snr=np.array(snr),rsr=np.array(12.),regime=np.array(regime),method_names=np.array(list(outputs)),
              estimates=arr,errors=errors,truth_energy=energy,nmse=errors/energy[None],selection_sha256=np.array(freeze))
            write_json(OUT/'status.json',dict(phase='testing',last_cell=path.stem,completed_cells=len(list((OUT/'test').glob('*.npz')))))
            print('Test cell complete',path.stem,flush=True)
    write_json(OUT/'status.json',dict(phase='test_complete',cells=15,selection_sha256=freeze))


@torch.no_grad()
def final_timing():
    selection,freeze=check_selection();fast=solver();models=load_learned_models();obslist=timing_observations(final=True)
    names=list(models)+list(CAP_NAMES)+['em100','optshrink','fast_full'];values={name:[] for name in names}
    def call(name,obs):
        if name in models:return models[name](*obs.args())
        if name in CAP_NAMES:return fast.estimate(obs,selected_config(selection,obs,name))
        if name=='em100':return em_estimate(obs)
        if name=='optshrink':return frozen_optshrink(obs,OPT_LOOKUP)
        return fast.estimate(obs,FULL)
    for name in names:call(name,obslist[0][1])
    rng=np.random.default_rng(20261017441)
    for wi,(meta,obs) in enumerate(obslist):
        for repeat in range(3):
            for ni in rng.permutation(len(names)):
                name=names[ni];ms,g=measured(lambda:call(name,obs));assert torch.isfinite(g).all()
                values[name].append(dict(ms=ms,repeat=repeat,**meta))
        print('Final timing case',wi+1,'/',len(obslist),flush=True)
    def stats(v):return dict(median_ms=float(np.median(v)),p90_ms=float(np.quantile(v,.9)),mean_ms=float(np.mean(v)))
    methods={name:dict(**stats([x['ms'] for x in vals]),per_snr={f'{snr:g}':stats([x['ms'] for x in vals if x['snr']==snr]) for snr in SNRS},samples=vals) for name,vals in values.items()}
    learned=[x['ms'] for name in models for x in values[name]]
    methods['learned_mean']=dict(**stats(learned),per_snr={f'{snr:g}':stats([x['ms'] for name in models for x in values[name] if x['snr']==snr]) for snr in SNRS})
    overshoot={name:dict(cap_ms=selection['caps_ms'][name],actual_p90_ms=methods[name]['p90_ms'],
       pooled_cap_exceeded=methods[name]['p90_ms']>selection['caps_ms'][name],
       per_snr_cap_exceeded={f'{snr:g}':methods[name]['per_snr'][f'{snr:g}']['p90_ms']>selection['caps_ms'][name] for snr in SNRS}) for name in CAP_NAMES}
    write_json(OUT/'final_timing.json',dict(selection_sha256=freeze,methods=methods,cap_verification=overshoot,
      no_reselection_after_test=True,completed_unix=time.time(),timing_worlds=len(obslist),repeats=3))
    print('Final timing complete',flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['calibrate','develop','test','final-timing']);args=parser.parse_args()
    thread_setup()
    {'calibrate':calibrate,'develop':develop,'test':test,'final-timing':final_timing}[args.phase]()

if __name__=='__main__':main()
