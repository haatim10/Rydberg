"""Fresh observation-only calibration and test of frozen SAHURE hard bypass.

Generation, policy calibration, testing and isolated timing are separate phases.
No neural parameters are trained. Truth enters test scoring only.
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
from sahure.data import Observations,split_observations
from sahure.math import nll
from sahure_expansion.data import world
from tcom_followups_20261017.common import REPO,sha,write_json,frozen_inputs,effective_trial_id

OUT=REPO/'results/tcom_runtime_bypass_20261017/bypass'
DEV_SEED=2026101761;TEST_SEED=2026101762;TIMING_SEED=2026101764
DEV_COUNT=24;TEST_COUNT=64
SNRS=(10.,15.,20.,22.5,25.,27.5,30.,32.5,35.,40.,45.,50.,60.)
RSRS=(5.,12.,20.)
REGIMES=(('L2',2),('L4',4),('random3to7',None))
CUTOFFS=(20.,22.5,25.,27.5,30.,32.5,35.,40.,45.,50.,60.)
SEEDS=(9101,9102,9103)
NAMES=tuple([f'learned_{s}' for s in SEEDS]+[f'off_{s}' for s in SEEDS]+[f'fast_bypass_{s}' for s in SEEDS]+['em100']+[f'em100_bypass_{s}' for s in SEEDS])


def engine():
    from .bypass import BypassEstimator
    return BypassEstimator()


def save(path,**arrays):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp.npz');np.savez_compressed(temp,**arrays);temp.replace(path)


def load(path):
    with np.load(path,allow_pickle=False) as a:return {k:a[k] for k in a.files}


def cellname(regime,rsr,snr):return f'{regime}_rsr{rsr:g}_snr{snr:g}'


def freeze_protocol():
    originals=frozen_inputs()
    for k in ('development_seed','test_seed','data_seed'):originals.pop(k,None)
    sources=[Path(__file__),REPO/'tcom_runtime_bypass_20261017/bypass.py',
      REPO/'tcom_runtime_bypass_20261017/sparse_fast.py',REPO/'tcom_followups_20261017/common.py',
      REPO/'sahure/data.py',REPO/'sahure/train.py']
    spec=dict(version=1,name='Validation-calibrated hard bypass for frozen SAHURE',
      development_seed=DEV_SEED,test_seed=TEST_SEED,timing_seed=TIMING_SEED,
      N=32,K=3,P=30,snrs=SNRS,rsrs=RSRS,regimes=REGIMES,
      development_anchors_per_regime_reference=DEV_COUNT,test_anchors_per_regime_reference=TEST_COUNT,
      cutoffs=CUTOFFS,dispatch='Receiver nominal SNR proxy10log10(K/supplied_variance), reference proxy from known B. Three reference bins5/12/20. One cutoff per reference, shared by three fits and all path regimes. Strict greater-than cutoff with1e-10dB tolerance. Always preserve original for proxy<=20.',
      model='All original checkpoints frozen. Same-step fallback deep-copies identical weights and steps, disables only Hankel filtering and unused controller computation. Separate EM100 fallback uses the exact optimized unpenalized EM implementation.',
      policy_selection='Separate same_steps and em100 cutoffs minimize equal-cell mean heldout RicianNLL over all declared SNRs and path strata for each reference. Three original fits averaged as losses, not predictions. Ties prefer larger cutoff. Cutoff60 means no bypass through60dB, but still switches above60 if deployed.',
      fit='Only21 context pilots are available for initialization, features and all updates; remaining9 used only to score predictions. Refit with all30 for final test. No channel truth, true path count, test errors or previous-study crossover enters policy selection.',
      masks='Fixed per-anchor random21/9 split, seed20261017610+100*regime_index+reference_index; same split across SNR and all branches/candidates.',
      coupling='Generate separate10dB anchor at each reference/regime. Retain G,S,B and standardized W/sqrt(v10); rescale the same noise across SNR. Reference draws differ but random path-count vectors recur at same local index, so bootstrap shares local indices across SNR and reference within regime.',
      freeze='Freeze source/checkpoint hashes before development. Freeze policy and full development loss-table hashes before generating any test anchor. No changes after observing test results.',
      test_methods=NAMES,score='Per-cell ratio of summed squared channel errors to summed truth energy, converted to dB. Fit-group averages three scores, never predictions. Primary aggregate averages cells with SNR>20 equally. Also show each cell, ordinary range, pooled NMSE, severeNMSE>1, p95/p99 per-caseNMSE, and negative-gain cells.',
      uncertainty='3000 paired whole-anchor bootstrap replicates, shared case indices across SNR and reference within each regime and across every method/fit. Conditional on frozen fits and selected policies. Two aggregate comparisons vs original use Bonferroni simultaneous95 intervals; per-cell intervals are pointwise95.',
      ordinary_range='Require bitwise identity for every selected prediction at true and nominal SNR10/15/20. Verify an actual engine dispatch against reused branch predictions for one case per cell and all fits/fallbacks.',
      sensitivity=dict(actual_snrs=[20.,30.,60.],rsr=12.,regime='random3to7',supplied_variance_offsets_db=[-3.,0.,3.],
        policy='Reuse fresh64 test anchors and identical measurements; fixed policy, no calibration updates. All original/off/fast-bypass/EM100/EM100-bypass branches. Separate diagnostic only; known-noise primary assumption remains explicit.'),
      timing=dict(seed=TIMING_SEED,local_index=100,snrs=[10.,20.,30.,45.,60.],rsrs=RSRS,regimes=REGIMES,repeats=3,
        policy='Isolated batch-one CPU, initialization and receiver-only lookup included, one selected branch only, no parallel forward. Randomized method order. Models preloaded; report median/p90 pooled and by SNR/reference/selected branch.'),
      requested_thread_environment={k:os.environ[k] for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS')},
      source_hashes={str(p.relative_to(REPO)):sha(p) for p in sources},frozen_original_inputs=originals)
    target=OUT/'protocol.json'
    if target.exists():
        old=json.loads(target.read_text());old.pop('frozen_unix',None)
        if old!=json.loads(json.dumps(spec)):raise RuntimeError('frozen protocol changed')
    else:write_json(target,dict(**spec,frozen_unix=time.time()))
    return sha(target)


def verify_protocol():
    path=OUT/'protocol.json';record=json.loads(path.read_text())
    for name,h in record['source_hashes'].items():assert sha(REPO/name)==h,name
    for name,h in record['frozen_original_inputs']['hashes'].items():assert sha(REPO/name)==h,name
    return sha(path)


def make_anchor(seed,ids,L,rsr):
    rows=[world(effective_trial_id(i,L),seed,N=32,K=3,P=30,snr=10.,rsr=rsr,L=L) for i in ids]
    a=dict(g=np.stack([w.G for w,p in rows]),s=np.stack([w.S for w,p in rows]),b=np.stack([w.B for w,p in rows]),
      unit_noise=np.stack([w.W/np.sqrt(w.sigma2) for w,p in rows]),anchor_z=np.stack([w.Z for w,p in rows]),
      anchor_variance=np.array([w.sigma2 for w,p in rows]),paths=np.array([p for w,p in rows]),
      indices=np.array([effective_trial_id(i,L) for i in ids]),seed=np.array(seed),rsr=np.array(rsr))
    np.testing.assert_allclose(np.abs(a['g']@a['s']+a['b']+a['unit_noise']*np.sqrt(a['anchor_variance'])[:,None,None]),a['anchor_z'],rtol=0,atol=1e-13)
    return a


def anchors(split,regime,L,rsr,digest):
    path=OUT/split/'anchors'/f'{regime}_rsr{rsr:g}.npz'
    if path.exists():
        a=load(path);assert str(a['protocol_sha256'])==digest
        if split=='test':assert str(a['selection_sha256'])==sha(OUT/'policy.json')
        return a
    if split=='test':
        selection,freeze=check_selection();seed=TEST_SEED;count=TEST_COUNT
    else:freeze='development';seed=DEV_SEED;count=DEV_COUNT
    a=make_anchor(seed,range(count),L,rsr);a.update(protocol_sha256=np.array(digest),selection_sha256=np.array(freeze),regime=np.array(regime))
    save(path,**a);return a


def observed(a,snr,variance_offset_db=0.):
    v=a['anchor_variance']*10**((10.-snr)/10.)
    z=np.abs(a['g']@a['s']+a['b']+a['unit_noise']*np.sqrt(v)[:,None,None])
    return Observations(torch.tensor(z,dtype=torch.float64),torch.tensor(a['s'],dtype=torch.complex128),
      torch.tensor(a['b'],dtype=torch.complex128),torch.tensor(v*10**(variance_offset_db/10.),dtype=torch.float64))


@torch.no_grad()
def develop():
    digest=freeze_protocol();est=engine();records=[]
    for ri,rsr in enumerate(RSRS):
        for pi,(regime,L) in enumerate(REGIMES):
            a=anchors('development',regime,L,rsr,digest)
            for snr in SNRS:
                path=OUT/'development'/'losses'/f'{cellname(regime,rsr,snr)}.npz'
                if path.exists():assert str(load(path)['protocol_sha256'])==digest;continue
                context,holdout=split_observations(observed(a,snr),20261017610+100*pi+ri)
                assert context.z.shape[-1]==21 and holdout.z.shape[-1]==9
                fits=[est.original(context,f'learned_{s}') for s in SEEDS]
                off=[est.unfiltered(context,f'learned_{s}') for s in SEEDS]
                em=est.em100(context)
                losses=np.stack([nll(holdout.z,g@holdout.s+holdout.b,holdout.sigma2).mean((-2,-1)).numpy() for g in fits+off+[em]])
                assert np.isfinite(losses).all()
                masks=np.stack([est.masks(context,{f'{r:g}':c for r in RSRS})['bypass'].numpy() for c in CUTOFFS])
                save(path,losses=losses,bypass_masks=masks,cutoffs=np.array(CUTOFFS),
                  names=np.array([f'learned_{s}' for s in SEEDS]+[f'off_{s}' for s in SEEDS]+['em100']),
                  regime=np.array(regime),rsr=np.array(rsr),snr=np.array(snr),indices=a['indices'],protocol_sha256=np.array(digest))
                print('Development loss complete',path.stem,flush=True)
    selections={kind:{} for kind in ('same_steps','em100')};tables={kind:{} for kind in selections}
    lossfiles=sorted((OUT/'development'/'losses').glob('*.npz'))
    assert len(lossfiles)==len(RSRS)*len(REGIMES)*len(SNRS)
    for rsr in RSRS:
        cells=[load(p) for p in lossfiles if float(load(p)['rsr'])==rsr]
        for kind in selections:
            scores=[]
            for ci,cutoff in enumerate(CUTOFFS):
                percell=[]
                for a in cells:
                    fallback=a['losses'][3:6] if kind=='same_steps' else np.repeat(a['losses'][6:7],3,axis=0)
                    loss=np.where(a['bypass_masks'][ci][None],fallback,a['losses'][:3])
                    percell.append(dict(regime=str(a['regime']),snr=float(a['snr']),mean_nll=float(loss.mean())))
                scores.append(dict(cutoff_db=cutoff,mean_nll=float(np.mean([x['mean_nll'] for x in percell])),cells=percell))
            best=min(scores,key=lambda x:(x['mean_nll'],-x['cutoff_db']))
            selections[kind][f'{rsr:g}']=best['cutoff_db'];tables[kind][f'{rsr:g}']=scores
    record=dict(protocol_sha256=digest,policies=selections,validation_tables=tables,
      loss_table_hashes={p.name:sha(p) for p in lossfiles},development_anchor_hashes={p.name:sha(p) for p in sorted((OUT/'development'/'anchors').glob('*.npz'))})
    path=OUT/'policy.json'
    if path.exists():
        old=json.loads(path.read_text());old.pop('frozen_unix',None);assert old==record
    else:
        assert not list((OUT/'test').glob('**/*.npz')),'test existed before policy freeze'
        write_json(path,dict(**record,frozen_unix=time.time()))
    print('Policies frozen',selections,flush=True)
    write_json(OUT/'status.json',dict(phase='development_complete',selection_sha256=sha(path),policies=selections))


def check_selection():
    digest=verify_protocol();path=OUT/'policy.json';p=json.loads(path.read_text());assert p['protocol_sha256']==digest
    for name,h in p['loss_table_hashes'].items():assert sha(OUT/'development'/'losses'/name)==h,name
    for name,h in p['development_anchor_hashes'].items():assert sha(OUT/'development'/'anchors'/name)==h,name
    return p,sha(path)


@torch.no_grad()
def base_predictions(est,obs):
    original=torch.stack([est.original(obs,f'learned_{s}') for s in SEEDS])
    off=torch.stack([est.unfiltered(obs,f'learned_{s}') for s in SEEDS])
    em=est.em100(obs)
    return original,off,em


def assemble(est,obs,policy,original,off,em):
    mf=est.masks(obs,policy['same_steps'])['bypass'];me=est.masks(obs,policy['em100'])['bypass']
    fast=torch.where(mf[None,:,None,None],off,original)
    slow=torch.where(me[None,:,None,None],em[None],original)
    return torch.cat([original,off,fast,em[None],slow]),mf,me


@torch.no_grad()
def test():
    policy,freeze=check_selection();digest=policy['protocol_sha256'];est=engine();checks=[]
    for rsr in RSRS:
        for regime,L in REGIMES:
            a=anchors('test',regime,L,rsr,digest)
            for snr in SNRS:
                name=cellname(regime,rsr,snr);path=OUT/'test'/'cells'/f'{name}.npz'
                if path.exists():assert str(load(path)['selection_sha256'])==freeze;continue
                obs=observed(a,snr);original,off,em=base_predictions(est,obs)
                pred,mf,me=assemble(est,obs,policy['policies'],original,off,em)
                if snr<=20:
                    assert torch.equal(pred[6:9],original) and torch.equal(pred[10:13],original)
                dispatch_max=0.
                for i,s in enumerate(SEEDS):
                    for kind,offset in [('same_steps',6),('em100',10)]:
                        actual=est.estimate(obs.batch(slice(0,1)),f'learned_{s}',policy['policies'][kind],fallback=kind)
                        diff=float(torch.max(torch.abs(actual-pred[offset+i,0:1])));assert diff<1e-10
                        dispatch_max=max(dispatch_max,diff)
                arr=pred.numpy();assert np.isfinite(arr).all()
                errors=np.sum(abs(arr-a['g'][None])**2,axis=(-2,-1));energy=np.sum(abs(a['g'])**2,axis=(-2,-1))
                save(path,method_names=np.array(NAMES),estimates=arr,errors=errors,truth_energy=energy,nmse=errors/energy[None],
                  bypass_fast=mf.numpy(),bypass_em100=me.numpy(),snr=np.array(snr),rsr=np.array(rsr),regime=np.array(regime),
                  indices=a['indices'],paths=a['paths'],selection_sha256=np.array(freeze),protocol_sha256=np.array(digest),
                  actual_dispatch_max_abs_error=np.array(dispatch_max),ordinary_range_exact_identity=np.array(snr<=20))
                print('Test complete',name,flush=True)
                write_json(OUT/'status.json',dict(phase='testing',last_cell=name,completed_cells=len(list((OUT/'test'/'cells').glob('*.npz')))))
    write_json(OUT/'status.json',dict(phase='test_complete',cells=len(SNRS)*len(RSRS)*len(REGIMES),selection_sha256=freeze))


@torch.no_grad()
def timing():
    policy,freeze=check_selection();est=engine();rows=[]
    for rsr in RSRS:
        for regime,L in REGIMES:
            a=make_anchor(TIMING_SEED,[100],L,rsr)
            for snr in (10.,20.,30.,45.,60.):rows.append((dict(snr=snr,rsr=rsr,regime=regime,index=100),observed(a,snr)))
    def call(name,obs):
        if name=='em100':return est.em100(obs)
        seed=int(name.rsplit('_',1)[1]);model=f'learned_{seed}'
        if name.startswith('learned_'):return est.original(obs,model)
        if name.startswith('off_'):return est.unfiltered(obs,model)
        kind='em100' if name.startswith('em100_bypass_') else 'same_steps'
        return est.estimate(obs,model,policy['policies'][kind],fallback=kind)
    for name in NAMES:call(name,rows[0][1])
    rng=np.random.default_rng(20261017640);samples={name:[] for name in NAMES}
    for wi,(meta,obs) in enumerate(rows):
        fastmask=bool(est.masks(obs,policy['policies']['same_steps'])['bypass'][0]);emmask=bool(est.masks(obs,policy['policies']['em100'])['bypass'][0])
        for repeat in range(3):
            for ni in rng.permutation(len(NAMES)):
                name=NAMES[ni];start=time.perf_counter_ns();g=call(name,obs);elapsed=(time.perf_counter_ns()-start)/1e6
                assert torch.isfinite(g).all()
                branch='original'
                if name.startswith('off_') or (name.startswith('fast_bypass_') and fastmask):branch='off'
                if name=='em100' or (name.startswith('em100_bypass_') and emmask):branch='em100'
                samples[name].append(dict(ms=elapsed,repeat=repeat,branch=branch,**meta))
        print('Timing world',wi+1,'/',len(rows),flush=True)
    def stats(values):return dict(median_ms=float(np.median(values)),p90_ms=float(np.quantile(values,.9)),mean_ms=float(np.mean(values)),calls=len(values))
    groups=dict(original_mean=[f'learned_{s}' for s in SEEDS],off_mean=[f'off_{s}' for s in SEEDS],fast_bypass_mean=[f'fast_bypass_{s}' for s in SEEDS],em100_bypass_mean=[f'em100_bypass_{s}' for s in SEEDS])
    values=dict(samples)
    for name,parts in groups.items():values[name]=[x for part in parts for x in samples[part]]
    methods={name:dict(**stats([x['ms'] for x in v]),by_branch={b:stats([x['ms'] for x in v if x['branch']==b]) for b in sorted(set(x['branch'] for x in v))},
      by_condition={f'rsr{r:g}_snr{s:g}':stats([x['ms'] for x in v if x['rsr']==r and x['snr']==s]) for r in RSRS for s in (10.,20.,30.,45.,60.)}) for name,v in values.items()}
    write_json(OUT/'timing.json',dict(selection_sha256=freeze,completed_unix=time.time(),worlds=len(rows),repeats=3,methods=methods,raw_samples=samples))
    print('Timing complete',flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['develop','test','timing']);args=parser.parse_args()
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    {'develop':develop,'test':test,'timing':timing}[args.phase]()

if __name__=='__main__':main()
