"""Independent verification of paired starting-estimate ablation artifacts.

Verification-only source is hashed into each record. Audit PASS verifies the
protocol and arithmetic; cap attainment and statistical outcomes are separate.
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key]='1'
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from tcom_followups_20261017.common import sha,write_json
from . import study
ROOT=study.OUT/'audit'
REGIONS=('all','weak_warm100','strong_high','strong_transition','ordinary','weak','rsr5','rsr12','rsr20',
    'all_L2','all_L4','all_random3to7','weak_warm100_L2','weak_warm100_L4','weak_warm100_random3to7')
CONTRASTS=dict(adaptive_vs_global=('global_mean','adaptive_mean'),adaptive_vs_classical=('classical','adaptive_mean'),
    adaptive_vs_spectral=('spectral','adaptive_mean'),adaptive_vs_em10=('em10','adaptive_mean'),
    adaptive_vs_latest=('latest_mean','adaptive_mean'),adaptive_vs_em100=('em100','adaptive_mean'),
    adaptive_vs_compact=('compact_sparse','adaptive_mean'),adaptive_vs_optshrink=('optshrink','adaptive_mean'),
    budget_adaptive_vs_global=('budget_global_mean','budget_adaptive_mean'),
    budget_adaptive_vs_classical=('budget_classical','budget_adaptive_mean'),
    budget_adaptive_vs_adaptive=('adaptive_mean','budget_adaptive_mean'))


def close(actual,expected,tolerance=1e-12):
    np.testing.assert_allclose(actual,expected,atol=tolerance,rtol=0)


def record(name,**values):
    result=dict(passed=True,audit_source_sha256=sha(__file__),**values)
    write_json(ROOT/f'{name}.json',result);return result


def completed(directory):
    return sorted(p for p in Path(directory).glob('*.npz') if '.tmp.' not in p.name)


def belongs(region,snr,rsr,regime):
    scopes=dict(all=True,weak_warm100=rsr==5 and snr in (37.5,40.,42.5,45.,50.,60.),
        strong_high=rsr in (12,20) and snr in (40.,42.5,45.,50.,60.),
        strong_transition=rsr in (12,20) and snr in (32.5,35.,37.5),
        ordinary=snr in (0.,5.,10.,15.,20.),weak=rsr==5)
    if region in scopes:return scopes[region]
    if region in ('rsr5','rsr12','rsr20'):return rsr==int(region[3:])
    for prefix in ('all','weak_warm100'):
        if region.startswith(prefix+'_'):return scopes[prefix] and regime==region[len(prefix)+1:]
    raise KeyError(region)


def stats(rows):
    a=np.array([r['ms'] for r in rows]);assert len(a)>0 and np.isfinite(a).all() and (a>0).all()
    return dict(p90_ms=float(np.quantile(a,.9)),median_ms=float(np.median(a)),mean_ms=float(a.mean()),calls=len(a))


def verify_stats(rows,expected):
    for key,value in stats(rows).items():close(expected[key],value)


def _summary_dataset(raw=False):
    policy,digest=study.check_selection();stem='raw_start' if raw else 'nmse'
    path=study.OUT/f'{stem}_summary.json';report=json.loads(path.read_text())
    assert report['selection_sha256']==digest and report['protocol_sha256']==policy['protocol_sha256']
    assert report['raw_start_diagnostic']==raw
    names=tuple(study.RAW_NAMES if raw else study.NAMES)
    groups={name:members for name,members in study.GROUPS.items() if all(n in names for n in members)}
    methods=list(names)+list(groups);index={n:i for i,n in enumerate(methods)}
    contrasts={k:v for k,v in CONTRASTS.items() if all(n in methods for n in v)}
    assert report['method_names']==methods and report['groups']=={k:list(v) for k,v in groups.items()}
    assert report['contrasts']=={k:list(v) for k,v in contrasts.items()}
    def expand(v):return np.concatenate([v,np.stack([v[[index[n] for n in members]].mean(0) for members in groups.values()])])
    rng=np.random.default_rng(2026102019)
    full_weights={r:rng.multinomial(128,np.full(128,1/128),size=20000).T.astype(float) for r,_ in study.REGIMES}
    weights={r:w[:,:32] for r,w in full_weights.items()}
    rows=report['cells'];assert len(rows)==153
    points=[];boots=[];errors=[];energies=[];maximum=0.;full_cell_checks=[]
    prediction_key='raw_estimates' if raw else 'estimates';error_key='raw_errors' if raw else 'errors';name_key='raw_names' if raw else 'method_names'
    for row in rows:
        cellpath=study.OUT/'test/cells'/f"{row['cell']}.npz";a=study.load(cellpath);assert sha(cellpath)==report['test_hashes'][cellpath.name]
        assert str(a['selection_sha256'])==digest
        np.testing.assert_array_equal(a[name_key],names)
        err=np.sum(np.abs(a[prediction_key]-a['truth_g'][None])**2,axis=(-2,-1));energy=np.sum(np.abs(a['truth_g'])**2,axis=(-2,-1))
        np.testing.assert_array_equal(err,a[error_key]);np.testing.assert_array_equal(energy,a['truth_energy'])
        point=expand(10*np.log10(err.sum(-1)/energy.sum()));boot=expand(10*np.log10((err@weights[row['regime']])/(energy@weights[row['regime']])[None]))
        nm=err/energy[None];p95,p99=(expand(np.quantile(nm,q,axis=1)) for q in (.95,.99));severe=expand((nm>1).mean(1))
        for name,i in index.items():
            for key,value in dict(nmse_db=point[i],case_nmse_p95=p95[i],case_nmse_p99=p99[i],severe_rate=severe[i]).items():close(row['methods'][name][key],value)
            maximum=max(maximum,abs(point[i]-row['methods'][name]['nmse_db']))
        for key,(reference,candidate) in contrasts.items():close(row['contrasts'][key]['gain_db'],point[index[reference]]-point[index[candidate]])
        if (row['rsr'],row['snr']) in ((5.,0.),(12.,40.)):
            all_boot=expand(10*np.log10((err@full_weights[row['regime']])/(energy@full_weights[row['regime']])[None]))
            for name,i in index.items():close(row['methods'][name]['conditional_ci95_db'],np.quantile(all_boot[i],[.025,.975]),1e-11)
            for key,(reference,candidate) in contrasts.items():close(row['contrasts'][key]['pointwise_ci95_db'],np.quantile(all_boot[index[reference]]-all_boot[index[candidate]],[.025,.975]),1e-11)
            full_cell_checks.append(row['cell'])
        points.append(point);boots.append(boot);errors.append(err);energies.append(energy)
    points=np.stack(points);boots=np.stack(boots);stored=study.load(study.OUT/f'{stem}_bootstrap.npz')
    np.testing.assert_array_equal(stored['method_names'],methods);np.testing.assert_array_equal(stored['regions'],REGIONS)
    np.testing.assert_array_equal(stored['cell_names'],[r['cell'] for r in rows]);close(stored['cell_scores'],points)
    booterror=0.;counts={}
    for ri,region in enumerate(REGIONS):
        keep=np.array([belongs(region,r['snr'],r['rsr'],r['regime']) for r in rows]);counts[region]=int(keep.sum())
        aggregate=report['aggregates'][region];assert aggregate['cells']==keep.sum() and aggregate['condition_cases']==keep.sum()*128
        point=points[keep].mean(0);err=np.concatenate([e for e,k in zip(errors,keep) if k],1);energy=np.concatenate([e for e,k in zip(energies,keep) if k]);nm=err/energy[None]
        pooled=expand(10*np.log10(err.sum(-1)/energy.sum()));p95,p99=(expand(np.quantile(nm,q,axis=1)) for q in (.95,.99));severe=expand((nm>1).mean(1))
        full=stored['aggregate_bootstraps'][ri];independent=boots[keep].mean(0)
        error=float(np.abs(full[:,:32]-independent).max());booterror=max(booterror,error);close(full[:,:32],independent,1e-11)
        for name,i in index.items():
            for key,value in dict(nmse_db=point[i],pooled_ratio_of_sums_db=pooled[i],case_nmse_p95=p95[i],case_nmse_p99=p99[i],severe_rate=severe[i]).items():close(aggregate['methods'][name][key],value)
            close(aggregate['methods'][name]['conditional_ci95_db'],np.quantile(full[i],[.025,.975]))
        for key,(reference,candidate) in contrasts.items():
            item=aggregate['contrasts'][key];d=full[index[reference]]-full[index[candidate]]
            close(item['gain_db'],point[index[reference]]-point[index[candidate]]);close(item['pointwise_ci95_db'],np.quantile(d,[.025,.975]))
            primary=not raw and region=='all' and key in ('adaptive_vs_global','adaptive_vs_classical')
            assert ('bonferroni4_interval_db' in item)==primary
            if primary:close(item['bonferroni4_interval_db'],np.quantile(d,[.05/8,1-.05/8]))
            for seed in study.SEEDS:
                left=f'{reference[:-5]}_{seed}' if reference.endswith('_mean') else reference
                right=f'{candidate[:-5]}_{seed}' if candidate.endswith('_mean') else candidate
                close(item['seed_specific_gains_db'][str(seed)],point[index[left]]-point[index[right]])
    assert counts==dict(all=153,weak_warm100=18,strong_high=30,strong_transition=18,ordinary=45,weak=51,rsr5=51,rsr12=51,rsr20=51,all_L2=51,all_L4=51,all_random3to7=51,weak_warm100_L2=6,weak_warm100_L4=6,weak_warm100_random3to7=6)
    assert report['numerical_tie_tolerance_db']==1e-10
    assert report['source_sha256']==sha(Path(__file__).with_name('analysis.py'))
    assert report['bootstrap_index_clusters']==384 and report['unique_test_anchors']==1152 and report['condition_cases']==153*128
    assert report['bootstrap']['samples']==20000 and report['bootstrap']['seed']==2026102019 and len(full_cell_checks)==6
    for key in contrasts:
        item=report['negative_cells'][key];gains=[r['contrasts'][key]['gain_db'] for r in rows]
        assert item['negative_cells']==[r['cell'] for r in rows if r['contrasts'][key]['gain_db'] < -1e-10]
        assert item['pointwise95_harm_cells']==[r['cell'] for r in rows if r['contrasts'][key]['pointwise_ci95_db'][1] < -1e-10]
        assert item['weak_warm100_negative_cells']==[r['cell'] for r in rows if belongs('weak_warm100',r['snr'],r['rsr'],r['regime']) and r['contrasts'][key]['gain_db'] < -1e-10]
        worst=int(np.argmin(gains));assert item['worst_cell']==rows[worst]['cell'];close(item['worst_gain_db'],gains[worst])
    return dict(cells=153,region_counts=counts,maximum_point_error=maximum,
        independently_recomputed_bootstrap_replicates=32,maximum_bootstrap_error=booterror,
        selected_cells_with_all20000_intervals_recomputed=full_cell_checks,
        all_points_pooled_tails_fit_gains_intervals_adverse_lists_verified=True,
        summary_sha256=sha(path),bootstrap_sha256=sha(study.OUT/f'{stem}_bootstrap.npz'))


def summary():
    policy,digest=study.check_selection();report=json.loads((study.OUT/'nmse_summary.json').read_text())
    assert report['raw_start_summary_sha256']==sha(study.OUT/'raw_start_summary.json')
    assert report['raw_start_bootstrap_sha256']==sha(study.OUT/'raw_start_bootstrap.npz')
    for window in (1,2):
        name=f'timing_window{window}.json';timing_record=json.loads((study.OUT/name).read_text())
        assert report['timing_hashes'][name]==sha(study.OUT/name)
        assert report['timing_windows'][f'window{window}']['methods']==timing_record['methods']
        assert report['timing_windows'][f'window{window}']['fixed_cap_ms']==10.74
        if window==1:assert report['timing']==timing_record['methods']
    raw=_summary_dataset(True);main=_summary_dataset(False)
    return record('summary',selection_sha256=digest,main=main,raw_starts=raw,
        summary_sha256=sha(study.OUT/'nmse_summary.json'),raw_start_summary_sha256=sha(study.OUT/'raw_start_summary.json'),
        final_timing_bindings_verified=True,only_two_declared_primary_nmse_intervals=True,
        scope='Conditional fixed-model/control bootstrap. All aggregate quantiles checked;32 draws reconstructed from raw predictions, all20k in six independently selected cells per dataset.')


@torch.no_grad()
def test():
    policy,digest=study.check_selection();paths=completed(study.OUT/'test/cells');anchorpaths=completed(study.OUT/'test/anchors')
    assert len(paths)==153 and len(anchorpaths)==9
    est=study.engine();anchors={};anchorhash={};hashes={};replays=0;maximum=0.;conditions=[]
    raw_replays=0;raw_maximum=0.;continuation_checks=0;continuation_maximum=0.
    from tcom_pilot_em_20261018 import continuation
    from tcom_incumbent_guard_20261019 import shrink
    for path in anchorpaths:
        a=study.load(path);regime=str(a['regime']);L=dict(study.REGIMES)[regime];ref=float(a['rsr'])
        assert int(a['seed'])==2026102012 and int(a['holdout_seed'])==2026102015
        assert str(a['selection_sha256'])==digest and str(a['protocol_sha256'])==policy['protocol_sha256']
        assert len(a['g'])==128 and path.stat().st_mtime>=policy['frozen_unix']
        regenerated=study.make_anchor(2026102012,[0,127],L,ref,truth=True)
        for key in ('g','s','b','paths','indices','local_indices','anchor_variance','unit_noise'):
            np.testing.assert_array_equal(a[key][[0,127]],regenerated[key])
        np.testing.assert_array_equal(a['z_by_snr'][:,[0,127]],regenerated['z_by_snr'])
        anchors[path.stem]=a;anchorhash[path.name]=sha(path)
    for path in paths:
        a=study.load(path);ref,snr,regime=float(a['rsr']),float(a['snr']),str(a['regime'])
        assert str(a['selection_sha256'])==digest and str(a['protocol_sha256'])==policy['protocol_sha256']
        assert int(a['seed'])==2026102012 and path.stat().st_mtime>=policy['frozen_unix']
        conditions.append((regime,ref,snr));anchor=anchors[f'{regime}_rsr{ref:g}'];obs=study.observed(anchor,snr)
        for key,other in (('truth_g','g'),('paths','paths'),('indices','indices'),('local_indices','local_indices')):
            np.testing.assert_array_equal(a[key],anchor[other])
        for key,value in zip(('z','s','b','sigma2'),obs.args()):np.testing.assert_array_equal(a[key],value.numpy())
        energy=np.sum(np.abs(a['truth_g'])**2,axis=(-2,-1));np.testing.assert_array_equal(energy,a['truth_energy'])
        for names,predkey,errkey,namekey in ((study.NAMES,'estimates','errors','method_names'),(study.RAW_NAMES,'raw_estimates','raw_errors','raw_names')):
            np.testing.assert_array_equal(a[namekey],names);pred=a[predkey]
            assert pred.shape==(len(names),128,32,3) and np.isfinite(pred).all() and pred.dtype==np.complex128
            err=np.sum(np.abs(pred-a['truth_g'][None])**2,axis=(-2,-1));np.testing.assert_array_equal(err,a[errkey])
            if predkey=='estimates' and 'nmse' in a:np.testing.assert_array_equal(a['nmse'],err/energy[None])
        last=obs.batch(slice(127,128))
        cfg=study.filter_config(ref,snr)
        for i,name in enumerate(study.RAW_NAMES):
            replay=study.start_estimate(last,name,policy,est).numpy()
            error=float(np.abs(replay-a['raw_estimates'][i,127:128]).max());raw_maximum=max(raw_maximum,error);raw_replays+=1
            np.testing.assert_allclose(replay,a['raw_estimates'][i,127:128],atol=1e-10,rtol=1e-10)
            raw=torch.from_numpy(a['raw_estimates'][i,127:128])
            count=policy['budget_steps'][study.family(name)]
            trajectory,_=continuation.trajectory(last,raw,(100,count))
            for target,steps in ((name,100),('budget_'+name,count)):
                expected=shrink.apply(last,trajectory[steps],c=cfg['c'],beta=cfg['beta']).numpy()
                stored=a['estimates'][study.NAMES.index(target),127:128]
                error=float(np.abs(expected-stored).max());continuation_maximum=max(continuation_maximum,error);continuation_checks+=1
                np.testing.assert_allclose(expected,stored,atol=1e-10,rtol=1e-10)
            if count==100:np.testing.assert_array_equal(a['estimates'][study.NAMES.index(name)],a['estimates'][study.NAMES.index('budget_'+name)])
        for i,name in enumerate(study.NAMES):
            replay=study.estimate_method(obs.batch(slice(127,128)),name,policy,est).numpy()
            error=float(np.abs(replay-a['estimates'][i,127:128]).max());maximum=max(maximum,error);replays+=1
            np.testing.assert_allclose(replay,a['estimates'][i,127:128],atol=1e-10,rtol=1e-10)
        hashes[path.name]=sha(path)
    expected={(r,float(ref),float(s)) for r,_ in study.REGIMES for ref in study.RSRS for s in study.SNRS}
    assert len(conditions)==len(set(conditions))==153 and set(conditions)==expected
    return record('test',selection_sha256=digest,cells=153,raw_starts_per_cell=9,final_methods_per_cell=24,
        independently_regenerated_channel_reference_anchors=18,all_observations_and_raw_final_metric_arrays_exact=True,
        independently_replayed_final_anchor_pipelines=replays,maximum_replay_error=maximum,
        independently_replayed_raw_starts=raw_replays,maximum_raw_replay_error=raw_maximum,
        common_and_selected_budget_tails_verified=continuation_checks,maximum_continuation_error=continuation_maximum,
        all_starts_forced_everywhere_same_filter_no_bypass=True,
        test_anchor_hashes=anchorhash,test_prediction_hashes=hashes)



def _grid(rows,indices):
    expected={(regime,float(ref),float(snr),index) for regime,_ in study.REGIMES for ref in study.RSRS for snr in study.SNRS for index in indices}
    actual=[(r['regime'],float(r['rsr']),float(r['snr']),int(r['index'])) for r in rows]
    assert len(actual)==len(set(actual))==len(expected) and set(actual)==expected
    return actual


@torch.no_grad()
def selection():
    policy,digest=study.check_selection();protocol=json.loads((study.OUT/'protocol.json').read_text())
    assert tuple(protocol['names'])==tuple(study.NAMES) and tuple(protocol['raw_names'])==tuple(study.RAW_NAMES)
    assert len(study.NAMES)==24 and len(study.RAW_NAMES)==9
    assert protocol['main']['continuation_steps']==study.COMMON_STEPS==100
    assert protocol['timing']['cap_ms']==10.74 and protocol['budget']['calibration_cap_ms']==10.
    assert protocol['test_seed']==2026102012 and protocol['holdout_seed']==2026102015 and protocol['bootstrap_seed']==2026102019
    assert protocol['primary']['contrasts']==['adaptive_vs_global','adaptive_vs_classical'] and protocol['primary']['regions']==['all']
    train_path=study.OUT/'training_protocol.json';train=json.loads(train_path.read_text())
    assert protocol['training_protocol_sha256']==sha(train_path)
    config=train['config'];assert config['parameters']==30 and config['epochs']==20 and config['batch_size']==32
    assert config['train_cases']==16384 and config['val_cases']==2048 and config['context_P']==30 and config['heldout_P']==9
    for category in ('source_hashes','input_hashes'):
        for name,value in train[category].items():assert sha(study.REPO/name)==value,name
    trained={}
    for seed in study.SEEDS:
        folder=study.OUT/f'checkpoints/global/seed_{seed}'
        latest=torch.load(folder/'em_global_latest.pt',map_location='cpu',weights_only=False)
        best=torch.load(folder/'em_global_best.pt',map_location='cpu',weights_only=False)
        assert latest['epoch']==20 and latest['optimizer_updates']==10240
        assert latest['protocol_sha256']==best['protocol_sha256']==sha(train_path)
        assert len(latest['history'])==20
        adaptive=json.loads((study.REPO/f'results/tcom_pilot_em_20261018/checkpoints/context30/seed_{seed}/em_adaptive_history.json').read_text())
        for left,right in zip(latest['history'],adaptive):assert left['evidence']==right['evidence']
        values=[latest['initial_val30']]+[row['val30_nll'] for row in latest['history']]
        selected=int(np.argmin(values));assert best['epoch']==selected and best['best_score']==min(values)
        assert sum(x.numel() for x in best['model'].values())==30
        trained[str(seed)]=dict(epochs=20,updates=10240,selected_epoch=selected,minimum_val30=min(values),
            best_sha256=sha(folder/'em_global_best.pt'),all20_masks_and_orders_match_adaptive=True)
    # Reconstruct the same final-filter table without invoking the runner's
    # filter lookup. This rule is common to all starts and never selects a start.
    from tcom_incumbent_guard_20261019 import shrink
    filters={}
    for ref in study.RSRS:
        for snr in study.SNRS:
            choice=study.latest.LATEST_POLICY['repair_choices'][f'{ref:g}'][f'{snr:g}']['choice']
            expected=dict(shrink.CONFIGS[choice]) if choice in shrink.CONFIGS else dict(c=0.,beta=0.)
            assert study.filter_config(ref,snr)==expected
            filters[f'rsr{ref:g}_snr{snr:g}']=expected
    classical_path=study.OUT/'classical_selection.json';classical=json.loads(classical_path.read_text())
    assert classical['protocol_sha256']==policy['protocol_sha256']
    grid=[dict(kind='relative',relative_tau=t,beta=b) for t in (.01,.03,.1,.3,1.) for b in (.1,.5,1.)]
    grid += [dict(kind='noise',c=c,beta=b) for c in (.5,1.,2.,4.,8.) for b in (.1,.5,1.)]
    assert len(classical['candidates'])==30 and list(study.CLASSICAL_GRID)==grid
    for row,cfg in zip(classical['candidates'],grid):
        assert row['config']==cfg
        values=np.asarray(row['stratum_scores']);assert values.shape==(126,) and np.isfinite(values).all()
        close(row['score'],values.mean())
    winner=min(range(30),key=lambda i:classical['candidates'][i]['score'])
    assert classical['selected_index']==winner and classical['config']==grid[winner] and policy['classical_config']==grid[winner]
    manifest_path=study.REPO/'results/tcom_pilot_em_20261018/training_data/val/manifest.json'
    assert classical['validation_manifest_sha256']==sha(manifest_path)
    manifest=json.loads(manifest_path.read_text())
    assert set(manifest['file_hashes'])=={'z.npy','s.npy','b.npy','sigma2.npy','conditions.npy'}
    for name,value in manifest['file_hashes'].items():assert sha(manifest_path.parent/name)==value
    # Independent likelihood evaluation for the selected and an unselected
    # boundary configuration, on every validation channel and all126 strata.
    from sahure.data import Observations
    from scipy.special import i0e
    bank,conditions=study.load_set('val');_,context,held=study.paired_split(bank,2026101810)
    context=Observations(*(x.to(torch.complex128 if i in (1,2) else torch.float64) for i,x in enumerate(context.args())))
    held=Observations(*(x.to(torch.complex128 if i in (1,2) else torch.float64) for i,x in enumerate(held.args())))
    ids=np.asarray(conditions);counts=np.bincount(ids,minlength=126);assert np.all(counts>0)
    # Reproduce each selected checkpoint's score in the declared training
    # precision using independent stratum aggregation.
    from sahure.train import load_model
    from sahure.math import nll as torch_nll
    _,fit32,held32=study.paired_split(bank,2026101810)
    selected_score_error=0.
    for seed in study.SEEDS:
        model,_=load_model(study.OUT/f'checkpoints/global/seed_{seed}/em_global_best.pt',dtype=torch.float32)
        scores=[]
        for start in range(0,len(fit32),32):
            section=slice(start,start+32);a=fit32.batch(section);h=held32.batch(section)
            g=model(*a.args());scores.extend(torch_nll(h.z,g@h.s+h.b,h.sigma2).mean((-2,-1)).double().numpy())
        score=float((np.bincount(ids,weights=scores,minlength=126)/counts).mean())
        error=abs(score-trained[str(seed)]['minimum_val30']);selected_score_error=max(selected_score_error,error)
        close(score,trained[str(seed)]['minimum_val30'],1e-10)
        trained[str(seed)]['independently_reproduced_val30']=score
    est=study.engine();rechecked=sorted({winner,29 if winner!=29 else 0});maximum=0.
    for ci in rechecked:
        values=[]
        for start in range(0,len(context),32):
            section=slice(start,start+32);obs=context.batch(section);h=held.batch(section)
            g=est['starts'].start('classical',obs,config=grid[ci]).numpy()
            z,s,b,v=(x.numpy() for x in h.args());amplitude=np.abs(g@s+b);x=2*z*amplitude/v[:,None,None]
            losses=(z-amplitude)**2/v[:,None,None]-np.log(i0e(x))
            values.extend(losses.mean((-2,-1)))
        strata=np.bincount(ids,weights=values,minlength=126)/counts
        error=float(np.abs(strata-classical['candidates'][ci]['stratum_scores']).max());maximum=max(maximum,error)
        close(strata,classical['candidates'][ci]['stratum_scores'],1e-9)
    budget_path=study.OUT/'budget_selection.json';budget=json.loads(budget_path.read_text())
    assert budget['protocol_sha256']==policy['protocol_sha256']
    assert set(budget['raw_samples'])==set(study.RAW_NAMES)
    gridsteps=(100,125,150,175,200);assert tuple(study.BUDGET_GRID)==gridsteps
    raw=budget['raw_samples'];callcount=0
    for name,counts_ in raw.items():
        assert set(counts_)=={str(c) for c in gridsteps}
        for count,rows in counts_.items():_grid(rows,[400]);stats(rows);callcount+=len(rows)
    assert callcount==9*5*153
    chosen={};fallback={}
    for group in ('adaptive','global','classical','spectral','em10'):
        members=[name for name in study.RAW_NAMES if (name.rsplit('_',1)[0] if name.startswith(('adaptive_','global_')) else name)==group]
        eligible=[]
        for count in gridsteps:
            rows=[r for member in members for r in raw[member][str(count)]]
            weak=[r for r in rows if r['rsr']==5 and r['snr']>=37.5]
            wholep=stats(rows)['p90_ms'];weakp=stats(weak)['p90_ms'];ok=wholep<=10. and weakp<=10.
            item=budget['scores'][group][str(count)]
            close(item['p90_ms'],wholep);close(item['weak_p90_ms'],weakp);assert item['eligible']==ok
            if ok:eligible.append(count)
        chosen[group]=max(eligible) if eligible else 100;fallback[group]=not bool(eligible)
    assert chosen==budget['budget_steps']==policy['budget_steps']
    assert fallback==budget['fallback_to_100']==policy['budget_fallback_to_100']
    assert policy['frozen_unix']>=max(classical['completed_unix'],budget['completed_unix'])
    assert classical['completed_unix']>=protocol['frozen_unix'] and budget['completed_unix']>=classical['completed_unix']
    return record('selection',selection_sha256=digest,protocol_sha256=policy['protocol_sha256'],
        classical_candidates=30,equal_validation_strata=126,classical_selected_index=winner,
        independently_recomputed_classical_configs=rechecked,maximum_independent_nll_error=maximum,
        training_and_validation_source_input_hashes_verified=True,paired_global_training=trained,
        maximum_selected_checkpoint_validation_error=selected_score_error,
        common_filter_configs=filters,common_tail_steps=100,
        budget_steps=chosen,budget_fallback_no_eligible=fallback,calibration_raw_calls=callcount,calibration_cap_ms=10.,
        full_grid_and_weak_quantiles_and_largest_eligible_rule_verified=True,
        classical_selection_sha256=sha(classical_path),budget_selection_sha256=sha(budget_path),
        scope='Classical selection optimizes raw-start heldout NLL; common postprocessing is fixed. Timing-only tail selection does not establish an accuracy improvement.')


def timing():
    policy,digest=study.check_selection();records={};hashes={};previous_end=0.;total=0
    for window in (1,2):
        path=study.OUT/f'timing_window{window}.json';data=json.loads(path.read_text());raw=data['raw_samples']
        assert data['window']==window and data['selection_sha256']==digest and data['protocol_sha256']==policy['protocol_sha256']
        indices=[500,501,502] if window==1 else [600,601,602]
        assert set(raw)==set(study.NAMES) and data['indices']==indices and data['worlds']==459
        assert data['started_unix']>=max(policy['frozen_unix'],previous_end) and data['completed_unix']>=data['started_unix']
        previous_end=data['completed_unix'];assert data['environment']['torch_threads']==1
        if window==1:
            times=[p.stat().st_mtime for p in completed(study.OUT/'test/anchors')]
            assert len(times)==9 and min(times)>=data['completed_unix']
        else:
            generated=completed(study.OUT/'test/cells');assert len(generated)==153
            assert max(p.stat().st_mtime for p in generated)<=data['started_unix']
            assert (study.OUT/'ber/summary.json').stat().st_mtime<=data['started_unix']
        worlds=[(r,float(ref),float(s),idx) for idx in indices for ref in study.RSRS for r,_ in study.REGIMES for s in study.SNRS]
        rng=np.random.default_rng(np.random.SeedSequence([2026102014,window]));order=rng.permutation(459)
        expected_order=[worlds[i] for i in order]
        for name,rows in raw.items():assert _grid(rows,indices)==expected_order
        expanded=dict(raw)
        expanded.update({n:[r for member in members for r in raw[member]] for n,members in study.GROUPS.items()})
        assert set(expanded)==set(data['methods'])==set(data['cap_verification'])
        target={};conditional={}
        for name,rows in expanded.items():
            item=data['methods'][name];verify_stats(rows,item)
            verify_stats([r for r in rows if r['rsr']==5 and r['snr']>=37.5],item['weak_warm100'])
            assert len(item['by_condition'])==51
            for ref in study.RSRS:
                for snr in study.SNRS:verify_stats([r for r in rows if r['rsr']==ref and r['snr']==snr],item['by_condition'][f'rsr{ref:g}_snr{snr:g}'])
            cap=data['cap_verification'][name];close(cap['p90_ms'],item['p90_ms']);assert cap['passed']==(item['p90_ms']<=10.74)
            target[name]=cap['passed'];conditional[name]=[key for key,v in item['by_condition'].items() if v['p90_ms']>10.74]
        calls=sum(map(len,raw.values()));assert calls==24*459;total+=calls
        records[f'window{window}']=dict(raw_calls=calls,cap_passed_by_method=target,conditional_overruns=conditional,
            p90_ms={n:v['p90_ms'] for n,v in data['methods'].items()},weak_p90_ms={n:v['weak_warm100']['p90_ms'] for n,v in data['methods'].items()})
        hashes[path.name]=sha(path)
    assert total==22032
    both={name:all(r['cap_passed_by_method'][name] for r in records.values()) for name in records['window1']['cap_passed_by_method']}
    return record('timing',selection_sha256=digest,timing_hashes=hashes,windows=records,literal_cap_ms=10.74,
        target_met_in_both_windows_by_method=both,raw_calls=total,every_grid_quantile_and_cap_flag_verified=True,
        scope='Correctness audit and observed cap attainment are distinct. All arms retained, including cap failures; no retries, normalization, or population latency guarantee.')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=('selection','test','summary','timing'));args=parser.parse_args()
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    print(json.dumps(globals()[args.phase]()),flush=True)
