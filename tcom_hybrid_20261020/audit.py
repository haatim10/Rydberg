"""Independent verification of heldout-selected hybrid artifacts.

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
REGIONS=('all','high_all','transition_all','weak_warm100','strong_high','strong_transition','ordinary','weak','rsr5','rsr12','rsr20',
    'all_L2','all_L4','all_random3to7','weak_warm100_L2','weak_warm100_L4','weak_warm100_random3to7')
PRIMARY=('hybrid_vs_latest','hybrid_vs_compact')
CONTRASTS=dict(hybrid_vs_latest=('latest_mean','hybrid_mean'),hybrid_vs_compact=('compact_sparse','hybrid_mean'),
    hybrid_vs_adaptive40=('adaptive40_mean','hybrid_mean'),hybrid_vs_global40=('global40_mean','hybrid_mean'),
    hybrid_vs_classical40=('classical40','hybrid_mean'),hybrid_vs_em100=('em100','hybrid_mean'),
    hybrid_vs_optshrink=('optshrink','hybrid_mean'),hybrid_vs_noem=('hybrid0_mean','hybrid_mean'),
    adaptive_vs_global=('global_mean','adaptive_mean'))


def close(actual,expected,tolerance=1e-12):
    np.testing.assert_allclose(actual,expected,atol=tolerance,rtol=0)


def record(name,**values):
    result=dict(passed=True,audit_source_sha256=sha(__file__),**values)
    write_json(ROOT/f'{name}.json',result);return result


def completed(directory):
    return sorted(p for p in Path(directory).glob('*.npz') if '.tmp.' not in p.name)


def belongs(region,snr,rsr,regime):
    scopes=dict(all=True,high_all=snr>=40,transition_all=20<snr<40,weak_warm100=rsr==5 and snr in (37.5,40.,42.5,45.,50.,60.),
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
    rng=np.random.default_rng(2026102058)
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
        pooled_linear=expand(err.sum(-1)/energy.sum());pooled=10*np.log10(pooled_linear);p95,p99=(expand(np.quantile(nm,q,axis=1)) for q in (.95,.99));severe=expand((nm>1).mean(1))
        full=stored['aggregate_bootstraps'][ri];independent=boots[keep].mean(0)
        error=float(np.abs(full[:,:32]-independent).max());booterror=max(booterror,error);close(full[:,:32],independent,1e-11)
        for name,i in index.items():
            for key,value in dict(nmse_db=point[i],pooled_ratio_of_sums_linear=pooled_linear[i],pooled_ratio_of_sums_db=pooled[i],case_nmse_p95=p95[i],case_nmse_p99=p99[i],severe_rate=severe[i]).items():close(aggregate['methods'][name][key],value)
            close(aggregate['methods'][name]['conditional_ci95_db'],np.quantile(full[i],[.025,.975]))
        for key,(reference,candidate) in contrasts.items():
            item=aggregate['contrasts'][key];d=full[index[reference]]-full[index[candidate]]
            close(item['gain_db'],point[index[reference]]-point[index[candidate]]);close(item['pointwise_ci95_db'],np.quantile(d,[.025,.975]))
            primary=not raw and region=='all' and key in ('hybrid_vs_latest','hybrid_vs_compact')
            assert ('bonferroni4_interval_db' in item)==primary
            if primary:close(item['bonferroni4_interval_db'],np.quantile(d,[.05/8,1-.05/8]))
            for seed in study.SEEDS:
                left=f'{reference[:-5]}_{seed}' if reference.endswith('_mean') else reference
                right=f'{candidate[:-5]}_{seed}' if candidate.endswith('_mean') else candidate
                close(item['seed_specific_gains_db'][str(seed)],point[index[left]]-point[index[right]])
    assert counts==dict(all=153,high_all=45,transition_all=63,weak_warm100=18,strong_high=30,strong_transition=18,ordinary=45,weak=51,rsr5=51,rsr12=51,rsr20=51,all_L2=51,all_L4=51,all_random3to7=51,weak_warm100_L2=6,weak_warm100_L4=6,weak_warm100_random3to7=6)
    assert report['numerical_tie_tolerance_db']==1e-10
    assert report['source_sha256']==sha(Path(__file__).with_name('analysis.py'))
    assert report['bootstrap_index_clusters']==384 and report['unique_test_anchors']==1152 and report['condition_cases']==153*128
    assert report['bootstrap']['samples']==20000 and report['bootstrap']['seed']==2026102058 and len(full_cell_checks)==6
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
    raw_replays=0;raw_maximum=0.;ordinary=0;zero_tail=0;high=0;continuation_checks=0;continuation_error=0.
    from tcom_pilot_em_20261018 import continuation
    protocol=json.loads((study.OUT/'protocol.json').read_text())
    for path in anchorpaths:
        a=study.load(path);regime=str(a['regime']);L=dict(study.REGIMES)[regime];ref=float(a['rsr'])
        assert int(a['seed'])==study.TEST_SEED
        assert str(a['selection_sha256'])==digest and str(a['protocol_sha256'])==policy['protocol_sha256']
        assert len(a['g'])==128 and path.stat().st_mtime>=policy['frozen_unix']
        regenerated=study.make_anchor(study.TEST_SEED,[0,127],L,ref,truth=True)
        for key in ('g','s','b','paths','indices','local_indices','anchor_variance','unit_noise'):
            np.testing.assert_array_equal(a[key][[0,127]],regenerated[key])
        np.testing.assert_array_equal(a['z_by_snr'][:,[0,127]],regenerated['z_by_snr'])
        anchors[path.stem]=a;anchorhash[path.name]=sha(path)
    for path in paths:
        a=study.load(path);ref,snr,regime=float(a['rsr']),float(a['snr']),str(a['regime'])
        assert str(a['selection_sha256'])==digest and str(a['protocol_sha256'])==policy['protocol_sha256']
        assert int(a['seed'])==study.TEST_SEED and path.stat().st_mtime>=policy['frozen_unix']
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
        for i,name in enumerate(study.RAW_NAMES):
            replay=study.start_estimate(last,name,policy,est).numpy()
            error=float(np.abs(replay-a['raw_estimates'][i,127:128]).max());raw_maximum=max(raw_maximum,error);raw_replays+=1
            np.testing.assert_allclose(replay,a['raw_estimates'][i,127:128],atol=1e-10,rtol=1e-10)
        for i,name in enumerate(study.NAMES):
            replay=study.estimate_method(last,name,policy,est).numpy()
            error=float(np.abs(replay-a['estimates'][i,127:128]).max());maximum=max(maximum,error);replays+=1
            np.testing.assert_allclose(replay,a['estimates'][i,127:128],atol=1e-10,rtol=1e-10)
        family='adaptive' if snr<=20 or snr<=policy['crossover_db'] else 'global'
        entry=policy['tails'][f'{ref:g}'][f'{snr:g}']
        steps=0 if snr<=20 else int(entry['steps'])
        assert steps in (0,10,20,40) and entry['family']==family
        if snr>=40:assert family=='global';high+=1
        for seed in study.SEEDS:
            raw=a['raw_estimates'][study.RAW_NAMES.index(f'{family}_{seed}')]
            start=torch.from_numpy(raw[127:128])
            zero=raw[127:128] if snr<=20 else study.finish(last,start).numpy()
            full=raw[127:128] if snr<=20 else study.finish(last,continuation.estimate(last,start,steps) if steps else start).numpy()
            for target,value in ((f'hybrid_{seed}',full),(f'hybrid0_{seed}',zero)):
                stored=a['estimates'][study.NAMES.index(target),127:128]
                continuation_error=max(continuation_error,float(np.abs(value-stored).max()));continuation_checks+=1
                np.testing.assert_allclose(value,stored,atol=1e-10,rtol=1e-10)
            for fixed in ('adaptive','global'):
                base=torch.from_numpy(a['raw_estimates'][study.RAW_NAMES.index(f'{fixed}_{seed}'),127:128])
                value=study.finish(last,continuation.estimate(last,base,40)).numpy()
                stored=a['estimates'][study.NAMES.index(f'{fixed}40_{seed}'),127:128]
                continuation_error=max(continuation_error,float(np.abs(value-stored).max()));continuation_checks+=1
                np.testing.assert_allclose(value,stored,atol=1e-10,rtol=1e-10)
            if steps==0:
                np.testing.assert_array_equal(a['estimates'][study.NAMES.index(f'hybrid_{seed}')],a['estimates'][study.NAMES.index(f'hybrid0_{seed}')])
            if snr<=20:
                for target in ('hybrid','hybrid0','latest'):
                    np.testing.assert_array_equal(a['estimates'][study.NAMES.index(f'{target}_{seed}')],raw)
        if snr<=20:ordinary+=1
        if steps==0:zero_tail+=1
        hashes[path.name]=sha(path)
    expected={(r,float(ref),float(s)) for r,_ in study.REGIMES for ref in study.RSRS for s in study.SNRS}
    assert len(conditions)==len(set(conditions))==153 and set(conditions)==expected and ordinary==45 and high==45
    return record('test',selection_sha256=digest,cells=153,raw_starts_per_cell=6,final_methods_per_cell=19,
        independently_regenerated_channel_reference_anchors=18,all_observations_and_raw_final_metric_arrays_exact=True,
        independently_replayed_final_anchor_pipelines=replays,maximum_replay_error=maximum,
        independently_replayed_raw_starts=raw_replays,maximum_raw_replay_error=raw_maximum,
        exact_ordinary_hybrid_noem_latest_rawadaptive_cells=ordinary,high_global_family_cells=high,
        exact_zero_tail_hybrid_noem_cells=zero_tail,independently_replayed_continuations=continuation_checks,maximum_continuation_error=continuation_error,
        test_anchor_hashes=anchorhash,test_prediction_hashes=hashes)


def _grid(rows,indices):
    expected={(regime,float(ref),float(snr),index) for regime,_ in study.REGIMES for ref in study.RSRS for snr in study.SNRS for index in indices}
    actual=[(r['regime'],float(r['rsr']),float(r['snr']),int(r['index'])) for r in rows]
    assert len(actual)==len(set(actual))==len(expected) and set(actual)==expected
    return actual


@torch.no_grad()
def selection():
    from scipy.special import i0e
    from scipy.stats import t as student_t
    from tcom_pilot_em_20261018 import continuation
    from tcom_incumbent_guard_20261019 import risk_data
    policy,digest=study.check_selection()
    protocol=json.loads((study.OUT/'protocol.json').read_text())
    assert len(study.NAMES)==19 and len(study.RAW_NAMES)==6
    assert protocol['test_seed']==2026102054 and protocol['bootstrap_seed']==2026102058
    assert protocol['dev_seeds']=={'A':2026102051,'B':2026102052,'C':2026102053}
    assert protocol['holdout_seed']==2026102057 and protocol['timing_seed']==2026102056
    assert protocol['primary']['contrasts']==['hybrid_vs_latest','hybrid_vs_compact']
    assert protocol['primary']['regions']==['all']
    assert protocol['timing']['cap_ms']==10.74 and protocol['timing']['calibration_cap_ms']==10.
    assert tuple(protocol['names'])==study.NAMES and tuple(protocol['raw_names'])==study.RAW_NAMES
    assert protocol['dev_cases_per_reference_regime']==32
    assert not completed(study.OUT/'test/cells') and not completed(study.OUT/'test/anchors')
    artifacts={name:json.loads((study.OUT/name).read_text()) for name in
        ('runtime_calibration.json','crossover_selection.json','tail_proposals.json','tail_confirmation.json')}
    cal,A,B,C=(artifacts[n] for n in artifacts)
    for name,item in artifacts.items():
        assert item['protocol_sha256']==policy['protocol_sha256']
        assert sha(study.OUT/name)==policy['selection_evidence'][name]
        assert item['completed_unix']>=item['started_unix']>=protocol['frozen_unix']
    assert A['started_unix']>=cal['completed_unix']
    assert B['started_unix']>=A['completed_unix']
    assert C['started_unix']>=B['completed_unix']
    assert policy['frozen_unix']>=C['completed_unix']
    assert B['crossover_sha256']==sha(study.OUT/'crossover_selection.json')
    assert B['runtime_calibration_sha256']==sha(study.OUT/'runtime_calibration.json')
    assert C['proposal_sha256']==sha(study.OUT/'tail_proposals.json')

    # Reconstruct the exact randomized calibration order, separately from its
    # family-level quantiles and eligibility flags.
    rows=cal['records'];assert len(rows)==153*2*3*4
    worlds=[(r,float(ref),float(s),401) for ref in study.RSRS for r,_ in study.REGIMES for s in study.SNRS]
    menu=[(f,seed,t) for f in ('adaptive','global') for seed in study.SEEDS for t in (0,10,20,40)]
    rng=np.random.default_rng(np.random.SeedSequence([2026102056,0]));expected=[]
    for wi in rng.permutation(153):
        for mi in rng.permutation(24):expected.append(worlds[wi]+menu[mi])
    observed_order=[(r['regime'],float(r['rsr']),float(r['snr']),r['index'],r['family'],r['seed'],r['steps']) for r in rows]
    assert observed_order==expected
    eligibility={}
    for family in ('adaptive','global'):
        eligibility[family]={}
        for tail in (0,10,20,40):
            selected=[r for r in rows if r['family']==family and r['steps']==tail]
            item=cal['scores'][family][str(tail)]
            for seed in study.SEEDS:_grid([r for r in selected if r['seed']==seed],[401])
            verify_stats(selected,item)
            weak=stats([r for r in selected if r['rsr']==5 and r['snr']>=37.5])['p90_ms']
            close(item['weak_p90_ms'],weak)
            eligible=stats(selected)['p90_ms']<=10. and weak<=10.
            assert item['eligible']==eligible;eligibility[family][tail]=eligible

    est=study.engine();banks={};losses={};hashes={};replays=0;maximum=0.
    forbidden={'g','G','truth_g','unit_noise','paths','mu'}
    prior_ends={'A':cal['completed_unix'],'B':A['completed_unix'],'C':B['completed_unix']}
    for bank,record_ in (('A',A),('B',B),('C',C)):
        expected_snr=[s for s in study.SNRS if 20<s<40] if bank=='A' else [s for s in study.SNRS if s>20]
        bankpaths=completed(study.OUT/f'validation/{bank}/anchors')
        cellpaths=completed(study.OUT/f'validation/{bank}/cells')
        assert len(bankpaths)==9 and len(cellpaths)==9*len(expected_snr)
        expected_hashes={str(p.relative_to(study.OUT)):sha(p) for p in bankpaths+cellpaths}
        assert expected_hashes==record_['validation_hashes'];hashes.update(expected_hashes)
        for path in bankpaths:
            anchor=study.load(path);regime=str(anchor['regime']);ref=float(anchor['rsr'])
            assert not forbidden.intersection(anchor)
            assert int(anchor['seed'])==study.DEV_SEEDS[bank] and int(anchor['holdout_seed'])==2026102057
            assert str(anchor['protocol_sha256'])==policy['protocol_sha256']
            assert path.stat().st_mtime>=prior_ends[bank]
            assert anchor['s'].shape==(32,3,39) and anchor['b'].shape==(32,32,39)
            assert anchor['z_by_snr'].shape==(17,32,32,39)
            regenerated=study.make_anchor(study.DEV_SEEDS[bank],[0,31],dict(study.REGIMES)[regime],ref,truth=False)
            for key in ('s','b','indices','local_indices','anchor_variance'):
                np.testing.assert_array_equal(anchor[key][[0,31]],regenerated[key])
            np.testing.assert_array_equal(anchor['z_by_snr'][:,[0,31]],regenerated['z_by_snr'])
            banks[bank,regime,ref]=anchor
        for path in cellpaths:
            item=study.load(path);regime=str(item['regime']);ref=float(item['rsr']);snr=float(item['snr'])
            assert not forbidden.intersection(item) and snr in expected_snr
            assert str(item['protocol_sha256'])==policy['protocol_sha256']
            assert path.stat().st_mtime>=prior_ends[bank]
            arr=item['losses'];assert arr.shape==((4,3,32) if bank=='B' else (2,3,32)) and np.isfinite(arr).all()
            losses[bank,regime,ref,snr]=arr
            if bank=='A':
                np.testing.assert_array_equal(item['families'],('adaptive','global'))
                options=[(family,0) for family in ('adaptive','global')]
            else:
                family='adaptive' if snr<=A['crossover_db'] else 'global'
                assert str(item['family'])==family
                steps=(0,10,20,40) if bank=='B' else (0,B['proposals'][f'{ref:g}'][f'{snr:g}']['proposed_steps'])
                np.testing.assert_array_equal(item['steps'],steps)
                options=[(family,step) for step in steps]
            if snr not in (32.5,50.):continue
            # Independently replay a prespecified endpoint channel; heldout
            # magnitudes enter the scorer only after the estimate is finished.
            obs,held=risk_data.fit_holdout(study.observed(banks[bank,regime,ref],snr,P=39))
            obs=obs.batch(slice(31,32));held=held.batch(slice(31,32))
            for fi,seed in enumerate(study.SEEDS):
                starts={family:est['starts'].start(family,obs,seed=seed) for family,_ in options}
                for oi,(family,steps) in enumerate(options):
                    initial=starts[family]
                    pred=study.finish(obs,continuation.estimate(obs,initial,steps) if steps else initial)
                    z,s,b,v=(x.numpy() for x in held.args());mu=pred.numpy()@s+b
                    amplitude=np.abs(mu);variance=v[:,None,None]
                    score=np.mean((z-amplitude)**2/variance-np.log(i0e(2*z*amplitude/variance)))
                    error=abs(float(score)-float(arr[oi,fi,31]));maximum=max(maximum,error);replays+=1
                    np.testing.assert_allclose(score,arr[oi,fi,31],atol=1e-9,rtol=1e-11)
    for regime,_ in study.REGIMES:
        for ref in study.RSRS:
            for x,y in (('A','B'),('A','C'),('B','C')):
                assert not np.array_equal(banks[x,regime,ref]['s'],banks[y,regime,ref]['s'])
                assert not np.array_equal(banks[x,regime,ref]['z_by_snr'],banks[y,regime,ref]['z_by_snr'])

    def argmin(scores,keys):
        smallest=min(scores[k] for k in keys)
        return min(k for k in keys if scores[k]<=smallest+1e-12)

    def bound(delta,alpha,actual):
        assert delta.shape==(32,)
        mean=float(delta.mean());se=float(delta.std(ddof=1)/np.sqrt(32))
        critical=float(student_t.ppf(1-alpha,31));lower=mean-critical*se
        for key,value in dict(mean_improvement=mean,standard_error=se,critical=critical,alpha=alpha,lower_bound=lower).items():
            close(actual[key],value)
        passed=mean>1e-12 and lower>0.
        assert actual['passed']==passed
        return passed

    scores={}
    for cut in (20.,25.,30.,35.):
        selected=[losses['A',regime,ref,snr][0 if snr<=cut else 1].mean()
                  for ref in study.RSRS for regime,_ in study.REGIMES for snr in study.SNRS if 20<snr<40]
        scores[cut]=float(np.mean(selected));close(A['scores'][f'{cut:g}'],scores[cut])
    chosen=argmin(scores,scores)
    assert A['cells']==63 and A['crossover_db']==chosen==policy['crossover_db']
    proposals={};confirmed={};guards_checked=0
    for ref in study.RSRS:
        rkey=f'{ref:g}'
        for snr in study.SNRS:
            skey=f'{snr:g}'
            if snr<=20:
                assert policy['tails'][rkey][skey]==dict(family='adaptive',steps=0)
                continue
            family='adaptive' if snr<=chosen else 'global'
            arrays=[losses['B',regime,ref,snr].mean(1) for regime,_ in study.REGIMES]
            means={step:float(np.mean([a[i] for a in arrays])) for i,step in enumerate((0,10,20,40))}
            eligible=[step for step in (0,10,20,40) if eligibility[family][step]]
            if 0 not in eligible:eligible=[0]+eligible
            best=argmin(means,eligible)
            item=B['proposals'][rkey][skey]
            assert item['family']==family and item['eligible_steps']==eligible and item['best_steps']==best
            for step,value in means.items():close(item['scores'][str(step)],value)
            checks=[bound(a[0]-a[(0,10,20,40).index(best)],.05/3,item['guards'][regime]) for (regime,_),a in zip(study.REGIMES,arrays)]
            proposed=best if best>0 and all(checks) else 0;guards_checked+=3
            assert item['proposed_steps']==proposed;proposals[f'{rkey}/{skey}']=proposed
            confirmation=C['confirmed'][rkey][skey]
            assert confirmation['family']==family and confirmation['proposed_steps']==proposed
            checks=[]
            for regime,_ in study.REGIMES:
                arr=losses['C',regime,ref,snr].mean(1)
                if proposed==0:np.testing.assert_array_equal(arr[0],arr[1])
                checks.append(bound(arr[0]-arr[1],.05/108,confirmation['guards'][regime]))
            accepted=proposed>0 and all(checks);steps=proposed if accepted else 0;guards_checked+=3
            assert confirmation['accepted']==accepted and confirmation['steps']==steps
            assert policy['tails'][rkey][skey]==dict(family=family,steps=steps)
            confirmed[f'{rkey}/{skey}']=steps
    assert len(proposals)==len(confirmed)==36 and guards_checked==216
    return record('selection',selection_sha256=digest,protocol_sha256=policy['protocol_sha256'],
        selection_evidence_hashes=policy['selection_evidence'],runtime_calibration_calls=len(rows),
        runtime_calibration_quantiles_and_eligibility_verified=True,validation_anchor_hashes={k:v for k,v in hashes.items() if '/anchors/' in k},
        validation_loss_hashes={k:v for k,v in hashes.items() if '/cells/' in k},
        independent_observable_validation_banks=3,independently_regenerated_validation_anchors=54,
        independently_replayed_likelihood_scores=replays,maximum_likelihood_replay_error=maximum,
        no_truth_exports_and_disjoint30plus9_pilots_verified=True,crossover_db=chosen,
        equal_cell_equal_fit_selection_verified=True,approximate_student_t_guards_verified=guards_checked,
        proposals=proposals,confirmed=confirmed,only_B_selected_candidate_confirmed_no_runnerup=True,
        ordinary_exact_rule_verified=True,test_artifacts_absent_at_selection_audit=True,
        scope='Verification of the prespecified approximate likelihood screen; no population familywise guarantee or NMSE/BER improvement guarantee.')


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
        worlds=[(r,float(ref),float(s),idx) for ref in study.RSRS for r,_ in study.REGIMES for idx in indices for s in study.SNRS]
        rng=np.random.default_rng(np.random.SeedSequence([study.TIMING_SEED,window]));order=rng.permutation(459)
        expected_order=[worlds[i] for i in order]
        for name,rows in raw.items():assert _grid(rows,indices)==expected_order
        expanded=dict(raw)
        expanded.update({n:[r for member in members for r in raw[member]] for n,members in study.GROUPS.items() if all(member in raw for member in members)})
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
        calls=sum(map(len,raw.values()));assert calls==19*459;total+=calls
        records[f'window{window}']=dict(raw_calls=calls,cap_passed_by_method=target,conditional_overruns=conditional,
            p90_ms={n:v['p90_ms'] for n,v in data['methods'].items()},weak_p90_ms={n:v['weak_warm100']['p90_ms'] for n,v in data['methods'].items()})
        hashes[path.name]=sha(path)
    assert total==17442
    both={name:all(r['cap_passed_by_method'][name] for r in records.values()) for name in records['window1']['cap_passed_by_method']}
    return record('timing',selection_sha256=digest,timing_hashes=hashes,windows=records,literal_cap_ms=10.74,
        target_met_in_both_windows_by_method=both,raw_calls=total,every_grid_quantile_and_cap_flag_verified=True,
        scope='Correctness audit and observed cap attainment are distinct. All arms retained, including cap failures; no retries, normalization, or population latency guarantee.')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=('selection','test','summary','timing'));args=parser.parse_args()
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    print(json.dumps(globals()[args.phase]()),flush=True)
