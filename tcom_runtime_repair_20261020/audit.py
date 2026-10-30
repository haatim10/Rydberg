"""Independent audits for an implementation-only, fixed-policy comparison.

Audit correctness is separate from the literal two-window runtime target.
No audit chooses a policy, tolerance, timing sample, or numerical precision.
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


def close(actual,expected,tolerance=1e-12):
    np.testing.assert_allclose(actual,expected,atol=tolerance,rtol=0)


def record(name,**values):
    result=dict(passed=True,audit_source_sha256=sha(__file__),**values)
    write_json(ROOT/f'{name}.json',result)
    return result


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


def chosen(policy,ref,snr):
    grid=min(study.SNRS,key=lambda s:(abs(s-snr),s))
    return 'original' if snr<=20+1e-10 else policy['repair_choices'][f'{ref:g}'][f'{grid:g}']['choice']


def selection():
    policy,digest=study.check_selection();p=json.loads((study.OUT/'protocol.json').read_text())
    assert policy['repair_choices']==study.CURRENT_REPAIR['repair_choices']
    assert policy['compact_choices']==study.CURRENT_REPAIR['compact_choices']
    assert p['timing']['cap_ms']==10.74 and p['timing']['windows']==2
    assert p['parity']['atol']==p['parity']['rtol']==1e-10
    assert p['test_seed']==2026102002 and p['bootstrap_seed']==2026102009
    assert p['test_cases_per_reference_regime']==128
    assert tuple(p['methods'])==tuple(study.NAMES) and len(study.NAMES)==20
    assert p['primary']['scopes']==['all153','weak_warm10018'] and p['primary']['contrast']=='repair_vs_current'
    for key in ('source_hashes','checkpoint_hashes'):
        for name,value in p[key].items():assert sha(study.REPO/name)==value
    smoke=json.loads((study.OUT/'smoke_parity.json').read_text());assert smoke['passed']
    assert p['smoke_parity_sha256']==sha(study.OUT/'smoke_parity.json')
    for key in ('source_hashes','checkpoint_hashes'):
        for name,value in smoke[key].items():assert sha(study.REPO/name)==value
    assert smoke['latest_policy_sha256']==sha(study.CURRENT_REPAIR_PATH)
    for ref in study.RSRS:
        for snr in study.SNRS:
            assert chosen(policy,ref,snr)==chosen(study.CURRENT_REPAIR,ref,snr)
    return record('selection',selection_sha256=digest,protocol_sha256=policy['protocol_sha256'],
        inherited_repair_bins=51,inherited_compact_bins=51,no_new_statistical_selection=True,
        source_checkpoint_smoke_and_ber_hashes_verified=True,literal_runtime_cap_ms=10.74,
        coefficient_parity_atol=1e-10,coefficient_parity_rtol=1e-10)


@torch.no_grad()
def test():
    policy,digest=study.check_selection();est=study.engine()
    paths=sorted((study.OUT/'test/cells').glob('*.npz'));dense=sorted((study.OUT/'boundary/cells').glob('*.npz'))
    assert len(paths)==153 and len(dense)==54
    anchorpaths=sorted((study.OUT/'test/anchors').glob('*.npz'));assert len(anchorpaths)==9
    anchors={};anchorhash={};hashes={};boundaryhash={};maximum=0.;fraction=0.;replaymax=0.;bitwise=0;replays=0
    for path in anchorpaths:
        a=study.load(path);ref=float(a['rsr']);regime=str(a['regime']);L=dict(study.REGIMES)[regime]
        assert int(a['seed'])==2026102002 and int(a['holdout_seed'])==2026102005
        assert str(a['selection_sha256'])==digest and str(a['protocol_sha256'])==policy['protocol_sha256']
        assert len(a['g'])==128 and path.stat().st_mtime>=policy['frozen_unix']
        regenerated=study.make_anchor(2026102002,[0,127],L,ref,truth=True)
        for key in ('g','s','b','paths','indices','local_indices','anchor_variance','unit_noise'):
            np.testing.assert_array_equal(a[key][[0,127]],regenerated[key])
        np.testing.assert_array_equal(a['z_by_snr'][:,[0,127]],regenerated['z_by_snr'])
        anchors[path.stem]=a;anchorhash[path.name]=sha(path)
    ordinary=weak=warm=0;conditionkeys=[]
    for isdense,files in ((False,paths),(True,dense)):
        for path in files:
            a=study.load(path);ref,snr,regime=float(a['rsr']),float(a['snr']),str(a['regime'])
            assert str(a['selection_sha256'])==digest and str(a['protocol_sha256'])==policy['protocol_sha256']
            assert int(a['seed'])==2026102002 and path.stat().st_mtime>=policy['frozen_unix']
            np.testing.assert_array_equal(a['method_names'],study.NAMES)
            pred=a['estimates'];assert pred.shape==(20,128,32,3) and np.isfinite(pred).all()
            assert snr in (study.BOUNDARY_SNRS if isdense else study.SNRS)
            if isdense:assert ref in (12.,20.)
            conditionkeys.append((isdense,regime,ref,snr))
            anchor=anchors[f'{regime}_rsr{ref:g}'];obs=study.observed(anchor,snr)
            for key,other in (('truth_g','g'),('paths','paths'),('indices','indices'),('local_indices','local_indices')):
                np.testing.assert_array_equal(a[key],anchor[other])
            for key,value in zip(('z','s','b','sigma2'),obs.args()):np.testing.assert_array_equal(a[key],value.numpy())
            err=np.sum(np.abs(pred-a['truth_g'][None])**2,axis=(-2,-1));energy=np.sum(np.abs(a['truth_g'])**2,axis=(-2,-1))
            np.testing.assert_array_equal(err,a['errors']);np.testing.assert_array_equal(energy,a['truth_energy'])
            np.testing.assert_array_equal(err/energy[None],a['nmse'])
            diff=np.abs(pred[12:15]-pred[9:12]);limit=1e-10+1e-10*np.abs(pred[9:12])
            assert np.all(diff<=limit);maximum=max(maximum,float(diff.max()));fraction=max(fraction,float((diff/limit).max()))
            close(a['native_canonical_max_abs_error'],diff.max(),0)
            assert bool(a['native_canonical_bitwise_equal'])==np.array_equal(pred[12:15],pred[9:12])
            bitwise+=sum(np.array_equal(pred[12+i],pred[9+i]) for i in range(3))
            old=snr>study.OLD_POLICY[f'{ref:g}']+1e-10 and snr>20+1e-10
            np.testing.assert_array_equal(a['old_bypass'],np.full(128,old))
            np.testing.assert_array_equal(pred[3:6],pred[15:16].repeat(3,axis=0) if old else pred[:3])
            np.testing.assert_array_equal(pred[17],pred[18])
            assert str(a['repair_choice'])==chosen(policy,ref,snr)
            assert str(a['prior_repair_choice'])==chosen(study.PRIOR_REPAIR,ref,snr)
            # Independently replay both canonical and native full pipelines on
            # the final anchor (the production harness checked the first).
            for family in ('current_repair','repair'):
                for seed in study.SEEDS:
                    name=f'{family}_{seed}';i=study.NAMES.index(name)
                    replay=study.estimate_method(obs.batch(slice(127,128)),name,policy,est).numpy()
                    error=float(np.abs(replay-pred[i,127:128]).max());replaymax=max(replaymax,error);replays+=1
                    np.testing.assert_allclose(replay,pred[i,127:128],atol=1e-10,rtol=1e-10)
            if not isdense:
                if snr<=20:
                    for i in (3,6,12):np.testing.assert_array_equal(pred[i:i+3],pred[:3])
                    ordinary+=1
                if ref==5:np.testing.assert_array_equal(pred[12:15],pred[6:9]);weak+=1
                if ref==5 and snr>=37.5:assert str(a['repair_choice'])=='warm100';warm+=1
            (boundaryhash if isdense else hashes)[path.name]=sha(path)
    expected={(False,r,float(ref),float(s)) for r,_ in study.REGIMES for ref in study.RSRS for s in study.SNRS}
    expected|={(True,r,float(ref),float(s)) for r,_ in study.REGIMES for ref in (12,20) for s in study.BOUNDARY_SNRS}
    assert len(set(conditionkeys))==207 and set(conditionkeys)==expected
    assert (ordinary,weak,warm)==(45,51,18)
    first=min(p.stat().st_mtime for p in anchorpaths+paths+dense)
    timing1=json.loads((study.OUT/'timing_window1.json').read_text())
    assert first>=timing1['completed_unix']
    return record('test',selection_sha256=digest,main_cells=153,boundary_cells=54,
        independently_regenerated_channel_reference_anchors=18,all_observations_and_metric_arrays_exact=True,
        native_canonical_coefficient_checks=207*3*128*32*3,maximum_absolute_error=maximum,
        maximum_tolerance_fraction=fraction,bitwise_equal_cell_model_pairs=bitwise,total_cell_model_pairs=621,
        independent_final_anchor_pipeline_replays=replays,maximum_replay_error=replaymax,
        exact_ordinary_native_identity_cells=ordinary,exact_weak_prior_identity_cells=weak,
        fixed_weak_warm100_cells=warm,all_fresh_artifacts_after_first_timing_window=True,
        test_anchor_hashes=anchorhash,test_prediction_hashes=hashes,boundary_prediction_hashes=boundaryhash)


def stats(rows):
    a=np.array([r['ms'] for r in rows]);assert len(a)>0 and np.isfinite(a).all() and (a>0).all()
    return dict(p90_ms=float(np.quantile(a,.9)),median_ms=float(np.median(a)),mean_ms=float(a.mean()),calls=len(a))


def verify_stats(rows,expected):
    for key,value in stats(rows).items():close(expected[key],value)


def timing():
    policy,digest=study.check_selection();records={};hashes={};previous_end=0.;total=0
    for window in (1,2):
        path=study.OUT/f'timing_window{window}.json';data=json.loads(path.read_text());raw=data['raw_samples']
        assert data['window']==window and data['selection_sha256']==digest and data['protocol_sha256']==policy['protocol_sha256']
        indices=[200,201,202] if window==1 else [300,301,302]
        names=tuple(study.NAMES) if window==1 else tuple(n for n in study.NAMES if n.startswith(('current_repair_','repair_')) or n in ('em100','compact_sparse'))
        assert set(raw)==set(names) and data['indices']==indices and data['worlds']==459 and data['repeats']==1
        assert data['started_unix']>=max(policy['frozen_unix'],previous_end) and data['completed_unix']>=data['started_unix']
        previous_end=data['completed_unix'];assert data['environment']['torch_threads']==1
        if window==1:
            anchor_times=[p.stat().st_mtime for p in (study.OUT/'test/anchors').glob('*.npz')]
            assert len(anchor_times)==9 and min(anchor_times)>=data['completed_unix']
        else:
            generated=list((study.OUT/'test/cells').glob('*.npz'))+list((study.OUT/'boundary/cells').glob('*.npz'))
            assert len(generated)==207 and max(p.stat().st_mtime for p in generated)<=data['started_unix']
            assert (study.OUT/'ber/summary.json').stat().st_mtime<=data['started_unix']
        expected={(r,float(ref),float(s),idx,0) for r,_ in study.REGIMES for ref in study.RSRS for s in study.SNRS for idx in indices}
        # Recover randomized world order and RNG consumption independently.
        worlds=[(r,float(ref),float(s),idx,0) for idx in indices for ref in study.RSRS for r,_ in study.REGIMES for s in study.SNRS]
        rng=np.random.default_rng(np.random.SeedSequence([2026102004,window]));order=rng.permutation(459)
        ordered=[worlds[i] for i in order]
        for _ in order:rng.permutation(len(names))
        for name,rows in raw.items():
            keys=[(x['regime'],float(x['rsr']),float(x['snr']),int(x['index']),int(x['repeat'])) for x in rows]
            assert len(keys)==len(set(keys))==459 and set(keys)==expected and keys==ordered
            for row in rows:
                ref,snr=row['rsr'],row['snr'];old=snr>study.OLD_POLICY[f'{ref:g}']+1e-10 and snr>20+1e-10
                branch=(chosen(policy,ref,snr) if name.startswith(('repair_','current_repair_')) else
                    chosen(study.PRIOR_REPAIR,ref,snr) if name.startswith('prior_repair_') else
                    ('spectral100' if old else 'original') if name.startswith('old_bypass_') else
                    'original' if name.startswith('original_') else name)
                assert row['branch']==branch
        expanded=dict(raw)
        expanded.update({n:[x for member in members for x in raw[member]] for n,members in study.GROUPS.items() if all(member in raw for member in members)})
        assert set(expanded)==set(data['methods'])
        for name,rows in expanded.items():
            item=data['methods'][name];verify_stats(rows,item)
            branches={r['branch'] for r in rows};assert set(item['by_branch'])==branches
            for branch in branches:verify_stats([r for r in rows if r['branch']==branch],item['by_branch'][branch])
            assert len(item['by_condition'])==51
            for ref in study.RSRS:
                for snr in study.SNRS:verify_stats([r for r in rows if r['rsr']==ref and r['snr']==snr],item['by_condition'][f'rsr{ref:g}_snr{snr:g}'])
            for region in ('weak_warm100','strong_high'):
                verify_stats([r for r in rows if belongs(region,r['snr'],r['rsr'],r['regime'])],item[region])
        cap=data['cap_verification'];assert cap['cap_ms']==10.74
        p90=data['methods']['repair_mean']['p90_ms'];close(cap['repair_p90_ms'],p90)
        assert cap['passed']==(p90<=10.74)
        close(cap['weak_warm100_p90_ms'],data['methods']['repair_mean']['weak_warm100']['p90_ms'])
        records[f'window{window}']=dict(p90_ms=p90,target_met=p90<=10.74,weak_warm100_p90_ms=cap['weak_warm100_p90_ms'],raw_calls=sum(map(len,raw.values())))
        hashes[path.name]=sha(path);total+=sum(map(len,raw.values()))
    assert total==12852
    return record('timing',selection_sha256=digest,timing_hashes=hashes,windows=records,literal_cap_ms=10.74,
        target_met_in_both_windows=all(r['target_met'] for r in records.values()),raw_calls=total,
        every_sample_route_grid_and_quantile_verified=True,three_fresh_indices_per_window=True,
        scope='Audit PASS verifies recorded arithmetic and protocol consistency, not target attainment or external host isolation. No timing retries or normalization are performed.')


def summary():
    """All point tables, all aggregate interval quantiles, 32 independent bootstraps."""
    policy,digest=study.check_selection();report=json.loads((study.OUT/'nmse_summary.json').read_text())
    assert report['selection_sha256']==digest
    for window in (1,2):
        name=f'timing_window{window}.json';timing_record=json.loads((study.OUT/name).read_text())
        assert report['timing_hashes'][name]==sha(study.OUT/name)
        assert report['timing_windows'][f'window{window}']['methods']==timing_record['methods']
        assert report['timing_windows'][f'window{window}']['fixed_cap_ms']==10.74
        if window==1:assert report['timing']==timing_record['methods']
    groups=study.GROUPS;methods=list(study.NAMES)+list(groups);index={n:i for i,n in enumerate(methods)}
    def expand(v):return np.concatenate([v,np.stack([v[[index[n] for n in part]].mean(0) for part in groups.values()])])
    rng=np.random.default_rng(2026102009)
    full_weights={r:rng.multinomial(128,np.full(128,1/128),size=20000).T.astype(float) for r,_ in study.REGIMES}
    weights={r:w[:,:32] for r,w in full_weights.items()}
    rows=report['cells'];assert len(rows)==153
    points=[];boots=[];errors=[];energies=[];maximum=0.;full_cell_checks=[]
    for row in rows:
        path=study.OUT/'test/cells'/f"{row['cell']}.npz";a=study.load(path);assert sha(path)==report['test_hashes'][path.name]
        assert str(a['selection_sha256'])==digest
        np.testing.assert_array_equal(a['method_names'],study.NAMES)
        err=(np.abs(a['estimates']-a['truth_g'][None])**2).sum((-2,-1));energy=(np.abs(a['truth_g'])**2).sum((-2,-1))
        point=expand(10*np.log10(err.sum(-1)/energy.sum()));boot=expand(10*np.log10((err@weights[row['regime']])/(energy@weights[row['regime']])[None]))
        nm=err/energy[None];p95,p99=(expand(np.quantile(nm,q,axis=1)) for q in (.95,.99));severe=expand((nm>1).mean(1))
        for name,i in index.items():
            values=dict(nmse_db=point[i],case_nmse_p95=p95[i],case_nmse_p99=p99[i],severe_rate=severe[i])
            for key,value in values.items():close(row['methods'][name][key],value)
            maximum=max(maximum,abs(point[i]-row['methods'][name]['nmse_db']))
        for key,(reference,candidate) in study.CONTRASTS.items():close(row['contrasts'][key]['gain_db'],point[index[reference]]-point[index[candidate]])
        if (row['rsr'],row['snr']) in ((12.,40.),(20.,60.)):
            all_boot=expand(10*np.log10((err@full_weights[row['regime']])/(energy@full_weights[row['regime']])[None]))
            for name,i in index.items():close(row['methods'][name]['conditional_ci95_db'],np.quantile(all_boot[i],[.025,.975]),1e-11)
            for key,(reference,candidate) in study.CONTRASTS.items():
                close(row['contrasts'][key]['pointwise_ci95_db'],np.quantile(all_boot[index[reference]]-all_boot[index[candidate]],[.025,.975]),1e-11)
            full_cell_checks.append(row['cell'])
        points.append(point);boots.append(boot);errors.append(err);energies.append(energy)
    points=np.stack(points);boots=np.stack(boots);stored=study.load(study.OUT/'nmse_bootstrap.npz')
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
            values=dict(nmse_db=point[i],pooled_ratio_of_sums_db=pooled[i],case_nmse_p95=p95[i],case_nmse_p99=p99[i],severe_rate=severe[i])
            for key,value in values.items():close(aggregate['methods'][name][key],value)
            close(aggregate['methods'][name]['conditional_ci95_db'],np.quantile(full[i],[.025,.975]))
        for key,(reference,candidate) in study.CONTRASTS.items():
            item=aggregate['contrasts'][key];d=full[index[reference]]-full[index[candidate]]
            close(item['gain_db'],point[index[reference]]-point[index[candidate]]);close(item['pointwise_ci95_db'],np.quantile(d,[.025,.975]))
            primary=region in ('all','weak_warm100') and key=='repair_vs_current'
            assert ('bonferroni4_interval_db' in item)==primary
            if primary:close(item['bonferroni4_interval_db'],np.quantile(d,[.05/8,1-.05/8]))
            for seed in study.SEEDS:
                source=f'{reference[:-5]}_{seed}' if reference.endswith('_mean') else reference
                close(item['seed_specific_gains_db'][str(seed)],point[index[source]]-point[index[f'repair_{seed}']])
    assert counts==dict(all=153,weak_warm100=18,strong_high=30,strong_transition=18,ordinary=45,weak=51,rsr5=51,rsr12=51,rsr20=51,all_L2=51,all_L4=51,all_random3to7=51,weak_warm100_L2=6,weak_warm100_L4=6,weak_warm100_random3to7=6)
    assert report['numerical_tie_tolerance_db']==1e-10
    assert report['source_sha256']==sha(Path(__file__).with_name('analysis.py'))
    parity=report['numerical_parity'];assert parity['passed'] and parity['atol']==parity['rtol']==1e-10
    assert parity['total_cell_model_pairs']==459 and len(parity['cells'])==153
    parity_rows={r['cell']:r for r in parity['cells']};parity_max=0.;parity_fraction=0.;parity_bits=0
    for row in rows:
        a=study.load(study.OUT/'test/cells'/f"{row['cell']}.npz")
        for i,seed in enumerate(study.SEEDS):
            ref=a['estimates'][9+i];new=a['estimates'][12+i];difference=np.abs(new-ref);limit=1e-10+1e-10*np.abs(ref)
            assert np.all(difference<=limit);item=parity_rows[row['cell']]['models'][i];assert item['seed']==seed
            close(item['maximum_absolute_error'],difference.max(),0);close(item['maximum_tolerance_fraction'],(difference/limit).max(),0)
            assert item['bitwise_equal']==np.array_equal(new,ref)
            parity_max=max(parity_max,float(difference.max()));parity_fraction=max(parity_fraction,float((difference/limit).max()));parity_bits+=np.array_equal(new,ref)
    close(parity['maximum_absolute_error'],parity_max,0);close(parity['maximum_tolerance_fraction'],parity_fraction,0)
    assert parity['bitwise_equal_cell_model_pairs']==parity_bits
    assert report['bootstrap_index_clusters']==384 and report['unique_test_anchors']==1152
    assert report['condition_cases']==153*128 and len(full_cell_checks)==6
    for key in study.CONTRASTS:
        item=report['negative_cells'][key];gains=[r['contrasts'][key]['gain_db'] for r in rows]
        assert item['negative_cells']==[r['cell'] for r in rows if r['contrasts'][key]['gain_db'] < -1e-10]
        assert item['pointwise95_harm_cells']==[r['cell'] for r in rows if r['contrasts'][key]['pointwise_ci95_db'][1] < -1e-10]
        assert item['target_negative_cells']==[r['cell'] for r in rows if belongs('weak_warm100',r['snr'],r['rsr'],r['regime']) and r['contrasts'][key]['gain_db'] < -1e-10]
        worst=int(np.argmin(gains));assert item['worst_cell']==rows[worst]['cell'];close(item['worst_gain_db'],gains[worst])
    result=dict(passed=True,selection_sha256=digest,cells=153,region_counts=counts,maximum_point_error=maximum,
        independently_recomputed_bootstrap_replicates=32,maximum_bootstrap_error=booterror,
        selected_cells_with_all_20000_intervals_independently_recomputed=full_cell_checks,
        all_points_pooled_nmse_tail_rates_per_fit_gains_verified=True,all_aggregate_intervals_and_only_declared_bonferroni4_verified=True,
        numerical_ties_and_every_adverse_cell_list_verified=True,final_timing_attached_and_hash_verified=True,
        summary_sha256=sha(study.OUT/'nmse_summary.json'),bootstrap_sha256=sha(study.OUT/'nmse_bootstrap.npz'),source_sha256=sha(__file__),analysis_sha256=sha(Path(__file__).with_name("analysis.py")),
        scope='Conditional fixed-model/policy bootstrap. All20000 aggregate quantiles checked;32 replicated from raw predictions with whole-anchor counts shared across references/SNR/method/fit within regime.')
    write_json(ROOT/'summary.json',result);return result


def boundary():
    """Recompute every boundary point, interval, paired step and endpoint change."""
    policy,digest=study.check_selection();path=study.OUT/'boundary/summary.json';report=json.loads(path.read_text())
    assert report['selection_sha256']==digest and report['protocol_sha256']==policy['protocol_sha256']
    assert report['source_sha256']==sha(Path(__file__).with_name('analysis.py'))
    snrs=(37.5,38.,38.5,38.75,39.,39.5,40.,40.5,41.,41.5,42.,42.5)
    extra=(38.,38.5,38.75,39.,39.5,40.5,41.,41.5,42.)
    methods=('current_repair_mean','repair_mean','em100')
    assert tuple(report['snrs'])==snrs and tuple(report['additional_snrs'])==extra
    assert report['cells']==72 and report['additional_cells']==54 and report['shared_main_cells']==18
    assert report['unique_channel_reference_anchors']==768 and report['bootstrap_index_clusters']==384
    assert report['condition_cases']==72*128 and report['historical_lookup_boundary_db']==38.75
    assert report['paired_anchor_identity'] and tuple(report['methods'])==methods
    assert report['bootstrap']['samples']==20000 and report['bootstrap']['seed']==2026102009
    rng=np.random.default_rng(2026102009)
    weights={r:rng.multinomial(128,np.full(128,1/128),size=20000).T.astype(float) for r,_ in study.REGIMES}
    group_members={name:tuple(study.NAMES.index(n) for n in study.GROUPS.get(name,(name,))) for name in methods}
    rows={(r['regime'],float(r['rsr']),float(r['snr'])):r for r in report['curves']}
    assert len(rows)==len(report['curves'])==72
    points={};samples={};hashes={};maximum=0.;anchors={}
    for regime,_ in study.REGIMES:
        for ref in (12.,20.):
            for snr in snrs:
                key=(regime,ref,snr);row=rows[key]
                directory='test/cells' if snr in (37.5,40.,42.5) else 'boundary/cells'
                cellpath=study.OUT/directory/f'{study.cellname(regime,ref,snr)}.npz'
                relative=str(cellpath.relative_to(study.OUT));assert row['source_directory']==directory
                hashes[relative]=sha(cellpath);assert report['cell_hashes'][relative]==hashes[relative]
                cell=study.load(cellpath);assert str(cell['selection_sha256'])==digest
                np.testing.assert_array_equal(cell['method_names'],study.NAMES)
                anchor_key=(regime,ref)
                if anchor_key not in anchors:
                    anchors[anchor_key]={k:cell[k].copy() for k in ('truth_g','indices','local_indices','s','b')}
                else:
                    for name,value in anchors[anchor_key].items():np.testing.assert_array_equal(cell[name],value)
                energy=np.einsum('bnk,bnk->b',cell['truth_g'].conj(),cell['truth_g']).real
                err=np.sum(np.abs(cell['estimates']-cell['truth_g'][None])**2,axis=(-2,-1))
                np.testing.assert_array_equal(err,cell['errors']);close(energy,cell['truth_energy'],1e-11)
                scores=[];boots=[]
                for method in methods:
                    selected=err[list(group_members[method])]
                    score=float(np.mean(10*np.log10(selected.sum(1)/energy.sum())))
                    boot=np.mean(10*np.log10((selected@weights[regime])/(energy@weights[regime])[None]),axis=0)
                    close(row['methods'][method]['nmse_db'],score,1e-11)
                    close(row['methods'][method]['conditional_ci95_db'],np.quantile(boot,[.025,.975]),1e-11)
                    maximum=max(maximum,abs(row['methods'][method]['nmse_db']-score));scores.append(score);boots.append(boot)
                points[key]=np.array(scores);samples[key]=np.stack(boots)
                close(row['repair_vs_current']['gain_db'],scores[0]-scores[1],1e-11)
                close(row['repair_vs_current']['pointwise_ci95_db'],np.quantile(boots[0]-boots[1],[.025,.975]),1e-11)
    assert set(hashes)==set(report['cell_hashes'])

    def verify_change(record,keys,lower,upper,scaled=False):
        difference=np.mean([points[(r,ref,upper)]-points[(r,ref,lower)] for r,ref in keys],axis=0)
        draws=np.mean([samples[(r,ref,upper)]-samples[(r,ref,lower)] for r,ref in keys],axis=0)
        assert record['lower_snr']==lower and record['upper_snr']==upper
        for i,method in enumerate(methods):
            close(record['methods'][method]['change_db'],difference[i],1e-11)
            close(record['methods'][method]['pointwise_ci95_db'],np.quantile(draws[i],[.025,.975]),1e-11)
            if scaled:close(record['methods'][method]['change_per_snr_db'],difference[i]/(upper-lower),1e-11)
        close(record['worsening_reduction_vs_current']['change_db'],difference[0]-difference[1],1e-11)
        close(record['worsening_reduction_vs_current']['pointwise_ci95_db'],np.quantile(draws[0]-draws[1],[.025,.975]),1e-11)

    adjacent={(r['regime'],r['rsr'],r['lower_snr'],r['upper_snr']):r for r in report['adjacent_changes']}
    endpoints={(r['regime'],r['rsr']):r for r in report['changes_37_5_to_40']}
    assert len(adjacent)==len(report['adjacent_changes'])==66 and len(endpoints)==len(report['changes_37_5_to_40'])==6
    for regime,_ in study.REGIMES:
        for ref in (12.,20.):
            for lower,upper in zip(snrs[:-1],snrs[1:]):verify_change(adjacent[(regime,ref,lower,upper)],[(regime,ref)],lower,upper,True)
            verify_change(endpoints[(regime,ref)],[(regime,ref)],37.5,40.,True)
    aggregates={(r['lower_snr'],r['upper_snr']):r for r in report['equal_cell_changes']}
    assert len(aggregates)==len(report['equal_cell_changes'])==12
    for lower,upper in list(zip(snrs[:-1],snrs[1:]))+[(37.5,40.)]:
        assert aggregates[(lower,upper)]['cells_at_each_endpoint']==6
        verify_change(aggregates[(lower,upper)],[(r,ref) for r,_ in study.REGIMES for ref in (12.,20.)],lower,upper)
    result=dict(passed=True,selection_sha256=digest,cells=72,additional_cells=54,shared_main_cells=18,
        maximum_point_error=maximum,all_points_and_all_20000_draw_intervals_independently_recomputed=True,
        paired_adjacent_changes=66,paired_endpoint_changes=6,equal_cell_changes=12,
        coupled_anchor_identity_verified=True,main_aggregate_exclusion_verified_by_scope=True,
        summary_sha256=sha(path),cell_hashes=hashes,source_sha256=sha(__file__),
        scope='Descriptive boundary only; higher-minus-lower SNR dB changes, not superiority hypotheses or an assertion of monotone sample paths.')
    write_json(ROOT/'boundary.json',result);return result



if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=('selection','test','summary','boundary','timing'));args=parser.parse_args()
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    print(json.dumps(globals()[args.phase]()),flush=True)
