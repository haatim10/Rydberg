"""Independent scoring audits; scientific analysis source remains frozen."""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[name]='1'
import argparse
import json
from pathlib import Path
import numpy as np
from tcom_followups_20261017.common import sha,write_json
from . import study
ROOT=study.OUT/'audit'
REGIONS=('ordinary','transition','stress','high_snr','all','weak_transition','weak_transition_L2','weak_transition_L4','weak_transition_random3to7',
    'strong_transition','strong_transition_L2','strong_transition_L4','strong_transition_random3to7',
    'strong_transition_snr32.5','strong_transition_snr35','strong_transition_snr37.5',
    'historical_flagged','historical_flagged_L2','historical_flagged_L4','historical_flagged_random3to7',
    'strong_high','strong_high_L2','strong_high_L4','strong_high_random3to7',
    'strong_high_snr40','strong_high_snr42.5','strong_high_snr45','strong_high_snr50','strong_high_snr60')
FLAGGED={('L2',12.,37.5),('L2',20.,32.5),('L2',20.,35.),('L2',20.,37.5),
    ('L4',12.,37.5),('L4',20.,35.),('L4',20.,37.5),('random3to7',12.,37.5)}


def close(actual,expected,tolerance=1e-12):
    np.testing.assert_allclose(actual,expected,atol=tolerance,rtol=0)


def belongs(region,snr,rsr,regime):
    scopes={'ordinary':snr in (0.,5.,10.,15.,20.),'transition':20<snr<50,
        'stress':snr in (50.,60.),'high_snr':snr>20,'all':True,
        'weak_transition':rsr==5 and snr in (32.5,35.,37.5,40.,42.5,45.),
        'strong_transition':rsr in (12,20) and snr in (32.5,35.,37.5),
        'strong_high':rsr in (12,20) and snr in (40.,42.5,45.,50.,60.),
        'historical_flagged':(regime,rsr,snr) in FLAGGED}
    if region in scopes:return scopes[region]
    for prefix in ('strong_high','strong_transition','weak_transition','historical_flagged'):
        if region.startswith(prefix+'_'):
            suffix=region[len(prefix)+1:]
            return scopes[prefix] and (snr==float(suffix[3:]) if suffix.startswith('snr') else regime==suffix)
    raise KeyError(region)


def summary():
    """All point tables, all aggregate interval quantiles, 32 independent bootstraps."""
    policy,digest=study.check_selection();report=json.loads((study.OUT/'nmse_summary.json').read_text())
    assert report['selection_sha256']==digest and report['timing_sha256']==sha(study.OUT/'timing.json')
    timing_record=json.loads((study.OUT/'timing.json').read_text());assert report['timing']==timing_record['methods']
    groups=study.GROUPS;methods=list(study.NAMES)+list(groups);index={n:i for i,n in enumerate(methods)}
    def expand(v):return np.concatenate([v,np.stack([v[[index[n] for n in part]].mean(0) for part in groups.values()])])
    rng=np.random.default_rng(2026101959)
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
            primary=region=='strong_high' and key in ('repair_vs_current','repair_vs_compact')
            assert ('bonferroni4_interval_db' in item)==primary
            if primary:close(item['bonferroni4_interval_db'],np.quantile(d,[.05/8,1-.05/8]))
            for seed in study.SEEDS:
                source=f'{reference[:-5]}_{seed}' if reference.endswith('_mean') else reference
                close(item['seed_specific_gains_db'][str(seed)],point[index[source]]-point[index[f'repair_{seed}']])
    assert counts==dict(ordinary=45,transition=90,stress=18,high_snr=108,all=153,weak_transition=18,
        weak_transition_L2=6,weak_transition_L4=6,weak_transition_random3to7=6,strong_transition=18,
        strong_transition_L2=6,strong_transition_L4=6,strong_transition_random3to7=6,
        **{'strong_transition_snr32.5':6,'strong_transition_snr35':6,'strong_transition_snr37.5':6},
        historical_flagged=8,historical_flagged_L2=4,historical_flagged_L4=3,historical_flagged_random3to7=1,
        strong_high=30,strong_high_L2=10,strong_high_L4=10,strong_high_random3to7=10,
        **{f'strong_high_snr{x}':6 for x in ('40','42.5','45','50','60')})
    assert report['numerical_tie_tolerance_db']==1e-10
    assert report['source_sha256']==sha(Path(__file__).with_name('analysis.py'))
    assert report['unchanged_main_cells']==123 and report['changed_region_main_cells']==30
    assert report['preserved_strong_transition_exact_identity'] and report['non_target_current_exact_identity']
    assert report['bootstrap_index_clusters']==384 and report['unique_test_anchors']==1152
    assert report['condition_cases']==153*128 and len(full_cell_checks)==6
    for key in study.CONTRASTS:
        item=report['negative_cells'][key];gains=[r['contrasts'][key]['gain_db'] for r in rows]
        assert item['negative_cells']==[r['cell'] for r in rows if r['contrasts'][key]['gain_db'] < -1e-10]
        assert item['pointwise95_harm_cells']==[r['cell'] for r in rows if r['contrasts'][key]['pointwise_ci95_db'][1] < -1e-10]
        assert item['target_negative_cells']==[r['cell'] for r in rows if belongs('strong_high',r['snr'],r['rsr'],r['regime']) and r['contrasts'][key]['gain_db'] < -1e-10]
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
    assert report['condition_cases']==72*128 and report['previous_lookup_boundary_db']==38.75
    assert report['paired_anchor_identity'] and tuple(report['methods'])==methods
    assert report['bootstrap']['samples']==20000 and report['bootstrap']['seed']==2026101959
    rng=np.random.default_rng(2026101959)
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
    write_json(ROOT/'boundary_summary.json',result);return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=('summary','boundary'));args=parser.parse_args()
    print(json.dumps(summary() if args.phase=='summary' else boundary()),flush=True)

