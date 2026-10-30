"""Independent verification of the frozen transition-repair study artifacts."""
import os
for _name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[_name]='1'
import argparse
import json
import math
import numpy as np
import torch
from sahure.data import Observations
from sahure.math import posterior_field,right_solve
from tcom_followups_20261017.common import sha,write_json
from . import study

ROOT=study.OUT/'audit'
REGIONS=('ordinary','transition','stress','high_snr','all','weak_transition','weak_transition_L2','weak_transition_L4','weak_transition_random3to7')


def numeric_stats(rows):
    values=np.array([x['ms'] for x in rows]);assert np.isfinite(values).all() and (values>0).all()
    return dict(p90_ms=float(np.quantile(values,.9)),median_ms=float(np.median(values)),mean_ms=float(values.mean()),calls=len(values))


def close(actual,expected,tolerance=1e-12):
    np.testing.assert_allclose(actual,expected,atol=tolerance,rtol=0)


def verify_stats(expected,record):
    for key,value in expected.items():close(record[key],value)


def expected_grid(index):
    return {(regime,float(ref),float(snr),repeat,index) for regime,_ in study.REGIMES for ref in study.RSRS for snr in study.SNRS for repeat in range(3)}


def check_grid(rows,index):
    keys=[(r['regime'],float(r['rsr']),float(r['snr']),int(r['repeat']),int(r['index'])) for r in rows]
    assert len(keys)==459 and len(set(keys))==459 and set(keys)==expected_grid(index)


def chosen(policy,ref,snr):
    return 'original' if snr<=20+1e-10 else policy['repair_choices'][f'{ref:g}'][f'{snr:g}']['choice']


def repair():
    """Independently reproduce all development guards before runtime calibration."""
    record=study.check_repair();protocol=json.loads((study.OUT/'protocol.json').read_text())
    anchors=sorted((study.OUT/'development/anchors').glob('*.npz'));assert len(anchors)==9
    for path in anchors:
        a=study.load(path)
        assert not {'g','G','mu','paths','unit_noise','truth_g'}.intersection(a)
        assert int(a['seed'])==2026101921 and a['z_by_snr'].shape==(17,96,32,39)
        assert a['s'].shape==(96,3,39) and a['b'].shape==(96,32,39)
        assert str(a['protocol_sha256'])==record['protocol_sha256']
        assert path.stat().st_mtime>=protocol['frozen_unix']
    files=sorted((study.OUT/'development/repair_losses').glob('*.npz'));assert len(files)==51
    rng=np.random.default_rng(2026101927)
    weights={r:rng.multinomial(96,np.full(96,1/96),size=4000) for r,_ in study.REGIMES}
    saved=study.load(study.OUT/'development/repair_bootstrap_weights.npz')
    for regime,value in weights.items():np.testing.assert_array_equal(saved[regime],value)
    counts={name:0 for name in study.MENU};admissible_counts=dict(counts);guard_rejections=0;numeric_error=0.;reviews=[]
    for ref in study.RSRS:
        for snr in study.SNRS:
            a=study.load(study.OUT/'development/repair_losses'/f'rsr{ref:g}_snr{snr:g}.npz')
            assert str(a['protocol_sha256'])==record['protocol_sha256']
            assert float(a['snr'])==snr and float(a['rsr'])==ref
            np.testing.assert_array_equal(a['names'],study.MENU);np.testing.assert_array_equal(a['seeds'],study.SEEDS)
            np.testing.assert_array_equal(a['regimes'],[r for r,_ in study.REGIMES])
            fit,loss=a['fit_nll'],a['heldout_nll'];assert fit.shape==loss.shape==(3,5,3,96)
            assert np.isfinite(fit).all() and np.isfinite(loss).all()
            assert np.all(fit[:,1:4]<=fit[:,0:1]+1e-10*(1+np.abs(fit[:,0:1])))
            scores=[]
            for mi,name in enumerate(study.MENU):
                # Difference within fit/case first, then average the three fixed fits.
                paired=(loss[:,mi]-loss[:,0]).mean(0)
                upper={r:float(np.quantile(paired[pi]@weights[r].T/96,.95)) for pi,(r,_) in enumerate(study.REGIMES)}
                valid=name=='original' or all(x<=0 for x in upper.values())
                score=dict(choice=name,mean_nll=float(loss[:,mi].mean()),mean_delta=float(paired.mean()),
                    by_regime_mean_delta={r:float(paired[pi].mean()) for pi,(r,_) in enumerate(study.REGIMES)},
                    regime_upper95=upper,admissible=valid,cost_order=study.COST_ORDER.index(name))
                stored=next(x for x in record['candidate_scores'][f'{ref:g}'][f'{snr:g}'] if x['choice']==name)
                for key in ('mean_nll','mean_delta'):close(stored[key],score[key])
                for key in ('by_regime_mean_delta','regime_upper95'):
                    for regime in upper:close(stored[key][regime],score[key][regime],1e-9)
                assert stored['admissible']==valid and stored['cost_order']==score['cost_order']
                assert name!='original' or (paired==0).all()
                admissible_counts[name]+=int(valid)
                guard_rejections+=int(name!='original' and all(x<=0 for x in score['by_regime_mean_delta'].values()) and not valid)
                scores.append(score)
            winner=scores[0] if snr<=20 else min((x for x in scores if x['admissible']),key=lambda x:(x['mean_nll'],x['cost_order']))
            selected=record['repair_choices'][f'{ref:g}'][f'{snr:g}'];assert selected['choice']==winner['choice']
            assert selected==next(x for x in record['candidate_scores'][f'{ref:g}'][f'{snr:g}'] if x['choice']==winner['choice'])
            counts[winner['choice']]+=1
            wi=study.MENU.index(winner['choice']);centered=[]
            for mi,name in enumerate(study.MENU):
                margin=math.fsum(map(float,(loss[:,mi]-loss[:,wi]).ravel()))/loss[:,mi].size
                direct=scores[mi]['mean_nll']-winner['mean_nll'];numeric_error=max(numeric_error,abs(direct-margin))
                if snr>20 and scores[mi]['admissible']:assert margin>=-max(1e-10,2*abs(direct-margin))
                centered.append(dict(choice=name,mean_nll_above_selected=margin))
            reviews.append(dict(rsr=ref,snr=snr,selected=winner['choice'],admissible=[x['choice'] for x in scores if x['admissible']],centered_margins=centered))
    result=dict(passed=True,repair_selection_sha256=sha(study.OUT/'repair_selection.json'),protocol_sha256=record['protocol_sha256'],
        observable_only_development_anchors_checked=9,repair_loss_tables_checked=51,guard_candidate_checks=255,
        bootstrap_weights_exactly_reconstructed=True,paired_fit_average_before_bootstrap=True,selected_counts=counts,
        admissible_counts=admissible_counts,mean_nonharm_but_upper_bound_rejected_candidates=guard_rejections,
        maximum_direct_vs_centered_margin_error=numeric_error,reviews=reviews,audit_source_sha256=sha(__file__),
        caveat='The per-regime development screen is not a population safety certificate; selection is conditional on three fixed trained models. No diagnostic changes any choice.')
    write_json(ROOT/'repair_selection.json',result);return result


def calibration():
    record=study.check_calibration();policy=study.check_repair();configs=study.prior.candidates();raw=record['raw_samples']
    expected=set(f'repair_{s}' for s in study.SEEDS)|set(f'compact_{k}' for k in configs)
    assert set(raw)==expected and set(record['candidates'])==set(configs)
    for name,rows in raw.items():
        check_grid(rows,100);verify_stats(numeric_stats(rows),record['methods'][name])
        for row in rows:assert row['repair_choice']==chosen(policy,row['rsr'],row['snr'])
    pooled=numeric_stats([row for seed in study.SEEDS for row in raw[f'repair_{seed}']])
    verify_stats(pooled,record['repair_timing']);assert record['cap_ms']==pooled['p90_ms']
    eligible=[]
    for name,config in configs.items():
        assert record['candidates'][name]['config']==config
        values=numeric_stats(raw[f'compact_{name}']);verify_stats(values,record['candidates'][name])
        if values['p90_ms']<=record['cap_ms']:eligible.append(name)
    assert record['eligible']==eligible
    assert record['completed_unix']>=policy['frozen_unix']
    result=dict(passed=True,protocol_sha256=record['protocol_sha256'],calibration_sha256=sha(study.OUT/'calibration.json'),
        candidates=len(configs),eligible=eligible,raw_calls=sum(map(len,raw.values())),cap_ms=record['cap_ms'],
        cap_rule='Pooled actual selected-repair p90 across three fixed fits and all153conditions.',
        raw_samples_quantiles_routes_and_eligibility_verified=True,audit_source_sha256=sha(__file__),
        caveat='Observed pooled latency quantiles do not establish a worst-case deadline or prove host isolation.')
    write_json(ROOT/'calibration.json',result);return result


def selection():
    review=repair();policy,digest=study.check_selection();cal=study.check_calibration();configs=study.prior.candidates()
    paths=sorted((study.OUT/'development/compact_losses').glob('*.npz'));assert len(paths)==51
    for ref in study.RSRS:
        for snr in study.SNRS:
            path=study.OUT/'development/compact_losses'/f'rsr{ref:g}_snr{snr:g}.npz';a=study.load(path)
            assert str(a['calibration_sha256'])==policy['calibration_sha256']
            assert float(a['rsr'])==ref and float(a['snr'])==snr
            np.testing.assert_array_equal(a['names'],cal['eligible'])
            assert a['losses'].shape==(len(cal['eligible']),3,96) and np.isfinite(a['losses']).all()
            scores=[]
            for mi,key in enumerate(cal['eligible']):
                score=dict(candidate=key,config=configs[key],mean_nll=float(a['losses'][mi].mean()),
                    by_regime=a['losses'][mi].mean(-1).tolist(),p90_ms=cal['candidates'][key]['p90_ms'])
                stored=next(x for x in policy['compact_tables'][f'{ref:g}'][f'{snr:g}'] if x['candidate']==key)
                assert stored==score;scores.append(score)
            winner=min(scores,key=lambda x:(x['mean_nll'],x['p90_ms'],x['candidate']))
            assert policy['compact_choices'][f'{ref:g}'][f'{snr:g}']==winner
            assert path.stat().st_mtime>=cal['completed_unix']
    assert policy['frozen_unix']>=cal['completed_unix']
    result=dict(passed=True,selection_sha256=digest,protocol_sha256=policy['protocol_sha256'],
        repair_review_sha256=sha(ROOT/'repair_selection.json'),repair_choices_checked=51,compact_choices_checked=51,
        selected_counts=review['selected_counts'],all_per_regime_upper95_guards_recomputed=True,
        source_sha256=sha(__file__),caveat=review['caveat'])
    write_json(ROOT/'selection.json',result);return result


@torch.no_grad()
def test():
    policy,digest=study.check_selection();files=sorted((study.OUT/'test/cells').glob('*.npz'));assert len(files)==153
    anchors=sorted((study.OUT/'test/anchors').glob('*.npz'));assert len(anchors)==9
    anchor_map={};anchor_hashes={};hashes={};ordinary=warm_checks=original_checks=spectral_checks=0;warm_error=0.
    for path in anchors:
        a=study.load(path);assert int(a['seed'])==2026101922 and str(a['selection_sha256'])==digest and len(a['g'])==128
        anchor_map[path.stem]=a;anchor_hashes[path.name]=sha(path)
    for path in files:
        a=study.load(path);assert str(a['selection_sha256'])==digest and str(a['protocol_sha256'])==policy['protocol_sha256']
        np.testing.assert_array_equal(a['method_names'],study.NAMES);assert a['estimates'].shape==(14,128,32,3)
        ref,snr,regime=float(a['rsr']),float(a['snr']),str(a['regime']);anchor=anchor_map[f'{regime}_rsr{ref:g}']
        assert path.stem==study.cellname(regime,ref,snr)
        for key,anchor_key in (('truth_g','g'),('indices','indices'),('local_indices','local_indices'),('paths','paths')):
            np.testing.assert_array_equal(a[key],anchor[anchor_key])
        obs=study.observed(anchor,snr,P=30)
        for key,value in zip(('z','s','b','sigma2'),obs.args()):np.testing.assert_array_equal(a[key],value.numpy())
        pred=a['estimates'];assert np.isfinite(pred).all()
        errors=(np.abs(pred-a['truth_g'][None])**2).sum((-2,-1));energy=(np.abs(a['truth_g'])**2).sum((-2,-1))
        np.testing.assert_array_equal(errors,a['errors']);np.testing.assert_array_equal(energy,a['truth_energy']);np.testing.assert_array_equal(errors/energy[None],a['nmse'])
        old_on=snr>study.OLD_POLICY[f'{ref:g}']+1e-10 and snr>20+1e-10
        np.testing.assert_array_equal(a['old_bypass'],np.full(128,old_on))
        np.testing.assert_array_equal(pred[3:6],np.repeat(pred[9:10],3,axis=0) if old_on else pred[:3])
        choice=chosen(policy,ref,snr);assert str(a['repair_choice'])==choice
        if choice=='original':np.testing.assert_array_equal(pred[6:9],pred[:3]);original_checks+=1
        elif choice=='spectral100':np.testing.assert_array_equal(pred[6:9],np.repeat(pred[9:10],3,axis=0));spectral_checks+=1
        else:
            # Independent canonical EM replay on archived measurements and saved learned starts.
            small=Observations(*(torch.from_numpy(a[k][:1]) for k in ('z','s','b','sigma2')))
            op=right_solve(small.s.mH,small.s@small.s.mH)
            for i in range(3):
                g=torch.from_numpy(pred[i,:1].copy())
                for _ in range(int(choice[4:])):g=(posterior_field(small.z,g@small.s+small.b,small.sigma2)-small.b)@op
                error=float(np.abs(g.numpy()-pred[i+6,:1]).max());warm_error=max(warm_error,error);assert error<1e-10;warm_checks+=1
        if snr<=20:
            np.testing.assert_array_equal(pred[3:6],pred[:3]);np.testing.assert_array_equal(pred[6:9],pred[:3]);ordinary+=1
        assert float(a['actual_dispatch_max_abs_error'])<1e-10
        hashes[path.name]=sha(path)
    earliest=min(p.stat().st_mtime for p in files+anchors);assert earliest>=policy['frozen_unix']
    result=dict(passed=True,selection_sha256=digest,protocol_sha256=policy['protocol_sha256'],test_cells=153,test_anchors=9,
        ordinary_exact_identity_cells=ordinary,original_repair_identity_cells=original_checks,spectral_repair_identity_cells=spectral_checks,
        canonical_warm_replays=warm_checks,maximum_canonical_warm_error=warm_error,
        raw_errors_observations_truth_and_branches_verified=True,earliest_test_file_after_freeze_seconds=earliest-policy['frozen_unix'],
        anchor_hashes=anchor_hashes,test_prediction_hashes=hashes,audit_source_sha256=sha(__file__))
    write_json(ROOT/'test.json',result);return result


def belongs(region,snr,ref,regime):
    weak=ref==5 and 32.5<=snr<=45
    if region.startswith('weak_transition_'):return weak and regime==region[len('weak_transition_'):]
    return dict(ordinary=snr<=20,transition=20<snr<=45,stress=snr>=50,high_snr=snr>20,all=True,weak_transition=weak)[region]


def summary():
    """All point tables, all aggregate interval quantiles, 32 independent bootstraps."""
    policy,digest=study.check_selection();report=json.loads((study.OUT/'nmse_summary.json').read_text())
    assert report['selection_sha256']==digest and report['timing_sha256']==sha(study.OUT/'timing.json')
    timing_record=json.loads((study.OUT/'timing.json').read_text());assert report['timing']==timing_record['methods']
    groups=study.GROUPS;methods=list(study.NAMES)+list(groups);index={n:i for i,n in enumerate(methods)}
    def expand(v):return np.concatenate([v,np.stack([v[[index[n] for n in part]].mean(0) for part in groups.values()])])
    rng=np.random.default_rng(2026101929)
    weights={r:rng.multinomial(128,np.full(128,1/128),size=20000)[:32].T.astype(float) for r,_ in study.REGIMES}
    rows=report['cells'];assert len(rows)==153
    points=[];boots=[];errors=[];energies=[];maximum=0.
    for row in rows:
        path=study.OUT/'test/cells'/f"{row['cell']}.npz";a=study.load(path);assert sha(path)==report['test_hashes'][path.name]
        err=(np.abs(a['estimates']-a['truth_g'][None])**2).sum((-2,-1));energy=(np.abs(a['truth_g'])**2).sum((-2,-1))
        point=expand(10*np.log10(err.sum(-1)/energy.sum()));boot=expand(10*np.log10((err@weights[row['regime']])/(energy@weights[row['regime']])[None]))
        nm=err/energy[None];p95,p99=(expand(np.quantile(nm,q,axis=1)) for q in (.95,.99));severe=expand((nm>1).mean(1))
        for name,i in index.items():
            values=dict(nmse_db=point[i],case_nmse_p95=p95[i],case_nmse_p99=p99[i],severe_rate=severe[i])
            for key,value in values.items():close(row['methods'][name][key],value)
            maximum=max(maximum,abs(point[i]-row['methods'][name]['nmse_db']))
        for key,(reference,candidate) in study.CONTRASTS.items():close(row['contrasts'][key]['gain_db'],point[index[reference]]-point[index[candidate]])
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
            primary=region=='weak_transition' and key in ('repair_vs_old','repair_vs_original')
            assert ('bonferroni4_interval_db' in item)==primary
            if primary:close(item['bonferroni4_interval_db'],np.quantile(d,[.05/8,1-.05/8]))
            for seed in study.SEEDS:
                source=f'{reference[:-5]}_{seed}' if reference.endswith('_mean') else reference
                close(item['seed_specific_gains_db'][str(seed)],point[index[source]]-point[index[f'repair_{seed}']])
    assert counts==dict(ordinary=45,transition=90,stress=18,high_snr=108,all=153,weak_transition=18,
        weak_transition_L2=6,weak_transition_L4=6,weak_transition_random3to7=6)
    assert report['numerical_tie_tolerance_db']==1e-10
    for key in study.CONTRASTS:
        item=report['negative_cells'][key];gains=[r['contrasts'][key]['gain_db'] for r in rows]
        assert item['negative_cells']==[r['cell'] for r in rows if r['contrasts'][key]['gain_db'] < -1e-10]
        assert item['pointwise95_harm_cells']==[r['cell'] for r in rows if r['contrasts'][key]['pointwise_ci95_db'][1] < -1e-10]
        assert item['target_negative_cells']==[r['cell'] for r in rows if belongs('weak_transition',r['snr'],r['rsr'],r['regime']) and r['contrasts'][key]['gain_db'] < -1e-10]
        worst=int(np.argmin(gains));assert item['worst_cell']==rows[worst]['cell'];close(item['worst_gain_db'],gains[worst])
    result=dict(passed=True,selection_sha256=digest,cells=153,region_counts=counts,maximum_point_error=maximum,
        independently_recomputed_bootstrap_replicates=32,maximum_bootstrap_error=booterror,
        all_points_pooled_nmse_tail_rates_per_fit_gains_verified=True,all_aggregate_intervals_and_only_declared_bonferroni4_verified=True,
        numerical_ties_and_every_adverse_cell_list_verified=True,final_timing_attached_and_hash_verified=True,
        summary_sha256=sha(study.OUT/'nmse_summary.json'),bootstrap_sha256=sha(study.OUT/'nmse_bootstrap.npz'),source_sha256=sha(__file__),
        scope='Conditional fixed-model/policy bootstrap. All20000 aggregate quantiles checked;32 replicated from raw predictions with whole-anchor counts shared across references/SNR/method/fit within regime.')
    write_json(ROOT/'summary.json',result);return result


def timing():
    policy,digest=study.check_selection();record=json.loads((study.OUT/'timing.json').read_text());raw=record['raw_samples']
    assert record['selection_sha256']==digest and record['worlds']==153 and record['repeats']==3 and set(raw)==set(study.NAMES)
    for name,rows in raw.items():
        check_grid(rows,200)
        for row in rows:
            choice=chosen(policy,row['rsr'],row['snr']);assert row['repair_choice']==choice
            old=row['snr']>study.OLD_POLICY[f"{row['rsr']:g}"]+1e-10 and row['snr']>20+1e-10
            expected=choice if name.startswith('repair_') else ('spectral100' if old else 'original') if name.startswith('old_bypass_') else 'original' if name.startswith('original_') else name
            assert row['branch']==expected
    combined=dict(raw)
    for name,parts in study.GROUPS.items():combined[name]=[r for part in parts for r in raw[part]]
    for name,rows in combined.items():
        verify_stats(numeric_stats(rows),record['methods'][name])
        for branch in {r['branch'] for r in rows}:verify_stats(numeric_stats([r for r in rows if r['branch']==branch]),record['methods'][name]['by_branch'][branch])
        for ref in study.RSRS:
            for snr in study.SNRS:verify_stats(numeric_stats([r for r in rows if r['rsr']==ref and r['snr']==snr]),record['methods'][name]['by_condition'][f'rsr{ref:g}_snr{snr:g}'])
    cap=study.check_calibration()['cap_ms'];repair_value=record['methods']['repair_mean']['p90_ms'];compact=record['methods']['compact_sparse']['p90_ms']
    expected=dict(calibrated_cap_ms=cap,repair_p90_ms=repair_value,compact_p90_ms=compact,
        repair_exceeds_calibrated_cap=repair_value>cap,compact_exceeds_calibrated_cap=compact>cap,compact_exceeds_fresh_repair=compact>repair_value)
    assert record['cap_verification']==expected
    result=dict(passed=True,selection_sha256=digest,worlds=153,repeats=3,individual_methods=14,raw_calls=sum(map(len,raw.values())),
        all_quantiles_condition_counts_and_branches_verified=True,cap_verification=expected,timing_sha256=sha(study.OUT/'timing.json'),audit_source_sha256=sha(__file__),
        caveat='Audit verifies the retained latency samples, not host isolation or a worst-case runtime guarantee.')
    write_json(ROOT/'timing.json',result);return result


def sensitivity():
    """Independent routing and zero-offset parity for the frozen mismatch bank."""
    from . import sensitivity as module
    policy,digest=study.check_selection();base=json.loads((module.ROOT/'audit.json').read_text());assert base['passed']
    report=json.loads((module.ROOT/'summary.json').read_text());lookup={(r['regime'],r['rsr'],r['actual_snr_db'],r['supplied_variance_offset_db']):r for r in report['cells']}
    assert len(lookup)==45 and report['unique_channel_reference_anchors']==1152 and report['coupled_case_index_clusters']==384
    zero=0;maximum=0.;routes={}
    for condition in module.conditions():
        regime,ref,snr,offset=(condition[k] for k in ('regime','rsr','actual_snr','offset'))
        a=study.load(module.cellpath(regime,ref,snr,offset));assert str(a['selection_sha256'])==digest
        proxy=snr-offset;grid=min(study.SNRS,key=lambda x:(abs(x-proxy),x))
        choice='original' if proxy<=20+1e-10 else policy['repair_choices'][f'{ref:g}'][f'{grid:g}']['choice']
        assert str(a['repair_choice'])==choice;routes[choice]=routes.get(choice,0)+1
        row=lookup[(regime,ref,snr,offset)];assert row['repair_choice']==choice and row['supplied_snr_proxy_db']==proxy
        point=10*np.log10(a['errors'].sum(1)/a['truth_energy'].sum())
        for group,names in module.GROUPS.items():
            ids=[module.NAMES.index(n) for n in names];expected=float(point[ids].mean());difference=abs(row['methods'][group]['nmse_db']-expected)
            maximum=max(maximum,difference);assert difference<1e-12
        if offset==0:
            main=study.load(study.OUT/'test/cells'/f'{study.cellname(regime,ref,snr)}.npz')
            for i,name in enumerate(module.NAMES):np.testing.assert_array_equal(a['estimates'][i],main['estimates'][study.NAMES.index(name)])
            zero+=1
    result=dict(passed=True,conditions=45,zero_offset_full_prediction_identity_cells=zero,independently_checked_nearest_bin_routes=routes,
        maximum_point_error=maximum,selection_sha256=digest,sensitivity_summary_sha256=sha(module.ROOT/'summary.json'),
        sensitivity_primary_audit_sha256=sha(module.ROOT/'audit.json'),audit_source_sha256=sha(__file__))
    write_json(ROOT/'sensitivity.json',result);return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=('repair','calibration','selection','test','summary','timing','sensitivity'))
    args=parser.parse_args();torch.set_num_threads(1)
    try:torch.set_num_interop_threads(1)
    except RuntimeError:pass
    result=globals()[args.phase]()
    print(json.dumps({k:v for k,v in result.items() if k not in ('reviews','anchor_hashes','test_prediction_hashes')},indent=2))
