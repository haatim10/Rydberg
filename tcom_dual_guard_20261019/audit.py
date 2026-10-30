"""Independent verification of the frozen dual-guard study artifacts."""
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
REGIONS=('ordinary','transition','stress','high_snr','all','weak_transition','weak_transition_L2','weak_transition_L4','weak_transition_random3to7',
    'strong_transition','strong_transition_L2','strong_transition_L4','strong_transition_random3to7',
    'strong_transition_snr32.5','strong_transition_snr35','strong_transition_snr37.5',
    'historical_flagged','historical_flagged_L2','historical_flagged_L4','historical_flagged_random3to7')
HISTORICAL=('L2_rsr20_snr32.5','L2_rsr20_snr42.5','L2_rsr20_snr45','L2_rsr20_snr50','L2_rsr20_snr60',
    'L4_rsr12_snr32.5','L4_rsr20_snr32.5','random3to7_rsr12_snr32.5','random3to7_rsr20_snr32.5','random3to7_rsr20_snr37.5')


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


def verify_bank(path,digest):
    """Rebuild each saved context/holdout split from observable anchor rows."""
    bank=study.load(path);ref=float(bank['rsr']);snr=float(bank['snr']);count=int(bank['cases_per_regime'])
    assert count==(384 if ref in (12.,20.) and snr in (32.5,35.,37.5) else 96)
    assert str(bank['protocol_sha256'])==digest
    assert not {'g','G','mu','unit_noise','paths','truth_g'}.intersection(bank)
    for pi,(regime,_) in enumerate(study.REGIMES):
        a=study.load(study.OUT/'development/anchors'/f'{regime}_rsr{ref:g}.npz')
        assert not {'g','G','mu','unit_noise','paths','truth_g'}.intersection(a)
        assert int(a['seed'])==2026101931 and len(a['s'])==(384 if ref in (12.,20.) else 96)
        obs=study.observed(a,snr,P=39).batch(slice(0,count))
        ri=list(study.RSRS).index(ref)
        order=torch.rand(count,39,generator=torch.Generator().manual_seed(2026101935+100*pi+ri)).argsort(-1)
        assert torch.equal(order.sort(1).values,torch.arange(39).expand(count,39))
        for prefix,cols in (('context',order[:,:30]),('heldout',order[:,30:])):
            expected=obs.columns(cols)
            for key,value in zip(('z','s','b','sigma2'),expected.args()):
                np.testing.assert_array_equal(bank[f'{prefix}_{key}'][pi*count:(pi+1)*count],value.numpy())
    return count


def repair():
    """Recompute the six dual guards, all fixed bins and fallback evidence."""
    record=study.check_repair();protocol=json.loads((study.OUT/'protocol.json').read_text())
    assert len(record['development_anchor_hashes'])==6 and len(record['development_bank_hashes'])==6
    assert len(record['loss_table_hashes'])==6
    for name in record['development_bank_hashes']:
        path=study.OUT/'development/banks'/name;assert verify_bank(path,record['protocol_sha256'])==384
        assert path.stat().st_mtime>=protocol['frozen_unix']
    rng=np.random.default_rng(2026101937)
    weights={r:rng.multinomial(384,np.full(384,1/384),size=4000) for r,_ in study.REGIMES}
    saved=study.load(study.OUT/'development/repair_bootstrap_weights.npz')
    for regime,value in weights.items():np.testing.assert_array_equal(saved[regime],value)
    counts={name:0 for name in study.MENU};statuses={};reviews=[];numeric_error=0.;constraint_count=0
    for ref in study.RSRS:
        for snr in study.SNRS:
            selected=record['repair_choices'][f'{ref:g}'][f'{snr:g}']
            if ref not in (12.,20.) or snr not in (32.5,35.,37.5):
                if ref==5.:
                    choice=study.PRIOR_REPAIR['repair_choices']['5'][f'{snr:g}']['choice'];status='fixed_prior_repair'
                elif snr<=30.:choice='original';status='fixed_original'
                else:assert snr>=40.;choice='spectral100';status='fixed_spectral100'
                assert selected==dict(choice=choice,selection_status=status,dual_admissible=None,failed_constraints=[])
                assert record['candidate_scores'][f'{ref:g}'][f'{snr:g}']==[]
            else:
                a=study.load(study.OUT/'development/repair_losses'/f'rsr{ref:g}_snr{snr:g}.npz')
                assert str(a['protocol_sha256'])==record['protocol_sha256'] and int(a['cases_per_regime'])==384
                np.testing.assert_array_equal(a['names'],study.MENU);np.testing.assert_array_equal(a['seeds'],study.SEEDS)
                np.testing.assert_array_equal(a['regimes'],[r for r,_ in study.REGIMES])
                fit,loss=a['fit_nll'],a['heldout_nll'];assert fit.shape==loss.shape==(3,9,3,384)
                assert np.isfinite(fit).all() and np.isfinite(loss).all()
                assert np.all(fit[:,1:-1]<=fit[:,0:1]+1e-10*(1+np.abs(fit[:,0:1])))
                scores=[]
                for mi,name in enumerate(study.MENU):
                    references={};failed=[]
                    for baseline,bi in (('original',0),('old_bypass',8)):
                        paired=(loss[:,mi]-loss[:,bi]).mean(0);by_regime={}
                        for pi,(regime,_) in enumerate(study.REGIMES):
                            upper=float(np.quantile(paired[pi]@weights[regime].T/384,.95))
                            by_regime[regime]=dict(mean_delta=float(paired[pi].mean()),upper95=upper)
                            if upper>0:failed.append(f'{baseline}:{regime}')
                            constraint_count+=1
                        references[baseline]=dict(mean_delta=float(paired.mean()),by_regime=by_regime)
                    score=dict(choice=name,mean_nll=float(loss[:,mi].mean()),references=references,
                        dual_admissible=not failed,failed_constraints=failed,preference_order=study.PREFERENCE_ORDER.index(name))
                    stored=next(x for x in record['candidate_scores'][f'{ref:g}'][f'{snr:g}'] if x['choice']==name)
                    close(stored['mean_nll'],score['mean_nll'])
                    for baseline,values in references.items():
                        close(stored['references'][baseline]['mean_delta'],values['mean_delta'])
                        for regime,values2 in values['by_regime'].items():
                            for key,value in values2.items():close(stored['references'][baseline]['by_regime'][regime][key],value,1e-10)
                    for key in ('dual_admissible','failed_constraints','preference_order'):assert stored[key]==score[key]
                    scores.append(score)
                feasible=[x for x in scores if x['dual_admissible']]
                if feasible:
                    optimum=min(x['mean_nll'] for x in feasible)
                    allowed=[x for x in feasible if x['mean_nll']<=optimum+.001]
                    winner=min(allowed,key=lambda x:x['preference_order']);status='dual_admissible'
                    close(selected['admissible_optimum_mean_nll'],optimum)
                    assert selected['dual_admissible'] and not selected['failed_constraints']
                else:
                    winner=scores[8];status='fallback_old';assert selected['admissible_optimum_mean_nll'] is None
                    assert not selected['dual_admissible'] and selected['failed_constraints']
                assert selected['choice']==winner['choice'] and selected['selection_status']==status and selected['preference_band']==.001
                for key in ('mean_nll','references','dual_admissible','failed_constraints','preference_order'):
                    assert selected[key]==next(x for x in record['candidate_scores'][f'{ref:g}'][f'{snr:g}'] if x['choice']==winner['choice'])[key]
                wi=study.MENU.index(winner['choice']);margins=[]
                for mi,name in enumerate(study.MENU):
                    margin=math.fsum(map(float,(loss[:,mi]-loss[:,wi]).ravel()))/loss[:,mi].size
                    direct=scores[mi]['mean_nll']-winner['mean_nll'];numeric_error=max(numeric_error,abs(direct-margin))
                    margins.append(dict(choice=name,mean_nll_above_selected=margin))
                reviews.append(dict(rsr=ref,snr=snr,choice=winner['choice'],selection_status=status,
                    admissible=[x['choice'] for x in feasible],failed_constraints=selected['failed_constraints'],centered_margins=margins))
            counts[selected['choice']]+=1;statuses[selected['selection_status']]=statuses.get(selected['selection_status'],0)+1
    assert constraint_count==324
    result=dict(passed=True,repair_selection_sha256=sha(study.OUT/'repair_selection.json'),protocol_sha256=record['protocol_sha256'],
        observable_only_development_anchors_checked=6,repair_loss_tables_checked=6,guard_candidate_checks=54,guard_constraint_checks=324,
        bootstrap_weights_exactly_reconstructed=True,paired_fit_average_before_bootstrap=True,selected_counts=counts,selection_status_counts=statuses,
        original_not_automatically_admissible=True,preference_band_only_after_strict_guards=True,fallback_failed_constraints_retained=True,
        maximum_direct_vs_centered_margin_error=numeric_error,reviews=reviews,audit_source_sha256=sha(__file__),
        caveat='Fallback_old declares comparator priority when no candidate passes both empirical guards; it is not evidence of original-estimator nonharm. Fixed bins make no new guard claim.')
    write_json(ROOT/'repair_selection.json',result);return result


def calibration():
    record=study.check_calibration();policy=study.check_repair();configs=study.base.candidates();raw=record['raw_samples']
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
    review=repair();policy,digest=study.check_selection();cal=study.check_calibration();configs=study.base.candidates()
    paths=sorted((study.OUT/'development/compact_losses').glob('*.npz'));assert len(paths)==51
    assert len(policy['development_anchor_hashes'])==9 and len(policy['development_bank_hashes'])==51 and len(policy['compact_loss_hashes'])==51
    for ref in study.RSRS:
        for snr in study.SNRS:
            path=study.OUT/'development/compact_losses'/f'rsr{ref:g}_snr{snr:g}.npz';a=study.load(path)
            assert str(a['calibration_sha256'])==policy['calibration_sha256']
            assert float(a['rsr'])==ref and float(a['snr'])==snr
            np.testing.assert_array_equal(a['names'],cal['eligible'])
            count=384 if ref in (12.,20.) and snr in (32.5,35.,37.5) else 96
            assert int(a['cases_per_regime'])==count and a['losses'].shape==(len(cal['eligible']),3,count) and np.isfinite(a['losses']).all()
            assert verify_bank(study.OUT/'development/banks'/path.name,policy['protocol_sha256'])==count
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
        selected_counts=review['selected_counts'],selection_status_counts=review['selection_status_counts'],all_dual_comparator_per_regime_upper95_guards_recomputed=True,
        paired_actual_context_holdout_banks_verified=51,development_counts_384_target_96_elsewhere=True,
        source_sha256=sha(__file__),caveat=review['caveat'])
    write_json(ROOT/'selection.json',result);return result


@torch.no_grad()
def test():
    policy,digest=study.check_selection();files=sorted((study.OUT/'test/cells').glob('*.npz'));assert len(files)==153
    anchors=sorted((study.OUT/'test/anchors').glob('*.npz'));assert len(anchors)==9
    anchor_map={};anchor_hashes={};hashes={};ordinary=weak=stronglow=stronghigh=warm_checks=original_checks=spectral_checks=0;warm_error=0.
    for path in anchors:
        a=study.load(path);assert int(a['seed'])==2026101932 and str(a['selection_sha256'])==digest and len(a['g'])==128
        anchor_map[path.stem]=a;anchor_hashes[path.name]=sha(path)
    for path in files:
        a=study.load(path);assert str(a['selection_sha256'])==digest and str(a['protocol_sha256'])==policy['protocol_sha256'] and int(a['seed'])==2026101932
        np.testing.assert_array_equal(a['method_names'],study.NAMES);assert a['estimates'].shape==(17,128,32,3)
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
        np.testing.assert_array_equal(pred[3:6],np.repeat(pred[12:13],3,axis=0) if old_on else pred[:3])
        priorchoice=chosen(study.PRIOR_REPAIR,ref,snr);choice=chosen(policy,ref,snr)
        assert str(a['prior_repair_choice'])==priorchoice and str(a['repair_choice'])==choice
        for start,which in ((6,priorchoice),(9,choice)):
            if which=='original':np.testing.assert_array_equal(pred[start:start+3],pred[:3]);original_checks+=1
            elif which=='spectral100':np.testing.assert_array_equal(pred[start:start+3],np.repeat(pred[12:13],3,axis=0));spectral_checks+=1
            else:
                # Independent canonical EM replay on archived measurements and saved learned starts.
                small=Observations(*(torch.from_numpy(a[k][:1]) for k in ('z','s','b','sigma2')))
                op=right_solve(small.s.mH,small.s@small.s.mH)
                for i in range(3):
                    g=torch.from_numpy(pred[i,:1].copy())
                    for _ in range(int(which[4:])):g=(posterior_field(small.z,g@small.s+small.b,small.sigma2)-small.b)@op
                    error=float(np.abs(g.numpy()-pred[i+start,:1]).max());warm_error=max(warm_error,error);assert error<1e-10;warm_checks+=1
        if snr<=20:
            for start in (3,6,9):np.testing.assert_array_equal(pred[start:start+3],pred[:3])
            ordinary+=1
        if ref==5:np.testing.assert_array_equal(pred[9:12],pred[6:9]);weak+=1
        if ref in (12.,20.) and snr<=30:np.testing.assert_array_equal(pred[9:12],pred[:3]);stronglow+=1
        if ref in (12.,20.) and snr>=40:np.testing.assert_array_equal(pred[9:12],np.repeat(pred[12:13],3,axis=0));stronghigh+=1
        assert bool(a['ordinary_range_exact_identity'])==(snr<=20) and bool(a['weak_reference_exact_identity'])==(ref==5)
        assert float(a['actual_dispatch_max_abs_error'])<1e-10
        hashes[path.name]=sha(path)
    assert (ordinary,weak,stronglow,stronghigh)==(45,51,54,30)
    earliest=min(p.stat().st_mtime for p in files+anchors);assert earliest>=policy['frozen_unix']
    result=dict(passed=True,selection_sha256=digest,protocol_sha256=policy['protocol_sha256'],test_cells=153,test_anchors=9,
        ordinary_exact_identity_cells=ordinary,weak_reference_prior_repair_identity_cells=weak,strong_reference_original_identity_cells=stronglow,
        strong_reference_spectral_identity_cells=stronghigh,original_repair_identity_cells=original_checks,spectral_repair_identity_cells=spectral_checks,
        canonical_warm_replays=warm_checks,maximum_canonical_warm_error=warm_error,
        raw_errors_observations_truth_and_branches_verified=True,earliest_test_file_after_freeze_seconds=earliest-policy['frozen_unix'],
        anchor_hashes=anchor_hashes,test_prediction_hashes=hashes,audit_source_sha256=sha(__file__))
    write_json(ROOT/'test.json',result);return result


def belongs(region,snr,ref,regime):
    weak=ref==5 and 32.5<=snr<=45;strong=ref in (12.,20.) and 32.5<=snr<=37.5
    historical=study.cellname(regime,ref,snr) in HISTORICAL
    if region.startswith('strong_transition_snr'):return strong and snr==float(region[len('strong_transition_snr'):])
    for prefix,match in (('weak_transition_',weak),('strong_transition_',strong),('historical_flagged_',historical)):
        if region.startswith(prefix):return match and regime==region[len(prefix):]
    return dict(ordinary=snr<=20,transition=20<snr<=45,stress=snr>=50,high_snr=snr>20,all=True,
        weak_transition=weak,strong_transition=strong,historical_flagged=historical)[region]


def summary():
    """All point tables, all aggregate interval quantiles, 32 independent bootstraps."""
    policy,digest=study.check_selection();report=json.loads((study.OUT/'nmse_summary.json').read_text())
    assert report['selection_sha256']==digest and report['timing_sha256']==sha(study.OUT/'timing.json')
    timing_record=json.loads((study.OUT/'timing.json').read_text());assert report['timing']==timing_record['methods']
    groups=study.GROUPS;methods=list(study.NAMES)+list(groups);index={n:i for i,n in enumerate(methods)}
    def expand(v):return np.concatenate([v,np.stack([v[[index[n] for n in part]].mean(0) for part in groups.values()])])
    rng=np.random.default_rng(2026101939)
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
            primary=region=='strong_transition' and key in ('repair_vs_old','repair_vs_prior')
            assert ('bonferroni4_interval_db' in item)==primary
            if primary:close(item['bonferroni4_interval_db'],np.quantile(d,[.05/8,1-.05/8]))
            for seed in study.SEEDS:
                source=f'{reference[:-5]}_{seed}' if reference.endswith('_mean') else reference
                close(item['seed_specific_gains_db'][str(seed)],point[index[source]]-point[index[f'repair_{seed}']])
    assert counts==dict(ordinary=45,transition=90,stress=18,high_snr=108,all=153,weak_transition=18,
        weak_transition_L2=6,weak_transition_L4=6,weak_transition_random3to7=6,strong_transition=18,
        strong_transition_L2=6,strong_transition_L4=6,strong_transition_random3to7=6,
        **{'strong_transition_snr32.5':6,'strong_transition_snr35':6,'strong_transition_snr37.5':6},
        historical_flagged=10,historical_flagged_L2=5,historical_flagged_L4=2,historical_flagged_random3to7=3)
    assert report['numerical_tie_tolerance_db']==1e-10
    for key in study.CONTRASTS:
        item=report['negative_cells'][key];gains=[r['contrasts'][key]['gain_db'] for r in rows]
        assert item['negative_cells']==[r['cell'] for r in rows if r['contrasts'][key]['gain_db'] < -1e-10]
        assert item['pointwise95_harm_cells']==[r['cell'] for r in rows if r['contrasts'][key]['pointwise_ci95_db'][1] < -1e-10]
        assert item['target_negative_cells']==[r['cell'] for r in rows if belongs('strong_transition',r['snr'],r['rsr'],r['regime']) and r['contrasts'][key]['gain_db'] < -1e-10]
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
            priorchoice=chosen(study.PRIOR_REPAIR,row['rsr'],row['snr']);assert row['prior_repair_choice']==priorchoice
            old=row['snr']>study.OLD_POLICY[f"{row['rsr']:g}"]+1e-10 and row['snr']>20+1e-10
            expected=choice if name.startswith('repair_') else priorchoice if name.startswith('prior_repair_') else ('spectral100' if old else 'original') if name.startswith('old_bypass_') else 'original' if name.startswith('original_') else name
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
    result=dict(passed=True,selection_sha256=digest,worlds=153,repeats=3,individual_methods=17,raw_calls=sum(map(len,raw.values())),
        all_quantiles_condition_counts_and_branches_verified=True,cap_verification=expected,timing_sha256=sha(study.OUT/'timing.json'),audit_source_sha256=sha(__file__),
        caveat='Audit verifies the retained latency samples, not host isolation or a worst-case runtime guarantee.')
    write_json(ROOT/'timing.json',result);return result


def sensitivity():
    """Independent raw points, conditional intervals, routing and main parity."""
    from . import sensitivity as module
    policy,digest=study.check_selection();base=json.loads((module.ROOT/'audit.json').read_text());assert base['passed']
    report=json.loads((module.ROOT/'summary.json').read_text());lookup={(r['regime'],r['rsr'],r['actual_snr_db'],r['supplied_variance_offset_db']):r for r in report['cells']}
    assert len(lookup)==45 and report['unique_channel_reference_anchors']==1152 and report['coupled_case_index_clusters']==384
    assert report['selection_sha256']==digest and base['source_sha256']==sha(module.__file__)
    rng=np.random.default_rng(2026101938)
    weights={r:rng.multinomial(128,np.full(128,1/128),size=4000).T for r,_ in study.REGIMES}
    zero=weak=0;maximum=0.;routes={};boots={};cell_hashes={}
    for condition in module.conditions():
        regime,ref,snr,offset=(condition[k] for k in ('regime','rsr','actual_snr','offset'))
        path=module.cellpath(regime,ref,snr,offset);a=study.load(path);assert str(a['selection_sha256'])==digest
        anchorpath=study.OUT/'test/anchors'/f'{regime}_rsr{ref:g}.npz';anchor=study.load(anchorpath)
        assert str(a['anchor_sha256'])==sha(anchorpath)
        errors=(np.abs(a['estimates']-anchor['g'][None])**2).sum((-2,-1));energy=(np.abs(anchor['g'])**2).sum((-2,-1))
        np.testing.assert_array_equal(errors,a['errors']);np.testing.assert_array_equal(energy,a['truth_energy'])
        proxy=snr-offset;grid=min(study.SNRS,key=lambda x:(abs(x-proxy),x))
        choice='original' if proxy<=20+1e-10 else policy['repair_choices'][f'{ref:g}'][f'{grid:g}']['choice']
        assert str(a['repair_choice'])==choice;routes[choice]=routes.get(choice,0)+1
        row=lookup[(regime,ref,snr,offset)];assert row['repair_choice']==choice and row['supplied_snr_proxy_db']==proxy
        point=10*np.log10(errors.sum(1)/energy.sum());fitboot=10*np.log10((errors@weights[regime])/(energy@weights[regime])[None]);groupboot={}
        for group,names in module.GROUPS.items():
            ids=[module.NAMES.index(n) for n in names];expected=float(point[ids].mean());difference=abs(row['methods'][group]['nmse_db']-expected)
            maximum=max(maximum,difference);assert difference<1e-12
            close(row['methods'][group]['per_fit_nmse_db'],point[ids]);groupboot[group]=fitboot[ids].mean(0)
            close(row['methods'][group]['bootstrap95'],np.quantile(groupboot[group],[.025,.975]),1e-10)
            close(row['methods'][group]['severe_nmse_above1_rate'],float((errors[ids]/energy[None]>1).mean()))
        for reference in ('original_mean','old_bypass_mean','prior_repair_mean','em100','compact_sparse'):
            item=row['contrasts']['repair_vs_'+reference]
            close(item['gain_db'],row['methods'][reference]['nmse_db']-row['methods']['repair_mean']['nmse_db'])
            close(item['bootstrap95'],np.quantile(groupboot[reference]-groupboot['repair_mean'],[.025,.975]),1e-10)
        boots[(regime,ref,snr,offset)]=groupboot
        if ref==5:np.testing.assert_array_equal(a['estimates'][9:12],a['estimates'][6:9]);weak+=1
        if offset==0:
            main=study.load(study.OUT/'test/cells'/f'{study.cellname(regime,ref,snr)}.npz')
            for i,name in enumerate(module.NAMES):np.testing.assert_array_equal(a['estimates'][i],main['estimates'][study.NAMES.index(name)])
            zero+=1
        cell_hashes[path.name]=sha(path)
    for key,row in lookup.items():
        correctkey=(*key[:3],0.);correct=lookup[correctkey]
        for group,value in row['methods'].items():
            close(value['degradation_vs_correct_variance_db'],value['nmse_db']-correct['methods'][group]['nmse_db'])
            close(value['degradation_bootstrap95'],np.quantile(boots[key][group]-boots[correctkey][group],[.025,.975]),1e-10)
    assert zero==15 and weak==27
    result=dict(passed=True,conditions=45,zero_offset_full_prediction_identity_cells=zero,weak_reference_prior_identity_cells=weak,
        independently_checked_nearest_bin_routes=routes,maximum_point_error=maximum,selection_sha256=digest,
        independently_reconstructed_all_4000_bootstrap_intervals=True,cell_hashes=cell_hashes,
        sensitivity_summary_sha256=sha(module.ROOT/'summary.json'),sensitivity_primary_audit_sha256=sha(module.ROOT/'audit.json'),audit_source_sha256=sha(__file__))
    write_json(ROOT/'sensitivity.json',result);return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=('repair','calibration','selection','test','summary','timing','sensitivity'))
    args=parser.parse_args();torch.set_num_threads(1)
    try:torch.set_num_interop_threads(1)
    except RuntimeError:pass
    result=globals()[args.phase]()
    print(json.dumps({k:v for k,v in result.items() if k not in ('reviews','anchor_hashes','test_prediction_hashes')},indent=2))
