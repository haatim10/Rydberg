"""Independent verification of the frozen four-incumbent guard study artifacts."""
import os
for _name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[_name]='1'
import argparse
import json
import math
import numpy as np
import torch
from sahure.data import Observations
from sahure.math import posterior_field,right_solve,nll
from tcom_followups_20261017.common import sha,write_json
from . import study

ROOT=study.OUT/'audit'
REGIONS=('ordinary','transition','stress','high_snr','all','weak_transition','weak_transition_L2','weak_transition_L4','weak_transition_random3to7',
    'strong_transition','strong_transition_L2','strong_transition_L4','strong_transition_random3to7',
    'strong_transition_snr32.5','strong_transition_snr35','strong_transition_snr37.5',
    'historical_flagged','historical_flagged_L2','historical_flagged_L4','historical_flagged_random3to7')
HISTORICAL=('L2_rsr12_snr37.5','L2_rsr20_snr32.5','L2_rsr20_snr35','L2_rsr20_snr37.5','L4_rsr12_snr37.5','L4_rsr20_snr35','L4_rsr20_snr37.5','random3to7_rsr12_snr37.5')


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
        assert int(a['seed'])==2026101941 and len(a['s'])==(384 if ref in (12.,20.) else 96)
        obs=study.observed(a,snr,P=39).batch(slice(0,count))
        for prefix,expected in zip(('context','heldout'),(obs.columns(torch.arange(30).expand(count,-1)),obs.columns(torch.arange(30,39).expand(count,-1)))):
            for key,value in zip(('z','s','b','sigma2'),expected.args()):
                np.testing.assert_array_equal(bank[f'{prefix}_{key}'][pi*count:(pi+1)*count],value.numpy())
    return count


@torch.no_grad()
def repair():
    record=study.check_repair();protocol=json.loads((study.OUT/'protocol.json').read_text())
    assert len(record['development_anchor_hashes'])==len(record['development_bank_hashes'])==len(record['loss_table_hashes'])==6
    rng=np.random.default_rng(2026101947)
    weights={r:rng.multinomial(384,np.full(384,1/384),size=4000) for r,_ in study.REGIMES}
    saved=study.load(study.OUT/'development/repair_bootstrap_weights.npz')
    for regime,w in weights.items():np.testing.assert_array_equal(saved[regime],w)
    counts={name:0 for name in study.MENU};statuses={};reviews=[];constraints=0;est=study.engine();risk_replays=0
    for ref in study.RSRS:
        for snr in study.SNRS:
            r,k=f'{ref:g}',f'{snr:g}';selected=record['repair_choices'][r][k]
            if ref not in (12.,20.) or snr not in (32.5,35.,37.5):
                assert selected['choice']==study.CURRENT_REPAIR['repair_choices'][r][k]['choice'] and selected['selection_status']=='fixed_current'
                assert record['candidate_scores'][r][k]==[]
            else:
                path=study.OUT/'development/banks'/f'rsr{r}_snr{k}.npz';assert verify_bank(path,record['protocol_sha256'])==384
                assert path.stat().st_mtime>=protocol['frozen_unix']
                a=study.load(study.OUT/'development/repair_losses'/path.name)
                assert str(a['protocol_sha256'])==record['protocol_sha256']
                np.testing.assert_array_equal(a['names'],study.MENU);np.testing.assert_array_equal(a['seeds'],study.SEEDS)
                assert a['fit_nll'].shape==a['heldout_nll'].shape==(3,36,3,384) and a['paired_risk'].shape==(3,36,4,3,384)
                assert all(np.isfinite(a[q]).all() for q in ('fit_nll','heldout_nll','paired_risk'))
                # Separate explicit intensity-moment calculation on six anchors
                # per bin validates saved risks and heldout likelihoods.
                bank=study.load(path);ids=np.array([0,1,384,385,768,769]);local=np.array([0,1])
                c,h=(Observations(*(torch.from_numpy(bank[f'{prefix}_{key}'][ids]) for key in ('z','s','b','sigma2'))) for prefix in ('context','heldout'))
                refs=study.reference_choices(ref,snr)
                for fi,seed in enumerate(study.SEEDS):
                    values=est.candidate_predictions(c,f'learned_{seed}')
                    for mi,name in enumerate(study.MENU):
                        left=values[name].numpy();pilots=h.s.numpy();reference=h.b.numpy()
                        actual_nll=nll(h.z,values[name]@h.s+h.b,h.sigma2).mean((-2,-1)).numpy()
                        expected_nll=a['heldout_nll'][fi,mi,:,local].T.reshape(-1)
                        close(actual_nll,expected_nll,1e-9)
                        for bi,baseline in enumerate(study.REFERENCES):
                            right=values[refs[baseline]].numpy();midpoint=(left+right)/2
                            moment=(h.z.numpy()**2-abs(reference+midpoint@pilots)**2-h.sigma2.numpy()[:,None,None])/reference.conj()
                            score=-2*np.real(((left-right)@pilots).conj()*moment).sum((-2,-1))/(9*96)
                            expected=a['paired_risk'][fi,mi,bi,:,local].T.reshape(-1)
                            close(score,expected,1e-10);risk_replays+=len(ids)
                feasible=[]
                for mi,name in enumerate(study.MENU):
                    stored=record['candidate_scores'][r][k][mi];assert stored['choice']==name
                    failed=[]
                    for bi,baseline in enumerate(study.REFERENCES):
                        paired=a['paired_risk'][:,mi,bi].mean(0)
                        close(stored['references'][baseline]['mean_delta'],paired.mean())
                        for pi,(regime,_) in enumerate(study.REGIMES):
                            upper=float(np.quantile(paired[pi]@weights[regime].T/384,.95))
                            entry=stored['references'][baseline]['by_regime'][regime]
                            close(entry['mean_delta'],paired[pi].mean());close(entry['upper95'],upper,1e-12)
                            if upper>0:failed.append(f'{baseline}:{regime}')
                            constraints+=1
                    close(stored['mean_nll'],a['heldout_nll'][:,mi].mean())
                    close(stored['objective_mean_risk_vs_current'],a['paired_risk'][:,mi,3].mean())
                    assert stored['failed_constraints']==failed and stored['all_incumbent_admissible']==(not failed)
                    if not failed:feasible.append(stored)
                if feasible:
                    winner=min(feasible,key=lambda x:(x['objective_mean_risk_vs_current'],x['menu_index']));status='all_incumbent_admissible'
                else:
                    winner=record['candidate_scores'][r][k][study.MENU.index(study.CURRENT_REPAIR['repair_choices'][r][k]['choice'])];status='fallback_current'
                assert selected==dict(winner,selection_status=status)
                reviews.append(dict(rsr=ref,snr=snr,choice=winner['choice'],selection_status=status,admissible=[x['choice'] for x in feasible]))
            counts[selected['choice']]+=1;statuses[selected['selection_status']]=statuses.get(selected['selection_status'],0)+1
    assert constraints==2592
    result=dict(passed=True,repair_selection_sha256=sha(study.OUT/'repair_selection.json'),protocol_sha256=record['protocol_sha256'],
        selected_counts=counts,selection_status_counts=statuses,reviews=reviews,guard_constraint_checks=constraints,risk_and_nll_anchor_replays=risk_replays,
        observable_development_banks_verified=6,strict_four_incumbent_guards_and_objective_reconstructed=True,
        source_sha256=sha(__file__),caveat='Empirical development screening; fallback preserves current policy but does not satisfy missing guards.')
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
    assert len(policy['development_anchor_hashes'])==len(policy['development_bank_hashes'])==len(policy['compact_loss_hashes'])==6
    changed=fixed=0
    for ref in study.RSRS:
        for snr in study.SNRS:
            r,k=f'{ref:g}',f'{snr:g}'
            if ref not in (12.,20.) or snr not in (32.5,35.,37.5):
                assert policy['compact_choices'][r][k]==study.PRIOR_COMPACT['compact_choices'][r][k] and policy['compact_tables'][r][k]==[];fixed+=1;continue
            path=study.OUT/'development/compact_losses'/f'rsr{r}_snr{k}.npz';a=study.load(path)
            assert str(a['calibration_sha256'])==policy['calibration_sha256'];np.testing.assert_array_equal(a['names'],cal['eligible'])
            assert a['losses'].shape==(len(cal['eligible']),3,384) and np.isfinite(a['losses']).all()
            scores=[dict(candidate=key,config=configs[key],mean_nll=float(a['losses'][i].mean()),by_regime=a['losses'][i].mean(-1).tolist(),p90_ms=cal['candidates'][key]['p90_ms']) for i,key in enumerate(cal['eligible'])]
            assert policy['compact_tables'][r][k]==scores
            assert policy['compact_choices'][r][k]==min(scores,key=lambda x:(x['mean_nll'],x['p90_ms'],x['candidate']))
            assert path.stat().st_mtime>=cal['completed_unix'];changed+=1
    assert fixed==45 and changed==6 and policy['frozen_unix']>=cal['completed_unix']
    result=dict(passed=True,selection_sha256=digest,protocol_sha256=policy['protocol_sha256'],repair_review_sha256=sha(ROOT/'repair_selection.json'),
        repair_choices_checked=51,compact_choices_checked=51,compact_retuned_bins=changed,compact_fixed_bins=fixed,
        selected_counts=review['selected_counts'],selection_status_counts=review['selection_status_counts'],source_sha256=sha(__file__))
    write_json(ROOT/'selection.json',result);return result


@torch.no_grad()
def test():
    policy,digest=study.check_selection();files=sorted((study.OUT/'test/cells').glob('*.npz'));assert len(files)==153
    anchors=sorted((study.OUT/'test/anchors').glob('*.npz'));assert len(anchors)==9
    amap={};hashes={};ordinary=weak=outside=0;maximum=0.;est=study.engine()
    for path in anchors:
        a=study.load(path);assert int(a['seed'])==2026101942 and str(a['selection_sha256'])==digest and len(a['g'])==128
        amap[path.stem]=a
    for path in files:
        a=study.load(path);assert str(a['selection_sha256'])==digest and int(a['seed'])==2026101942
        np.testing.assert_array_equal(a['method_names'],study.NAMES);pred=a['estimates'];assert pred.shape==(20,128,32,3) and np.isfinite(pred).all()
        ref,snr,regime=float(a['rsr']),float(a['snr']),str(a['regime']);anchor=amap[f'{regime}_rsr{ref:g}']
        for key,other in (('truth_g','g'),('indices','indices'),('local_indices','local_indices'),('paths','paths')):np.testing.assert_array_equal(a[key],anchor[other])
        obs=study.observed(anchor,snr)
        for key,value in zip(('z','s','b','sigma2'),obs.args()):np.testing.assert_array_equal(a[key],value.numpy())
        err=(abs(pred-anchor['g'][None])**2).sum((-2,-1));energy=(abs(anchor['g'])**2).sum((-2,-1))
        np.testing.assert_array_equal(a['errors'],err);np.testing.assert_array_equal(a['truth_energy'],energy);np.testing.assert_array_equal(a['nmse'],err/energy[None])
        old=snr>study.OLD_POLICY[f'{ref:g}']+1e-10 and snr>20+1e-10
        np.testing.assert_array_equal(a['old_bypass'],np.full(128,old));np.testing.assert_array_equal(pred[3:6],pred[15:16].repeat(3,axis=0) if old else pred[:3])
        for i,name in enumerate(study.NAMES):
            if not name.startswith(('prior_repair_','current_repair_','repair_')):continue
            replay=study.estimate_method(obs.batch(slice(0,1)),name,policy,est).numpy()
            error=float(abs(replay-pred[i,:1]).max());maximum=max(maximum,error);assert error<1e-10
        if snr<=20:
            for j in (3,6,9,12):np.testing.assert_array_equal(pred[j:j+3],pred[:3])
            ordinary+=1
        if ref==5:np.testing.assert_array_equal(pred[12:15],pred[6:9]);weak+=1
        if ref not in (12,20) or snr not in (32.5,35,37.5):np.testing.assert_array_equal(pred[12:15],pred[9:12]);outside+=1
        hashes[path.name]=sha(path)
    assert (ordinary,weak,outside)==(45,51,135)
    earliest=min(p.stat().st_mtime for p in files+anchors);assert earliest>=policy['frozen_unix']
    result=dict(passed=True,selection_sha256=digest,protocol_sha256=policy['protocol_sha256'],test_cells=153,test_anchors=9,
        ordinary_exact_identity_cells=ordinary,weak_prior_identity_cells=weak,outside_current_identity_cells=outside,
        maximum_scalar_dispatch_error=maximum,raw_errors_and_observations_verified=True,earliest_test_file_after_freeze_seconds=earliest-policy['frozen_unix'],
        test_prediction_hashes=hashes,source_sha256=sha(__file__))
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
    rng=np.random.default_rng(2026101949)
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
            primary=region=='strong_transition' and key in ('repair_vs_old','repair_vs_prior','repair_vs_current')
            assert ('bonferroni6_interval_db' in item)==primary
            if primary:close(item['bonferroni6_interval_db'],np.quantile(d,[.05/12,1-.05/12]))
            for seed in study.SEEDS:
                source=f'{reference[:-5]}_{seed}' if reference.endswith('_mean') else reference
                close(item['seed_specific_gains_db'][str(seed)],point[index[source]]-point[index[f'repair_{seed}']])
    assert counts==dict(ordinary=45,transition=90,stress=18,high_snr=108,all=153,weak_transition=18,
        weak_transition_L2=6,weak_transition_L4=6,weak_transition_random3to7=6,strong_transition=18,
        strong_transition_L2=6,strong_transition_L4=6,strong_transition_random3to7=6,
        **{'strong_transition_snr32.5':6,'strong_transition_snr35':6,'strong_transition_snr37.5':6},
        historical_flagged=8,historical_flagged_L2=4,historical_flagged_L4=3,historical_flagged_random3to7=1)
    assert report['numerical_tie_tolerance_db']==1e-10
    for key in study.CONTRASTS:
        item=report['negative_cells'][key];gains=[r['contrasts'][key]['gain_db'] for r in rows]
        assert item['negative_cells']==[r['cell'] for r in rows if r['contrasts'][key]['gain_db'] < -1e-10]
        assert item['pointwise95_harm_cells']==[r['cell'] for r in rows if r['contrasts'][key]['pointwise_ci95_db'][1] < -1e-10]
        assert item['target_negative_cells']==[r['cell'] for r in rows if belongs('strong_transition',r['snr'],r['rsr'],r['regime']) and r['contrasts'][key]['gain_db'] < -1e-10]
        worst=int(np.argmin(gains));assert item['worst_cell']==rows[worst]['cell'];close(item['worst_gain_db'],gains[worst])
    result=dict(passed=True,selection_sha256=digest,cells=153,region_counts=counts,maximum_point_error=maximum,
        independently_recomputed_bootstrap_replicates=32,maximum_bootstrap_error=booterror,
        all_points_pooled_nmse_tail_rates_per_fit_gains_verified=True,all_aggregate_intervals_and_only_declared_bonferroni6_verified=True,
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
            expected=choice if name.startswith('repair_') else chosen(study.CURRENT_REPAIR,row['rsr'],row['snr']) if name.startswith('current_repair_') else priorchoice if name.startswith('prior_repair_') else ('spectral100' if old else 'original') if name.startswith('old_bypass_') else 'original' if name.startswith('original_') else name
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
    result=dict(passed=True,selection_sha256=digest,worlds=153,repeats=3,individual_methods=20,raw_calls=sum(map(len,raw.values())),
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
    rng=np.random.default_rng(2026101948)
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
        for reference in ('original_mean','old_bypass_mean','prior_repair_mean','current_repair_mean','em100','compact_sparse'):
            item=row['contrasts']['repair_vs_'+reference]
            close(item['gain_db'],row['methods'][reference]['nmse_db']-row['methods']['repair_mean']['nmse_db'])
            close(item['bootstrap95'],np.quantile(groupboot[reference]-groupboot['repair_mean'],[.025,.975]),1e-10)
        boots[(regime,ref,snr,offset)]=groupboot
        if ref==5:np.testing.assert_array_equal(a['estimates'][12:15],a['estimates'][6:9]);weak+=1
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
