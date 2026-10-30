"""Independent recomputation of frozen selections, predictions and reported scores."""
import os
for _name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[_name]='1'
import argparse
import json
import math
from pathlib import Path
import numpy as np
from tcom_followups_20261017.common import sha,write_json
from . import study

ROOT=study.OUT/'audit'


def numeric_stats(rows):
    values=np.array([x['ms'] for x in rows]);assert np.isfinite(values).all() and (values>0).all()
    return dict(p90_ms=float(np.quantile(values,.9)),median_ms=float(np.median(values)),mean_ms=float(values.mean()),calls=len(values))


def verify_stats(expected,record):
    for key,value in expected.items():np.testing.assert_allclose(record[key],value,atol=1e-12,rtol=0)


def calibration():
    record=study.check_calibration();configs=study.candidates();raw=record['raw_samples']
    assert set(record['candidates'])==set(configs)
    for name,rows in raw.items():
        expected=54 if name.startswith('em100branch_') else 108
        assert len(rows)==expected and all(int(r['index'])==100 for r in rows)
        assert set(float(r['snr']) for r in rows)==({40.,60.} if name.startswith('em100branch_') else {0.,20.,40.,60.})
        verify_stats(numeric_stats(rows),record['methods'][name])
    branch={name:numeric_stats([row for seed in study.SEEDS for row in raw[f'{name}_{seed}']]) for name in ('filtered','em100branch')}
    for name,values in branch.items():verify_stats(values,record['branch_timings'][name])
    cap=max(values['p90_ms'] for values in branch.values());assert cap==record['cap_ms']
    eligible=[]
    for name,config in configs.items():
        assert record['candidates'][name]['config']==config
        values=numeric_stats(raw[f'compact_{name}']);verify_stats(values,record['candidates'][name])
        if values['p90_ms']<=cap:eligible.append(name)
    assert eligible==record['eligible']
    result=dict(passed=True,protocol_sha256=record['protocol_sha256'],calibration_sha256=sha(study.OUT/'calibration.json'),
        candidate_count=len(configs),eligible=eligible,cap_ms=cap,cap_recomputed_from_actual_branch_p90=True,
        every_raw_latency_and_quantile_checked=True,audit_source_sha256=sha(__file__),
        caveat='Pooled calibration quantiles establish eligibility; they are not hard deadlines or proof of timing isolation. Final observed cap overshoots remain reportable.')
    write_json(ROOT/'calibration.json',result);return result


def selection():
    """Recompute cutoff and sparse choices; bootstrap calibration-bank sensitivity."""
    policy,digest=study.check_selection();calibration_record=study.check_calibration();configs=study.candidates()
    lossfiles=sorted((study.OUT/'development/losses').glob('*.npz'));assert len(lossfiles)==51
    anchorfiles=sorted((study.OUT/'development/anchors').glob('*.npz'));assert len(anchorfiles)==9
    for path in anchorfiles:
        data=study.load(path)
        assert not {'g','G','unit_noise','paths','mu','truth_g'}.intersection(data)
        assert int(data['seed'])==study.DEV_SEED and data['z_by_snr'].shape==(17,48,32,39)
        assert data['s'].shape==(48,3,39) and data['b'].shape==(48,32,39)
        assert path.stat().st_mtime>=calibration_record['completed_unix']
    rng=np.random.default_rng(2026101827)
    indices=[rng.integers(0,48,size=(4000,48)) for _ in study.REGIMES]
    reviews={};compact_checked=0;numerical_max=0.
    for ref in study.RSRS:
        cells=[study.load(study.OUT/'development/losses'/f'rsr{ref:g}_snr{snr:g}.npz') for snr in study.SNRS]
        expected_names=[f'original_{s}' for s in study.SEEDS]+[f'off_{s}' for s in study.SEEDS]+['em100']+[f'compact_{key}' for key in calibration_record['eligible']]
        for cell,snr in zip(cells,study.SNRS):
            np.testing.assert_array_equal(cell['names'],expected_names)
            np.testing.assert_array_equal(cell['regimes'],[r for r,_ in study.REGIMES])
            assert float(cell['snr'])==snr and float(cell['rsr'])==ref
            assert cell['losses'].shape==(len(expected_names),3,48) and np.isfinite(cell['losses']).all()
            assert str(cell['protocol_sha256'])==policy['protocol_sha256'] and str(cell['calibration_sha256'])==policy['calibration_sha256']
            scores=[]
            for key in calibration_record['eligible']:
                loss=cell['losses'][expected_names.index(f'compact_{key}')]
                score=dict(candidate=key,mean_nll=float(loss.mean()),p90_ms=calibration_record['candidates'][key]['p90_ms'])
                scores.append(score)
                recorded=next(r for r in policy['compact_tables'][f'{ref:g}'][f'{snr:g}'] if r['candidate']==key)
                np.testing.assert_allclose(recorded['mean_nll'],score['mean_nll'],atol=1e-12,rtol=0)
                np.testing.assert_allclose(recorded['by_regime'],loss.mean(-1),atol=1e-12,rtol=0)
                assert recorded['config']==configs[key]
            winner=min(scores,key=lambda x:(x['mean_nll'],x['p90_ms'],x['candidate']))
            selected=policy['compact_choices'][f'{ref:g}'][f'{snr:g}']
            assert winner['candidate']==selected['candidate'] and selected['config']==configs[winner['candidate']]
            compact_checked+=1
        for kind in ('same_steps','em100'):
            values=[];direct=[];excess=[];branches=[]
            for cutoff in study.CUTOFFS:
                by_snr=[];direct_snr=[];excess_snr=[]
                for cell in cells:
                    original=cell['losses'][:3];fallback=cell['losses'][3:6] if kind=='same_steps' else np.broadcast_to(cell['losses'][6:7],original.shape)
                    use=float(cell['snr'])>cutoff+1e-10 and float(cell['snr'])>20+1e-10
                    chosen=fallback if use else original
                    by_snr.append(chosen.mean(0));direct_snr.append(float(chosen.mean()));excess_snr.append(float((chosen-original).mean()))
                values.append(np.stack(by_snr));direct.append(float(np.mean(direct_snr)));excess.append(float(np.mean(excess_snr)))
            values=np.stack(values);direct=np.array(direct);excess=np.array(excess)
            selected=float(policy['policies'][kind][f'{ref:g}']);selected_index=list(study.CUTOFFS).index(selected)
            winner=min(range(len(study.CUTOFFS)),key=lambda i:(excess[i],-study.CUTOFFS[i]))
            assert selected_index==winner
            recorded=policy['validation_tables'][kind][f'{ref:g}']
            np.testing.assert_allclose([r['mean_nll'] for r in recorded],direct,atol=1e-12,rtol=0)
            np.testing.assert_allclose([r['excess_nll'] for r in recorded],excess,atol=1e-12,rtol=0)
            # Subtract the frozen chosen candidate before summation: shared large high-SNR terms cancel locally.
            centered=values-values[selected_index:selected_index+1]
            margins=np.array([math.fsum(map(float,row.ravel()))/row.size for row in centered])
            direct_margin=direct-direct[selected_index];excess_margin=excess-excess[selected_index]
            numeric=float(max(np.abs(direct_margin-margins).max(),np.abs(excess_margin-margins).max()))
            numerical_max=max(numerical_max,numeric)
            assert margins.min()>=-max(1e-10,numeric*2),'centered arithmetic changes cutoff winner'
            boot=np.zeros((len(study.CUTOFFS),4000))
            # SNR trajectories share indices, so average SNR first to keep memory bounded.
            trajectory_means=centered.mean(1)
            for region,ids in enumerate(indices):boot+=trajectory_means[:,region,ids].mean(-1)/len(study.REGIMES)
            # Reversing candidate order implements the declared larger-cutoff tie break.
            selected_boot=len(study.CUTOFFS)-1-np.argmin(boot[::-1],axis=0)
            counts=np.bincount(selected_boot,minlength=len(study.CUTOFFS))
            positive=margins[margins>0];smallest=float(positive.min()) if len(positive) else None
            for snr,cell in zip(study.SNRS,cells):
                original=cell['losses'][:3];fallback=cell['losses'][3:6] if kind=='same_steps' else np.broadcast_to(cell['losses'][6:7],original.shape)
                branches.append(dict(snr=snr,fallback_minus_original_nll=float((fallback-original).mean()),
                    by_regime=(fallback-original).mean((0,2)).tolist()))
            reviews[f'{kind}_rsr{ref:g}']=dict(selected_cutoff_db=selected,mean_nll=float(direct[selected_index]),
                maximum_direct_or_excess_vs_centered_margin_error=numeric,smallest_positive_candidate_margin=smallest,
                numerical_error_to_smallest_margin_ratio=numeric/smallest if smallest else None,
                bootstrap_selected_cutoff_frequencies={f'{cutoff:g}':float(count/4000) for cutoff,count in zip(study.CUTOFFS,counts)},
                selected_cutoff_bootstrap_frequency=float(counts[selected_index]/4000),
                candidate_margins=[dict(cutoff_db=float(c),centered_mean_nll_above_selected=float(margins[i]),
                    conditional_development_bootstrap95=np.quantile(boot[i],[.025,.975]).tolist()) for i,c in enumerate(study.CUTOFFS)],
                branch_loss_differences=branches)
    result=dict(passed=True,selection_sha256=digest,protocol_sha256=policy['protocol_sha256'],
        observable_only_development_anchors_checked=9,loss_tables_checked=51,compact_selections_checked=compact_checked,
        reviews=reviews,maximum_numerical_margin_error=numerical_max,audit_source_sha256=sha(__file__),
        bootstrap=dict(replicates=4000,seed=2026101827,unit='48 anchor indices per path regime, shared across SNR/reference/candidate/fitted seed',
            scope='Finite-development-bank stability conditional on the three trained models and declared grid; not training-population or deployment uncertainty.'),
        interpretation='Cutoff and compact selections independently recomputed from observable heldout losses. Positive centered margin favors selected cutoff. No diagnostic retunes a policy.')
    write_json(ROOT/'selection.json',result);return result


def test():
    policy,digest=study.check_selection()
    files=sorted((study.OUT/'test/cells').glob('*.npz'))
    assert len(files)==len(study.SNRS)*len(study.RSRS)*len(study.REGIMES)==153
    anchors=sorted((study.OUT/'test/anchors').glob('*.npz'));assert len(anchors)==9
    ordinary=0;hashes={};anchor_hashes={}
    for path in anchors:
        a=study.load(path)
        assert int(a['seed'])==study.TEST_SEED and str(a['selection_sha256'])==digest
        assert len(a['g'])==study.TEST_COUNT
        assert a['s'].shape[-1]>=30 and a['b'].shape[-1]>=30
        anchor_hashes[path.name]=sha(path)
    for path in files:
        cell=study.load(path)
        assert str(cell['selection_sha256'])==digest
        assert str(cell['protocol_sha256'])==policy['protocol_sha256']
        np.testing.assert_array_equal(cell['method_names'],study.NAMES)
        assert cell['estimates'].shape==(len(study.NAMES),study.TEST_COUNT,32,3)
        ref,snr,regime=float(cell['rsr']),float(cell['snr']),str(cell['regime'])
        anchor=study.load(study.OUT/'test/anchors'/f'{regime}_rsr{ref:g}.npz')
        truth=cell['truth_g']
        np.testing.assert_array_equal(truth,anchor['g'])
        np.testing.assert_array_equal(cell['indices'],anchor['indices'])
        errors=(np.abs(cell['estimates']-truth[None])**2).sum((-2,-1));energy=(np.abs(truth)**2).sum((-2,-1))
        np.testing.assert_array_equal(errors,cell['errors']);np.testing.assert_array_equal(energy,cell['truth_energy'])
        if 'nmse' in cell:np.testing.assert_array_equal(errors/energy[None],cell['nmse'])
        for kind,prefix in (('same_steps','fast_bypass'),('em100','em100_bypass')):
            use=snr>float(policy['policies'][kind][f'{ref:g}'])+1e-10 and snr>20+1e-10
            for seed in study.SEEDS:
                original=cell['estimates'][study.NAMES.index(f'original_{seed}')]
                fallback=cell['estimates'][study.NAMES.index(f'off_{seed}' if kind=='same_steps' else 'em100')]
                value=cell['estimates'][study.NAMES.index(f'{prefix}_{seed}')]
                np.testing.assert_array_equal(value,fallback if use else original)
        if snr<=20:ordinary+=1
        hashes[path.name]=sha(path)
    freeze_time=float(policy['frozen_unix'])
    earliest=min(p.stat().st_mtime for p in files+anchors)
    assert earliest>=freeze_time,'test data predated policy freeze'
    result=dict(passed=True,selection_sha256=digest,protocol_sha256=policy['protocol_sha256'],
        test_cells=len(files),test_anchors=len(anchors),ordinary_exact_identity_cells=ordinary,
        raw_errors_and_truth_energy_exactly_recomputed=True,all_branch_predictions_exactly_recomputed=True,
        earliest_test_file_after_selection_seconds=earliest-freeze_time,
        anchor_hashes=anchor_hashes,test_prediction_hashes=hashes,audit_source_sha256=sha(__file__),
        timestamp_caveat='Local timestamps support sequence; frozen source/checkpoint/selection hashes are the complementary provenance record.')
    write_json(ROOT/'test.json',result);return result


def summary():
    """All point scores plus 32 replicates of the independent conditional bootstrap."""
    policy,digest=study.check_selection();reported=json.loads((study.OUT/'nmse_summary.json').read_text())
    assert reported['selection_sha256']==digest
    groups={name:list(names) for name,names in study.GROUPS.items()}
    methods=list(study.NAMES)+list(groups);index={name:i for i,name in enumerate(methods)}
    def expand(values):return np.concatenate([values,np.stack([values[[index[n] for n in names]].mean(0) for names in groups.values()])])
    def belongs(region,snr):return dict(ordinary=snr<=20,transition=20<snr<=45,stress=snr>=50,high_snr=snr>20,all=True)[region]
    rng=np.random.default_rng(2026101829)
    weights={name:rng.multinomial(64,np.full(64,1/64),size=20000)[:32].T.astype(float) for name,_ in study.REGIMES}
    # Generating every count vector preserves the declared RNG stream; only32 are multiplied.
    points=[];boots=[];raw=[];energies=[];rows=reported['cells'];maximum=0.
    for row in rows:
        path=study.OUT/'test/cells'/f"{row['cell']}.npz";cell=study.load(path)
        assert sha(path)==reported['test_hashes'][path.name]
        err=(np.abs(cell['estimates']-cell['truth_g'][None])**2).sum((-2,-1));energy=(np.abs(cell['truth_g'])**2).sum((-2,-1))
        point=expand(10*np.log10(err.sum(-1)/energy.sum()));points.append(point)
        boot=expand(10*np.log10((err@weights[row['regime']])/(energy@weights[row['regime']])[None]));boots.append(boot)
        for name,i in index.items():
            difference=abs(float(point[i])-row['methods'][name]['nmse_db']);maximum=max(maximum,difference)
            assert difference<1e-12
        for key,(reference,candidate) in study.CONTRASTS.items():
            np.testing.assert_allclose(row['contrasts'][key]['gain_db'],point[index[reference]]-point[index[candidate]],atol=1e-12,rtol=0)
        raw.append(err);energies.append(energy)
    points=np.stack(points);boots=np.stack(boots);regions=('ordinary','transition','stress','high_snr','all')
    stored=study.load(study.OUT/'nmse_bootstrap.npz')
    np.testing.assert_array_equal(stored['method_names'],methods);np.testing.assert_array_equal(stored['regions'],regions)
    np.testing.assert_allclose(stored['cell_scores'],points,atol=1e-12,rtol=0)
    booterror=0.
    for region_number,region in enumerate(regions):
        keep=np.array([belongs(region,row['snr']) for row in rows]);expected=points[keep].mean(0)
        aggregate=reported['aggregates'][region]
        assert aggregate['cells']==int(keep.sum())
        pooled_err=np.concatenate([e for e,k in zip(raw,keep) if k],axis=1)
        pooled_energy=np.concatenate([e for e,k in zip(energies,keep) if k])
        pooled=expand(10*np.log10(pooled_err.sum(-1)/pooled_energy.sum()))
        nmse=pooled_err/pooled_energy[None]
        tail95,tail99=(expand(np.quantile(nmse,q,axis=1)) for q in (.95,.99))
        severe=expand((nmse>1).mean(1))
        for name,i in index.items():
            expected_values=dict(nmse_db=expected[i],pooled_ratio_of_sums_db=pooled[i],case_nmse_p95=tail95[i],case_nmse_p99=tail99[i],severe_rate=severe[i])
            for key,value in expected_values.items():np.testing.assert_allclose(aggregate['methods'][name][key],value,atol=1e-12,rtol=0)
        expected_boot=boots[keep].mean(0)
        saved=stored['aggregate_bootstraps'][region_number,:,:32]
        difference=float(np.abs(expected_boot-saved).max());booterror=max(booterror,difference)
        np.testing.assert_allclose(expected_boot,saved,atol=1e-11,rtol=0)
        full_boot=stored['aggregate_bootstraps'][region_number]
        for key,(reference,candidate) in study.CONTRASTS.items():
            contrast=full_boot[index[reference]]-full_boot[index[candidate]]
            np.testing.assert_allclose(aggregate['contrasts'][key]['pointwise_ci95_db'],np.quantile(contrast,[.025,.975]),atol=1e-12,rtol=0)
            if region=='high_snr':np.testing.assert_allclose(aggregate['contrasts'][key]['bonferroni6_interval_db'],np.quantile(contrast,[.05/12,1-.05/12]),atol=1e-12,rtol=0)
    tolerance=float(reported['numerical_tie_tolerance_db']);assert tolerance==1e-10
    for key in list(study.CONTRASTS)+['fast_bypass_vs_original']:
        entries=[row['contrasts'][key] if key in study.CONTRASTS else row[key] for row in rows]
        negative=[row['cell'] for row,value in zip(rows,entries) if value['gain_db'] < -tolerance]
        harmful=[row['cell'] for row,value in zip(rows,entries) if value['pointwise_ci95_db'][1] < -tolerance]
        assert reported['negative_cells'][key]['negative_cells']==negative
        assert reported['negative_cells'][key]['pointwise95_harm_cells']==harmful
        worst=int(np.argmin([value['gain_db'] for value in entries]))
        assert reported['negative_cells'][key]['worst_cell']==rows[worst]['cell']
    result=dict(passed=True,selection_sha256=digest,cells=len(rows),regions=len(regions),
        maximum_point_error=maximum,independently_recomputed_bootstrap_replicates=32,maximum_bootstrap_error=booterror,
        all_pooled_nmse_tail_quantiles_severe_rates_recomputed=True,interval_quantiles_and_bonferroni6_verified=True,
        numerical_tie_tolerance_and_adverse_cell_counts_verified=True,
        summary_sha256=sha(study.OUT/'nmse_summary.json'),bootstrap_sha256=sha(study.OUT/'nmse_bootstrap.npz'),source_sha256=sha(__file__),
        scope='Conditional fixed-model/policy bootstrap; one independent count matrix per path regime, shared across SNR/reference/method/fit.32 replicates recomputed from raw predictions; all aggregate intervals checked from retained20000 replicates.')
    write_json(ROOT/'summary.json',result);return result


def timing():
    policy,digest=study.check_selection();record=json.loads((study.OUT/'timing.json').read_text())
    assert record['selection_sha256']==digest
    raw=record['raw_samples'];assert set(raw)==set(study.NAMES)
    assert record['worlds']==153 and record['repeats']==3
    for name,rows in raw.items():
        assert len(rows)==459 and all(int(row['index'])==200 for row in rows)
        assert set(float(row['snr']) for row in rows)==set(study.SNRS)
        for row in rows:
            branch=name
            if name.startswith('original_'):branch='original'
            elif name.startswith('off_'):branch='off'
            elif name=='em100':branch='em100'
            elif name.startswith('fast_bypass_') or name.startswith('em100_bypass_'):
                kind='same_steps' if name.startswith('fast_bypass_') else 'em100'
                on=row['snr']>policy['policies'][kind][f"{row['rsr']:g}"]+1e-10 and row['snr']>20+1e-10
                branch=('off' if kind=='same_steps' else 'em100') if on else 'original'
            assert row['branch']==branch
    combined=dict(raw)
    for name,parts in study.GROUPS.items():combined[name]=[x for part in parts for x in raw[part]]
    for name,rows in combined.items():
        verify_stats(numeric_stats(rows),record['methods'][name])
        for branch in set(row['branch'] for row in rows):verify_stats(numeric_stats([r for r in rows if r['branch']==branch]),record['methods'][name]['by_branch'][branch])
        for ref in study.RSRS:
            for snr in study.SNRS:verify_stats(numeric_stats([r for r in rows if r['rsr']==ref and r['snr']==snr]),record['methods'][name]['by_condition'][f'rsr{ref:g}_snr{snr:g}'])
    cap=study.check_calibration()['cap_ms'];primary=record['methods']['em100_bypass_mean']['p90_ms'];compact=record['methods']['compact_sparse']['p90_ms']
    expected=dict(calibrated_cap_ms=cap,compact_p90_ms=compact,primary_pipeline_p90_ms=primary,
        compact_exceeds_calibrated_cap=compact>cap,compact_exceeds_fresh_pipeline=compact>primary,primary_pipeline_exceeds_calibrated_cap=primary>cap)
    assert record['cap_verification']==expected
    result=dict(passed=True,selection_sha256=digest,worlds=153,repeats=3,individual_methods=len(raw),
        raw_calls=sum(map(len,raw.values())),all_quantiles_and_branch_routes_recomputed=True,
        cap_verification=expected,timing_sha256=sha(study.OUT/'timing.json'),audit_source_sha256=sha(__file__),
        caveat='This independently verifies retained samples and summaries, not host isolation or a real-time worst-case bound.')
    write_json(ROOT/'timing.json',result);return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=('calibration','selection','test','summary','timing'))
    arguments=parser.parse_args();result=globals()[arguments.phase]()
    print(json.dumps({k:v for k,v in result.items() if k not in ('anchor_hashes','test_prediction_hashes')},indent=2))
