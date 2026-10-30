"""Independent artifact audits for fixed high-SNR filter continuation."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key]='1'
import argparse
import json
import numpy as np
import torch
from sahure.data import Observations
from sahure.math import hankel,unhankel,nll
from tcom_followups_20261017.common import sha,write_json,em_estimate
from . import study

OUT=study.OUT/'audit'


def close(actual,expected,atol=1e-12):
    np.testing.assert_allclose(actual,expected,rtol=0,atol=atol)


def record(name,**values):
    result=dict(passed=True,audit_source_sha256=sha(__file__),**values)
    write_json(OUT/f'{name}.json',result);return result


def stats(rows):
    a=np.array([r['ms'] for r in rows]);assert np.isfinite(a).all() and (a>0).all()
    return dict(p90_ms=float(np.quantile(a,.9)),median_ms=float(np.median(a)),mean_ms=float(a.mean()),calls=len(a))


def verify_stats(rows,summary):
    for key,value in stats(rows).items():close(summary[key],value)


def chosen(policy,ref,snr):
    grid=min(study.SNRS,key=lambda s:(abs(s-snr),s))
    return 'original' if snr<=20+1e-10 else policy['repair_choices'][f'{ref:g}'][f'{grid:g}']['choice']


def grid_check(rows,index):
    expected={(regime,float(ref),float(snr),repeat,index) for regime,_ in study.REGIMES for ref in study.RSRS for snr in study.SNRS for repeat in range(3)}
    actual=[(r['regime'],float(r['rsr']),float(r['snr']),int(r['repeat']),int(r['index'])) for r in rows]
    assert len(actual)==len(set(actual))==459 and set(actual)==expected


def protocol_hashes():
    p=json.loads((study.OUT/'protocol.json').read_text())
    for key in ('source_hashes','checkpoint_hashes'):
        for path,digest in p[key].items():assert sha(study.REPO/path)==digest,path
    assert p['current_repair_policy_sha256']==sha(study.CURRENT_REPAIR_PATH)
    assert p['previous_repair_policy_sha256']==sha(study.PRIOR_REPAIR_PATH)
    return p


def verify_bank(path,digest):
    bank=study.load(path);ref=float(bank['rsr']);snr=float(bank['snr'])
    assert ref in (12.,20.) and snr in (40.,42.5,45.,50.,60.)
    assert int(bank['cases_per_regime'])==384 and str(bank['protocol_sha256'])==digest
    assert not {'g','G','truth_g','paths','unit_noise'}.intersection(bank)
    for pi,(regime,_) in enumerate(study.REGIMES):
        anchor=study.load(study.OUT/'development/anchors'/f'{regime}_rsr{ref:g}.npz')
        assert int(anchor['seed'])==2026101951 and int(anchor['holdout_seed'])==2026101955 and len(anchor['s'])==384
        assert not {'g','G','truth_g','paths','unit_noise'}.intersection(anchor)
        full=study.observed(anchor,snr,P=39)
        for prefix,columns in (('context',slice(0,30)),('heldout',slice(30,39))):
            for key,value in zip(('z','s','b','sigma2'),full.args()):
                expected=value if key=='sigma2' else value[...,columns]
                np.testing.assert_array_equal(bank[f'{prefix}_{key}'][pi*384:(pi+1)*384],expected.numpy())
    return bank


def svd_filter(obs,g):
    """Independent explicit spectral formula, not the deployed solve helper."""
    h=hankel(g.mT);u,s,vh=torch.linalg.svd(h,full_matrices=False)
    diag=torch.linalg.inv(obs.s@obs.s.mH).diagonal(dim1=-2,dim2=-1).real
    tau=4*(2*obs.sigma2[:,None]*diag*17).sqrt()
    filtered=u@torch.diag_embed(s**3/(s*s+tau[...,None]**2)).to(g.dtype)@vh
    # Retain the actual beta=1 expression to include its rounding convention.
    return g+(unhankel(filtered).mT-g)


@torch.no_grad()
def repair():
    p=study.check_repair();protocol=protocol_hashes();digest=p['protocol_sha256']
    assert len(p['development_anchor_hashes'])==6 and len(p['development_bank_hashes'])==10 and len(p['loss_table_hashes'])==10
    rng=np.random.default_rng(2026101957)
    weights={r:rng.multinomial(384,np.full(384,1/384),size=4000) for r,_ in study.REGIMES}
    saved=study.load(study.OUT/'development/repair_bootstrap_weights.npz')
    for regime,w in weights.items():np.testing.assert_array_equal(saved[regime],w)
    expected=[];replays=0;maximum=0.
    for ref in (12.,20.):
        for snr in (40.,42.5,45.,50.,60.):
            name=f'rsr{ref:g}_snr{snr:g}.npz';path=study.OUT/'development/repair_losses'/name
            a=study.load(path);bank=verify_bank(study.OUT/'development/banks'/name,digest)
            assert a['paired_risk'].shape==(3,384) and np.isfinite(a['paired_risk']).all()
            assert str(a['protocol_sha256'])==digest and path.stat().st_mtime>=protocol['frozen_unix']
            for pi,(regime,_) in enumerate(study.REGIMES):
                delta=a['paired_risk'][pi];upper=float(np.quantile(weights[regime]@delta/384,.95))
                expected.append(dict(rsr=ref,snr=snr,regime=regime,mean_delta=float(delta.mean()),upper95=upper,passed=upper<=0))
            # Recompute selected raw risks using canonical Torch EM and an
            # explicit SVD filter, then a separate intensity-moment expression.
            ids=np.array([0,1,384,385,768,769])
            c,h=(Observations(*(torch.from_numpy(bank[f'{prefix}_{key}'][ids]) for key in ('z','s','b','sigma2'))) for prefix in ('context','heldout'))
            old=em_estimate(c,100);new=svd_filter(c,old)
            ds=(new-old)@h.s;mid=(new+old)/2@h.s
            innovation=(h.z.square()-(h.b+mid).abs().square()-h.sigma2[:,None,None])/h.b.conj()
            actual=(-2*(ds.conj()*innovation).real.sum((-2,-1))/(9*32*3)).numpy()
            stored=a['paired_risk'][:,:2].reshape(-1)
            error=float(abs(actual-stored).max());maximum=max(maximum,error);close(actual,stored,1e-10);replays+=len(ids)
    assert p['guards']==expected and all(x['passed'] for x in expected)
    assert p['repair_choices']==study.candidate_table()
    for ref in study.RSRS:
        for snr in study.SNRS:
            actual=p['repair_choices'][f'{ref:g}'][f'{snr:g}']
            if ref in (12.,20.) and snr>=40:assert actual['choice']=='shrink_c4_beta1'
            else:assert actual==study.CURRENT_REPAIR['repair_choices'][f'{ref:g}'][f'{snr:g}']
    assert p['frozen_unix']>=max((study.OUT/'development/repair_losses'/n).stat().st_mtime for n in p['loss_table_hashes'])
    return record('repair_selection',protocol_sha256=digest,repair_selection_sha256=sha(study.OUT/'repair_selection.json'),
        guards=30,all_guards_passed=True,canonical_em_svd_risk_replays=replays,maximum_independent_risk_error=maximum,
        shared_bootstrap_weights_exact=True,independent_heldout_banks_verified=True,all51_fixed_table_bins_verified=True)


def calibration():
    p=study.check_calibration();policy=study.check_repair();raw=p['raw_samples'];configs=study.base.candidates()
    expected_names={f'repair_{s}' for s in study.SEEDS}|{f'compact_{k}' for k in configs}
    assert set(raw)==expected_names
    for name,rows in raw.items():
        grid_check(rows,100);verify_stats(rows,p['methods'][name])
        for row in rows:assert row['repair_choice']==chosen(policy,row['rsr'],row['snr'])
    pooled=[r for seed in study.SEEDS for r in raw[f'repair_{seed}']]
    verify_stats(pooled,p['repair_timing']);close(p['cap_ms'],stats(pooled)['p90_ms'])
    eligible=[]
    for key,cfg in configs.items():
        assert p['candidates'][key]['config']==cfg;verify_stats(raw[f'compact_{key}'],p['candidates'][key])
        if stats(raw[f'compact_{key}'])['p90_ms']<=p['cap_ms']:eligible.append(key)
    assert p['eligible']==eligible and sum(map(len,raw.values()))==8721
    assert p['completed_unix']>=policy['frozen_unix']
    return record('calibration',calibration_sha256=sha(study.OUT/'calibration.json'),candidates=16,eligible=eligible,
        raw_calls=8721,cap_ms=p['cap_ms'],all_quantiles_routes_and_eligibility_verified=True,
        caveat='Retained samples establish observed pooled quantiles, not host isolation or a worst-case deadline.')


def selection():
    protocol_hashes();p,digest=study.check_selection();cal=study.check_calibration();configs=study.base.candidates()
    assert len(p['development_anchor_hashes'])==6 and len(p['development_bank_hashes'])==len(p['compact_loss_hashes'])==10
    changed=fixed=0
    for ref in study.RSRS:
        for snr in study.SNRS:
            r,k=f'{ref:g}',f'{snr:g}'
            if ref not in (12.,20.) or snr not in (40.,42.5,45.,50.,60.):
                assert p['compact_choices'][r][k]==study.PRIOR_COMPACT['compact_choices'][r][k]
                assert p['compact_tables'][r][k]==[];fixed+=1;continue
            path=study.OUT/'development/compact_losses'/f'rsr{r}_snr{k}.npz';a=study.load(path)
            assert str(a['calibration_sha256'])==p['calibration_sha256']
            np.testing.assert_array_equal(a['names'],cal['eligible']);assert a['losses'].shape==(len(cal['eligible']),3,384)
            assert np.isfinite(a['losses']).all()
            scores=[dict(candidate=key,config=configs[key],mean_nll=float(a['losses'][i].mean()),
                by_regime=a['losses'][i].mean(-1).tolist(),p90_ms=cal['candidates'][key]['p90_ms']) for i,key in enumerate(cal['eligible'])]
            assert p['compact_tables'][r][k]==scores
            assert p['compact_choices'][r][k]==min(scores,key=lambda x:(x['mean_nll'],x['p90_ms'],x['candidate']))
            assert path.stat().st_mtime>=cal['completed_unix'];changed+=1
    assert changed==10 and fixed==41 and p['frozen_unix']>=cal['completed_unix']
    return record('selection',selection_sha256=digest,repair_choices_checked=51,compact_choices_checked=51,
        compact_refreshed_bins=changed,compact_fixed_bins=fixed,ber_bindings_and_all_frozen_hashes_verified=True)


@torch.no_grad()
def test():
    policy,digest=study.check_selection();protocol_hashes();est=study.engine()
    paths=sorted((study.OUT/'test/cells').glob('*.npz'));boundary=sorted((study.OUT/'boundary/cells').glob('*.npz'))
    assert len(paths)==153 and len(boundary)==54
    anchorpaths=sorted((study.OUT/'test/anchors').glob('*.npz'));assert len(anchorpaths)==9
    anchors={};outside=early=weak=ordinary=target=0;maximum=0.;filter_error=0.;hashes={};boundary_hashes={}
    for path in anchorpaths:
        a=study.load(path);assert int(a['seed'])==2026101952 and str(a['selection_sha256'])==digest and len(a['g'])==128
        assert int(a['holdout_seed'])==2026101955 and str(a['protocol_sha256'])==policy['protocol_sha256']
        anchors[path.stem]=a
        regime=str(a['regime']);L=dict(study.REGIMES)[regime];ref=float(a['rsr'])
        regenerate=study.make_anchor(2026101952,[0],L,ref,truth=True)
        for key in ('g','s','b','paths','indices','local_indices'):
            np.testing.assert_array_equal(a[key][:1],regenerate[key])
        np.testing.assert_array_equal(a['z_by_snr'][:,:1],regenerate['z_by_snr'])
    for is_boundary,files in ((False,paths),(True,boundary)):
        for path in files:
            a=study.load(path);assert str(a['selection_sha256'])==digest and int(a['seed'])==2026101952
            np.testing.assert_array_equal(a['method_names'],study.NAMES);pred=a['estimates']
            assert pred.shape==(20,128,32,3) and np.isfinite(pred).all()
            ref,snr,regime=float(a['rsr']),float(a['snr']),str(a['regime']);anchor=anchors[f'{regime}_rsr{ref:g}']
            assert snr in (study.BOUNDARY_SNRS if is_boundary else study.SNRS)
            for key,other in (('truth_g','g'),('indices','indices'),('local_indices','local_indices'),('paths','paths')):
                np.testing.assert_array_equal(a[key],anchor[other])
            obs=study.observed(anchor,snr)
            for key,value in zip(('z','s','b','sigma2'),obs.args()):np.testing.assert_array_equal(a[key],value.numpy())
            errors=(abs(pred-anchor['g'][None])**2).sum((-2,-1));energy=(abs(anchor['g'])**2).sum((-2,-1))
            np.testing.assert_array_equal(a['errors'],errors);np.testing.assert_array_equal(a['truth_energy'],energy)
            np.testing.assert_array_equal(a['nmse'],errors/energy[None])
            old=snr>study.OLD_POLICY[f'{ref:g}']+1e-10 and snr>20+1e-10
            np.testing.assert_array_equal(a['old_bypass'],np.full(128,old))
            np.testing.assert_array_equal(pred[3:6],pred[15:16].repeat(3,axis=0) if old else pred[:3])
            assert str(a['repair_choice'])==chosen(policy,ref,snr)
            for i,name in enumerate(study.NAMES):
                if not name.startswith(('prior_repair_','current_repair_','repair_')):continue
                replay=study.estimate_method(obs.batch(slice(0,1)),name,policy,est).numpy()
                error=float(abs(replay-pred[i,:1]).max());maximum=max(maximum,error);assert error<1e-10
            if is_boundary or (ref in (12.,20.) and snr>=40):
                expected=svd_filter(obs,torch.from_numpy(pred[15])).numpy()
                error=float(abs(pred[12:15]-expected[None]).max());filter_error=max(filter_error,error)
                close(pred[12:15],np.broadcast_to(expected,(3,*expected.shape)),1e-10)
            if not is_boundary:
                if snr<=20:
                    for j in (3,6,9,12):np.testing.assert_array_equal(pred[j:j+3],pred[:3])
                    ordinary+=1
                if ref==5:np.testing.assert_array_equal(pred[12:15],pred[6:9]);weak+=1
                if ref not in (12,20) or snr<40:np.testing.assert_array_equal(pred[12:15],pred[9:12]);outside+=1
                else:target+=1
                if ref in (12,20) and snr in (32.5,35.,37.5):np.testing.assert_array_equal(pred[12:15],pred[9:12]);early+=1
            (boundary_hashes if is_boundary else hashes)[path.name]=sha(path)
    assert (outside,early,weak,ordinary,target)==(123,18,51,45,30)
    earliest=min(p.stat().st_mtime for p in paths+boundary+anchorpaths);assert earliest>=policy['frozen_unix']
    return record('test',selection_sha256=digest,test_cells=153,boundary_cells=54,test_anchor_banks=9,
        independently_regenerated_anchor_samples=9,outside_exact_current_cells=outside,earlier_transition_exact_cells=early,
        weak_exact_prior_cells=weak,ordinary_exact_original_cells=ordinary,changed_high_snr_cells=target,
        maximum_actual_scalar_dispatch_error=maximum,maximum_independent_svd_filter_error=filter_error,
        all_raw_errors_energy_observations_and_shared_boundary_anchors_verified=True,
        earliest_test_after_freeze_seconds=earliest-policy['frozen_unix'],test_prediction_hashes=hashes,boundary_prediction_hashes=boundary_hashes)


def cross_window_drift(calibration,final,policy):
    """Describe retained timing drift; never rerun or normalize measurements."""
    old=calibration['raw_samples'];new=final['raw_samples']
    old_repair=[r for seed in study.SEEDS for r in old[f'repair_{seed}']]
    new_repair=[r for seed in study.SEEDS for r in new[f'repair_{seed}']]
    def compare(a,b):
        x,y=stats(a),stats(b)
        return dict(calibration=x,final=y,median_ratio=y['median_ms']/x['median_ms'],p90_ratio=y['p90_ms']/x['p90_ms'])
    branches={branch:compare([r for r in old_repair if r['repair_choice']==branch],
        [r for r in new_repair if r['branch']==branch]) for branch in sorted({r['branch'] for r in new_repair})}
    # Reconstruct the final selected compact mix using calibration calls for
    # the identical configuration and condition, without selecting new controls.
    compact=[]
    for ref in study.RSRS:
        for snr in study.SNRS:
            config=policy['compact_choices'][f'{ref:g}'][f'{snr:g}']['config']
            key=next(k for k,v in calibration['candidates'].items() if v['config']==config)
            compact.extend(r for r in old[f'compact_{key}'] if r['rsr']==ref and r['snr']==snr)
    assert len(compact)==459
    high=lambda rows:[r for r in rows if r['rsr'] in (12.,20.) and r['snr']>=40]
    return dict(repair=compare(old_repair,new_repair),repair_by_branch=branches,
        em100=compare(old['compact_em100'],new['em100']),
        selected_compact_identical_configuration_mix=compare(compact,new['compact_sparse']),
        final_strong_high=dict(repair=stats(high(new_repair)),
            incumbent=stats(high([r for seed in study.SEEDS for r in new[f'current_repair_{seed}']])),
            direct_em100=stats(high(new['em100'])),compact=stats(high(new['compact_sparse']))),
        interpretation='Descriptive cross-window comparisons use different prespecified timing anchors (100 versus 200). Shared slowdown is evidence against uniquely attributing the overrun to the new filter, but does not identify its host-level cause. No normalization, retiming, or reselection is performed.')


def timing():
    policy,digest=study.check_selection();p=json.loads((study.OUT/'timing.json').read_text());raw=p['raw_samples']
    assert p['selection_sha256']==digest and p['worlds']==153 and p['repeats']==3 and set(raw)==set(study.NAMES)
    for name,rows in raw.items():
        grid_check(rows,200)
        for row in rows:
            ref,snr=row['rsr'],row['snr'];choice=chosen(policy,ref,snr);prior=chosen(study.PRIOR_REPAIR,ref,snr)
            assert row['repair_choice']==choice and row['prior_repair_choice']==prior
            old=snr>study.OLD_POLICY[f'{ref:g}']+1e-10 and snr>20+1e-10
            expected=choice if name.startswith('repair_') else chosen(study.CURRENT_REPAIR,ref,snr) if name.startswith('current_repair_') else prior if name.startswith('prior_repair_') else ('spectral100' if old else 'original') if name.startswith('old_bypass_') else 'original' if name.startswith('original_') else name
            assert row['branch']==expected
    combined=dict(raw)
    for name,parts in study.GROUPS.items():combined[name]=[r for part in parts for r in raw[part]]
    for name,rows in combined.items():
        verify_stats(rows,p['methods'][name])
        for branch in {r['branch'] for r in rows}:verify_stats([r for r in rows if r['branch']==branch],p['methods'][name]['by_branch'][branch])
        for ref in study.RSRS:
            for snr in study.SNRS:verify_stats([r for r in rows if r['rsr']==ref and r['snr']==snr],p['methods'][name]['by_condition'][f'rsr{ref:g}_snr{snr:g}'])
    cap=study.check_calibration()['cap_ms'];repair_p90=p['methods']['repair_mean']['p90_ms'];compact=p['methods']['compact_sparse']['p90_ms']
    expected=dict(calibrated_cap_ms=cap,repair_p90_ms=repair_p90,compact_p90_ms=compact,
        repair_exceeds_calibrated_cap=repair_p90>cap,compact_exceeds_calibrated_cap=compact>cap,compact_exceeds_fresh_repair=compact>repair_p90)
    assert p['cap_verification']==expected and sum(map(len,raw.values()))==9180
    return record('timing',selection_sha256=digest,raw_calls=9180,worlds=153,individual_methods=20,repeats=3,
        all_quantiles_condition_counts_and_routes_verified=True,cap_verification=expected,timing_sha256=sha(study.OUT/'timing.json'),
        cross_window_drift=cross_window_drift(study.check_calibration(),p,policy),
        caveat='Retained samples do not independently prove host isolation or a worst-case deadline.')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=('repair','calibration','selection','test','timing'))
    args=parser.parse_args();torch.set_num_threads(1)
    try:torch.set_num_interop_threads(1)
    except RuntimeError:pass
    result=globals()[args.phase]()
    print(json.dumps({k:v for k,v in result.items() if not k.endswith('prediction_hashes')},indent=2))
