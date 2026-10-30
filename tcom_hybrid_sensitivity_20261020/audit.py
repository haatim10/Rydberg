"""Independent frozen checks for the exploratory B-proposal sensitivity arm.

PASS verifies provenance and arithmetic, not superiority or cap attainment.
The parent confirmed policy and its primary analysis are never modified.
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key]='1'
import argparse
import copy
import json
from pathlib import Path
import time
import numpy as np
import torch
from tcom_followups_20261017.common import sha,write_json
from . import study

REGIONS=('all','high_all','transition_all','weak_warm100','strong_high','strong_transition','ordinary','weak','rsr5','rsr12','rsr20',
    'all_L2','all_L4','all_random3to7','weak_warm100_L2','weak_warm100_L4','weak_warm100_random3to7')
CONTRASTS={f'proposal_vs_{suffix}':(reference,'proposal_mean') for suffix,reference in
    (('hybrid','hybrid_mean'),('latest','latest_mean'),('compact','compact_sparse'),('adaptive40','adaptive40_mean'),
     ('global40','global40_mean'),('classical40','classical40'),('em100','em100'),('optshrink','optshrink'))}


def close(a,b,tol=1e-11):np.testing.assert_allclose(a,b,atol=tol,rtol=0)

def load(path):
    with np.load(path,allow_pickle=False) as a:return {k:a[k] for k in a.files}

def completed(path):return sorted(p for p in Path(path).glob('*.npz') if '.tmp.' not in p.name)

def belongs(scope,snr,ref,regime):
    core=dict(all=True,high_all=snr>=40,transition_all=20<snr<40,weak_warm100=ref==5 and snr>=37.5,
        strong_high=ref in (12,20) and snr>=40,strong_transition=ref in (12,20) and 32.5<=snr<=37.5,
        ordinary=snr<=20,weak=ref==5)
    if scope in core:return core[scope]
    if scope in ('rsr5','rsr12','rsr20'):return ref==int(scope[3:])
    for prefix in ('all','weak_warm100'):
        if scope.startswith(prefix+'_'):return core[prefix] and regime==scope[len(prefix)+1:]
    raise KeyError(scope)

def record(phase,**kwargs):
    selection,digest=study.check_selection()
    result=dict(passed=True,audit_source_sha256=sha(__file__),selection_sha256=digest,protocol_sha256=digest,
        parent_selection_sha256=selection['parent_selection_sha256'],completed_unix=time.time(),**kwargs)
    path=study.OUT/'ber/audit.json' if phase=='ber' else study.OUT/'audit'/f'{phase}.json'
    write_json(path,result);return result

def no_adjusted(value):
    if isinstance(value,dict):
        assert not any('bonferroni' in k.lower() or 'familywise' in k.lower() for k in value)
        for v in value.values():no_adjusted(v)
    elif isinstance(value,list):
        for v in value:no_adjusted(v)

def schema(ber=False):
    parent=study.PARENT
    raw=(['perfect_csi'] if ber else [])+list(parent.NAMES)+list(study.NAMES)
    groups={k:v for k,v in parent.GROUPS.items() if all(n in raw for n in v)}
    groups.update({'proposal_mean':list(study.NAMES)})
    return raw,groups,raw+list(groups)

def grid():
    return [(r,float(ref),float(s)) for r,_ in study.PARENT.REGIMES for ref in study.PARENT.RSRS for s in study.PARENT.SNRS]

def selection():
    wrapped,digest=study.check_selection();parent=study.PARENT
    main,main_hash=parent.check_selection();proposals=json.loads((parent.OUT/'tail_proposals.json').read_text())
    confirmed=json.loads((parent.OUT/'tail_confirmation.json').read_text())
    expected=copy.deepcopy(main);nonzero=[];failed=[]
    for ref in parent.RSRS:
        for snr in parent.SNRS:
            rk,sk=f'{ref:g}',f'{snr:g}'
            if snr<=20:assert expected['tails'][rk][sk]==dict(family='adaptive',steps=0);continue
            p=proposals['proposals'][rk][sk];c=confirmed['confirmed'][rk][sk]
            assert p['family']==main['tails'][rk][sk]['family']
            expected['tails'][rk][sk]['steps']=p['proposed_steps']
            if p['proposed_steps']:
                nonzero.append(f'{rk}/{sk}');assert c['proposed_steps']==p['proposed_steps']
                if not c['accepted']:failed.append(f'{rk}/{sk}')
    assert len(nonzero)==len(failed)==21 and all(x['steps']==0 for rs in main['tails'].values() for x in rs.values())
    assert wrapped['main_policy']==main and wrapped['proposal_policy']==expected
    assert wrapped['parent_selection_sha256']==main_hash==sha(parent.OUT/'policy.json')
    assert digest==sha(study.OUT/'protocol.json')
    assert not completed(parent.OUT/'test/cells') and not completed(parent.OUT/'test/anchors')
    assert not completed(study.OUT/'test/cells')
    assert parent.PRIMARY==('hybrid_vs_latest','hybrid_vs_compact')
    return record('selection',parent_policy_unchanged=True,exact_B_proposals_without_reselection=True,
        ordinary_unchanged=True,C_rejected_proposal_bins=failed,C_rejected_proposals=21,nonzero_B_proposals=21,
        parent_test_not_generated_before_secondary_freeze=True,parent_policy_sha256=main_hash,
        parent_proposals_sha256=sha(parent.OUT/'tail_proposals.json'),parent_confirmation_sha256=sha(parent.OUT/'tail_confirmation.json'),
        scope='Exploratory policy sensitivity only. All21 nonzero proposals failed independent C confirmation; parent primary policy unchanged.')


@torch.no_grad()
def test():
    wrapped,digest=study.check_selection();parent=study.PARENT;paths=completed(study.OUT/'test/cells')
    assert len(paths)==153;est=parent.engine();hashes={};maximum=0.;replays=0;identities=0;seen=[]
    from tcom_pilot_em_20261018 import continuation
    frozen=json.loads((study.OUT/'protocol.json').read_text())['frozen_unix']
    for path in paths:
        a=load(path);pp=parent.OUT/'test/cells'/path.name;p=load(pp)
        regime,ref,snr=str(a['regime']),float(a['rsr']),float(a['snr']);seen.append((regime,ref,snr))
        assert str(a['selection_sha256'])==str(a['protocol_sha256'])==digest
        assert str(a['parent_selection_sha256'])==wrapped['parent_selection_sha256']
        assert str(a['parent_source_sha256'])==sha(pp)
        assert str(a['parent_ber_cell_sha256'])==sha(parent.OUT/'ber/cells'/path.name)
        assert str(a['parent_ber_data_sha256'])==sha(parent.OUT/'ber/data'/path.name)
        assert pp.stat().st_mtime>=frozen and path.stat().st_mtime>=frozen
        np.testing.assert_array_equal(a['method_names'],study.NAMES)
        for key in ('indices','local_indices','seed','truth_energy'):np.testing.assert_array_equal(a[key],p[key])
        pred=a['estimates'];assert pred.shape==(3,128,32,3) and pred.dtype==np.complex128 and np.isfinite(pred).all()
        errors=np.sum(abs(pred-p['truth_g'][None])**2,axis=(-2,-1));energy=np.sum(abs(p['truth_g'])**2,axis=(-2,-1))
        np.testing.assert_array_equal(a['errors'],errors);np.testing.assert_array_equal(a['truth_energy'],energy)
        np.testing.assert_array_equal(a['nmse'],errors/energy[None])
        from sahure.data import Observations
        obs=Observations(*(torch.from_numpy(p[k][127:128]) for k in ('z','s','b','sigma2')))
        branch=wrapped['proposal_policy']['tails'][f'{ref:g}'][f'{snr:g}']
        for i,seed in enumerate(parent.SEEDS):
            raw=est['starts'].start(branch['family'],obs,seed=seed)
            expected=raw if snr<=20 else parent.finish(obs,continuation.estimate(obs,raw,branch['steps']) if branch['steps'] else raw)
            error=float(np.abs(expected.numpy()-pred[i,127:128]).max());maximum=max(maximum,error);replays+=1
            np.testing.assert_allclose(expected.numpy(),pred[i,127:128],atol=1e-10,rtol=1e-10)
            if branch['steps']==0:
                np.testing.assert_array_equal(pred[i],p['estimates'][list(map(str,p['method_names'])).index(f'hybrid_{seed}')])
                identities+=1
        hashes[path.name]=sha(path)
    assert len(set(seen))==153 and set(seen)==set(grid())
    return record('test',cells=153,test_prediction_hashes=hashes,independently_replayed_predictions=replays,
        maximum_replay_error=maximum,exact_zero_tail_parent_fit_cells=identities,
        same_parent_worlds_observations_and_data_hashes_verified=True,all_error_arrays_exact=True)


@torch.no_grad()
def ber():
    from tcom_followups_20261017.ber_detector import detect,hypotheses
    wrapped,digest=study.check_selection();parent=study.PARENT;paths=completed(study.OUT/'ber/cells')
    assert len(paths)==153;x,bits=hypotheses();hashes={};decodes=0;data_checks=0
    for path in paths:
        a=load(path);source=load(study.OUT/'test/cells'/path.name);p=load(parent.OUT/'test/cells'/path.name);data=load(parent.OUT/'ber/data'/path.name)
        assert str(a['protocol_sha256'])==str(a['selection_sha256'])==digest
        assert str(a['parent_selection_sha256'])==wrapped['parent_selection_sha256']
        assert str(a['source_sha256'])==sha(study.OUT/'test/cells'/path.name)
        assert str(a['parent_source_sha256'])==sha(parent.OUT/'test/cells'/path.name)
        assert str(a['parent_ber_sha256'])==sha(parent.OUT/'ber/cells'/path.name)
        assert str(a['parent_data_sha256'])==sha(parent.OUT/'ber/data'/path.name)
        np.testing.assert_array_equal(a['method_names'],study.NAMES)
        np.testing.assert_array_equal(a['bits_per_user'],np.full((128,3),128))
        counts=a['errors_per_user'];assert counts.shape==(3,128,3) and np.issubdtype(counts.dtype,np.integer)
        assert (counts>=0).all() and (counts<=128).all()
        regime,ref=str(source['regime']),float(source['rsr'])
        for case in (0,127):
            rng=np.random.default_rng(np.random.SeedSequence([2026102055,('L2','L4','random3to7').index(regime),(5.,12.,20.).index(ref),case]))
            noise=(rng.standard_normal((32,64))+1j*rng.standard_normal((32,64)))/np.sqrt(2)
            z=np.abs(p['truth_g'][case]@x.T+p['b'][case,:,:1]+np.sqrt(p['sigma2'][case])*noise)
            np.testing.assert_array_equal(data['z'][case],z);data_checks+=1
            for i in range(3):
                decoded=detect(z,source['estimates'][i,case],data['b'][case],float(data['sigma2'][case]))
                np.testing.assert_array_equal(np.count_nonzero(decoded!=bits,axis=(0,2)),counts[i,case]);decodes+=1
        hashes[path.name]=sha(path)
    return record('ber',cells=153,cell_hashes=hashes,independently_redecoded_predictions=decodes,
        independently_regenerated_data_cases=data_checks,parent_data_exact=True,no_approximate_count_reuse=True)


def timing():
    wrapped,digest=study.check_selection();parent=study.PARENT;windows={};hashes={};total=0
    protocol=json.loads((study.OUT/'protocol.json').read_text())
    names=tuple(f'{f}_{seed}' for f in ('proposal','hybrid','latest') for seed in parent.SEEDS)+('compact_sparse',)
    groups={f'{f}_mean':[f'{f}_{seed}' for seed in parent.SEEDS] for f in ('proposal','hybrid','latest')}
    def stats(rows):
        x=np.array([r['ms'] for r in rows]);assert (x>0).all() and np.isfinite(x).all()
        return dict(p90_ms=float(np.quantile(x,.9)),median_ms=float(np.median(x)),mean_ms=float(x.mean()),calls=len(x))
    previous_end=0
    for window in (1,2):
        path=study.OUT/f'timing_window{window}.json';a=json.loads(path.read_text());raw=a['raw_samples']
        assert a['window']==window and a['selection_sha256']==digest and a['protocol_sha256']==digest
        ids=[500,501,502] if window==1 else [600,601,602]
        assert a['indices']==ids and a['worlds']==459 and set(raw)==set(names)
        assert a['started_unix']>=max(previous_end,protocol['frozen_unix']) and a['completed_unix']>=a['started_unix'];previous_end=a['completed_unix']
        assert a['environment']['torch_threads']==1
        worlds=[(r,float(ref),float(s),i) for ref in parent.RSRS for r,_ in parent.REGIMES for i in ids for s in parent.SNRS]
        rng=np.random.default_rng(np.random.SeedSequence([2026102056,0x4250524F,window]));order=[worlds[i] for i in rng.permutation(459)]
        for rows in raw.values():assert [(r['regime'],float(r['rsr']),float(r['snr']),r['index']) for r in rows]==order
        full=dict(raw);full.update({g:[r for n in ns for r in raw[n]] for g,ns in groups.items()})
        assert set(a['methods'])==set(a['cap_verification'])==set(full)
        flags={};overruns={}
        for name,rows in full.items():
            item=a['methods'][name]
            for k,v in stats(rows).items():close(item[k],v)
            for k,v in stats([r for r in rows if r['rsr']==5 and r['snr']>=37.5]).items():close(item['weak_warm100'][k],v)
            assert len(item['by_condition'])==51
            for ref in parent.RSRS:
                for snr in parent.SNRS:
                    for k,v in stats([r for r in rows if r['rsr']==ref and r['snr']==snr]).items():close(item['by_condition'][f'rsr{ref:g}_snr{snr:g}'][k],v)
            cap=a['cap_verification'][name];close(cap['p90_ms'],item['p90_ms']);assert cap['passed']==(item['p90_ms']<=10.74)
            flags[name]=cap['passed'];overruns[name]=[k for k,v in item['by_condition'].items() if v['p90_ms']>10.74]
        if window==1:assert min(p.stat().st_mtime for p in completed(parent.OUT/'test/anchors'))>=a['completed_unix']
        else:
            assert max(p.stat().st_mtime for p in completed(study.OUT/'test/cells'))<=a['started_unix']
            assert (study.OUT/'ber/summary.json').stat().st_mtime<=a['started_unix']
        calls=sum(map(len,raw.values()));assert calls==4590;total+=calls
        windows[f'window{window}']=dict(raw_calls=calls,cap_passed_by_method=flags,conditional_overruns=overruns,p90_ms={n:m['p90_ms'] for n,m in a['methods'].items()})
        hashes[path.name]=sha(path)
    assert total==9180
    return record('timing',timing_hashes=hashes,windows=windows,raw_calls=total,literal_cap_ms=10.74,
        target_met_in_both_windows_by_method={n:all(w['cap_passed_by_method'][n] for w in windows.values()) for n in windows['window1']['cap_passed_by_method']})


def summary():
    wrapped,digest=study.check_selection();parent=study.PARENT
    npth=study.OUT/'nmse_summary.json';bpth=study.OUT/'ber/summary.json'
    n=json.loads(npth.read_text());b=json.loads(bpth.read_text())
    nraw,groups,nnames=schema();braw,bgroups,bnames=schema(True)
    assert groups==bgroups and len(nnames)==28 and len(bnames)==29
    ni,bi=({name:i for i,name in enumerate(names)} for names in (nnames,bnames))
    for report_,names in ((n,nnames),(b,bnames)):
        assert report_['selection_sha256']==report_['protocol_sha256']==digest
        assert report_['parent_selection_sha256']==wrapped['parent_selection_sha256']
        assert report_['exploratory'] and report_['primary'] is False and report_['adjusted_intervals'] is False
        assert report_['method_names']==names and report_['groups']==groups
        assert report_['contrasts']=={k:list(v) for k,v in CONTRASTS.items()}
        assert report_['source_sha256']==sha(Path(__file__).with_name('analysis.py'))
        assert report_['bootstrap']['samples']==20000 and report_['bootstrap']['seed']==2026102058
        assert report_['unique_test_anchors']==1152 and report_['condition_cases']==19584 and report_['bootstrap_index_clusters']==384
        no_adjusted(report_)
    assert n['test_hashes']==b['test_hashes']
    assert n['numerical_tie_tolerance_db']==1e-10
    assert len(n['cells'])==153 and len(b['cells'])==153*29 and len(b['cell_contrasts'])==153*8
    brows={(r['regime'],r['rsr_db'],r['snr_db'],r['method']):r for r in b['cells']}
    bcomp={(r['cell'],r['contrast']):r for r in b['cell_contrasts']}
    assert len(brows)==153*29 and len(bcomp)==153*8
    def expand(values,raw):
        return np.concatenate([values,np.stack([values[[raw.index(x) for x in members]].mean(0) for members in groups.values()])])
    rng=np.random.default_rng(2026102058)
    weights={r:rng.multinomial(128,np.full(128,1/128),size=20000).T.astype(float) for r,_ in parent.REGIMES}
    errors=[];energies=[];counts=[];npoints=[];bpoints=[];nboots=[];bboots=[];full_cell_checks=[]
    for row in n['cells']:
        label=row['cell'];regime,ref,snr=row['regime'],row['rsr'],row['snr']
        assert label==parent.cellname(regime,ref,snr)
        paths={'proposal':study.OUT/'test/cells'/f'{label}.npz','parent':parent.OUT/'test/cells'/f'{label}.npz',
            'proposal_ber':study.OUT/'ber/cells'/f'{label}.npz','parent_ber':parent.OUT/'ber/cells'/f'{label}.npz',
            'data':parent.OUT/'ber/data'/f'{label}.npz'}
        assert n['test_hashes'][label]=={k:sha(p) for k,p in paths.items()}
        pa,a,pb,ab=(load(paths[k]) for k in ('proposal','parent','proposal_ber','parent_ber'))
        err=np.concatenate([np.sum(abs(x['estimates']-a['truth_g'][None])**2,axis=(-2,-1)) for x in (a,pa)])
        energy=np.sum(abs(a['truth_g'])**2,axis=(-2,-1))
        ct=np.concatenate([ab['errors_per_user'],pb['errors_per_user']]).sum(-1)
        assert err.shape==(22,128) and ct.shape==(23,128)
        point=expand(10*np.log10(err.sum(1)/energy.sum()),nraw)
        boot=expand(10*np.log10((err@weights[regime][:,:32])/(energy@weights[regime][:,:32])[None]),nraw)
        rate=expand(ct/384,braw);bp=rate.mean(1);bb=rate@weights[regime][:,:32]/128
        case=err/energy[None];p95,p99=(expand(np.quantile(case,q,axis=1),nraw) for q in (.95,.99));severe=expand((case>1).mean(1),nraw)
        for method,i in ni.items():
            for key,value in dict(nmse_db=point[i],case_nmse_p95=p95[i],case_nmse_p99=p99[i],severe_rate=severe[i]).items():
                close(row['methods'][method][key],value)
        for method,i in bi.items():
            item=brows[regime,ref,snr,method];close(item['ber'],bp[i])
            assert item['bits_per_fit']==128*384 and item['channel_anchors']==128
            if method in groups:assert item['errors_per_training_seed']=={m:int(ct[braw.index(m)].sum()) for m in groups[method]}
            else:assert item['bit_errors']==ct[i].sum()
        for key,(left,right) in CONTRASTS.items():
            close(row['contrasts'][key]['gain_db'],point[ni[left]]-point[ni[right]])
            close(bcomp[label,key]['ber_reduction'],bp[bi[left]]-bp[bi[right]])
        if (ref,snr) in ((5.,0.),(12.,40.)):
            nf=expand(10*np.log10((err@weights[regime])/(energy@weights[regime])[None]),nraw)
            bf=rate@weights[regime]/128
            for method,i in ni.items():close(row['methods'][method]['conditional_ci95_db'],np.quantile(nf[i],[.025,.975]))
            for method,i in bi.items():close(brows[regime,ref,snr,method]['bootstrap95'],np.quantile(bf[i],[.025,.975]))
            for key,(left,right) in CONTRASTS.items():
                close(row['contrasts'][key]['pointwise_ci95_db'],np.quantile(nf[ni[left]]-nf[ni[right]],[.025,.975]))
                close(bcomp[label,key]['bootstrap95'],np.quantile(bf[bi[left]]-bf[bi[right]],[.025,.975]))
            full_cell_checks.append(label)
        errors.append(err);energies.append(energy);counts.append(ct)
        npoints.append(point);bpoints.append(bp);nboots.append(boot);bboots.append(bb)
    assert len(full_cell_checks)==6
    npoints,bpoints,nboots,bboots=map(np.stack,(npoints,bpoints,nboots,bboots))
    savedn=load(study.OUT/'nmse_bootstrap.npz');savedb=load(study.OUT/'ber/bootstrap.npz')
    for saved,names,points in ((savedn,nnames,npoints),(savedb,bnames,bpoints)):
        np.testing.assert_array_equal(saved['method_names'],names);np.testing.assert_array_equal(saved['regions'],REGIONS)
        np.testing.assert_array_equal(saved['cell_names'],[r['cell'] for r in n['cells']]);close(saved['cell_scores'],points)
        assert str(saved['selection_sha256'])==digest
    maximum=0.;scope_counts={}
    for ri,region in enumerate(REGIONS):
        keep=np.array([belongs(region,r['snr'],r['rsr'],r['regime']) for r in n['cells']])
        total=int(keep.sum());scope_counts[region]=total;nr=n['aggregates'][region];br=b['regions'][region]
        assert nr['cells']==br['cells']==total and nr['condition_cases']==br['operating_cases']==total*128
        assert br['coupled_clusters']==128*len({r['regime'] for r,k in zip(n['cells'],keep) if k})
        point,bp=npoints[keep].mean(0),bpoints[keep].mean(0)
        nb,bb=savedn['aggregate_bootstraps'][ri],savedb['aggregate_bootstraps'][ri]
        for full,recomputed in ((nb,nboots[keep].mean(0)),(bb,bboots[keep].mean(0))):
            maximum=max(maximum,float(np.abs(full[:,:32]-recomputed).max()));close(full[:,:32],recomputed)
        err=np.concatenate([a for a,k in zip(errors,keep) if k],1);energy=np.concatenate([a for a,k in zip(energies,keep) if k])
        linear=expand(err.sum(1)/energy.sum(),nraw);pooled=10*np.log10(linear);case=err/energy[None]
        p95,p99=(expand(np.quantile(case,q,axis=1),nraw) for q in (.95,.99));severe=expand((case>1).mean(1),nraw)
        ct=sum((a.sum(1) for a,k in zip(counts,keep) if k))
        for method,i in ni.items():
            item=nr['methods'][method]
            for key,value in dict(nmse_db=point[i],pooled_ratio_of_sums_linear=linear[i],pooled_ratio_of_sums_db=pooled[i],
                                  case_nmse_p95=p95[i],case_nmse_p99=p99[i],severe_rate=severe[i]).items():close(item[key],value)
            close(item['conditional_ci95_db'],np.quantile(nb[i],[.025,.975]))
        for method,i in bi.items():
            item=br['methods'][method];close(item['ber'],bp[i]);close(item['bootstrap95'],np.quantile(bb[i],[.025,.975]))
            assert item['bits_per_fit']==total*128*384
            if method in groups:
                assert item['errors_per_training_seed']=={m:int(ct[braw.index(m)]) for m in groups[method]}
                close(item['training_seed_ber_range'],[min(bp[bi[m]] for m in groups[method]),max(bp[bi[m]] for m in groups[method])])
            else:assert item['bit_errors']==ct[i]
        for key,(left,right) in CONTRASTS.items():
            nc,bc=nr['contrasts'][key],br['contrasts'][key]
            close(nc['gain_db'],point[ni[left]]-point[ni[right]])
            close(nc['pointwise_ci95_db'],np.quantile(nb[ni[left]]-nb[ni[right]],[.025,.975]))
            for seed in parent.SEEDS:
                lf=f'{left[:-5]}_{seed}' if left.endswith('_mean') else left
                rf=f'{right[:-5]}_{seed}' if right.endswith('_mean') else right
                close(nc['seed_specific_gains_db'][str(seed)],point[ni[lf]]-point[ni[rf]])
            close(bc['ber_reduction'],bp[bi[left]]-bp[bi[right]])
            close(bc['bootstrap95'],np.quantile(bb[bi[left]]-bb[bi[right]],[.025,.975]))
            assert bc['both_methods_zero_errors']==bool(bp[bi[left]]==bp[bi[right]]==0)
    assert scope_counts==dict(all=153,high_all=45,transition_all=63,weak_warm100=18,strong_high=30,strong_transition=18,
        ordinary=45,weak=51,rsr5=51,rsr12=51,rsr20=51,all_L2=51,all_L4=51,all_random3to7=51,
        weak_warm100_L2=6,weak_warm100_L4=6,weak_warm100_random3to7=6)
    assert b['overall']==b['regions']['all']
    for key in CONTRASTS:
        gains=[r['contrasts'][key]['gain_db'] for r in n['cells']];worst=int(np.argmin(gains));item=n['negative_cells'][key]
        assert item['negative_cells']==[r['cell'] for r,v in zip(n['cells'],gains) if v < -1e-10]
        assert item['pointwise95_harm_cells']==[r['cell'] for r in n['cells'] if r['contrasts'][key]['pointwise_ci95_db'][1]<-1e-10]
        assert item['worst_cell']==n['cells'][worst]['cell'];close(item['worst_gain_db'],gains[worst])
        rows=[r for r in b['cell_contrasts'] if r['contrast']==key];worst=min(rows,key=lambda x:x['ber_reduction']);item=b['negative_cells'][key]
        assert item['negative_cells']==[r['cell'] for r in rows if r['ber_reduction']<-1e-15]
        assert item['pointwise95_harm_cells']==[r['cell'] for r in rows if r['bootstrap95'][1]<-1e-15]
        assert item['worst_cell']==worst['cell'];close(item['worst_ber_reduction'],worst['ber_reduction'])
    return record('summary',summary_sha256=sha(npth),ber_summary_sha256=sha(bpth),region_counts=scope_counts,
        nmse_bootstrap_sha256=sha(study.OUT/'nmse_bootstrap.npz'),ber_bootstrap_sha256=sha(study.OUT/'ber/bootstrap.npz'),
        all_points_pooled_tails_fit_gains_counts_adverse_lists_verified=True,no_primary_or_adjusted_intervals=True,
        independently_recomputed_bootstrap_draws=32,maximum_bootstrap_error=maximum,
        all20000_cell_intervals_recomputed=full_cell_checks,all_aggregate_interval_quantiles_verified=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=('selection','test','ber','summary','timing'));args=parser.parse_args()
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    print(json.dumps(globals()[args.phase]()),flush=True)
