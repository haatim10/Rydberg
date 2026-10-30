"""Independent BER artifact, data, decoding and coupled-interval audit."""
import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[_key]='1'
import json
from pathlib import Path
import time
import numpy as np
from . import ber
from tcom_followups_20261017.ber_detector import detect,hypotheses,verify_detector
from tcom_pilot_em_20261018.training_protocol import sha,write_json


def run():
    out=ber.OUT;summary=json.loads((out/'summary.json').read_text());protocol=json.loads((out/'protocol.json').read_text())
    for name,digest in protocol['source_hashes'].items():assert sha(ber.REPO/name)==digest
    digest=sha(out/'protocol.json');selected=sha(ber.SOURCE/'policy.json')
    assert summary['protocol_sha256']==digest and summary['selection_sha256']==selected
    x,bits=hypotheses(3)
    assert x.shape==(64,3) and len(np.unique(x,axis=0))==64
    np.testing.assert_allclose(np.abs(x)**2,1,rtol=0,atol=1e-15)
    np.testing.assert_array_equal(bits.sum(0),np.full((3,2),32))
    names=ber.NAMES+tuple(ber.GROUPS);lookup={(r['regime'],r['rsr_db'],r['snr_db'],r['method']):r for r in summary['cells']}
    assert len(lookup)==153*len(names)
    records=[];anchors={};decodes=0;reuses=0;max_data_error=0.;cell_hashes={}
    primary_deltas={r:np.zeros((3,64)) for r in ber.REGIMES}
    for regime,rsr,snr in ber.cells():
        label=ber.label(regime,rsr,snr);path=out/'cells'/(label+'.npz');a=ber.load(path)
        sourcepath=ber.SOURCE/'test/cells'/(label+'.npz');source=ber.load(sourcepath)
        assert sourcepath.stat().st_mtime>=protocol['frozen_unix']
        assert str(a['protocol_sha256'])==digest and str(a['selection_sha256'])==selected
        assert str(a['source_sha256'])==sha(sourcepath)
        datapath=out/'data'/(label+'.npz');assert str(a['data_sha256'])==sha(datapath)
        data=ber.load(datapath);np.testing.assert_array_equal(data['transmitted_bits'],bits)
        np.testing.assert_array_equal(data['hypothesis_symbols'],x)
        assert tuple(map(str,a['method_names']))==ber.NAMES
        source_names=list(map(str,source['method_names']))
        np.testing.assert_array_equal(a['bits_per_user'],np.full((64,3),128))
        assert np.all(a['errors_per_user']>=0) and np.all(a['errors_per_user']<=128)
        anchor_key=(regime,rsr)
        current={k:source[k] for k in ('truth_g','b','indices','local_indices','paths')}
        if anchor_key in anchors:
            for key,value in current.items():np.testing.assert_array_equal(value,anchors[anchor_key][key])
        else:anchors[anchor_key]=current
        for case in (0,63):
            # Reconstruct the documented random stream directly, separately
            # from the production data_for helper.
            rng=np.random.default_rng(np.random.SeedSequence([2026101823,ber.REGIMES.index(regime),ber.RSRS.index(rsr),case]))
            unit=(rng.standard_normal((32,64))+1j*rng.standard_normal((32,64)))/np.sqrt(2)
            z=np.abs(source['truth_g'][case]@x.T+source['b'][case,:,:1]+np.sqrt(source['sigma2'][case])*unit)
            np.testing.assert_array_equal(z,data['z'][case]);max_data_error=max(max_data_error,float(np.max(abs(z-data['z'][case]))))
            for name in ('perfect_csi','original_9101','off_9102','em100_bypass_9103','compact_sparse'):
                channel=source['truth_g'][case] if name=='perfect_csi' else source['estimates'][source_names.index(name),case]
                decoded=detect(data['z'][case],channel,data['b'][case],float(data['sigma2'][case]))
                errors=np.count_nonzero(decoded!=bits,axis=(0,2))
                np.testing.assert_array_equal(errors,a['errors_per_user'][ber.NAMES.index(name),case]);decodes+=1
        for mi,name in enumerate(ber.NAMES):
            other=str(a['duplicate_of'][mi])
            if other:
                channel=source['truth_g'] if name=='perfect_csi' else source['estimates'][source_names.index(name)]
                other_channel=source['truth_g'] if other=='perfect_csi' else source['estimates'][source_names.index(other)]
                np.testing.assert_array_equal(channel,other_channel)
                np.testing.assert_array_equal(a['errors_per_user'][mi],a['errors_per_user'][ber.NAMES.index(other)]);reuses+=1
        counts=a['errors_per_user'].sum(-1);rates=counts/384
        rates=np.concatenate((rates,np.stack([rates[[ber.NAMES.index(n) for n in members]].mean(0) for members in ber.GROUPS.values()])))
        if snr<=20:
            for seed in ber.SEEDS:
                for branch in ('fast_bypass','em100_bypass'):
                    np.testing.assert_array_equal(counts[ber.NAMES.index(f'original_{seed}')],counts[ber.NAMES.index(f'{branch}_{seed}')])
        for mi,name in enumerate(names):
            row=lookup[(regime,rsr,snr,name)]
            assert abs(row['ber']-float(rates[mi].mean()))<1e-16
            if name in ber.GROUPS:
                assert row['errors_per_training_seed']=={n:int(counts[ber.NAMES.index(n)].sum()) for n in ber.GROUPS[name]}
            else:assert row['bit_errors']==int(counts[mi].sum())
        if snr>20:
            for ci,(reference,candidate) in enumerate(ber.CONTRASTS.values()):
                primary_deltas[regime][ci]+=rates[names.index(reference)]-rates[names.index(candidate)]
        records.append(dict(regime=regime,rsr=rsr,snr=snr,rates=rates.mean(-1),errors=counts.sum(-1)))
        cell_hashes[label]=sha(path)
    scopes={
        'overall':(summary['overall'],lambda r:True),
        'primary':(summary['primary'],lambda r:r['snr']>20),
        'ordinary':(summary['ordinary'],lambda r:r['snr']<=20),
        'transition':(summary['transition'],lambda r:20<r['snr']<=45),
        'stress':(summary['stress'],lambda r:r['snr'] in (50,60))}
    for snr in ber.SNRS:scopes[f'snr{snr}']=(summary['by_snr'][f'{snr:g}'],lambda r,s=snr:r['snr']==s)
    for rsr in ber.RSRS:
        scopes[f'rsr{rsr}']=(summary['by_rsr'][f'{rsr:g}'],lambda r,s=rsr:r['rsr']==s)
        scopes[f'high_rsr{rsr}']=(summary['primary_by_rsr'][f'{rsr:g}'],lambda r,s=rsr:r['rsr']==s and r['snr']>20)
    for regime in ber.REGIMES:
        scopes[regime]=(summary['by_regime'][regime],lambda r,s=regime:r['regime']==s)
        scopes['high_'+regime]=(summary['primary_by_regime'][regime],lambda r,s=regime:r['regime']==s and r['snr']>20)
    for key,(saved,include) in scopes.items():
        subset=[r for r in records if include(r)];assert saved['cells']==len(subset)
        point=np.mean([r['rates'] for r in subset],axis=0);counts=np.sum([r['errors'] for r in subset],axis=0)
        for mi,name in enumerate(names):
            actual=saved['methods'][name];assert abs(actual['ber']-float(point[mi]))<1e-16
            assert actual['bits_per_fit']==len(subset)*24576
            if name in ber.GROUPS:assert actual['errors_per_training_seed']=={n:int(counts[ber.NAMES.index(n)]) for n in ber.GROUPS[name]}
            else:assert actual['bit_errors']==int(counts[mi])
        for contrast,(reference,candidate) in ber.CONTRASTS.items():
            assert abs(saved['contrasts'][contrast]['ber_reduction']-(point[names.index(reference)]-point[names.index(candidate)]))<1e-16
    assert summary['primary']['cells']==108 and summary['ordinary']['cells']==45
    assert summary['transition']['cells']==90 and summary['stress']['cells']==18
    # Recompute primary coupled bootstrap from per-anchor summed effects rather
    # than the production code's per-cell bootstrap accumulation.
    rng=np.random.default_rng(2026101829);replicates=np.zeros((3,20000))
    for regime in ('L2','L4','random3to7'):
        w=rng.multinomial(64,np.full(64,1/64),size=20000)
        replicates+=primary_deltas[regime]@w.T/(64*108)
    max_interval_error=0.
    for ci,contrast in enumerate(ber.CONTRASTS):
        saved=summary['primary']['contrasts'][contrast]
        for key,q in [('bootstrap95',[.025,.975]),('bonferroni6_interval',[.05/12,1-.05/12])]:
            actual=np.quantile(replicates[ci],q);error=float(np.max(abs(actual-np.asarray(saved[key]))))
            assert error<1e-15,(contrast,key,error);max_interval_error=max(max_interval_error,error)
    result=dict(passed=True,completed_unix=time.time(),protocol_sha256=digest,selection_sha256=selected,
        source_hashes_match=True,cells=153,physical_anchor_sets=9,ordinary_cells_exact=45,
        regenerated_data_cases=306,maximum_data_reconstruction_error=max_data_error,
        independently_redecoded_case_methods=decodes,exact_prediction_reuses_verified=reuses,
        aggregate_scopes_verified=len(scopes),primary_coupled_intervals_recomputed=True,
        maximum_primary_interval_discrepancy=max_interval_error,detector_checks=verify_detector(),
        cell_hashes=cell_hashes,audit_source_sha256=sha(__file__),
        zero_error_interpretation='No independent-bit rare-event bound is asserted.')
    write_json(out/'audit.json',result);print(json.dumps({k:v for k,v in result.items() if k!='cell_hashes'}),flush=True)


if __name__=='__main__':run()
