"""Frozen-policy supplied-noise diagnostic on common physical test observations."""
import os
for _name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[_name]='1'
import argparse
import json
import numpy as np
import torch
from tcom_followups_20261017.common import sha,write_json
from . import study

ROOT=study.OUT/'sensitivity'
OFFSETS=(-3.,0.,3.)
BOOTSTRAPS=4000
BOOTSTRAP_SEED=2026101828
NAMES=tuple([f'original_{s}' for s in study.SEEDS]+[f'fast_bypass_{s}' for s in study.SEEDS]+
    [f'em100_bypass_{s}' for s in study.SEEDS]+['em100','compact_sparse'])
GROUPS={key:study.GROUPS[key] for key in ('original_mean','fast_bypass_mean','em100_bypass_mean')}
GROUPS.update(em100=['em100'],compact_sparse=['compact_sparse'])


def conditions(selection):
    rows=[]
    for ref in study.RSRS:
        cutoff=float(selection['policies']['em100'][f'{ref:g}'])
        # The maximum candidate denotes no bypass within the declared 0--60 dB domain.
        snrs=sorted(set([20.,60.]+[max(min(s,max(study.SNRS)),min(study.SNRS)) for s in (cutoff-2.5,cutoff,cutoff+2.5)]))
        rows.extend(dict(rsr=float(ref),actual_snr=float(snr),offset=float(offset)) for snr in snrs for offset in OFFSETS)
    return rows


def prepare():
    selection,digest=study.check_selection()
    record=dict(version=1,selection_sha256=digest,parent_protocol_sha256=selection['protocol_sha256'],
        source_sha256=sha(__file__),conditions=conditions(selection),methods=NAMES,groups=GROUPS,
        anchors='Same64 random3to7 test anchors for each known reference; no new channel draws or policy adaptation.',
        measurement_control='Actual channel, Z, S and B unchanged; supplied variance multiplied by10^(offset/10) for every estimator and lookup.',
        scope='Diagnostic NMSE and branch fractions only. Finite tested mismatch does not prove robust deployment.',
        bootstrap=dict(replicates=BOOTSTRAPS,seed=BOOTSTRAP_SEED,unit='Shared whole-anchor indices across reference, SNR, offsets, estimators and trained fits; fixed weights and selected policies.'))
    path=ROOT/'protocol.json'
    if path.exists():
        assert json.loads(path.read_text())==json.loads(json.dumps(record)),'sensitivity protocol changed'
    else:write_json(path,record)
    return selection,sha(path)


def cellpath(ref,snr,offset):
    return ROOT/'cells'/f'random3to7_rsr{ref:g}_snr{snr:g}_offset{offset:+g}.npz'


@torch.no_grad()
def evaluate():
    selection,digest=prepare();selection_sha=sha(study.OUT/'policy.json');est=study.engine()
    for condition in conditions(selection):
        ref,snr,offset=(condition[k] for k in ('rsr','actual_snr','offset'))
        target=cellpath(ref,snr,offset)
        if target.exists():
            assert str(study.load(target)['protocol_sha256'])==digest
            continue
        anchor_path=study.OUT/'test/anchors'/f'random3to7_rsr{ref:g}.npz'
        anchor=study.load(anchor_path)
        assert int(anchor['seed'])==study.TEST_SEED and str(anchor['selection_sha256'])==selection_sha
        assert len(anchor['g'])==study.TEST_COUNT
        original_obs=study.observed(anchor,snr,P=30)
        obs=study.observed(anchor,snr,variance_offset_db=offset,P=30)
        for a,b in zip(original_obs.args()[:3],obs.args()[:3]):assert torch.equal(a,b)
        torch.testing.assert_close(obs.sigma2,original_obs.sigma2*10**(offset/10),atol=0,rtol=0)
        predictions=torch.stack([study.estimate_method(obs,name,selection,est) for name in NAMES])
        assert torch.isfinite(predictions).all()
        masks={kind:est.masks(obs,selection['policies'][kind])['bypass'].numpy() for kind in ('same_steps','em100')}
        estimates=predictions.numpy();truth=anchor['g']
        errors=(np.abs(estimates-truth[None])**2).sum((-2,-1));energy=(np.abs(truth)**2).sum((-2,-1))
        study.save(target,method_names=np.array(NAMES),estimates=estimates,errors=errors,truth_energy=energy,
            nmse=errors/energy[None],bypass_fast=masks['same_steps'],bypass_em100=masks['em100'],
            supplied_snr_proxy_db=np.array(snr-offset),actual_snr=np.array(snr),rsr=np.array(ref),
            supplied_variance_offset_db=np.array(offset),indices=anchor['indices'],
            anchor_sha256=np.array(sha(anchor_path)),selection_sha256=np.array(selection_sha),protocol_sha256=np.array(digest))
        print(json.dumps(dict(phase='noise_sensitivity',**condition,
            fast_fraction=float(masks['same_steps'].mean()),em100_fraction=float(masks['em100'].mean()))),flush=True)


def analyze():
    selection,digest=prepare()
    ids=np.random.default_rng(BOOTSTRAP_SEED).integers(0,study.TEST_COUNT,size=(BOOTSTRAPS,study.TEST_COUNT))
    rows=[];boot_records={}
    for condition in conditions(selection):
        ref,snr,offset=(condition[k] for k in ('rsr','actual_snr','offset'))
        result=study.load(cellpath(ref,snr,offset));assert str(result['protocol_sha256'])==digest
        errors,energy=result['errors'],result['truth_energy']
        fit_db=10*np.log10(errors.sum(-1)/energy.sum())
        fit_boot=10*np.log10(errors[:,ids].sum(-1)/energy[ids].sum(-1)[None])
        methods={};group_boot={}
        for group,names in GROUPS.items():
            indices=[NAMES.index(n) for n in names];values=fit_boot[indices].mean(0);group_boot[group]=values
            methods[group]=dict(nmse_db=float(fit_db[indices].mean()),per_fit_nmse_db=fit_db[indices].tolist(),
                bootstrap95=np.quantile(values,[.025,.975]).tolist(),
                severe_nmse_above1_rate=float((result['nmse'][indices]>1).mean()))
        row=dict(rsr=ref,actual_snr_db=snr,supplied_variance_offset_db=offset,supplied_snr_proxy_db=snr-offset,
            fast_bypass_fraction=float(result['bypass_fast'].mean()),em100_bypass_fraction=float(result['bypass_em100'].mean()),methods=methods)
        for group in ('fast_bypass_mean','em100_bypass_mean'):
            gain=group_boot['original_mean']-group_boot[group]
            row[group+'_gain_vs_same_supplied_noise_original_db']=methods['original_mean']['nmse_db']-methods[group]['nmse_db']
            row[group+'_gain_bootstrap95']=np.quantile(gain,[.025,.975]).tolist()
        rows.append(row);boot_records[(ref,snr,offset)]=group_boot
    lookup={(r['rsr'],r['actual_snr_db'],r['supplied_variance_offset_db']):r for r in rows}
    for row in rows:
        ref,snr,offset=row['rsr'],row['actual_snr_db'],row['supplied_variance_offset_db']
        correct=lookup[(ref,snr,0.)]
        for group,value in row['methods'].items():
            value['degradation_vs_correct_variance_db']=value['nmse_db']-correct['methods'][group]['nmse_db']
            diff=boot_records[(ref,snr,offset)][group]-boot_records[(ref,snr,0.)][group]
            value['degradation_bootstrap95']=np.quantile(diff,[.025,.975]).tolist()
    output=dict(passed=True,protocol_sha256=digest,selection_sha256=sha(study.OUT/'policy.json'),
        cells=rows,conditions=len(rows),independent_anchor_count=study.TEST_COUNT*len(study.RSRS),
        coupled_case_index_clusters=study.TEST_COUNT,
        interpretation='Positive bypass gain favors bypass over original supplied the same variance; positive degradation means mismatch harms the same method. Changes include lookup routing and estimator response to supplied variance.',
        caveat='Ordinary preservation applies to receiver proxy, not unknown actual SNR under miscalibration. Pointwise intervals condition on frozen models/policies. No BER or general mismatch robustness claim.')
    write_json(ROOT/'summary.json',output)
    return output


def audit():
    selection,digest=prepare();count=0;ordinary=0
    for condition in conditions(selection):
        ref,snr,offset=(condition[k] for k in ('rsr','actual_snr','offset'))
        data=study.load(cellpath(ref,snr,offset));anchor_path=study.OUT/'test/anchors'/f'random3to7_rsr{ref:g}.npz'
        anchor=study.load(anchor_path)
        assert str(data['protocol_sha256'])==digest and str(data['anchor_sha256'])==sha(anchor_path)
        np.testing.assert_array_equal(data['method_names'],NAMES)
        errors=(np.abs(data['estimates']-anchor['g'][None])**2).sum((-2,-1));energy=(np.abs(anchor['g'])**2).sum((-2,-1))
        np.testing.assert_array_equal(errors,data['errors']);np.testing.assert_array_equal(energy,data['truth_energy'])
        for kind,key in (('same_steps','bypass_fast'),('em100','bypass_em100')):
            expected=snr-offset>selection['policies'][kind][f'{ref:g}']+1e-10 and snr-offset>20+1e-10
            np.testing.assert_array_equal(data[key],np.full(study.TEST_COUNT,expected))
        if snr-offset<=20+1e-10:
            np.testing.assert_array_equal(data['estimates'][:3],data['estimates'][3:6])
            np.testing.assert_array_equal(data['estimates'][:3],data['estimates'][6:9]);ordinary+=1
        count+=1
    result=dict(passed=True,cells=count,proxy_ordinary_identity_cells=ordinary,raw_errors_and_gate_decisions_recomputed=True,
        source_sha256=sha(__file__),protocol_sha256=digest)
    write_json(ROOT/'audit.json',result);return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=('prepare','evaluate','analyze','audit'))
    args=parser.parse_args();torch.set_num_threads(1)
    try:torch.set_num_interop_threads(1)
    except RuntimeError:pass
    result=globals()[args.phase]()
    if args.phase in ('analyze','audit'):print(json.dumps({k:v for k,v in result.items() if k!='cells'},indent=2))
