"""Prespecified fixed-policy noise mismatch on common fresh test observations."""
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
BOOTSTRAP_SEED=2026101928
OLD_POLICY_PATH=study.REPO/'results/tcom_retrained_bypass_20261018/policy.json'
OLD_POLICY=json.loads(OLD_POLICY_PATH.read_text())['policies']['em100']
NAMES=tuple([f'original_{s}' for s in study.SEEDS]+[f'old_bypass_{s}' for s in study.SEEDS]+
    [f'repair_{s}' for s in study.SEEDS]+['em100','compact_sparse'])
GROUPS={key:study.GROUPS[key] for key in ('original_mean','old_bypass_mean','repair_mean')}
GROUPS.update(em100=['em100'],compact_sparse=['compact_sparse'])


def conditions():
    return [dict(regime=regime,rsr=float(ref),actual_snr=float(snr),offset=float(offset))
        for regime,_ in study.REGIMES for ref in study.RSRS
        for snr in ((35.,40.,45.) if ref==5. else (32.5,)) for offset in OFFSETS]


def prepare():
    selection,digest=study.check_selection()
    record=dict(version=1,selection_sha256=digest,parent_protocol_sha256=selection['protocol_sha256'],
        source_sha256=sha(__file__),old_policy_sha256=sha(OLD_POLICY_PATH),conditions=conditions(),methods=NAMES,groups=GROUPS,
        anchors='Same128 fresh test anchors per path regime/reference; no new channel draws, selection or policy adaptation.',
        measurement_control='Physical channel, Z, S and B remain fixed. Every estimator and dispatch receives actual variance multiplied by10^(offset/10).',
        scope='Diagnostic NMSE and selected branches only, with no BER. This finite mismatch grid is not a general robustness guarantee.',
        bootstrap=dict(replicates=BOOTSTRAPS,seed=BOOTSTRAP_SEED,
            unit='128 whole-anchor indices per path regime, shared across reference/SNR/offset/method/fitted seed; fixed models and selected policies.'))
    path=ROOT/'protocol.json'
    if path.exists():assert json.loads(path.read_text())==json.loads(json.dumps(record)),'sensitivity protocol changed'
    else:write_json(path,record)
    return selection,sha(path)


def cellpath(regime,ref,snr,offset):
    return ROOT/'cells'/f'{regime}_rsr{ref:g}_snr{snr:g}_offset{offset:+g}.npz'


@torch.no_grad()
def evaluate():
    selection,digest=prepare();selection_sha=sha(study.OUT/'policy.json');est=study.engine()
    for condition in conditions():
        regime,ref,snr,offset=(condition[k] for k in ('regime','rsr','actual_snr','offset'))
        target=cellpath(regime,ref,snr,offset)
        if target.exists():
            assert str(study.load(target)['protocol_sha256'])==digest
            continue
        anchor_path=study.OUT/'test/anchors'/f'{regime}_rsr{ref:g}.npz';anchor=study.load(anchor_path)
        assert int(anchor['seed'])==study.TEST_SEED and str(anchor['selection_sha256'])==selection_sha
        assert len(anchor['g'])==study.TEST_COUNT==128
        correct=study.observed(anchor,snr,P=30)
        obs=study.observed(anchor,snr,variance_offset_db=offset,P=30)
        for a,b in zip(correct.args()[:3],obs.args()[:3]):assert torch.equal(a,b)
        torch.testing.assert_close(obs.sigma2,correct.sigma2*10**(offset/10),atol=0,rtol=0)
        predictions=torch.stack([study.estimate_method(obs,name,selection,est) for name in NAMES])
        assert torch.isfinite(predictions).all()
        old_mask=est.masks(obs,OLD_POLICY)['bypass'].numpy()
        choice=study.selected_repair_choice(obs,selection)
        estimates=predictions.numpy();truth=anchor['g']
        errors=(np.abs(estimates-truth[None])**2).sum((-2,-1));energy=(np.abs(truth)**2).sum((-2,-1))
        study.save(target,method_names=np.array(NAMES),estimates=estimates,errors=errors,truth_energy=energy,
            nmse=errors/energy[None],old_bypass=old_mask,repair_choice=np.array(choice),regime=np.array(regime),
            supplied_snr_proxy_db=np.array(snr-offset),actual_snr=np.array(snr),rsr=np.array(ref),
            supplied_variance_offset_db=np.array(offset),indices=anchor['indices'],
            anchor_sha256=np.array(sha(anchor_path)),selection_sha256=np.array(selection_sha),protocol_sha256=np.array(digest))
        print(json.dumps(dict(phase='sensitivity',**condition,old_bypass_fraction=float(old_mask.mean()),repair_choice=choice)),flush=True)


def analyze():
    selection,digest=prepare();rng=np.random.default_rng(BOOTSTRAP_SEED)
    weights={name:rng.multinomial(128,np.full(128,1/128),size=BOOTSTRAPS).T.astype(float) for name,_ in study.REGIMES}
    rows=[];boot_records={}
    for condition in conditions():
        regime,ref,snr,offset=(condition[k] for k in ('regime','rsr','actual_snr','offset'))
        data=study.load(cellpath(regime,ref,snr,offset));assert str(data['protocol_sha256'])==digest
        errors,energy=data['errors'],data['truth_energy'];fit_db=10*np.log10(errors.sum(-1)/energy.sum())
        fit_boot=10*np.log10((errors@weights[regime])/(energy@weights[regime])[None])
        methods={};group_boot={}
        for group,names in GROUPS.items():
            ids=[NAMES.index(name) for name in names];values=fit_boot[ids].mean(0);group_boot[group]=values
            methods[group]=dict(nmse_db=float(fit_db[ids].mean()),per_fit_nmse_db=fit_db[ids].tolist(),
                bootstrap95=np.quantile(values,[.025,.975]).tolist(),severe_nmse_above1_rate=float((data['nmse'][ids]>1).mean()))
        contrasts={}
        for reference in ('original_mean','old_bypass_mean','em100','compact_sparse'):
            difference=group_boot[reference]-group_boot['repair_mean']
            contrasts['repair_vs_'+reference]=dict(gain_db=methods[reference]['nmse_db']-methods['repair_mean']['nmse_db'],
                bootstrap95=np.quantile(difference,[.025,.975]).tolist())
        row=dict(regime=regime,rsr=ref,actual_snr_db=snr,supplied_variance_offset_db=offset,supplied_snr_proxy_db=snr-offset,
            old_bypass_fraction=float(data['old_bypass'].mean()),repair_choice=str(data['repair_choice']),methods=methods,contrasts=contrasts)
        rows.append(row);boot_records[(regime,ref,snr,offset)]=group_boot
    lookup={(r['regime'],r['rsr'],r['actual_snr_db'],r['supplied_variance_offset_db']):r for r in rows}
    for row in rows:
        regime,ref,snr,offset=row['regime'],row['rsr'],row['actual_snr_db'],row['supplied_variance_offset_db']
        correct=lookup[(regime,ref,snr,0.)]
        for group,value in row['methods'].items():
            value['degradation_vs_correct_variance_db']=value['nmse_db']-correct['methods'][group]['nmse_db']
            difference=boot_records[(regime,ref,snr,offset)][group]-boot_records[(regime,ref,snr,0.)][group]
            value['degradation_bootstrap95']=np.quantile(difference,[.025,.975]).tolist()
    result=dict(protocol_sha256=digest,selection_sha256=sha(study.OUT/'policy.json'),cells=rows,conditions=len(rows),
        unique_channel_reference_anchors=128*len(study.REGIMES)*len(study.RSRS),coupled_case_index_clusters=128*len(study.REGIMES),
        interpretation='Positive gain favors repair over the comparator supplied the same variance; positive degradation means noise mismatch worsens the same method. Routing and estimator responses both change.',
        caveat='Descriptive pointwise intervals condition on frozen weights/policies. No sensitivity BER, universal non-degradation, training-population or arbitrary-mismatch guarantee.')
    write_json(ROOT/'summary.json',result);return result


def audit():
    selection,digest=prepare();count=0;identity=0;spectral=0
    for condition in conditions():
        regime,ref,snr,offset=(condition[k] for k in ('regime','rsr','actual_snr','offset'))
        data=study.load(cellpath(regime,ref,snr,offset));anchor_path=study.OUT/'test/anchors'/f'{regime}_rsr{ref:g}.npz'
        anchor=study.load(anchor_path)
        assert str(data['protocol_sha256'])==digest and str(data['anchor_sha256'])==sha(anchor_path)
        np.testing.assert_array_equal(data['method_names'],NAMES)
        errors=(np.abs(data['estimates']-anchor['g'][None])**2).sum((-2,-1));energy=(np.abs(anchor['g'])**2).sum((-2,-1))
        np.testing.assert_array_equal(errors,data['errors']);np.testing.assert_array_equal(energy,data['truth_energy'])
        np.testing.assert_array_equal(errors/energy[None],data['nmse'])
        on=snr-offset>OLD_POLICY[f'{ref:g}']+1e-10 and snr-offset>20+1e-10
        np.testing.assert_array_equal(data['old_bypass'],np.full(128,on))
        expected=data['estimates'][9:10].repeat(3,axis=0) if on else data['estimates'][:3]
        np.testing.assert_array_equal(data['estimates'][3:6],expected)
        obs=study.observed(anchor,snr,variance_offset_db=offset,P=30)
        choice=study.selected_repair_choice(obs,selection);assert str(data['repair_choice'])==choice
        if choice=='original':np.testing.assert_array_equal(data['estimates'][6:9],data['estimates'][:3]);identity+=1
        if choice=='spectral100':np.testing.assert_array_equal(data['estimates'][6:9],data['estimates'][9:10].repeat(3,axis=0));spectral+=1
        count+=1
    result=dict(passed=True,cells=count,original_repair_identity_cells=identity,spectral_repair_identity_cells=spectral,
        raw_errors_and_branch_decisions_checked=True,source_sha256=sha(__file__),protocol_sha256=digest)
    write_json(ROOT/'audit.json',result);return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=('prepare','evaluate','analyze','audit'))
    args=parser.parse_args();torch.set_num_threads(1)
    try:torch.set_num_interop_threads(1)
    except RuntimeError:pass
    result=globals()[args.phase]()
    if args.phase in ('analyze','audit'):print(json.dumps({k:v for k,v in result.items() if k!='cells'},indent=2))
