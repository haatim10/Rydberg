"""Frozen-model high-SNR operating-range study; no training or test tuning."""
import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[_key] = '1'
import copy
from dataclasses import replace
import json
from pathlib import Path
import time
import numpy as np
import torch
from sahure.data import Observations
from sahure_expansion.data import world
from fair_optshrink.estimator import trajectory, PARAMS, lookup_key
from .common import (REPO, ROOT, TEST_SEED, effective_trial_id, load_learned_models,
                     frozen_optshrink, frozen_inputs, sha, write_json)

OUT = ROOT / 'high_snr'
SNRS = [10.,15.,20.,25.,30.,35.,40.,50.,60.]
RSRS = [5.,12.,20.]
REGIMES = [('random3to7',None),('L2',2),('L4',4)]
COUNT = 64
BOOT = 2000

def save(path, **values):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp.npz')
    np.savez_compressed(temp, **values)
    temp.replace(path)

def protocol():
    record=dict(version=1, experiment='frozen high-SNR filtering intervention',
      test_seed=TEST_SEED, local_indices=list(range(COUNT)), N=32,K=3,P=30,
      rsrs=RSRS,snrs=SNRS,path_regimes=REGIMES,bootstrap_samples=BOOT,
      method_names=['learned_9101','learned_9102','learned_9103','off_9101','off_9102','off_9103','em10','em100','optshrink'],
      intervention='Filter-off copies all same checkpoint weights and learned step sizes, changes only config.structure to none; unused controller parameters retained but not computed. No timings claimed.',
      coupling='Generate anchor at SNR10 separately for each RSR/path regime. Retain G,S,B and W/sqrt(v10). Rescale that same noise for all SNRs. SNR cells are dependent; 64 distinct worlds per RSR/regime. SNR10-anchor overlap with other studies must not be counted as independent replication.',
      reference='RSR normalized as existing study; three RSR values in original training domain',
      optshrink='Frozen prior offline lookup; SNR above20 uses20dB settings, no extrapolation tuning or claim optimized out of range.',
      diagnostics='All10 stages: beta, absolute tau, relative tau, normalized channel correction, likelihood, learned gamma; truth used only for scoring.',
      scoring='Per-model dB ratio of summed squared errors to summed channel powers; learned-group summaries average three model scores, never predictions. Severe per-caseNMSE>1.',
      intervals='Pointwise conditional paired channel bootstrap, same indices for all models and seeds. Exclude training-population uncertainty and multiplicity correction.',
      refinement='After coarse sweep, find adjacent mean filtering-benefit sign changes vs same-weight filter-off. Add2.5dB points within each bracket if at least one endpoint pointwise95%CI excludes0. Selection is exploratory curve refinement, never selection of a deployment policy. At most27 new cells.',
      original_inputs=frozen_inputs(),source_sha256=sha(__file__))
    target=OUT/'protocol.json'
    if target.exists():
        old=json.loads(target.read_text()); old.pop('frozen_unix',None)
        if old != json.loads(json.dumps(record)):
            raise RuntimeError('Protocol/source changed after freeze')
    else:
        write_json(target,dict(**record,frozen_unix=time.time()))
    return sha(target)


def anchors(regime,L,rsr,digest):
    path=OUT/'anchors'/f'{regime}_rsr{rsr:g}.npz'
    if path.exists():
        a=dict(np.load(path,allow_pickle=False))
        if str(a['protocol_sha256'])!=digest: raise RuntimeError('anchor protocol mismatch')
        return a
    rows=[]
    for i in range(COUNT):
        w,p=world(effective_trial_id(i,L),TEST_SEED,N=32,K=3,P=30,snr=10.,rsr=rsr,L=L)
        rows.append((w,p))
    a=dict(g=np.stack([w.G for w,p in rows]),s=np.stack([w.S for w,p in rows]),
      b=np.stack([w.B for w,p in rows]),unit_noise=np.stack([w.W/np.sqrt(w.sigma2) for w,p in rows]),
      anchor_z=np.stack([w.Z for w,p in rows]),anchor_variance=np.array([w.sigma2 for w,p in rows]),
      paths=np.array([p for w,p in rows]),indices=np.array([effective_trial_id(i,L) for i in range(COUNT)]),
      protocol_sha256=np.array(digest))
    np.testing.assert_allclose(np.abs(a['g']@a['s']+a['b']+a['unit_noise']*np.sqrt(a['anchor_variance'])[:,None,None]),a['anchor_z'],rtol=0,atol=1e-13)
    save(path,**a)
    return a


def observed(a,snr):
    v=a['anchor_variance']*10**((10.-snr)/10.)
    z=np.abs(a['g']@a['s']+a['b']+a['unit_noise']*np.sqrt(v)[:,None,None])
    return Observations(torch.tensor(z,dtype=torch.float64),torch.tensor(a['s'],dtype=torch.complex128),
                        torch.tensor(a['b'],dtype=torch.complex128),torch.tensor(v,dtype=torch.float64))


@torch.no_grad()
def learned_diagnostics(model,obs):
    raw=[]
    handle=model.net.register_forward_hook(lambda _module,_args,output: raw.append(output.detach()))
    try: g,trace=model(*obs.args(),diagnostics=True)
    finally: handle.remove()
    return g,dict(beta=np.stack([t['beta'].numpy() for t in trace]),
      tau=np.stack([t['tau'].numpy() for t in trace]),
      relative_tau=(.001+torch.nn.functional.softplus(torch.stack(raw)[...,0])).numpy(),
      correction=np.stack([t['correction'].numpy() for t in trace]),
      gamma=np.stack([t['gamma'].numpy() for t in trace]),
      nll=np.stack([t['nll'].numpy() for t in trace]))


def cellname(regime,rsr,snr): return f'{regime}_rsr{rsr:g}_snr{snr:g}'


@torch.no_grad()
def run_cell(models,off,a,regime,rsr,snr,digest,refined=False):
    name=cellname(regime,rsr,snr); path=OUT/'cells'/f'{name}.npz'
    if path.exists():
        if str(np.load(path,allow_pickle=False)['protocol_sha256'])!=digest: raise RuntimeError('cell protocol mismatch')
        return
    obs=observed(a,snr); outputs={}; diag=[]
    for key,model in models.items():
        outputs[key],d=learned_diagnostics(model,obs);diag.append(d)
    for key,model in off.items(): outputs[key]=model(*obs.args())
    em=trajectory(obs,[PARAMS[0]]*len(obs),(10,100))
    outputs['em10']=em[10];outputs['em100']=em[100]
    outputs['optshrink']=frozen_optshrink(obs)
    estimates=np.stack([v.numpy() for v in outputs.values()])
    if not np.isfinite(estimates).all():raise FloatingPointError(name)
    errors=np.sum(np.abs(estimates-a['g'][None])**2,axis=(-2,-1));energy=np.sum(abs(a['g'])**2,axis=(-2,-1))
    diagnostic={k:np.stack([d[k] for d in diag]) for k in diag[0]}
    save(path,estimates=estimates,errors=errors,truth_energy=energy,nmse=errors/energy[None],
      names=np.array(list(outputs)),**diagnostic,indices=a['indices'],paths=a['paths'],snr=np.array(snr),
      rsr=np.array(rsr),regime=np.array(regime),exploratory_refinement=np.array(refined),
      optshrink_lookup=np.array(lookup_key(obs)),protocol_sha256=np.array(digest))
    write_json(OUT/'status.json',dict(phase='sweep',last_cell=name,completed_cells=len(list((OUT/'cells').glob('*.npz'))),updated_unix=time.time()))
    print('Completed',name,flush=True)


def summarize_cell(path):
    a=dict(np.load(path,allow_pickle=False)); e=a['errors'];truth=a['truth_energy']
    rng=np.random.default_rng(2026101799); ids=rng.integers(0,len(truth),(BOOT,len(truth)))
    nmse=10*np.log10(e.sum(1)/truth.sum()); boot=10*np.log10(e[:,ids].sum(-1)/truth[ids].sum(-1)[None])
    def contrast(other):
        gain=np.mean(nmse[other]-nmse[:3]); b=np.mean(boot[other]-boot[:3],axis=0)
        return dict(gain_db=float(gain),ci95_db=np.quantile(b,[.025,.975]).tolist(),positive_means='learned filtered better')
    methods={str(n):dict(nmse_db=float(nmse[i]),severe=float(np.mean(a['nmse'][i]>1))) for i,n in enumerate(a['names'])}
    methods['learned_mean']=dict(nmse_db=float(nmse[:3].mean()),severe=float(np.mean(a['nmse'][:3]>1)))
    methods['off_mean']=dict(nmse_db=float(nmse[3:6].mean()),severe=float(np.mean(a['nmse'][3:6]>1)))
    return dict(cell=path.stem,regime=str(a['regime']),rsr=float(a['rsr']),snr=float(a['snr']),
      exploratory_refinement=bool(a['exploratory_refinement']),methods=methods,
      learned_vs_same_weight_filteroff=contrast(slice(3,6)),learned_vs_em100=contrast(7),learned_vs_em10=contrast(6),
      diagnostics=dict(final_beta_median=float(np.median(a['beta'][:,-1])),final_relative_tau_median=float(np.median(a['relative_tau'][:,-1])),
        final_relative_correction_median=float(np.median(a['correction'][:,-1])),final_tau_median=float(np.median(a['tau'][:,-1]))),
      optshrink_lookup=sorted(set(map(str,a['optshrink_lookup']))))


def refinement_plan(rows):
    plan=[];brackets=[]
    for regime,L in REGIMES:
        for rsr in RSRS:
            cells=sorted([r for r in rows if r['regime']==regime and r['rsr']==rsr and not r['exploratory_refinement']],key=lambda r:r['snr'])
            for a,b in zip(cells,cells[1:]):
                x=a['learned_vs_same_weight_filteroff'];y=b['learned_vs_same_weight_filteroff']
                if x['gain_db']*y['gain_db']<0:
                    confident=any(ci[0]*ci[1]>0 for ci in [x['ci95_db'],y['ci95_db']])
                    brackets.append(dict(regime=regime,rsr=rsr,lo=a['snr'],hi=b['snr'],one_endpoint_excludes_zero=confident))
                    if confident:
                        for snr in np.arange(a['snr']+2.5,b['snr'],2.5):
                            if len(plan)<27: plan.append(dict(regime=regime,L=L,rsr=rsr,snr=float(snr)))
    return dict(created_after_coarse=True,exploratory=True,brackets=brackets,cells=plan)


def report(rows,digest):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    summary=dict(protocol_sha256=digest,cells=rows,complete=True,
      independent_worlds=COUNT*len(RSRS)*len(REGIMES),condition_cases=COUNT*len(rows),
      method_notes='All learned summaries average three fitted model scores, not estimates. Pointwise CIs conditional on fixed checkpoints. Different SNRs reuse each anchor world.',
      cell_hashes={p.name:sha(p) for p in sorted((OUT/'cells').glob('*.npz'))})
    write_json(OUT/'summary.json',summary)
    fig,axes=plt.subplots(3,3,figsize=(14,11),sharex=True)
    for i,(regime,L) in enumerate(REGIMES):
        for j,rsr in enumerate(RSRS):
            ax=axes[i,j];rr=sorted([r for r in rows if r['regime']==regime and r['rsr']==rsr],key=lambda r:r['snr'])
            for key,label in [('learned_mean','Frozen SAHURE'),('off_mean','Same weights, no filter'),('em10','EM10'),('em100','EM100'),('optshrink','Frozen offline OptShrink')]:
                ax.plot([r['snr'] for r in rr],[r['methods'][key]['nmse_db'] for r in rr],marker='.',label=label)
            ax.axvline(20,color='.65',ls='--',lw=.8);ax.set_title(f'{regime}, RSR {rsr:g} dB');ax.grid(alpha=.2)
            if i==2:ax.set_xlabel('SNR (dB)')
            if j==0:ax.set_ylabel('Channel NMSE (dB)')
    handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=5)
    fig.suptitle('Frozen-model operating range: filtering bias versus measurement iterations\n64 paired channels per panel; same channels/noise across SNR; training ends at20 dB')
    fig.tight_layout(rect=(0,.04,1,.95));fig.savefig(OUT/'high_snr_nmse.png',dpi=180);fig.savefig(OUT/'high_snr_nmse.svg');plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(14,4.5),sharey=True)
    for ax,(regime,L) in zip(axes,REGIMES):
        for rsr in RSRS:
            rr=sorted([r for r in rows if r['regime']==regime and r['rsr']==rsr],key=lambda r:r['snr'])
            x=[r['snr'] for r in rr];c=[r['learned_vs_same_weight_filteroff'] for r in rr]
            ax.plot(x,[r['gain_db'] for r in c],marker='.',label=f'RSR {rsr:g}')
            ax.fill_between(x,[r['ci95_db'][0] for r in c],[r['ci95_db'][1] for r in c],alpha=.12)
        ax.axhline(0,color='k',lw=.7);ax.axvline(20,color='.6',ls='--');ax.set_title(regime);ax.set_xlabel('SNR (dB)');ax.grid(alpha=.2)
    axes[0].set_ylabel('Filtering benefit vs same weights, filter off (dB)');axes[-1].legend();fig.suptitle('Positive: filtering helps; bands: pointwise conditional paired-bootstrap95% intervals')
    fig.tight_layout();fig.savefig(OUT/'high_snr_filtering_gain.png',dpi=180);fig.savefig(OUT/'high_snr_filtering_gain.svg');plt.close(fig)


@torch.no_grad()
def main():
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    start=time.time();digest=protocol();models=load_learned_models();off={}
    for name,m in models.items():
        n=copy.deepcopy(m);n.config=replace(n.config,structure='none');off[name.replace('learned_','off_')]=n
        for key,value in m.state_dict().items(): assert torch.equal(value,n.state_dict()[key])
    # Meaningful control check: optimized filter-off equals same model gate_override=0.
    a=anchors('random3to7',None,12.,digest);small=observed(a,10.).batch(slice(0,2));checks={}
    for name,m in models.items():
        direct=m(*small.args(),gate_override=0);fast=off[name.replace('learned_','off_')](*small.args())
        difference=float(torch.max(torch.abs(direct-fast)));assert difference<1e-12
        checked,diag=learned_diagnostics(m,small);ordinary=m(*small.args())
        assert torch.equal(checked,ordinary)
        checks[name]=dict(filteroff_max_difference=difference,diagnostic_forward_exact=True,
          parameter_count=sum(p.numel() for p in m.parameters()),learned_gammas=(2*torch.sigmoid(m.steps)).tolist())
    write_json(OUT/'preflight.json',dict(passed=True,controls=checks,protocol_sha256=digest))
    for regime,L in REGIMES:
        for rsr in RSRS:
            a=anchors(regime,L,rsr,digest)
            for snr in SNRS:run_cell(models,off,a,regime,rsr,snr,digest)
    coarse=[summarize_cell(p) for p in sorted((OUT/'cells').glob('*.npz'))]
    plan_path=OUT/'exploratory_refinement.json'
    if plan_path.exists():plan=json.loads(plan_path.read_text())
    else:plan=refinement_plan(coarse);write_json(plan_path,plan)
    for item in plan['cells']:
        a=anchors(item['regime'],item['L'],item['rsr'],digest)
        run_cell(models,off,a,item['regime'],item['rsr'],item['snr'],digest,True)
    rows=[summarize_cell(p) for p in sorted((OUT/'cells').glob('*.npz'))];report(rows,digest)
    write_json(OUT/'status.json',dict(phase='complete',cells=len(rows),elapsed_seconds=time.time()-start,protocol_sha256=digest))
    print(json.dumps(dict(complete=True,cells=len(rows),seconds=time.time()-start,summary=str(OUT/'summary.json'))),flush=True)

if __name__=='__main__': main()
