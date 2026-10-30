"""Descriptive paired review summaries; no selection or model changes."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='1'
import argparse,json,time
from pathlib import Path
import numpy as np
from .run import OUT,REPO,old,SEEDS,VARIANTS,ROBUST_ARMS,load,write,frozen,sha
B=10000

def summary(errors,bits,energy):
 # arrays: cells,fits,methods,cases;energy:cells,cases
 e=np.asarray(errors);b=np.asarray(bits);d=np.asarray(energy)
 cell=10*np.log10(e.sum(-1)/d.sum(-1)[:,None,None]);pooled=e.sum((0,3))/d.sum();ber=b.sum((0,3))/(len(e)*e.shape[-1]*384)
 return dict(mean_setting_nmse_db=cell.mean((0,1)).tolist(),pooled_nmse=pooled.mean(0).tolist(),ber=ber.mean(0).tolist(),fit_mean_nmse_db=cell.mean(0).tolist(),fit_ber=ber.tolist())

def scopes(meta):
 return dict(all=np.ones(len(meta),bool),ordinary=np.array([r['snr']<=20 for r in meta]),weak_target=np.array([r['ref']==5 and r['snr']>30 for r in meta]),high=np.array([r['snr']>=40 for r in meta]))

def intervals(errors,bits,energy,meta,names,baseline=0):
 e=np.asarray(errors);b=np.asarray(bits);d=np.asarray(energy);rng=np.random.default_rng(2026102203);n=e.shape[-1];gains=np.zeros((B,len(names)));bg=np.zeros_like(gains)
 # Each cell belongs to an independent reference/path block. SNRs remain paired.
 for ref,regime in dict.fromkeys((m['ref'],m['regime']) for m in meta):
  idx=[i for i,m in enumerate(meta) if m['ref']==ref and m['regime']==regime];weights=rng.multinomial(n,np.full(n,1/n),size=B)
  be=np.einsum('bn,cfmn->bcfm',weights,e[idx],optimize=True);bd=weights@d[idx].T;bb=np.einsum('bn,cfmn->bcfm',weights,b[idx],optimize=True)
  nm=10*np.log10(be/bd[:,:,None,None]);means=nm.mean(2);gains+=(means-means[:,:,baseline,None]).sum(1)/len(e)
  rates=bb.mean(2)/(n*384);bg+=(rates-rates[:,:,baseline,None]).sum(1)/len(e)
 return {name:dict(nmse_gain_reference_minus_full_ci95=np.quantile(gains[:,mi],[.025,.975]).tolist(),ber_reference_minus_full_ci95=np.quantile(bg[:,mi],[.025,.975]).tolist()) for mi,name in enumerate(names) if mi!=baseline}

def ablation():
 paths=sorted((OUT/'ablation/cells').glob('*.npz'));assert len(paths)==153
 arrays=[load(p) for p in paths];meta=[dict(ref=float(a['rsr']),snr=float(a['snr']),regime=str(a['regime'])) for a in arrays];e=np.stack([a['errors'] for a in arrays]);b=np.stack([a['bit_errors'] for a in arrays]);d=np.stack([a['truth_energy'] for a in arrays]);result={}
 for scope,mask in scopes(meta).items():
  if mask.any():result[scope]=dict(settings=int(mask.sum()),scores=summary(e[mask],b[mask],d[mask]))
 result['intervals_descriptive']=intervals(e,b,d,meta,VARIANTS);result['variants']=VARIANTS;result['maximum_base_parity_error']=max(float(a['base_parity'].max()) for a in arrays);result['counts']=dict(cells=153,physical_anchors=1152,cases_per_fit=19584);write(OUT/'ablation/summary.json',result);print('Ablation summary',json.dumps(result,indent=2),flush=True)

def robust():
 files=sorted((OUT/'robustness/cells').glob('*.npz'));assert len(files)==198
 groups={'receiver':(0,2,4),'prior':(1,3,5),'compact_sparse':(6,),'em100':(7,)};names=list(groups);result={}
 for arm,_,_,phase in ROBUST_ARMS:
  arrays=[a for p in files if str((a:=load(p))['arm'])==arm];assert len(arrays)==18
  # Single classical estimates are repeated across the fit axis for aggregation only.
  e=np.stack([np.stack([a['errors'][groups[m][min(f,len(groups[m])-1)]] for f in range(3) for m in names]).reshape(3,4,-1) for a in arrays]);b=np.stack([np.stack([a['bit_errors'][groups[m][min(f,len(groups[m])-1)]] for f in range(3) for m in names]).reshape(3,4,-1) for a in arrays]);d=np.stack([a['truth_energy'] for a in arrays]);aligned=np.stack([np.stack([a['phase_adjusted_errors'][groups[m][min(f,len(groups[m])-1)]] for f in range(3) for m in names]).reshape(3,4,-1) for a in arrays]);meta=[dict(ref=float(a['rsr']),snr=float(a['snr']),regime=str(a['regime'])) for a in arrays]
  result[arm]={scope:dict(settings=int(mask.sum()),scores=summary(e[mask],b[mask],d[mask]),phase_adjusted_scores=summary(aligned[mask],b[mask],d[mask])) for scope,mask in scopes(meta).items() if mask.any()}
 write(OUT/'robustness/summary.json',dict(methods=names,arms=result,scope='All intervals/offset comparisons diagnostic;192physicalanchors;576conditioncases perarm. Phase adjustment uses known experimental bias for scoring only.'))
 print('Robustness summary complete',flush=True)

def urformer():
 paths=sorted((OUT/'urformer/test/cells').glob('*.npz'));assert len(paths)==153
 e=[];b=[];d=[];meta=[]
 for p in paths:
  u=load(p);a=load(old.OUT/'test/cells'/p.name);bc=load(old.OUT/'ber/cells'/p.name);mi=list(a['method_names']);bi=list(bc['method_names'])
  errors=[];counts=[]
  for fit,seed in enumerate(SEEDS):
   errors.append(np.stack((a['errors'][mi.index(f'receiver_{seed}')],u['errors'][fit],a['errors'][mi.index(f'prior_{seed}')],a['errors'][mi.index('compact_sparse')],a['errors'][mi.index('em100')])));counts.append(np.stack((bc['errors_per_user'][bi.index(f'receiver_{seed}')].sum(-1),u['bit_errors'][fit],bc['errors_per_user'][bi.index(f'prior_{seed}')].sum(-1),bc['errors_per_user'][bi.index('compact_sparse')].sum(-1),bc['errors_per_user'][bi.index('em100')].sum(-1))))
  e.append(errors);b.append(counts);d.append(a['truth_energy']);meta.append(dict(ref=float(a['rsr']),snr=float(a['snr']),regime=str(a['regime'])))
 e=np.array(e);b=np.array(b);d=np.array(d);names=['receiver','urformer','prior','compact_sparse','em100'];result={'methods':names,'scopes':{}}
 for scope,mask in scopes(meta).items():
  if mask.any():result['scopes'][scope]=dict(settings=int(mask.sum()),scores=summary(e[mask],b[mask],d[mask]))
 result['intervals_descriptive']=intervals(e,b,d,meta,names);write(OUT/'urformer/summary.json',result);print('URformer comparison summarized',flush=True)

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('phase',choices=('ablation','robust','urformer'));args=parser.parse_args();frozen();globals()[args.phase]()
