import json,hashlib
from pathlib import Path
import numpy as np
root=Path('/Users/haatim/.codex/.chatgpt-projects/g-p-6aa7dfe2b65c8191b548f107c7ddcd75/tmp/Rydberg')
p=root/'results/tcom_hybrid_20261020';seeds=(9101,9102,9103)
rows=[];strong=[];source_hashes={}
for path in sorted((p/'test/cells').glob('*.npz')):
 if '.tmp.' in path.name:continue
 with np.load(path,allow_pickle=False) as a:
  ref,snr=float(a['rsr']),float(a['snr']);regime=str(a['regime'])
  if not ((ref==5 and snr>=37.5) or (ref in (12,20) and snr>=40)):continue
  names=list(map(str,a['method_names']));rawnames=list(map(str,a['raw_names']));energy=a['truth_energy']
  if ref==5:
   source_hashes[path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
   for seed in seeds:
    errors={f:a['errors'][names.index(f'{f}_{seed}')] for f in ('global40','adaptive40','latest')}
    errors['global_raw']=a['raw_errors'][rawnames.index(f'global_{seed}')]
    errors['adaptive_raw']=a['raw_errors'][rawnames.index(f'adaptive_{seed}')]
    metrics={}
    for family,e in errors.items():
     nm=e/energy;order=np.argsort(e)[::-1]
     metrics[family]=dict(nmse_db=float(10*np.log10(e.sum()/energy.sum())),median_case_nmse=float(np.median(nm)),p95_case_nmse=float(np.quantile(nm,.95)),p99_case_nmse=float(np.quantile(nm,.99)),maximum_case_nmse=float(nm.max()),severe_nmse_over1=int((nm>1).sum()),top1_error_share=float(e[order[:1]].sum()/e.sum()),top5_error_share=float(e[order[:5]].sum()/e.sum()))
    top=np.argsort(errors['global40'])[::-1][:5]
    anchors=[dict(local_index=int(a['local_indices'][i]),effective_index=int(a['indices'][i]),truth_energy=float(energy[i]),error_share=float(errors['global40'][i]/errors['global40'].sum()),case_nmse={f:float(e[i]/energy[i]) for f,e in errors.items()},global40_to_raw_error_ratio=float(errors['global40'][i]/errors['global_raw'][i])) for i in top]
    rows.append(dict(cell=path.stem,regime=regime,snr=snr,seed=seed,metrics=metrics,top_global40_anchors=anchors))
  else:
   for seed in seeds:
    eg=a['errors'][names.index(f'global40_{seed}')];el=a['errors'][names.index(f'latest_{seed}')]
    ng=eg/energy;nl=el/energy
    strong.append(dict(cell=path.stem,regime=regime,rsr=ref,snr=snr,seed=seed,global40_nmse_db=float(10*np.log10(eg.sum()/energy.sum())),latest_nmse_db=float(10*np.log10(el.sum()/energy.sum())),global_gain_db=float(10*np.log10(el.sum()/eg.sum())),maximum_abs_case_nmse_difference=float(np.max(abs(ng-nl)))))
result=dict(posthoc=True,no_new_estimation_or_selection=True,scope='Saved fresh primary predictions only; weak reference SNR>=37.5 and strong reference SNR>=40. Top1/top5 use raw squared-error contribution. Existing NMSE>1 severe threshold retained.',source_hashes=source_hashes,weak=rows,strong=strong)
out=p/'posthoc_weak_error_concentration.json';out.write_text(json.dumps(result,indent=2)+'\n')
print('weak condition-fit rows',len(rows),'strong',len(strong))
for seed in seeds:
 group=[r for r in rows if r['seed']==seed]
 print('seed',seed,'mean-db',{f:round(np.mean([r['metrics'][f]['nmse_db'] for r in group]),6) for f in ('global40','adaptive40','latest')},'global top1 median/range',np.median([r['metrics']['global40']['top1_error_share'] for r in group]),(min(r['metrics']['global40']['top1_error_share'] for r in group),max(r['metrics']['global40']['top1_error_share'] for r in group)))
for regime in ('L2','L4','random3to7'):
 for seed in seeds:
  r=next(r for r in rows if r['regime']==regime and r['seed']==seed and r['snr']==60)
  print('SNR60',regime,seed,'metrics',json.dumps(r['metrics']),'top',json.dumps(r['top_global40_anchors'][:2]))
print('strong mean gains by seed',{seed:float(np.mean([r['global_gain_db'] for r in strong if r['seed']==seed])) for seed in seeds})
print('strong gain range',min(r['global_gain_db'] for r in strong),max(r['global_gain_db'] for r in strong))
print('strong worst',min(strong,key=lambda r:r['global_gain_db']))
print('output',out)
