"""Read-only review of frozen experiments; new analysis outputs only."""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:
    os.environ[key]='1'
from pathlib import Path
import json
import hashlib
import numpy as np
import torch
from sahure.data import load_observations,split_observations
from sahure.train import load_model
from sahure_checks.train import channel_loss
from sahure_expansion.protocol import CONFIG

BASE=Path('results/sahure_checks'); OLD=Path('results/sahure_expansion')
OUT=Path('results/sahure_submission_review')
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

@torch.no_grad()
def main():
    torch.set_num_threads(1);OUT.mkdir(parents=True,exist_ok=True)
    frozen=read(BASE/'frozen_selection.json');summary=read(BASE/'summary.json');protocol=read(BASE/'protocol.json')
    checked=0
    for path,digest in {**protocol['sources'],**protocol['previous_sha256'],**read(BASE/'replicate_data_hashes.json'),**read(BASE/'test_hashes.json')}.items():
        assert sha(path)==digest,path
        checked+=1
    for entry in frozen['models'].values():assert sha(entry['path'])==entry['sha256']
    out=dict(unchanged_input_hashes=checked,checkpoint_hashes=len(frozen['models']),banks={},training={})
    methods=['adaptive','supervised','supervised_nll','supervised_final','final_only','final_only_final','gate_one','gate_one_final']
    for bank,bdata in summary['banks'].items():
        rep=int(bank[-1]);groups={m:[n for n,e in frozen['models'].items() if e['cohort']=='matched' and e['method']==m] for m in methods}
        cells={};pool={m:[] for m in methods};rng=np.random.default_rng(2026092209+rep)
        for cell,stored in bdata['cells'].items():
            arr={}
            for m,names in groups.items():
                records=[np.load(BASE/'evaluation'/bank/cell/f'{n}.npz') for n in names]
                arr[m]=np.stack([r['error'].copy() for r in records]);energy=records[0]['energy'].copy()
                for r in records:r.close()
                point=10*np.log10(arr[m].sum(1)/energy.sum())
                np.testing.assert_allclose(point,stored['matched/'+m]['seed_nmse_db'],rtol=1e-12,atol=1e-12)
                if cell.startswith('snr_'):pool[m].append((arr[m],energy))
            counts=rng.multinomial(len(energy),np.ones(len(energy))/len(energy),size=4000)
            boot={m:10*np.log10((x@counts.T)/(energy@counts.T)[None,:]).mean(0) for m,x in arr.items()}
            scores={m:float(np.mean(10*np.log10(x.sum(1)/energy.sum()))) for m,x in arr.items()}
            gain=scores['adaptive']-scores['gate_one']
            cells[cell]=dict(nmse_db=scores,gate_one_gain_db=gain,
                gate_one_seed_gains_db=(10*np.log10(arr['adaptive'].sum(1)/arr['gate_one'].sum(1))).tolist(),
                gate_one_ci95_exploratory=np.quantile(boot['adaptive']-boot['gate_one'],[.025,.975]).tolist(),
                gate_one_severe_frequency=float((arr['gate_one']/energy>1).mean()),
                adaptive_severe_frequency=float((arr['adaptive']/energy>1).mean()))
        pooled={}
        for m,entries in pool.items():
            e=np.concatenate([a for a,v in entries],axis=1);v=np.concatenate([v for a,v in entries])
            pooled[m]=dict(mean_ratio_of_sums_db=float(np.mean(10*np.log10(e.sum(1)/v.sum()))),
                mean_per_world_nmse_db=float(np.mean(10*np.log10((e/v).mean(1)))),
                equal_snr_macro_db=float(np.mean([cells[c]['nmse_db'][m] for c in cells if c.startswith('snr_')])))
        out['banks'][bank]=dict(cells=cells,metrics=pooled)
    for name,e in frozen['models'].items():
        if e['selection']=='final' or e['method'].endswith('_nll'):continue
        model,state=load_model(e['path'])
        path=Path(e['path'])
        latest=path.with_name('latest.pt') if path.name=='best.pt' else path.with_name(path.name.replace('_best','_latest'))
        full=torch.load(latest,map_location='cpu',weights_only=False)
        hist=full['history'];key='val_nmse' if e['method']=='supervised' else 'val_nll'
        out['training'][name]=dict(selected_epoch=e['epoch'],final_epoch=full['epoch'],
            primary_selection=e['selection'],last_five=[h[key] for h in hist[-5:]],
            first=hist[0][key],last=hist[-1][key],min_recorded=min(h[key] for h in hist),
            train_first=hist[0].get('loss',hist[0].get('train_nll')),train_last=hist[-1].get('loss',hist[-1].get('train_nll')),
            saved_nll_best=full.get('best_nll'))
    val=load_observations(OLD/'data/val_observations.npz');context,_=split_observations(val,CONFIG['validation_mask_seed'])
    conditions=np.load(OLD/'data/val_conditions.npy')
    with np.load(OLD/'data/val_truth.npz') as f:truth=torch.tensor(f['g'].copy(),dtype=torch.complex64)
    from sahure_expansion.data import condition_grid
    grid=condition_grid();snrs=np.array([grid[int(c)]['snr'] for c in conditions])
    out['validation_objective_diagnostic']={}
    for name,e in frozen['models'].items():
        if e['cohort']!='matched' or e['method'] not in ['adaptive','supervised']:continue
        model,_=load_model(e['path'],dtype=torch.float32);nmse=[]
        for i in range(0,len(val),32):
            ids=slice(i,i+32);nmse.append(channel_loss(model(*context.batch(ids).args()),truth[ids]).numpy())
        nmse=np.concatenate(nmse)
        strata=np.array([nmse[conditions==c].mean() for c in range(126)])
        low=sum(strata[i] for i,c in enumerate(grid) if c['snr']<=0)/strata.sum()
        out['validation_objective_diagnostic'][name]=dict(equal_stratum_nmse=float(strata.mean()),
            fraction_nmse_objective_from_snr_at_most_zero=float(low),
            by_snr={str(s):float(nmse[snrs==s].mean()) for s in np.unique(snrs)})
        all_pilot=[]
        for i in range(0,len(val),32):
            ids=slice(i,i+32);all_pilot.append(channel_loss(model(*val.batch(ids).args()),truth[ids]).numpy())
        all_pilot=np.concatenate(all_pilot)
        all_strata=np.array([all_pilot[conditions==c].mean() for c in range(126)])
        out['validation_objective_diagnostic'][name]['all_30_pilot_equal_stratum_nmse']=float(all_strata.mean())
        out['validation_objective_diagnostic'][name]['all_30_pilot_by_snr']={str(s):float(all_pilot[snrs==s].mean()) for s in np.unique(snrs)}
    timings=read(BASE/'timing.json')['rows'];out['timing_ms']={}
    for method in ['adaptive','supervised','final_only','gate_one']:
        names={n for n,e in frozen['models'].items() if e['cohort']=='matched' and e['method']==method}
        values=[np.median(r['seconds'])*1000 for r in timings if r['method'] in names]
        out['timing_ms'][method]=float(np.median(values))
    out['timing_ms']['hs_gs']=float(np.median([np.median(r['seconds'])*1000 for r in timings if r['method']=='hs_gs']))
    inputs=[BASE/'summary.json',BASE/'frozen_selection.json',BASE/'timing.json',BASE/'audit.json',OLD/'summary.json']
    out['input_sha256']={str(p):sha(p) for p in inputs}
    (OUT/'review.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(hashes=checked,timing=out['timing_ms'],
        metrics={b:v['metrics'] for b,v in out['banks'].items()},
        stress={b:{c:v['cells'][c] for c in ['weak_reference','two_paths']} for b,v in out['banks'].items()},
        objective=out['validation_objective_diagnostic']),indent=2))

if __name__=='__main__':main()
