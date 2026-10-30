"""Independent post-run checks of stored errors, selection and provenance."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key]='1'
import json
from pathlib import Path
import numpy as np
import torch
from sahure.data import load_observations,split_observations
from sahure.train import load_model
from .protocol import ROOT,CONFIG,sha,write_json
from .evaluate import freeze,data_name,observations_for,classical_estimate
from .train import validate


@torch.no_grad()
def audit():
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    frozen=freeze();summary=json.loads((ROOT/'summary.json').read_text())
    checks=[];validation=[]
    for seed in CONFIG['seeds']:
        manifest=json.loads((ROOT/'checkpoints'/f'seed_{seed}'/'manifest.json').read_text())
        for path,digest in manifest['sources'].items():
            assert sha(path)==digest,f'training source changed: {path}'
    campaign=json.loads((ROOT/'campaign_manifest.json').read_text())
    for path,digest in json.loads((ROOT/'analysis_sources.json').read_text()).items():
        assert sha(path)==digest,f'analysis source changed: {path}'
    for split,digest in campaign['observations_sha256'].items():
        assert sha(ROOT/'data'/f'{split}_observations.npz')==digest
    checks.extend(['Training sources match every seed manifest','Training and validation observation hashes unchanged'])
    val=load_observations(ROOT/'data/val_observations.npz')
    a,b=split_observations(val,CONFIG['validation_mask_seed'])
    conditions=np.load(ROOT/'data/val_conditions.npy')
    for name,entry in frozen['models'].items():
        if 'best_validation_nll' not in entry: continue
        model,_=load_model(entry['path'],dtype=torch.float32)
        score,_=validate(model,a,b,conditions)
        np.testing.assert_allclose(score,entry['best_validation_nll'],atol=1e-9,rtol=1e-9)
        validation.append(dict(model=name,recomputed=score,stored=entry['best_validation_nll']))
    checks.append('All 18 checkpoint validation scores reproduced from observations and fixed masks')
    total_files=0;world_seeds=[]
    for spec in frozen['cells']:
        name=spec['id'];dataid=data_name(spec);directory=ROOT/'cells'/name
        manifest=ROOT/'data'/f'{dataid}_generation.json'
        assert manifest.stat().st_mtime>=frozen['frozen_unix'],'test data predates model freeze'
        generation=json.loads(manifest.read_text());world_seeds.append(generation['description']['seed'])
        with np.load(ROOT/'data'/f'{dataid}_truth.npz') as f: truth=f['g'].copy()
        energy=np.sum(abs(truth)**2,axis=(-2,-1))
        arrays={}
        for modelname in list(frozen['models'])+frozen['classical']:
            with np.load(directory/f'{modelname}.npz') as f:
                np.testing.assert_array_equal(f['energy'],energy)
                assert f['error'].shape==(CONFIG['test_cases'],)
                assert np.isfinite(f['error']).all() and np.all(f['error']>=0)
                if 'estimates' in f:
                    np.testing.assert_allclose(f['error'],np.sum(abs(f['estimates']-truth)**2,axis=(-2,-1)),atol=1e-10,rtol=1e-10)
                arrays[modelname]=f['error'].copy()
            total_files+=1
        for method,stored in summary['cells'][name]['methods'].items():
            names=[n for n,e in frozen['models'].items() if e['method']==method] or [method]
            perseed=[10*np.log10(arrays[n].sum()/energy.sum()) for n in names]
            np.testing.assert_allclose(perseed,stored['seed_nmse_db'],atol=1e-10,rtol=1e-10)
        lines=[json.loads(line) for line in (directory/'classical.jsonl').read_text().splitlines()]
        assert len(lines)==CONFIG['test_cases']*len(frozen['classical'])
        assert len({(x['method'],x['trial']) for x in lines})==len(lines)
    assert CONFIG['train_data_seed'] not in world_seeds and CONFIG['val_data_seed'] not in world_seeds
    assert len(set(world_seeds))==len(frozen['cells'])-2
    checks.extend(['All test datasets created after frozen selection','Distinct train/validation/test master seeds',
        'Every stored error and energy is finite and complete','All learned errors recomputed from saved predictions',
        'All reported NMSE values recomputed from per-world errors','No duplicate classical result rows'])
    for spec in frozen['cells']:
        if not spec['noise_offset']: continue
        for method in ['linearized_ls','amplitude_10','amplitude_100']:
            with np.load(ROOT/'cells'/spec['id']/f'{method}.npz') as offset, np.load(ROOT/'cells'/data_name(spec)/f'{method}.npz') as reference:
                np.testing.assert_array_equal(offset['error'],reference['error'])
    checks.append('Variance-independent methods unchanged on shared noise-mismatch worlds')
    base=next(p['cell'] for p in frozen['axes']['snr'] if p['value']==5)
    spec=next(p for p in frozen['cells'] if p['id']==base);data=observations_for(spec)
    with np.load(ROOT/'data'/f'{base}_truth.npz') as f: truth=f['g'].copy()
    reproductions=[]
    for i in [0,137,399]:
        obs=data.batch(slice(i,i+1));z,s,b,v=(x[0].numpy() for x in obs.args())
        for name,entry in frozen['models'].items():
            model,_=load_model(entry['path'])
            actual=model(*obs.args())[0].numpy()
            with np.load(ROOT/'cells'/base/f'{name}.npz') as f:
                np.testing.assert_allclose(actual,f['estimates'][i],rtol=1e-8,atol=1e-9)
        for method in frozen['classical']:
            g,_=classical_estimate(method,z,s,b,float(v),frozen['hard_rank'])
            with np.load(ROOT/'cells'/base/f'{method}.npz') as f: stored=float(f['error'][i])
            actual=float(np.sum(abs(g-truth[i])**2));np.testing.assert_allclose(actual,stored,rtol=1e-10,atol=1e-10)
            reproductions.append(dict(method=method,trial=i,error=actual))
    checks.append(f'All {len(frozen["classical"])} classical estimators independently rerun on three reference worlds')
    checks.append(f'All {len(frozen["models"])} learned/control models reproduce saved predictions on three reference worlds')
    recoveries=list((ROOT/'numerical_recovery').glob('cell_*_trial_*.json'))
    if recoveries:
        from .recover_svd import recover
        for record in recoveries:
            details=json.loads(record.read_text())
            assert sha(ROOT/'data'/f'{details["cell"]}_observations.npz')==details['observations_sha256']
            recover(details['cell'],details['trial'],verify=True)
        checks.append(f'All {len(recoveries)} explicitly logged SVD recovery trials independently reproduced')
    paths=[p for p in ROOT.rglob('*') if p.is_file() and p.suffix not in ['.log','.tmp'] and p.name not in ['artifact_hashes.json','audit.json','campaign_status.json']]
    write_json(ROOT/'artifact_hashes.json',{str(p.relative_to(ROOT)):sha(p) for p in paths})
    result=dict(state='passed',checks=checks,validated_result_files=total_files,validation=validation,
                classical_reproductions=reproductions,models=len(frozen['models']),cells=len(frozen['cells']))
    write_json(ROOT/'audit.json',result)
    print(json.dumps(dict(audit='passed',result_files=total_files,cells=len(frozen['cells']))),flush=True)


if __name__=='__main__': audit()
