"""Read-only experiment audit; writes only this review and generated table.

Run from the repository root. No experiment driver is launched or changed.
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key] = '1'
from pathlib import Path
import hashlib
import json
import sys
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
os.chdir(REPO)
OUT = Path(__file__).resolve().parent
torch.set_num_threads(1)

def read(p): return json.loads(Path(p).read_text())
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def arrays(p):
    with np.load(p) as f: return {k:f[k].copy() for k in f.files}
def db(e,v): return float(10*np.log10(e.sum()/v.sum()))

main = Path('results/sahure_expansion')
frozen = read(main/'frozen_selection.json')
summary = read(main/'summary.json')
protected = {}
for study in ('sahure_checks','sahure_features'):
    p = read(Path('results')/study/'protocol.json')
    for key in ('sources','previous_sha256','inputs'):
        protected.update(p.get(key, {}))
for name, expected in protected.items():
    assert sha(name) == expected, ('changed frozen input',name)
checkpoint_count = 0
for study in ('sahure_expansion','sahure_checks','sahure_features'):
    for entry in read(Path('results')/study/'frozen_selection.json')['models'].values():
        assert sha(entry['path']) == entry['sha256'], entry['path']
        checkpoint_count += 1

# Check every saved estimate/error pair and every published cell mean.
from sahure_expansion.evaluate import data_name
methods = list(summary['aggregate_snr'])
recomputed = {}
files = 0
for spec in frozen['cells']:
    cell = spec['id']; truth = arrays(main/'data'/f'{data_name(spec)}_truth.npz')['g']
    energy = np.sum(abs(truth)**2,axis=(-2,-1)); recomputed[cell] = {}
    for method in methods:
        names = [n for n,x in frozen['models'].items() if x['method']==method] or [method]
        scores=[]; severe=[]
        for name in names:
            a=arrays(main/'cells'/cell/f'{name}.npz');files+=1
            np.testing.assert_array_equal(a['energy'],energy)
            assert np.isfinite(a['error']).all() and (a['error']>=0).all()
            if 'estimates' in a:
                np.testing.assert_allclose(np.sum(abs(a['estimates']-truth)**2,axis=(-2,-1)),a['error'],rtol=1e-10,atol=1e-10)
            scores.append(db(a['error'],energy));severe.append(float(np.mean(a['error']/energy>1)))
        point=float(np.mean(scores))
        np.testing.assert_allclose(point,summary['cells'][cell]['methods'][method]['nmse_db'],rtol=0,atol=1e-10)
        recomputed[cell][method]={'nmse_db':point,'severe_percent':100*float(np.mean(severe))}

followup_files={}
for study in ('sahure_checks','sahure_features'):
    root=Path('results')/study; fs=read(root/'frozen_selection.json'); saved=read(root/'summary.json')
    count=0
    for bank in (1,2):
        for directory in sorted((root/'evaluation'/f'bank{bank}').iterdir()):
            if not directory.is_dir(): continue
            cell=directory.name
            truthpath=(root/f'test{bank}'/'data'/f'{cell}_truth.npz') if study=='sahure_checks' else root/'data'/f'bank{bank}'/f'{cell}_truth.npz'
            truth=arrays(truthpath)['g']; energy=np.sum(abs(truth)**2,axis=(-2,-1)); groups={}
            for p in sorted(directory.glob('*.npz')):
                a=arrays(p);count+=1
                np.testing.assert_allclose(a['energy'],energy,rtol=1e-12,atol=1e-12)
                pred=a.get('estimates',a.get('pred'))
                if pred is not None:
                    np.testing.assert_allclose(np.sum(abs(pred-truth)**2,axis=(-2,-1)),a['error'],rtol=1e-10,atol=1e-10)
                assert np.isfinite(a['error']).all() and (a['error']>=0).all()
                if study=='sahure_checks':
                    entry=fs['models'].get(p.stem)
                    key=f"{entry['cohort']}/{entry['method']}" if entry else p.stem
                else:
                    entry=fs['models'][p.stem];key=(entry['selection'],entry['arm'])
                    np.testing.assert_array_equal(a['trial_ids'],np.arange(len(truth)))
                    assert a['checkpoint_sha256'].item()==entry['sha256']
                groups.setdefault(key,[]).append(db(a['error'],energy))
            for key,values in groups.items():
                cell_saved=saved['banks'][f'bank{bank}']['cells'][cell]
                target=cell_saved[key]['nmse_db'] if study=='sahure_checks' else cell_saved[key[0]][key[1]]['nmse_db']
                np.testing.assert_allclose(np.mean(values),target,rtol=0,atol=1e-10)
    followup_files[study]=count
assert files==1120 and followup_files=={'sahure_checks':702,'sahure_features':432}

# Three primary checkpoint replays on a shared reference subset.
from sahure.data import load_observations
from sahure.train import load_model
data=load_observations(main/'data/cell_03_observations.npz',torch.float64).batch(slice(0,8))
replays=[]
with torch.no_grad():
    for seed in (9101,9102,9103):
        name=f'em_adaptive_s{seed}'; model,_=load_model(frozen['models'][name]['path'])
        pred=model(*data.args()).numpy();saved=arrays(main/'cells/cell_03'/f'{name}.npz')['estimates'][:8]
        np.testing.assert_allclose(pred,saved,rtol=1e-8,atol=1e-9)
        replays.append({'seed':seed,'cases':8,'max_absolute_difference':float(np.max(abs(pred-saved)))})

# Independent latency aggregation and descriptive all-comparator table.
rows=[json.loads(line) for line in (main/'timings.jsonl').read_text().splitlines()]
snr_ids=[p['cell'] for p in frozen['axes']['snr']]
all_results={}
for method in methods:
    names=[n for n,x in frozen['models'].items() if x['method']==method] or [method]
    times=[]
    for name in names:
        records=sorted([r for r in rows if r['method']==name],key=lambda x:x['trial'])
        assert [r['trial'] for r in records]==list(range(100))
        times.extend(float(np.median(r['seconds'])) for r in records)
    latency=float(np.median(times))
    np.testing.assert_allclose(latency,summary['timing'][method]['median_seconds'],rtol=0,atol=1e-12)
    mean=float(np.mean([recomputed[c][method]['nmse_db'] for c in snr_ids]))
    hs=float(np.mean([recomputed[c]['hs_gs']['nmse_db'] for c in snr_ids]))
    np.testing.assert_allclose(hs-mean,summary['aggregate_snr'][method]['mean_per_snr_gain_db'],rtol=0,atol=1e-10)
    all_results[method]={'macro_nmse_db':mean,'median_ms':latency*1000,'severe_percent':float(np.mean([recomputed[c][method]['severe_percent'] for c in snr_ids]))}
labels={'em_adaptive':'EM adaptive (primary)','em_none':'EM learned steps only','em_global':'EM global filtering','em_constant_control':'EM constant-output control',
        'gd_adaptive':'GD adaptive (secondary)','gd_none':'GD learned steps only','gd_global':'GD global filtering','gd_constant_control':'GD constant-output control',
        'hs_gs':'HS-GS, rank search','em_10':'EM, 10 updates','em_100':'EM, 100 updates','rician_10':'Rician GD, 10 updates',
        'rician_100':'Rician GD, 100 updates','rician_converged':'Rician GD, stopping rule','amplitude_10':'Amplitude GD, 10 updates',
        'amplitude_100':'Amplitude GD, 100 updates','amplitude_converged':'Amplitude GD, stopping rule','rician_hard':'Rician GD, hard Hankel','linearized_ls':'Linearized least squares'}
table=[]
for method,label in labels.items():
    x=all_results[method]
    table.append(f"{label} & {x['macro_nmse_db']:.2f} & {x['median_ms']:.2f} & {x['severe_percent']:.2f} \\")
# Two backslashes are required to terminate a TeX table row.
table=[line+'\\' for line in table]
(OUT.parent/'ieee/baseline_rows.tex').write_text('\n'.join(table)+'\n')
result={'state':'passed','protected_hashes':len(protected),'checkpoint_references':checkpoint_count,
        'main_result_files':files,'followup_result_files':followup_files,'checkpoint_replays':replays,
        'all_comparators':all_results,'limitations':['This does not certify novelty or simulate new datasets.','Existing bootstrap intervals were not regenerated by this audit.','Timings were recalculated from saved measurements, not rebenchmarked.']}
(OUT/'evidence_audit.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='all_comparators'},indent=2))
