"""Fresh-test evaluator. All channel truth is confined to evaluation here."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key]='1'
import argparse
import json
import time
import numpy as np
import torch
from sahure.data import load_observations
from sahure.train import load_model
from sahure_expansion.evaluate import classical_estimate
from sahure_expansion.protocol import write_json,sha
from .protocol import ROOT,OLD,CONFIG,jobs,cells,check_sources

CLASSICAL=['hs_gs','em_10','em_100']

def save_npz(path,**arrays):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.npz.tmp')
    with tmp.open('wb') as stream:np.savez_compressed(stream,**arrays)
    tmp.replace(path)

def freeze():
    check_sources();path=ROOT/'frozen_selection.json'
    if path.exists():
        frozen=json.loads(path.read_text())
        for entry in frozen['models'].values():
            if sha(entry['path'])!=entry['sha256']:raise RuntimeError('Frozen checkpoint changed')
        return frozen
    models={}
    def add(name,path,cohort,method,seed,selection):
        model,state=load_model(path)
        models[name]=dict(path=str(path),sha256=sha(path),cohort=cohort,method=method,seed=seed,
            selection=selection,epoch=state.get('epoch'),parameters=sum(p.numel() for p in model.parameters()))
    for seed in CONFIG['seeds']:
        for arm in ['adaptive','global','none']:
            add(f'old_{arm}_{seed}',OLD/'checkpoints'/f'seed_{seed}'/f'em_{arm}_best.pt',
                'matched',arm,seed,'heldout_nll')
    for job in jobs():
        out=ROOT/'checkpoints'/job['id']
        latest=torch.load(out/'latest.pt',map_location='cpu',weights_only=False)
        if latest['epoch']!=CONFIG['epochs']:raise RuntimeError(f'Incomplete job {job["id"]}')
        selection='channel_nmse' if job['arm']=='supervised' else 'heldout_nll'
        add(job['id'],out/'best.pt',job['cohort'],job['arm'],job['seed'],selection)
        add(job['id']+'_final',out/'latest.pt',job['cohort'],job['arm']+'_final',job['seed'],'final')
        if job['arm']=='supervised':
            add(job['id']+'_nll',out/'nll_best.pt',job['cohort'],'supervised_nll',job['seed'],'heldout_nll')
    frozen=dict(models=models,cells=cells(),classical=CLASSICAL,protocol_sha256=sha(ROOT/'protocol.json'),
        frozen_unix=time.time())
    write_json(path,frozen);return frozen

def selected_models(frozen,rep):
    return {n:e for n,e in frozen['models'].items() if e['cohort'] in ('matched',f'rep{rep}')}

def prepare():
    frozen=freeze()
    import sahure_expansion.data as generator
    previous=generator.ROOT
    try:
        for rep in CONFIG['replicates']:
            generator.ROOT=ROOT/f'test{rep}'
            for i,cell in enumerate(frozen['cells']):
                generator.generate_set(cell['id'],CONFIG['test_cases'],2026110000+rep*100+i,spec=cell)
                print(f'Generated fresh test bank {rep}: {cell["id"]}',flush=True)
    finally:generator.ROOT=previous
    records={str(p):sha(p) for rep in CONFIG['replicates'] for p in (ROOT/f'test{rep}'/'data').iterdir() if p.is_file()}
    write_json(ROOT/'test_hashes.json',records)

def observations(rep,cell):
    base=ROOT/f'test{rep}'/'data'
    return load_observations(base/f'{cell}_observations.npz',torch.float64)

@torch.no_grad()
def run_cell(rep,cell):
    torch.set_num_threads(1);frozen=freeze();data=observations(rep,cell)
    with np.load(ROOT/f'test{rep}'/'data'/f'{cell}_truth.npz') as f:truth=f['g'].copy()
    energy=np.sum(abs(truth)**2,axis=(-2,-1));out=ROOT/'evaluation'/f'bank{rep}'/cell
    out.mkdir(parents=True,exist_ok=True)
    for name,entry in selected_models(frozen,rep).items():
        if (out/f'{name}.npz').exists():continue
        model,_=load_model(entry['path']);est=[]
        for start in range(0,len(data),32):est.append(model(*data.batch(slice(start,start+32)).args()).numpy())
        est=np.concatenate(est)
        if not np.isfinite(est).all():raise FloatingPointError(name)
        save_npz(out/f'{name}.npz',estimates=est,error=np.sum(abs(est-truth)**2,axis=(-2,-1)),energy=energy)
    done={};rows=out/'classical.jsonl'
    if rows.exists():
        for row in map(json.loads,rows.read_text().splitlines()):
            key=(row['method'],row['trial'])
            if key in done:raise RuntimeError('duplicate classical world')
            done[key]=row
    with rows.open('a') as stream:
        for i in range(len(data)):
            z,s,b,v=(x[i].numpy() for x in data.args())
            for name in CLASSICAL:
                if (name,i) in done:continue
                g,_=classical_estimate(name,z,s,b,float(v))
                error=float(np.sum(abs(g-truth[i])**2))
                if not np.isfinite(error):raise FloatingPointError(name)
                row=dict(method=name,trial=i,error=error,energy=float(energy[i]))
                stream.write(json.dumps(row)+'\n');stream.flush();done[(name,i)]=row
            if (i+1)%25==0:print(f'bank{rep} {cell}: {i+1}/{len(data)}',flush=True)
    for name in CLASSICAL:
        save_npz(out/f'{name}.npz',error=np.array([done[(name,i)]['error'] for i in range(len(data))]),energy=energy)
    write_json(out/'status.json',dict(state='complete',models=len(selected_models(frozen,rep)),cases=len(data)))

def summarize_arrays(errors,energy,counts):
    # Seed predictions stay paired; average dB scores, never predicted channels.
    errors=np.atleast_2d(errors)
    point=10*np.log10(errors.sum(1)/energy.sum())
    boot=10*np.log10((errors@counts.T)/(energy@counts.T)[None,:])
    return point,boot.mean(0)

def report():
    frozen=freeze();results={};rng=np.random.default_rng(2026113301)
    for rep in CONFIG['replicates']:
        groups={}
        for name,e in selected_models(frozen,rep).items():
            key=f'{e["cohort"]}/{e["method"]}'
            groups.setdefault(key,[]).append(name)
        for name in CLASSICAL:groups[name]=[name]
        bank={};boot_cells={}
        for cell in frozen['cells']:
            out=ROOT/'evaluation'/f'bank{rep}'/cell['id'];scores={};boots={}
            counts=rng.multinomial(CONFIG['test_cases'],np.full(CONFIG['test_cases'],1/CONFIG['test_cases']),
                                   size=CONFIG['bootstrap_resamples'])
            for group,names in groups.items():
                records=[np.load(out/f'{name}.npz') for name in names]
                errors=np.stack([r['error'] for r in records]);energy=records[0]['energy'].copy()
                point,boot=summarize_arrays(errors,energy,counts)
                scores[group]=dict(nmse_db=float(point.mean()),seed_nmse_db=point.tolist(),
                    severe_error_frequency=float((errors/energy>1).mean()),
                    ci95_world_conditional=np.quantile(boot,[.025,.975]).tolist())
                boots[group]=boot
                for r in records:r.close()
            bank[cell['id']]=scores
            if cell['primary']:boot_cells[cell['id']]=boots
        contrasts=[('matched/adaptive','matched/supervised'),('matched/adaptive','matched/final_only'),
            ('matched/gate_one','matched/adaptive'),(f'rep{rep}/adaptive',f'rep{rep}/global'),
            (f'rep{rep}/adaptive',f'rep{rep}/none'),(f'rep{rep}/adaptive','hs_gs')]
        comparison=[]
        for method,reference in contrasts:
            gains=[bank[c][reference]['nmse_db']-bank[c][method]['nmse_db'] for c in boot_cells]
            boot=np.mean([b[reference]-b[method] for b in boot_cells.values()],axis=0)
            comparison.append(dict(method=method,reference=reference,gain_db=float(np.mean(gains)),
                ci95_bonferroni_six_world_conditional=np.quantile(boot,[.05/12,1-.05/12]).tolist(),
                worst_cell_gain_db=float(min(gains))))
        results[f'bank{rep}']=dict(cells=bank,contrasts=comparison)
    write_json(ROOT/'summary.json',dict(banks=results,limitations=[
        'Intervals condition on trained models; seeds are not independent worlds.',
        'Two independent dataset replicates are reported separately, not a reliable population CI.',
        'Six primary contrasts use Bonferroni family-95% intervals within each bank; secondary cells are descriptive.',
        'Supervised primary uses labelled validation; supervised_nll uses common NLL selection.',
        'This does not establish novelty, convergence, hardware performance or universal robustness.']))
    lines=['# Submission checks: results','',
        'Fresh paired tests. Positive gain means lower error for the named method. Read both dataset banks.',
        '', '| Bank | Method | Reference | Mean gain (dB) | Conditional adjusted interval | Worst SNR gain |',
        '|---|---|---|---:|---|---:|']
    for bank,res in results.items():
        for c in res['contrasts']:
            lines.append(f'| {bank} | {c["method"]} | {c["reference"]} | {c["gain_db"]:.3f} | {c["ci95_bonferroni_six_world_conditional"]} | {c["worst_cell_gain_db"]:.3f} |')
    lines+=['','All per-cell, severe-error, selected/final and supervised-selection sensitivity results are in summary.json.',
        'Weak-reference and two-path stress cases are reported separately from the primary SNR average.',
        'These intervals capture test-world variation conditional on trained weights, not uncertainty across dataset populations.',
        'Do not claim equivalence from a nonsignificant contrast. Review audit.json and timing.json before manuscript use.']
    (ROOT/'REPORT.md').write_text('\n'.join(lines)+'\n')

@torch.no_grad()
def timing():
    torch.set_num_threads(1);frozen=freeze();rows=[]
    for rep in CONFIG['replicates']:
        data=observations(rep,'snr_5')
        entries={n:e for n,e in selected_models(frozen,rep).items() if e['selection']!='final'}
        models={n:load_model(e['path'])[0] for n,e in entries.items()}
        def predict(name,batch):
            if name in models:return models[name](*batch.args())
            z,s,b,v=(x[0].numpy() for x in batch.args())
            return classical_estimate(name,z,s,b,float(v))
        names=list(models)+CLASSICAL
        for name in names:predict(name,data.batch(slice(0,1)))
        for i in range(20):
            batch=data.batch(slice(i,i+1))
            for name in np.random.default_rng(555+rep*100+i).permutation(names):
                times=[]
                for _ in range(3):
                    tick=time.perf_counter();predict(name,batch);times.append(time.perf_counter()-tick)
                rows.append(dict(bank=rep,world=i,method=str(name),seconds=times))
        print(f'Timing bank {rep} complete',flush=True)
    write_json(ROOT/'timing.json',dict(scope='20 SNR5 worlds per bank, serial CPU batch-one, three repeats, initialization included',rows=rows))

@torch.no_grad()
def audit():
    torch.set_num_threads(1);protocol=check_sources();frozen=freeze();nfiles=0
    for path,digest in protocol['previous_sha256'].items():
        if sha(path)!=digest:raise RuntimeError(f'Previous experiment changed: {path}')
    for path,digest in json.loads((ROOT/'test_hashes.json').read_text()).items():
        if sha(path)!=digest:raise RuntimeError(f'Test data changed: {path}')
    for path,digest in json.loads((ROOT/'replicate_data_hashes.json').read_text()).items():
        if sha(path)!=digest:raise RuntimeError(f'Replicate data changed: {path}')
    for rep in CONFIG['replicates']:
        for cell in frozen['cells']:
            base=ROOT/f'test{rep}'/'data'
            if (base/f'{cell["id"]}_generation.json').stat().st_mtime < frozen['frozen_unix']:
                raise RuntimeError('Test generation predates selection freeze')
            with np.load(base/f'{cell["id"]}_truth.npz') as f:truth=f['g'].copy()
            energy=np.sum(abs(truth)**2,axis=(-2,-1))
            out=ROOT/'evaluation'/f'bank{rep}'/cell['id']
            for name in list(selected_models(frozen,rep))+CLASSICAL:
                with np.load(out/f'{name}.npz') as f:
                    np.testing.assert_array_equal(f['energy'],energy)
                    assert f['error'].shape==(CONFIG['test_cases'],)
                    assert np.isfinite(f['error']).all() and (f['error']>=0).all()
                    if 'estimates' in f:
                        np.testing.assert_allclose(f['error'],np.sum(abs(f['estimates']-truth)**2,axis=(-2,-1)),rtol=1e-10,atol=1e-10)
                nfiles+=1
        # Representative forward replay covers every selected/final model.
        data=observations(rep,'snr_5');out=ROOT/'evaluation'/f'bank{rep}'/'snr_5'
        for name,entry in selected_models(frozen,rep).items():
            model,_=load_model(entry['path'])
            with np.load(out/f'{name}.npz') as f:stored=f['estimates'][[0,99,199]]
            actual=model(*data.batch(torch.tensor([0,99,199])).args()).numpy()
            np.testing.assert_allclose(actual,stored,rtol=1e-8,atol=1e-9)
        with np.load(ROOT/f'test{rep}'/'data'/'snr_5_truth.npz') as f:truth=f['g'].copy()
        for i in [0,99,199]:
            z,s,b,v=(x[i].numpy() for x in data.args())
            for name in CLASSICAL:
                g,_=classical_estimate(name,z,s,b,float(v))
                with np.load(out/f'{name}.npz') as f:stored=float(f['error'][i])
                np.testing.assert_allclose(np.sum(abs(g-truth[i])**2),stored,rtol=1e-10,atol=1e-10)
    # Selection is independently verified from every saved validation history.
    for job in jobs():
        out=ROOT/'checkpoints'/job['id'];latest=torch.load(out/'latest.pt',weights_only=False,map_location='cpu')
        best=torch.load(out/'best.pt',weights_only=False,map_location='cpu')
        key='val_nmse' if job['arm']=='supervised' else 'val_nll'
        assert len(latest['history'])==20
        assert best['best_score']==min([latest['initial_score']]+[h[key] for h in latest['history']])
        from .train import data_root,supervised_validation,load_labels
        from sahure.data import split_observations
        from sahure_expansion.protocol import CONFIG as PREVIOUS
        from sahure_expansion.train import validate
        directory=data_root(job['cohort'])/'data'
        val=load_observations(directory/'val_observations.npz')
        a,b=split_observations(val,PREVIOUS['validation_mask_seed']);conditions=np.load(directory/'val_conditions.npy')
        model,_=load_model(out/'best.pt',dtype=torch.float32)
        if job['arm']=='supervised':
            actual=supervised_validation(model,a,load_labels(job['cohort'],job['arm'],'val'),conditions)
        else:actual=validate(model,a,b,conditions)[0]
        np.testing.assert_allclose(actual,best['best_score'],atol=1e-9,rtol=1e-9)
        nllmodel,nllstate=load_model(out/'nll_best.pt',dtype=torch.float32)
        np.testing.assert_allclose(validate(nllmodel,a,b,conditions)[0],latest['best_nll'],atol=1e-9,rtol=1e-9)
        for row in latest['history']:
            from .train import order_hash
            assert row['order_sha256']==order_hash(CONFIG['train_cases'],job['seed'],row['epoch']-1)
    before=sha(ROOT/'summary.json');report();assert before==sha(ROOT/'summary.json')
    write_json(ROOT/'audit.json',dict(state='passed',result_files=nfiles,training_jobs=15,epochs=300,
        checks=['old inputs unchanged','source and checkpoint hashes','test data frozen after selection',
                'saved learned errors recalculated','all models replayed on three worlds per bank',
                'all classical methods replayed on three worlds per bank','matched orders checked',
                'selected channel/NLL validation scores replayed from inputs',
                'summary regenerated identically']))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['freeze','prepare','cell','report','timing','audit'])
    p.add_argument('--rep',type=int);p.add_argument('--cell');a=p.parse_args()
    if a.phase=='cell':run_cell(a.rep,a.cell)
    else:globals()[a.phase]()
