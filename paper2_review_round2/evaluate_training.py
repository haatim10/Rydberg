"""Fresh paired tests after every training job is complete; never selects models."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key]='1'
import argparse
import fcntl
import json
from pathlib import Path
import time
import numpy as np
import torch
from sahure.data import load_observations
from sahure_expansion.data import world
from sahure_expansion.protocol import write_json
from codex_verification.solver import fit
from paper2_optshrink_loop.estimator import estimate as opt_estimate
from .common import ROOT,save,load,sha,verify_sources,aggregate
from .models import load_model,ARMS
from .training_study import CONFIG,GRID,JOBS

# Amendment before any test worlds were generated: the draft seed range in
# the training protocol overlapped separately run diagnostic phases. Neither
# the training data nor model-selection rule changes. See TEST_SEED_AMENDMENT.
TEST_SEED=20261008000001


def assert_split_separation():
    train_val={s+d for s in CONFIG['data_seeds'] for d in (0,10)}
    diagnostic={202610084201+100*j for j in range(6)}
    diagnostic|={202610087101+100*j for j in range(18)}
    diagnostic|={202610087201+100*j for j in range(3)}
    test={TEST_SEED+100*j for j in range(len(GRID))}
    assert len(test)==len(GRID)
    assert test.isdisjoint(train_val|diagnostic)


@torch.no_grad()
def run(wait=False):
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    assert_split_separation()
    directory=ROOT/'training';start=time.monotonic()
    while not (directory/'complete.json').exists():
        if not wait:raise RuntimeError('all training must finish before final-test generation')
        if (directory/'status.json').exists():
            status=json.loads((directory/'status.json').read_text())
            if status['state'] in ('failed','partial'):raise RuntimeError('training stopped: '+json.dumps(status))
        if time.monotonic()-start>7200:raise TimeoutError('training wait cap')
        time.sleep(15)
    protocol=json.loads((directory/'protocol.json').read_text());verify_sources(protocol)
    completion=json.loads((directory/'complete.json').read_text())
    for p,h in completion['hashes'].items():
        if sha(p)!=h:raise RuntimeError('trained checkpoint changed')
    out=ROOT/'training_test';out.mkdir(exist_ok=True)
    with (out/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        frozen=dict(training_protocol_sha256=sha(directory/'protocol.json'),checkpoints=completion['hashes'],
                    evaluation_source_sha256=sha(__file__),test_seed=TEST_SEED,
                    superseded_pre_generation_test_seed=CONFIG['test_seed'],
                    seed_amendment_sha256=sha('paper2_review_round2/TEST_SEED_AMENDMENT.txt'),cases=CONFIG['test_cases_per_cell'],
                    note='all test cases generated after selection; comparisons descriptive with conditional intervals')
        if (out/'protocol.json').exists():
            if json.loads((out/'protocol.json').read_text())!=frozen:raise RuntimeError('evaluation freeze mismatch')
        else:write_json(out/'protocol.json',frozen)
        if (out/'summary.json').exists():print('Fresh test already complete');return
        try:
            selection=json.loads(Path('results/paper2_optshrink_loop/frozen_selection.json').read_text())
            models={f'{job["name"]}/{which}':load_model(directory/job['name']/f'{which}.pt')[0]
                    for job in JOBS for which in ('selected','latest')}
            for cid,cell in enumerate(GRID):
                file=out/'data'/f'cell_{cid:02d}_observations.npz';truth=out/'data'/f'cell_{cid:02d}_truth.npz'
                if not file.exists():
                    values={k:[] for k in ('z','s','b','sigma2','g')}
                    for i in range(CONFIG['test_cases_per_cell']):
                        w,_=world(i,TEST_SEED+100*cid,**cell)
                        for k,v in zip(values,(w.Z,w.S,w.B,w.sigma2,w.G)):values[k].append(v)
                    save(truth,g=np.asarray(values.pop('g')),trial=np.arange(CONFIG['test_cases_per_cell']),seed=TEST_SEED+100*cid)
                    save(file,**{k:np.asarray(v) for k,v in values.items()})
                data=load_observations(file,dtype=torch.float64);g=load(truth)['g'];energy=np.sum(abs(g)**2,axis=(-2,-1))
                for name,model in models.items():
                    dest=out/'predictions'/name/f'cell_{cid:02d}.npz'
                    if dest.exists():continue
                    tick=time.perf_counter();estimates=[]
                    for i in range(0,len(data),32):estimates.append(model(*data.batch(slice(i,i+32)).args()).numpy())
                    estimates=np.concatenate(estimates)
                    if not np.isfinite(estimates).all():raise FloatingPointError(name)
                    save(dest,estimates=estimates,errors=np.sum(abs(estimates-g)**2,axis=(-2,-1)),energy=energy,
                        seconds=time.perf_counter()-tick,case=np.arange(len(data)),cell=json.dumps(cell))
                for method in ('opt_loop_frozen','em100'):
                    dest=out/'predictions'/method/f'cell_{cid:02d}.npz'
                    if dest.exists():continue
                    raw=load(file);tick=time.perf_counter();estimates=[]
                    for i in range(len(data)):
                        z,s,b,v=(raw[k][i] for k in ('z','s','b','sigma2'))
                        estimates.append(opt_estimate(z,s,b,v,selection['noise_aware'][str(cell['snr'])]['loop'])
                            if method=='opt_loop_frozen' else fit(s,z,b,v,iterations=100)[0])
                    estimates=np.stack(estimates)
                    save(dest,estimates=estimates,errors=np.sum(abs(estimates-g)**2,axis=(-2,-1)),energy=energy,
                        seconds=time.perf_counter()-tick,case=np.arange(len(data)),cell=json.dumps(cell))
                write_json(out/'status.json',dict(state='running',completed_cells=cid+1,total_cells=len(GRID),pid=os.getpid(),seconds=time.monotonic()-start))
                print(f'Test cell {cid+1}/{len(GRID)} saved',flush=True)
            summarize(out,directory)
            write_json(out/'status.json',dict(state='complete',cells=len(GRID),cases=len(GRID)*CONFIG['test_cases_per_cell']))
        except BaseException as e:
            write_json(out/'status.json',dict(state='failed',error=repr(e)));raise


def summarize(out,training):
    rng=np.random.default_rng(202610086999);cells={};all_hashes={};primary={a:[] for a in ARMS};boots={a:[] for a in ARMS}
    rep_scores={str(r):[] for r in range(3)}
    for cid,cell in enumerate(GRID):
        truth=load(out/'data'/f'cell_{cid:02d}_truth.npz');g=truth['g'];energy=np.sum(abs(g)**2,axis=(-2,-1))
        errors={};methods={}
        for p in sorted((out/'predictions').rglob(f'cell_{cid:02d}.npz')):
            row=load(p);name=str(p.parent.relative_to(out/'predictions'))
            np.testing.assert_allclose(row['errors'],np.sum(abs(row['estimates']-g)**2,axis=(-2,-1)),rtol=1e-11,atol=1e-11)
            errors[name]=row['errors'];methods[name]=aggregate(row['errors'],energy);all_hashes[str(p)]=sha(p)
        counts=rng.multinomial(len(g),np.ones(len(g))/len(g),size=2000)
        for arm in ARMS:
            ee=np.stack([errors[f'{arm}_d0_s{s}/selected'] for s in CONFIG['initializations']])
            val=aggregate(ee,energy);methods[arm+'_3seed']=val;primary[arm].append(val['nmse_db'])
            boots[arm].append(np.mean(10*np.log10((ee@counts.T)/(counts@energy)[None]),axis=0))
        for rep in range(3):
            ee=np.stack([errors[f'baseline_d{rep}_s{s}/selected'] for s in CONFIG['initializations'][:2]])
            rep_scores[str(rep)].append(aggregate(ee,energy)['nmse_db'])
        cells[str(cid)]=dict(condition=cell,methods=methods)
    comparisons={}
    for arm in ARMS:
        if arm=='baseline':continue
        delta=np.asarray(primary['baseline'])-np.asarray(primary[arm]);boot=np.mean(np.stack(boots['baseline'])-np.stack(boots[arm]),axis=0)
        comparisons[arm]=dict(macro_gain_db=float(delta.mean()),conditional_paired_interval=np.quantile(boot,[.025,.975]).tolist(),
            few_path_gain_db=float(np.mean([d for d,c in zip(delta,GRID) if c['L']==2])),
            low_reference_gain_db=float(np.mean([d for d,c in zip(delta,GRID) if c['rsr']==0])),
            worst_cell_gain_db=float(delta.min()))
    # Matched order checks exclude curriculum's intentional scheduling change.
    for seed in CONFIG['initializations']:
        histories={a:json.loads((training/f'{a}_d0_s{seed}'/'history.json').read_text()) for a in ARMS}
        for arm,hist in histories.items():
            assert len(hist)==CONFIG['epochs']
            for a,b in zip(hist,histories['baseline']):
                assert a['mask_seed']==b['mask_seed']
                if arm!='curriculum':assert a['order_hash']==b['order_hash']
    write_json(out/'summary.json',dict(cells=cells,comparisons=comparisons,
        macro_nmse_db={a:float(np.mean(v)) for a,v in primary.items()},
        independent_dataset_baseline_macros={r:float(np.mean(v)) for r,v in rep_scores.items()},
        caveats=['Reduced 4096-case/12-epoch study, not a convergence or optimal-depth demonstration',
                 'Three initialization seeds share dataset zero; extra datasets each use two initializations',
                 'Intervals condition on trained weights and are unadjusted exploratory intervals',
                 'Test cases shared across datasets; their predictions are not independent channel realizations',
                 'Curriculum has matched total per-case exposure but intentionally different order and masks assigned to cases',
                 'Mixture changes architecture and initialization; not an isolated claim about gating alone',
                 'Classical OptShrink settings transfer from the prior development mixture, not retuned on these tests']))
    write_json(out/'artifact_hashes.json',all_hashes)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--wait-for-training',action='store_true');run(p.parse_args().wait_for_training)
