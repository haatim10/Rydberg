"""Independent saved-output audit; never trains, selects, or changes evidence."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key]='1'
import json
from pathlib import Path
import numpy as np
import torch
from .common import sha, verify_sources
from .training_study import CONFIG, GRID, JOBS
from .evaluate_training import TEST_SEED, assert_split_separation
from .models import ARMS, load_model
from sahure_expansion.data import world

ROOT=Path('results/paper2_review_round2')

def read(p): return json.loads(Path(p).read_text())
def arrays(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}

def run():
    torch.set_num_threads(1)
    assert_split_separation()
    for phase in ('calibration','geometry','low_noise_replay','training'):
        verify_sources(read(ROOT/phase/'protocol.json'))
    training=ROOT/'training'; test=ROOT/'training_test'
    complete=read(training/'complete.json'); frozen=read(test/'protocol.json')
    assert complete['jobs']==25 and complete['epochs']==300
    assert frozen['training_protocol_sha256']==sha(training/'protocol.json')
    assert frozen['evaluation_source_sha256']==sha('paper2_review_round2/evaluate_training.py')
    assert frozen['seed_amendment_sha256']==sha('paper2_review_round2/TEST_SEED_AMENDMENT.txt')
    assert complete['hashes']==frozen['checkpoints']
    for name,digest in complete['hashes'].items():assert sha(name)==digest
    for manifest in (training/'data').glob('*manifest.json'):
        for name,digest in read(manifest)['hashes'].items():assert sha(name)==digest
    history={}; epochs={}
    for job in JOBS:
        name=job['name']; p=training/name
        h=read(p/'history.json'); history[name]=h
        assert len(h)==12 and [v['epoch'] for v in h]==list(range(1,13))
        assert all(np.isfinite(v['val_nll']) and np.isfinite(v['train_nll']) for v in h)
        final=torch.load(p/'latest.pt',weights_only=False,map_location='cpu')
        selected=torch.load(p/'selected.pt',weights_only=False,map_location='cpu')
        assert final['history']==h and final['epoch']==12
        e=selected['epoch']; assert 0<=e<=12
        assert selected['best_score']<=min(v['val_nll'] for v in h)+1e-12
        if e:assert selected['best_score']==h[e-1]['val_nll']
        assert final['protocol_sha256']==sha(training/'protocol.json')
        assert selected['protocol_sha256']==final['protocol_sha256']
        epochs[name]=e
    for seed in CONFIG['initializations']:
        for arm in ARMS:
            for h,b in zip(history[f'{arm}_d0_s{seed}'],history[f'baseline_d0_s{seed}']):
                assert h['mask_seed']==b['mask_seed']
                if arm!='curriculum':assert h['order_hash']==b['order_hash']
    saved=read(test/'summary.json'); saved_hashes=read(test/'artifact_hashes.json')
    scores={}; severe={}; case_nmse={}; errors_by_cell=[]; energies=[]; hashes={}
    for cid,cell in enumerate(GRID):
        truth_path=test/'data'/f'cell_{cid:02d}_truth.npz'
        obs_path=test/'data'/f'cell_{cid:02d}_observations.npz'
        truth=arrays(truth_path); obs=arrays(obs_path); g=truth['g']
        assert int(truth['seed'])==TEST_SEED+100*cid
        np.testing.assert_array_equal(truth['trial'],np.arange(64))
        en=(np.abs(g)**2).sum(axis=(1,2)); energies.append(en)
        assert len(g)==64 and np.isfinite(g).all()
        w,_=world(0,TEST_SEED+100*cid,**cell)
        for k,value in [('z',w.Z),('s',w.S),('b',w.B),('sigma2',w.sigma2)]:
            np.testing.assert_allclose(obs[k][0],value,rtol=1e-12,atol=1e-12)
        np.testing.assert_allclose(g[0],w.G,rtol=1e-12,atol=1e-12)
        hashes[str(truth_path)]=sha(truth_path);hashes[str(obs_path)]=sha(obs_path)
        paths=sorted((test/'predictions').rglob(f'cell_{cid:02d}.npz'));assert len(paths)==52
        ee={}
        for p in paths:
            name=str(p.parent.relative_to(test/'predictions'));row=arrays(p)
            assert np.isfinite(row['estimates']).all()
            assert json.loads(str(row['cell']))==cell
            np.testing.assert_array_equal(row['case'],truth['trial'])
            np.testing.assert_allclose(row['energy'],en,rtol=1e-12,atol=1e-12)
            err=(np.abs(row['estimates']-g)**2).sum(axis=(1,2));ee[name]=err
            np.testing.assert_allclose(err,row['errors'],rtol=1e-12,atol=1e-12)
            score=float(10*np.log10(err.sum()/en.sum()))
            rate=float(np.mean(err/en>1));linear=float(np.mean(err/en))
            sm=saved['cells'][str(cid)]['methods'][name]
            np.testing.assert_allclose([score,rate,linear],[sm['nmse_db'],sm['severe_fraction'],sm['mean_case_nmse']],rtol=1e-12,atol=1e-12)
            scores.setdefault(name,[]).append(score);severe.setdefault(name,[]).append(rate);case_nmse.setdefault(name,[]).append(linear)
            assert sha(p)==saved_hashes[str(p)]
        errors_by_cell.append(ee)
    assert len(saved_hashes)==2912
    rows={};snrs={};rng=np.random.default_rng(202610086999);boot={a:[] for a in ARMS}
    for cid in range(56):
        counts=rng.multinomial(64,np.ones(64)/64,size=2000)
        for arm in ARMS:
            values=[]
            for seed in CONFIG['initializations']:
                err=errors_by_cell[cid][f'{arm}_d0_s{seed}/selected']
                values.append(10*np.log10(counts@err/(counts@energies[cid])))
            boot[arm].append(np.mean(values,axis=0))
    timing=read(ROOT/'training_timing.json')
    for p,h in timing['checkpoint_hashes'].items():assert sha(p)==h
    assert sha('paper2_review_round2/timing_training.py')==timing['source_sha256']
    assert sha(test/'data/cell_34_observations.npz')==timing['data_sha256']
    for arm in ARMS:
        names=[f'{arm}_d0_s{s}' for s in CONFIG['initializations']]
        ss=np.array([scores[n+'/selected'] for n in names]);ff=np.array([scores[n+'/latest'] for n in names])
        base=np.array([scores[f'baseline_d0_s{s}/selected'] for s in CONFIG['initializations']])
        delta=np.mean(base-ss,axis=0)
        ci=np.quantile(np.mean(np.array(boot['baseline'])-np.array(boot[arm]),axis=0),[.025,.975])
        if arm!='baseline':
            c=saved['comparisons'][arm]
            np.testing.assert_allclose([delta.mean(),delta.min()],[c['macro_gain_db'],c['worst_cell_gain_db']],atol=1e-12)
            np.testing.assert_allclose(ci,c['conditional_paired_interval'],atol=1e-12)
        np.testing.assert_allclose(ss.mean(),saved['macro_nmse_db'][arm],atol=1e-12)
        raw=np.concatenate([timing['raw_seconds'][n] for n in names]);assert len(raw)==288
        for n in names:
            np.testing.assert_allclose(1000*np.mean(timing['raw_seconds'][n]),timing['summary'][n]['mean_ms'],rtol=1e-12)
            np.testing.assert_allclose(1000*np.median(timing['raw_seconds'][n]),timing['summary'][n]['median_ms'],rtol=1e-12)
        rows[arm]=dict(selected_nmse_db=float(ss.mean()),final_nmse_db=float(ff.mean()),
            selected_seed_macros=ss.mean(1).tolist(),final_seed_macros=ff.mean(1).tolist(),
            selected_epochs=[epochs[n] for n in names],gain_db=float(delta.mean()),interval=ci.tolist(),
            worst_cell_gain_db=float(delta.min()),worst_cell=GRID[int(delta.argmin())],
            low_reference_gain_db=float(delta[[c['rsr']==0 for c in GRID]].mean()),
            two_path_gain_db=float(delta[[c['L']==2 for c in GRID]].mean()),
            severe_percent=float(100*np.mean([severe[n+'/selected'] for n in names])),
            mean_case_nmse=float(np.mean([case_nmse[n+'/selected'] for n in names])),
            mean_ms=float(1000*raw.mean()),parameters=timing['summary'][names[0]]['parameters'])
        snrs[arm]={str(s):float(ss[:,[c['snr']==s for c in GRID]].mean()) for s in CONFIG['snrs']}
    datasets={}
    for rep in range(3):
        names=[f'baseline_d{rep}_s{s}/selected' for s in CONFIG['initializations'][:2]]
        vals=[float(np.mean(scores[n])) for n in names];datasets[str(rep)]=vals
        np.testing.assert_allclose(np.mean(vals),saved['independent_dataset_baseline_macros'][str(rep)],atol=1e-12)
    # Replays selected and final checkpoints in a different batch size at SNR 5, RSR 12.
    obs=arrays(test/'data/cell_34_observations.npz');replays={}
    with torch.no_grad():
        for job in JOBS:
            for which in ('selected','latest'):
                name=f'{job["name"]}/{which}';m,_=load_model(training/(name+'.pt'))
                got=m.estimate(obs['z'][0],obs['s'][0],obs['b'][0],float(obs['sigma2'][0])).numpy()
                expected=arrays(test/'predictions'/name/'cell_34.npz')['estimates'][0]
                diff=float(np.max(np.abs(got-expected)));np.testing.assert_allclose(got,expected,rtol=2e-7,atol=1e-8)
                replays[name]=diff
    controls={name:dict(nmse_db=float(np.mean(scores[name])),severe_percent=100*float(np.mean(severe[name])))
              for name in ('em100','opt_loop_frozen')}
    report=dict(status='passed',jobs=25,epoch_histories=300,prediction_files=2912,
        neural_selected_files=1400,neural_final_files=1400,classical_files=112,
        fresh_worlds=3584,regenerated_first_worlds=56,checkpoint_hashes_checked=50,
        replay_max_difference=max(replays.values()),replays=replays,rows=rows,
        per_snr_scores=snrs,independent_dataset_seed_macros=datasets,classical_transfer=controls,
        data_hashes=hashes,selected_epochs=epochs,
        caveats=saved['caveats'],source_sha256=sha(__file__))
    (ROOT/'FINAL_NUMERICAL_AUDIT.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('data_hashes','replays','selected_epochs')},indent=2))

if __name__=='__main__':run()
