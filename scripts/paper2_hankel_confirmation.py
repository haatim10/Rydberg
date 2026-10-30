"""Frozen 40-run study. Test access starts only after validation choices freeze."""
import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from scipy.stats import t
from torch.utils.data import DataLoader
import paper2_classical_start as prior
import paper2_original_channel as channel
from trackD_urformer.hankel import project_G
from rydberg_sim.track_b_proposed import hs_gs_auto
from rydberg_sim.gs import em_gs_channel_rows
base=prior.base
engine=base.engine
OUT=ROOT/'results/paper2_hankel_confirmation'
MODES=('literal_independent','clustered_5deg')
SEEDS=(2026092601,2026092602,2026092603,2026092604,2026092605)
ARMS=('standard_U','standard_H','weighted_U','weighted_H')
COUNTS={'train':16000,'val':1200,'test':1800,'tune':600}
EPOCHS=18
STARTS={'literal_independent':30000000,'clustered_5deg':40000000}
OFFSETS={'train':0,'val':1000000,'test':2000000,'tune':3000000}
STOPS=(1,3,5,10,20,50,100)
CLASSICAL=('emgs10','emgs100','emgs_tuned','hs7_10','hs7_50','paper1_auto','paper1_emgs50')
ORIGINAL_STEP=base.repro.train_step
WEIGHT_MEAN=1.
WORKER_LIMIT=15*3600
SEGMENT_START=None
PRIOR_SECONDS=0.
LAST_BUDGET_SAVE=0.


class BudgetReached(RuntimeError): pass


def check_budget(force=False):
    global LAST_BUDGET_SAVE
    if SEGMENT_START is None: return
    now=time.monotonic(); spent=PRIOR_SECONDS+now-SEGMENT_START
    if force or now-LAST_BUDGET_SAVE>30 or spent>=WORKER_LIMIT:
        engine.save_json(OUT/'budget.json',{'worker_seconds':spent,'limit_seconds':WORKER_LIMIT})
        LAST_BUDGET_SAVE=now
    if spent>=WORKER_LIMIT: raise BudgetReached('15-hour worker-time budget reached; saved partial work, no automatic extension.')


class Worlds(channel.PaperChannelData):
    def __init__(self,mode,split,count=None):
        super().__init__('val' if split=='tune' else split,mode,COUNTS[split] if count is None else count)
        self.split=split
        self.start=STARTS[mode]+OFFSETS[split]

    def raw(self,index):
        if not 0<=index<self.count: raise IndexError(index)
        trial=self.start+index
        rng=channel.rng_for(trial,5)
        snr=float(rng.uniform(-10,20)) if self.split=='train' else float(-10+5*(index%6)+rng.uniform(0,5))
        variance=self.signal_power/10**(snr/10)
        g=channel.draw_channel(trial,self.mode)
        return trial,snr,variance,g,channel.exact_forward(g,self.S,self.B,variance,rng_noise=channel.rng_for(trial,6))


class DiskData(torch.utils.data.Dataset):
    def __init__(self,mode,split):
        folder=OUT/mode/'data'/split
        self.meta=json.loads((folder/'complete.json').read_text())
        self.arrays={k:torch.from_numpy(np.load(folder/f'{k}.npy',mmap_mode='c')) for k in self.meta['sha256']}
    def __len__(self): return self.meta['count']
    def __getitem__(self,i): return {k:torch.as_tensor(v[i]) for k,v in self.arrays.items()}


def progress(**values):
    check_budget()
    engine.save_json(OUT/'status.json',{'updated_unix':time.time(),**values})


def cache(mode,split):
    if split=='test':
        assert (OUT/'selection.json').exists(),'Test access locked until selection freezes.'
    folder=OUT/mode/'data'/split; folder.mkdir(parents=True,exist_ok=True)
    path=folder/'complete.json'
    spec={'mode':mode,'split':split,'count':COUNTS[split],'start':STARTS[mode]+OFFSETS[split]}
    if path.exists():
        meta=json.loads(path.read_text()); assert all(meta[k]==v for k,v in spec.items())
        for k,h in meta['sha256'].items(): assert engine.hash_file(folder/f'{k}.npy')==h
        return DiskData(mode,split)
    ds=Worlds(mode,split); first=ds[0]
    arrays={k:np.lib.format.open_memmap(folder/f'{k}.npy',mode='w+',dtype=v.numpy().dtype,
               shape=(len(ds),*v.shape)) for k,v in first.items()}
    for i in range(len(ds)):
        check_budget()
        item=ds[i]
        for k,v in item.items(): arrays[k][i]=v.numpy()
        if (i+1)%1000==0: progress(state='generating_data',mode=mode,split=split,done=i+1,total=len(ds))
    for v in arrays.values(): v.flush()
    engine.save_json(path,{**spec,'sha256':{k:engine.hash_file(folder/f'{k}.npy') for k in arrays}})
    result=DiskData(mode,split)
    for i in (0,len(ds)//2,len(ds)-1):
        for k,v in ds[i].items(): torch.testing.assert_close(v,result[i][k],rtol=0,atol=0)
    return result


class HankelIdentityBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx,g):
        return project_G(g,rank=7,mode='fixed',n_iter=1)
    @staticmethod
    def backward(ctx,gradient):
        return gradient


class Model(prior.ClassicalStart):
    def __init__(self,original,arm):
        super().__init__(original)
        self.use_hankel=arm.endswith('_H'); self.weighted=arm.startswith('weighted')
    def forward(self,g,z,s,b,var):
        for former in self.formers:
            g=base.em_gs_layer(g,z,s,b,var,eps=self.eps)
            if self.use_hankel:
                # Exact projected forward values; same declared identity STE
                # without cancellation roundoff in g+(projection-g).
                g=HankelIdentityBackward.apply(g)
            g=g+former(g)
        return g


def build(arm):
    assert arm in ARMS
    original,_=base.repro.build_model(base.cfg())
    model=Model(original,arm)
    return model,{'architecture':'classical_start_PaperFormer','hankel':model.use_hankel,
        'weighted_loss':model.weighted,'weight_normalizer':WEIGHT_MEAN,
        'params':sum(p.numel() for p in model.parameters()),'rank':7,'sweeps':1,'svd_gradient':'identity_STE'}


def train_step(model,opt,batch,cfg):
    check_budget()
    if not model.weighted: return ORIGINAL_STEP(model,opt,batch,cfg)
    opt.zero_grad(set_to_none=True)
    pred=model(*(batch[k] for k in ('G0','Z','S','B','sigma2')))
    errors=abs(pred-batch['G_true']).square().sum((1,2))/abs(batch['G_true']).square().sum((1,2))
    weights=torch.where(batch['snr_db']>=15,10.,1.)/WEIGHT_MEAN
    loss=(weights*errors).mean()
    if not torch.isfinite(loss): raise RuntimeError('Non-finite weighted loss')
    loss.backward()
    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),float('inf'),error_if_nonfinite=True)
    opt.step()
    return float(errors.detach().mean()),float(norm)


@contextmanager
def configured(mode,seed):
    with patch.multiple(base,OUT=OUT/mode/f'seed{seed}',SEED=seed,ARMS=ARMS,build=build), \
         patch.object(base.repro,'train_step',train_step): yield


def folder(mode,seed,arm): return OUT/mode/f'seed{seed}'/'development'/arm
def load_json(path): return json.loads(Path(path).read_text())
def loader(data): return DataLoader(data,batch_size=32,shuffle=False,collate_fn=base.collate)


def save_npz(path,**values):
    temp=path.with_suffix('.partial-write.npz')
    np.savez(temp,**values)
    temp.replace(path)


def sources():
    result=prior.sources()
    for p in (Path(__file__),ROOT/'reports/paper2_hankel_confirmation_plan.md',ROOT/'tests/test_paper2_hankel_confirmation.py'):
        result[str(p.relative_to(ROOT))]=engine.hash_file(p)
    return result


@torch.no_grad()
def evaluate(model,data):
    start=time.monotonic()
    result=engine.test_model(model,loader(data))
    assert np.isfinite(result['numerator']).all() and (result['denominator']>0).all()
    result['elapsed_seconds']=np.asarray(time.monotonic()-start)
    return result


class Fixed(torch.nn.Module):
    def __init__(self,steps,hankel=False): super().__init__(); self.steps=steps; self.hankel=hankel
    def forward(self,g,z,s,b,var):
        for _ in range(self.steps):
            g=base.em_gs_layer(g,z,s,b,var,eps=1e-8)
            if self.hankel: g=project_G(g,rank=7,mode='fixed',n_iter=1)
        return g


def native_record(batch,method):
    args=[batch[k].numpy() for k in ('S','Z','B')]
    variance=float(batch['sigma2'])
    if method=='paper1_auto':
        fit=hs_gs_auto(*args,variance,max_iter=50,select_iter=25,cadzow_iter=4)
        return fit.G_hat,fit.L_hat
    return em_gs_channel_rows(*args,variance,max_iter=50,ridge=0.,G0=None).G_hat,16


def classical(mode,split,data,chosen):
    root=OUT/mode/'classical'/split; root.mkdir(parents=True,exist_ok=True)
    for method in CLASSICAL:
        check_budget()
        dest=root/f'{method}.npz'
        if dest.exists(): continue
        progress(state='classical_evaluation',mode=mode,split=split,method=method,done=0,total=len(data))
        if method.startswith('paper1'):
            part=root/f'{method}_partial.npz'
            keys=('numerator','denominator','trial','snr_db','rank')
            values={k:[] for k in keys}; start=0; elapsed=0.
            if part.exists():
                saved=np.load(part); values={k:saved[k].tolist() for k in keys}
                start=len(values['trial']); elapsed=float(saved['elapsed_seconds'])
            clock=time.monotonic()
            for i in range(start,len(data)):
                check_budget()
                batch=data[i]; g,rank=native_record(batch,method)
                truth=batch['G_true'].numpy()
                values['numerator'].append(float(np.sum(abs(g-truth)**2)))
                values['denominator'].append(float(np.sum(abs(truth.astype(np.complex128))**2)))
                values['trial'].append(int(batch['trial'])); values['snr_db'].append(float(batch['snr_db'])); values['rank'].append(rank)
                if (i+1)%8==0 or i+1==len(data):
                    save_npz(part,**values,elapsed_seconds=elapsed+time.monotonic()-clock)
                    progress(state='classical_evaluation',mode=mode,split=split,method=method,done=i+1,total=len(data))
            record={k:np.asarray(v) for k,v in values.items()}
            record['elapsed_seconds']=np.asarray(elapsed+time.monotonic()-clock)
        else:
            steps={'emgs10':10,'emgs100':100,'emgs_tuned':chosen,'hs7_10':10,'hs7_50':50}[method]
            record=evaluate(Fixed(steps,method.startswith('hs7')),data)
        assert np.isfinite(record['numerator']).all() and (record['denominator']>0).all()
        save_npz(dest,**record)


def errors(path):
    f=np.load(path); return f['numerator']/f['denominator']


def freeze_selection():
    selection={'modes':{},'source_hashes':sources(),'frozen_unix':time.time()}
    if (OUT/'selection.json').exists(): return load_json(OUT/'selection.json')
    for mode in MODES:
        val=cache(mode,'val'); tune=cache(mode,'tune')
        root=OUT/mode/'classical/tune'; root.mkdir(parents=True,exist_ok=True)
        means={}
        for steps in STOPS:
            dest=root/f'emgs{steps}.npz'
            if not dest.exists(): save_npz(dest,**evaluate(Fixed(steps),tune))
            means[steps]=float(errors(dest).mean())
        chosen=min(STOPS,key=lambda step:means[step])
        classical(mode,'val',val,chosen)
        model_means={}; selections={}
        for arm in ARMS:
            all_errors=[]
            for seed in SEEDS:
                p=folder(mode,seed,arm); r=load_json(p/'result.json'); v=np.load(p/'validation_errors.npz')
                ep=r['chosen_epoch']; selections[f'{seed}/{arm}']=ep
                all_errors.append(v['errors'][ep-1])
                with configured(mode,seed):
                    model,_=build(arm)
                    for check_epoch in set((ep,EPOCHS)):
                        model.load_state_dict(torch.load(p/f'epoch_{check_epoch:03d}.pt',weights_only=True)['model'])
                        np.testing.assert_array_equal(base.evaluate(model,val),v['errors'][check_epoch-1])
            model_means[arm]=float(np.mean(all_errors))
        u=min(('standard_U','weighted_U'),key=lambda a:model_means[a])
        h=min(('standard_H','weighted_H'),key=lambda a:model_means[a])
        classical_means={m:float(errors(OUT/mode/'classical/val'/f'{m}.npz').mean()) for m in CLASSICAL}
        selection['modes'][mode]={'primary_loss':u.split('_')[0],'best_U':u,'best_H':h,
            'best_classical':min(CLASSICAL,key=lambda m:classical_means[m]),'tuned_emgs_steps':chosen,
            'model_validation_means':model_means,'classical_validation_means':classical_means,'epochs':selections}
    engine.save_json(OUT/'selection.json',selection)
    return selection


def test_all(selection):
    for mode in MODES:
        data=cache(mode,'test')
        for seed in SEEDS:
            for arm in ARMS:
                p=folder(mode,seed,arm)
                with configured(mode,seed):
                    model,_=build(arm)
                    for label,ep in (('selected',selection['modes'][mode]['epochs'][f'{seed}/{arm}']),('final',EPOCHS)):
                        dest=p/f'test_{label}.npz'
                        if dest.exists(): continue
                        progress(state='test_evaluation',mode=mode,seed=seed,arm=arm,checkpoint=label)
                        model.load_state_dict(torch.load(p/f'epoch_{ep:03d}.pt',weights_only=True)['model'])
                        save_npz(dest,**evaluate(model,data))
        classical(mode,'test',data,selection['modes'][mode]['tuned_emgs_steps'])


def db(x): return float(10*np.log10(np.mean(x)))
def interval(values,alpha=.05):
    v=np.asarray(values); mean=float(v.mean())
    if len(v)<2: return [mean,mean]
    margin=float(t.ppf(1-alpha/2,len(v)-1)*v.std(ddof=1)/np.sqrt(len(v)))
    return [mean-margin,mean+margin]


def analyze(selection):
    result={'seeds':list(SEEDS),'limitations':'Five weight/order seeds; shared training data and fixed pilots. No generality or convergence guarantee.','modes':{}}
    for mode in MODES:
        snr=np.load(folder(mode,SEEDS[0],ARMS[0])/'test_selected.npz')['snr_db']
        masks={'all':np.ones(len(snr),bool),'snr_ge5':snr>=5}
        masks.update({f'[{lo},{lo+5})':(snr>=lo)&(snr<lo+5) for lo in range(-10,20,5)})
        record={'classical':{},'contrasts':{},'absolute':{},'neural_inference_seconds_per_case':{}}
        classical_errors={m:errors(OUT/mode/'classical/test'/f'{m}.npz') for m in CLASSICAL}
        for m,arr in classical_errors.items():
            timing=float(np.load(OUT/mode/'classical/test'/f'{m}.npz')['elapsed_seconds'])
            record['classical'][m]={'seconds_per_case':timing/len(arr),'bins':{k:{'nmse_db':db(arr[mask]),'severe_error_rate':float(np.mean(arr[mask]>1))} for k,mask in masks.items()}}
        for label in ('selected','final'):
            arrays={arm:np.stack([errors(folder(mode,seed,arm)/f'test_{label}.npz') for seed in SEEDS]) for arm in ARMS}
            for arm,arr in arrays.items():
                record['neural_inference_seconds_per_case'][f'{label}/{arm}']=[float(np.load(folder(mode,seed,arm)/f'test_{label}.npz')['elapsed_seconds'])/len(snr) for seed in SEEDS]
                record['absolute'][f'{label}/{arm}']={k:{'nmse_db':db(arr[:,mask]),'per_seed_nmse_db':[db(a[mask]) for a in arr],
                    'severe_error_rate':float(np.mean(arr[:,mask]>1))} for k,mask in masks.items()}
            for loss in ('standard','weighted'):
                u,h=arrays[loss+'_U'],arrays[loss+'_H']; bins={}
                for key,mask in masks.items():
                    gains=[db(a[mask])-db(b[mask]) for a,b in zip(u,h)]
                    pooled_u,pooled_h=u[:,mask].mean(axis=0),h[:,mask].mean(axis=0)
                    rng=np.random.default_rng(2026092699); boot=[]
                    for _ in range(1000):
                        ids=rng.integers(0,len(pooled_u),len(pooled_u)); boot.append(db(pooled_u[ids])-db(pooled_h[ids]))
                    bins[key]={'n_cases':int(mask.sum()),'per_seed_gain_db':gains,'mean_gain_db':float(np.mean(gains)),
                        'seed_sd_db':float(np.std(gains,ddof=1)) if len(gains)>1 else 0.,'seed_ci95':interval(gains),
                        'seed_ci975':interval(gains,.025),'case_bootstrap_ci95_conditional_on_models':np.quantile(boot,[.025,.975]).tolist(),
                        'per_seed_paired_median_db':[float(np.median(10*np.log10(a[mask]/b[mask]))) for a,b in zip(u,h)]}
                record['contrasts'][f'{label}/{loss}']=bins
            # No selection based on test outcomes: names come only from selection.json.
            best_h=arrays[selection['modes'][mode]['best_H']]
            classical_best=classical_errors[selection['modes'][mode]['best_classical']]
            record[f'{label}_bestH_vs_selected_classical_high_gain_db']=db(classical_best[masks['snr_ge5']])-db(best_h[:,masks['snr_ge5']])
        primary=record['contrasts']['selected/'+selection['modes'][mode]['primary_loss']]
        record['screen']={'primary_loss':selection['modes'][mode]['primary_loss'],
            'gain_at_least_05':primary['snr_ge5']['mean_gain_db']>=.5,
            'adjusted_interval_positive':primary['snr_ge5']['seed_ci975'][0]>0,
            'full_range_preserved':primary['all']['mean_gain_db']>=-.2,
            'beats_selected_classical_by_05':record['selected_bestH_vs_selected_classical_high_gain_db']>=.5}
        record['screen']['all_pass']=all(v for k,v in record['screen'].items() if k!='primary_loss')
        result['modes'][mode]=record
    engine.save_json(OUT/'summary.json',result)
    return result


def audit_training():
    count=0
    for mode in MODES:
        for seed in SEEDS:
            hashes=[]; orders=[]
            for arm in ARMS:
                p=folder(mode,seed,arm); r=load_json(p/'result.json'); arr=np.load(p/'validation_errors.npz')
                assert len(r['history'])==EPOCHS and len(list(p.glob('epoch_*.pt')))==EPOCHS
                assert arr['errors'].shape==(EPOCHS,COUNTS['val']) and np.isfinite(arr['errors']).all()
                curve=10*np.log10(arr['errors'].mean(axis=1))
                chosen,rule=base.select_epoch(curve,arr['errors'][curve.argmin()],'one_se')
                assert chosen+1==r['chosen_epoch'] and rule==r['selection']
                np.testing.assert_allclose(curve,[h['validation_nmse_db'] for h in r['history']],rtol=0,atol=1e-12)
                assert all(np.isfinite(h['max_gradient_norm']) for h in r['history'])
                hashes.append(r['initial_hash']); orders.append([h['data_order_sha256'] for h in r['history']])
                count+=1
            assert len(set(hashes))==1 and all(x==orders[0] for x in orders)
    return {'runs':count,'matched_weights_and_orders':True,'epochs':count*EPOCHS}


def audit_test():
    files=0
    for mode in MODES:
        data=DiskData(mode,'test')
        paths=[folder(mode,seed,arm)/f'test_{label}.npz' for seed in SEEDS for arm in ARMS for label in ('selected','final')]
        paths += [OUT/mode/'classical/test'/f'{m}.npz' for m in CLASSICAL]
        for path in paths:
            f=np.load(path)
            for k in ('trial','snr_db'): np.testing.assert_array_equal(f[k],data.arrays[k])
            assert np.isfinite(f['numerator']).all() and (f['denominator']>0).all()
            assert len(f['numerator'])==COUNTS['test']
            files+=1
    return {'test_files_verified':files,'test_case_alignment':True}


def report(summary,audit):
    lines=['# Large Hankel study: automated results report','',
      'All results are from the frozen independent simulation protocol, not an author-code reproduction.',
      f'Completed {audit["runs"]} runs and {audit["epochs"]} epochs. Matched parameter/order checks passed.',
      'Selected and final validation checkpoints replayed exactly before test access.','',
      '## Primary results','',
      '| Population | Selected loss | High-SNR gain (dB) | Adjusted seed interval | Screen |',
      '|---|---|---:|---|---|']
    for mode,r in summary['modes'].items():
        loss=r['screen']['primary_loss']; p=r['contrasts']['selected/'+loss]['snr_ge5']
        lines.append(f'| {mode} | {loss} | {p["mean_gain_db"]:.3f} | {p["seed_ci975"]} | {r["screen"]["all_pass"]} |')
    lines += ['', 'A passing screen is not a publication or novelty guarantee. A failed screen does not prove equivalence.',
      'Inspect summary.json for all bins, both losses, final checkpoints, severe errors and every classical control.',
      'Five seeds vary training randomness on shared data, not five independently sampled datasets. Case bootstrap intervals condition on the trained models.',
      'Training/validation curves must still be reviewed for continued trends and selection sensitivity; completing 18 epochs does not establish convergence.',
      'Inference timings include native Paper1 rank selection; architectures and numerical precision differ. No universal speed or hardware claim is justified.',
      'This automated report and the saved artifacts require a final human-facing scientific review before any manuscript change.']
    (OUT/'RESULTS_REPORT.md').write_text('\n'.join(lines)+'\n')


def run():
    global WEIGHT_MEAN,SEGMENT_START,PRIOR_SECONDS
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        torch.set_num_threads(1)
        assert load_json(OUT/'preflight.json')['passed']
        assert load_json(OUT/'preflight.json')['source_hashes']==sources(),'Preflight sources changed.'
        manifest={'source_hashes':sources(),'modes':list(MODES),'seeds':list(SEEDS),'arms':list(ARMS),
                  'counts':COUNTS,'epochs':EPOCHS,'starts':STARTS,'torch':torch.__version__,
                  'numpy':np.__version__,'platform':platform.platform(),'worker_limit_seconds':WORKER_LIMIT}
        path=OUT/'manifest.json'
        if path.exists(): assert load_json(path)==manifest,'Changed configuration/source: refusing resume.'
        else: engine.save_json(path,manifest)
        engine.save_json(OUT/'pid.json',{'pid':os.getpid(),'started_unix':time.time()})
        PRIOR_SECONDS=load_json(OUT/'budget.json')['worker_seconds'] if (OUT/'budget.json').exists() else 0.
        SEGMENT_START=time.monotonic()
        try:
            completed=0
            for seed in SEEDS:
                for mode in MODES:
                    data={split:cache(mode,split) for split in ('train','val')}
                    WEIGHT_MEAN=float(np.where(data['train'].arrays['snr_db']>=15,10.,1.).mean())
                    spec={'train':COUNTS['train'],'val':COUNTS['val'],'epochs':EPOCHS,'start':STARTS[mode],'snr':None}
                    with configured(mode,seed):
                        for arm in ARMS:
                            progress(state='training',mode=mode,seed=seed,arm=arm,runs_complete=completed,
                                     runs_total=len(MODES)*len(SEEDS)*len(ARMS),active_status=str(base.OUT/'status.json'))
                            base.train_arm('development',arm,spec,data); completed+=1
            audit=audit_training()
            selection=freeze_selection()
            test_all(selection)
            audit.update(audit_test())
            assert sources()==manifest['source_hashes']
            summary=analyze(selection); report(summary,audit)
            engine.save_json(OUT/'audit.json',audit)
            check_budget(force=True)
            progress(state='complete',runs=completed,scientific_review_pending=True)
        except BudgetReached as exc:
            engine.save_json(OUT/'status.json',{'state':'budget_exhausted','reason':str(exc),'incomplete':True})
        except Exception:
            engine.save_json(OUT/'status.json',{'state':'failed','error':traceback.format_exc()}); raise
        finally:
            SEGMENT_START=None


def launch():
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'launch.lock').open('a') as launch_lock:
        fcntl.flock(launch_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with (OUT/'run.lock').open('a') as worker_lock:
            try: fcntl.flock(worker_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:
                print('Worker already active; no duplicate.'); return
        if (OUT/'status.json').exists() and load_json(OUT/'status.json')['state'] in ('failed','complete','budget_exhausted'):
            print(json.dumps(load_json(OUT/'status.json'))); return
        with (OUT/'worker.log').open('ab') as log:
            p=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'run'],cwd=ROOT,
                stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        time.sleep(2)
        if p.poll() is not None: raise RuntimeError('Worker exited; inspect worker.log')
        print(json.dumps({'pid':p.pid,'output':str(OUT)}))


def status():
    r=load_json(OUT/'status.json')
    if r.get('active_status') and Path(r['active_status']).exists(): r['active']=load_json(r['active_status'])
    print(json.dumps(r,indent=2))


def preflight():
    import unittest
    import shutil
    torch.set_num_threads(1)
    suite=unittest.defaultTestLoader.discover(str(ROOT/'tests'),pattern='test_paper2_hankel_confirmation.py')
    # Tests import this file under its module name; the CLI itself is __main__.
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    assert result.wasSuccessful(),'Preflight unit/pipeline checks failed.'
    timing={}; distortion={}; calibration={}
    with configured(MODES[0],SEEDS[0]):
        ds=Worlds(MODES[0],'train',32); batch=base.collate([ds[i] for i in range(32)])
        for arm in ARMS:
            model,_=build(arm); opt=torch.optim.Adam(model.parameters(),lr=.001)
            train_step(model,opt,batch,base.cfg()); samples=[]
            for _ in range(5):
                start=time.monotonic(); train_step(model,opt,batch,base.cfg()); samples.append(time.monotonic()-start)
            timing[arm]=float(np.mean(samples))
    for mode in MODES:
        ds=Worlds(mode,'train',400); power=[]; sig=[]; noise=[]; distort=[]
        for i in range(400):
            trial,snr,var,g,ex=ds.raw(i)
            power.append(np.mean(abs(g)**2)); sig.append(np.mean(abs(g@ds.S)**2)); noise.append(np.mean(abs(ex.W)**2)/var)
            if i<100:
                x=torch.tensor(g[None],dtype=torch.complex128); y=project_G(x,rank=7)
                distort.append(float((abs(x-y).square().sum()/abs(x).square().sum())))
        assert abs(np.mean(power)/32-1)<.1 and abs(np.mean(sig)/ds.signal_power-1)<.1 and abs(np.mean(noise)-1)<.02
        calibration[mode]={'channel_power':float(np.mean(power)),'signal_power':float(np.mean(sig)),
            'expected_signal_power':ds.signal_power,'noise_ratio':float(np.mean(noise))}
        distortion[mode]=float(np.mean(distort))
    times=[]
    for i in range(2):
        start=time.monotonic(); native_record(Worlds(MODES[0],'train',2)[i],'paper1_auto'); times.append(time.monotonic()-start)
    hours=(len(MODES)*len(SEEDS)*EPOCHS*(COUNTS['train']/32)*sum(timing.values())+
           len(MODES)*(COUNTS['val']+COUNTS['test'])*np.mean(times))/3600
    assert hours<15,'Measured core workload alone exceeds15hours; revise prospectively before launch.'
    assert shutil.disk_usage(OUT.parent).free>12*1024**3,'Insufficient free disk headroom.'
    old_checks={}
    for name in ('paper2_original_channel','paper2_reproduction','paper2_classical_start','paper2_loss_weighting'):
        manifest=load_json(ROOT/'results'/name/'manifest.json')
        for path,h in manifest['source_hashes'].items(): assert engine.hash_file(ROOT/path)==h
        old_checks[name]=len(manifest['source_hashes'])
    saved={'passed':True,'tests_run':result.testsRun,'source_hashes':sources(),'seconds_per_training_batch':timing,
        'paper1_seconds_per_case':float(np.mean(times)),'core_hours_estimate':hours,
        'estimated_hours_with_overhead':[hours*1.1,hours*1.3],'hard_worker_hours':15,
        'calibration':calibration,'noiseless_rank7_distortion':distortion,'old_sources_verified':old_checks}
    engine.save_json(OUT/'preflight.json',saved); print(json.dumps(saved,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('command',choices=('run','launch','status','preflight'))
    args=parser.parse_args(); {'run':run,'launch':launch,'status':status,'preflight':preflight}[args.command]()
