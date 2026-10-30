"""Fixed small screen. Training never reads a truth file; test is frozen last."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key]='1'
import argparse
from dataclasses import asdict
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback
import numpy as np
import torch
from sahure.data import load_observations, split_observations
from sahure.model import AdaptiveHankelEstimator, ModelConfig
from sahure.train import heldout_loss, load_model
from sahure_expansion.data import world
from sahure_expansion.train import save_checkpoint, validate
from .noise import make_views

ROOT=Path('results/rician_pilot_screen')
CELLS=[dict(snr=s,rsr=r) for s in (-5.,5.,15.) for r in (5.,12.)]
TEST_CELLS=CELLS+[dict(snr=5.,rsr=0.),dict(snr=40.,rsr=12.)]
ARMS={'holdout':None,'r2r_05':.5,'r2r_10':1.}
CONFIG=dict(version=1,seeds=[10401,10402],epochs=10,train_count=2304,val_count=384,
    batch_size=32,lr=.001,stages=10,train_seed=20261014101001,val_seed=20261014102001,
    test_seed=20261014103001,test_per_cell=64,validation_mask_seed=202610140901,
    max_seconds=7200,cells=CELLS,test_cells=TEST_CELLS,arms=ARMS,
    primary='final epoch10, original-noise all30 input, equal6cell dB ratio-of-sums')

def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temp.replace(path)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def freeze():
    ROOT.mkdir(parents=True,exist_ok=True)
    paths=[Path(p) for p in ('rician_pilot_screen/run.py','rician_pilot_screen/noise.py',
        'rician_pilot_screen/test_noise.py','rician_pilot_screen/PROTOCOL.txt',
        'sahure/model.py','sahure/math.py','sahure/train.py','sahure/data.py',
        'sahure_expansion/data.py','sahure_expansion/train.py')]
    paths+=sorted(Path('rydberg_sim').rglob('*.py'))
    record=dict(config=CONFIG,sources={str(p):sha(p) for p in paths},torch=torch.__version__)
    target=ROOT/'manifest.json'
    if target.exists() and json.loads(target.read_text())!=record:
        raise RuntimeError('frozen source or configuration changed')
    if not target.exists():write(target,record)

def check_deadline(deadline):
    if time.monotonic()>deadline:raise TimeoutError('two-hour cap; complete epochs retained')

def prepare(name,count,seed,cells,truth=False,deadline=float('inf')):
    folder=ROOT/'data';folder.mkdir(parents=True,exist_ok=True)
    path=folder/f'{name}_observations.npz';manifest=folder/f'{name}_manifest.json'
    description=dict(count=count,seed=seed,cells=cells,truth_saved=truth)
    if manifest.exists():
        old=json.loads(manifest.read_text())
        if old['description']!=description:raise RuntimeError('data description changed')
        for p,digest in old['files'].items():
            if sha(p)!=digest:raise RuntimeError('data changed: '+p)
        return
    ids=np.tile(np.arange(len(cells)),count//len(cells))
    if len(ids)!=count:raise ValueError('balanced cells required')
    if name!='test':ids=np.random.default_rng(seed).permutation(ids)
    arrays={k:[] for k in ('z','s','b','sigma2')};gs=[]
    for i,c in enumerate(ids):
        check_deadline(deadline)
        w,_=world(i,seed,**cells[int(c)])
        for key,value in zip(arrays,(w.Z,w.S,w.B,w.sigma2)):arrays[key].append(value)
        if truth:gs.append(w.G)
    np.savez_compressed(path,**{k:np.asarray(v) for k,v in arrays.items()})
    cp=folder/f'{name}_cells.npy';np.save(cp,ids)
    files=[path,cp]
    if truth:
        tp=folder/f'{name}_truth.npy';np.save(tp,np.asarray(gs));files.append(tp)
    write(manifest,dict(description=description,files={str(p):sha(p) for p in files}))

def views(data,arm,seed):
    return split_observations(data,seed) if arm=='holdout' else make_views(data,ARMS[arm],seed=seed)

def train(deadline):
    data=load_observations(ROOT/'data/train_observations.npz')
    val=load_observations(ROOT/'data/val_observations.npz')
    conditions=np.load(ROOT/'data/val_cells.npy')
    va,vb=split_observations(val,CONFIG['validation_mask_seed'])
    epoch_times=[]
    for epoch in range(CONFIG['epochs']):
        for seed in CONFIG['seeds']:
            for arm in ARMS:
                folder=ROOT/'checkpoints'/f'{arm}_{seed}';folder.mkdir(parents=True,exist_ok=True)
                latest=folder/'latest.pt';best=folder/'best.pt'
                torch.manual_seed(seed);mc=ModelConfig()
                model=AdaptiveHankelEstimator(mc);opt=torch.optim.Adam(model.parameters(),lr=CONFIG['lr'])
                if latest.exists():
                    state=torch.load(latest,map_location='cpu',weights_only=False)
                    if state['epoch']>epoch:continue
                    if state['epoch']!=epoch:raise RuntimeError('epoch queue inconsistent')
                    model.load_state_dict(state['model']);opt.load_state_dict(state['optimizer'])
                else:
                    score,strata=validate(model,va,vb,conditions)
                    initial=hashlib.sha256(b''.join(p.detach().numpy().tobytes() for p in model.parameters())).hexdigest()
                    state=dict(epoch=0,best_score=score,best_epoch=0,history=[],seed=seed,arm=arm,
                        model_config=asdict(mc),initial_parameters_sha256=initial,
                        model=model.state_dict(),optimizer=opt.state_dict(),train_seconds=0.)
                    save_checkpoint(state,best);save_checkpoint(state,latest)
                model.train();a,b=views(data,arm,seed+10000+epoch)
                order=torch.randperm(len(data),generator=torch.Generator().manual_seed(seed+20000+epoch))
                order_sha=hashlib.sha256(order.numpy().tobytes()).hexdigest()
                tick=time.monotonic();total=0.
                for start in range(0,len(data),CONFIG['batch_size']):
                    if time.monotonic()>deadline:raise TimeoutError('two-hour cap; complete epochs retained')
                    ids=order[start:start+CONFIG['batch_size']];opt.zero_grad(set_to_none=True)
                    loss=heldout_loss(model,a.batch(ids),b.batch(ids))
                    if not torch.isfinite(loss):raise FloatingPointError('nonfinite loss')
                    loss.backward()
                    if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                        raise FloatingPointError('nonfinite gradient')
                    opt.step();total+=float(loss.detach())*len(ids)
                    if start%(8*CONFIG['batch_size'])==0:
                        done=epoch*len(ARMS)*len(CONFIG['seeds'])+CONFIG['seeds'].index(seed)*len(ARMS)+list(ARMS).index(arm)
                        write(ROOT/'status.json',dict(state='training',pid=os.getpid(),arm=arm,seed=seed,
                            epoch=epoch+1,batch=start//CONFIG['batch_size']+1,batches=72,
                            completed_run_epochs=done,total_run_epochs=60,updated_unix=time.time()))
                score,strata=validate(model,va,vb,conditions);seconds=time.monotonic()-tick
                improved=score<state['best_score']
                if improved:state.update(best_score=score,best_epoch=epoch+1)
                row=dict(epoch=epoch+1,train_nll=total/len(data),val_nll=score,seconds=seconds,
                    order_sha256=order_sha,augmentation_seed=seed+10000+epoch)
                state['history'].append(row);state['train_seconds']+=seconds
                state.update(epoch=epoch+1,model=model.state_dict(),optimizer=opt.state_dict())
                if improved:save_checkpoint(state,best)
                save_checkpoint(state,latest);write(folder/'history.json',state['history'])
                epoch_times.append(seconds)
                print(json.dumps(dict(arm=arm,seed=seed,**row)),flush=True)
    write(ROOT/'training_timing.json',dict(epoch_seconds=epoch_times))

def metrics(error,power,cells):
    nmse=error/power
    per={str(c):float(10*np.log10(error[cells==c].sum()/power[cells==c].sum())) for c in np.unique(cells)}
    primary=cells<len(CELLS)
    return dict(macro_db=float(np.mean([per[str(c)] for c in range(len(CELLS))])),cells_db=per,
        pooled_db=float(10*np.log10(error[primary].sum()/power[primary].sum())),
        severe_rate=float(np.mean(nmse[primary]>1)),stress_severe_rate=float(np.mean(nmse[~primary]>1)))

@torch.no_grad()
def evaluate(deadline):
    models={}
    for seed in CONFIG['seeds']:
        for arm in ARMS:
            for checkpoint in ('latest','best'):
                p=ROOT/'checkpoints'/f'{arm}_{seed}'/f'{checkpoint}.pt'
                models[str(p)]=sha(p)
    frozen=ROOT/'frozen_selection.json'
    if frozen.exists() and json.loads(frozen.read_text())!=models:raise RuntimeError('test checkpoint changed')
    if not frozen.exists():write(frozen,models)
    prepare('test',CONFIG['test_per_cell']*len(TEST_CELLS),CONFIG['test_seed'],TEST_CELLS,truth=True,deadline=deadline)
    data=load_observations(ROOT/'data/test_observations.npz',dtype=torch.float64)
    truth=torch.as_tensor(np.load(ROOT/'data/test_truth.npy'));cells=np.load(ROOT/'data/test_cells.npy')
    power=truth.abs().square().sum((-2,-1)).numpy();arrays=dict(power=power,cells=cells);records={}
    for seed in CONFIG['seeds']:
        for arm in ARMS:
            for checkpoint in ('latest','best'):
                model,state=load_model(ROOT/'checkpoints'/f'{arm}_{seed}'/f'{checkpoint}.pt')
                modes=('original','inflated_05','inflated_10') if checkpoint=='latest' else ('original',)
                for mode in modes:
                    incoming=data if mode=='original' else make_views(data,.5 if mode=='inflated_05' else 1.,seed=20261014104001)[0]
                    preds=[];tick=time.perf_counter()
                    for start in range(0,len(data),32):
                        check_deadline(deadline)
                        preds.append(model(*incoming.batch(slice(start,start+32)).args()))
                    g=torch.cat(preds);elapsed=time.perf_counter()-tick
                    error=(g-truth).abs().square().sum((-2,-1)).numpy()
                    key=f'{arm}_{seed}_{checkpoint}_{mode}'
                    arrays[key+'_error']=error;arrays[key+'_g']=g.numpy()
                    records[key]=dict(**metrics(error,power,cells),epoch=state['epoch'],seconds=elapsed)
                    write(ROOT/'status.json',dict(state='evaluating',key=key,pid=os.getpid(),updated_unix=time.time()))
    np.savez_compressed(ROOT/'test_predictions.npz',**arrays)
    changes={}
    for checkpoint in ('latest','best'):
        for arm in ('r2r_05','r2r_10'):
            gains=[records[f'holdout_{s}_{checkpoint}_original']['macro_db']-records[f'{arm}_{s}_{checkpoint}_original']['macro_db'] for s in CONFIG['seeds']]
            cells_gain={str(c):float(np.mean([records[f'holdout_{s}_{checkpoint}_original']['cells_db'][str(c)]-records[f'{arm}_{s}_{checkpoint}_original']['cells_db'][str(c)] for s in CONFIG['seeds']])) for c in range(len(TEST_CELLS))}
            changes[f'{arm}_{checkpoint}']=dict(gain_db=float(np.mean(gains)),seed_gains_db=gains,cells_gain_db=cells_gain)
    audits={}
    for seed in CONFIG['seeds']:
        states=[torch.load(ROOT/'checkpoints'/f'{arm}_{seed}'/'latest.pt',weights_only=False) for arm in ARMS]
        audits[str(seed)]=dict(initialization_matched=len({s['initial_parameters_sha256'] for s in states})==1,
            orders_matched=all(len({s['history'][e]['order_sha256'] for s in states})==1 for e in range(CONFIG['epochs'])),
            epochs_complete=all(s['epoch']==CONFIG['epochs'] for s in states))
    if not all(all(v.values()) for v in audits.values()):raise RuntimeError('matched-training audit failed')
    write(ROOT/'summary.json',dict(config=CONFIG,records=records,changes=changes,audits=audits,
        caveat='Exploratory small-budget screen; two initialization seeds share one dataset; no significance or novelty claim.'))
    lines=['SMALL ALL-PILOT RICIAN TRAINING SCREEN','Primary: final10epoch checkpoint, original-noise all30pilot inputs. Lower NMSE is better.','']
    for arm in ARMS:
        vals=[records[f'{arm}_{s}_latest_original'] for s in CONFIG['seeds']]
        lines.append(f"{arm}: macro {np.mean([v['macro_db'] for v in vals]):.4f}dB; NMSE>1 rate {np.mean([v['severe_rate'] for v in vals]):.3%}")
    for key,value in changes.items():lines.append(f"{key}: improvement versus holdout {value['gain_db']:.4f}dB; seed gains {value['seed_gains_db']}")
    lines+=['','Positive improvement means lower error; negative means worse.','best is secondary common21/9validation-selected; latest is the predeclared primary.',
        'Inflated-noise evaluations are diagnostics; compare methods only within the same mode.',
        'No OptShrink or recurrent comparison was rerun; this tests training alone.',
        'No historical manuscript or experiment changed. Phase diagnostic is separate under phase/.',
        'A positive small-screen result is not publication confirmation or proof of novelty.']
    (ROOT/'RESULTS.txt').write_text('\n'.join(lines)+'\n')

def launch():
    ROOT.mkdir(parents=True,exist_ok=True)
    with (ROOT/'launch.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        p=ROOT/'pid.json'
        if p.exists():
            try:os.kill(json.loads(p.read_text())['pid'],0)
            except ProcessLookupError:pass
            else:print('Existing process alive; no duplicate.');return
        if (ROOT/'status.json').exists() and json.loads((ROOT/'status.json').read_text()).get('state')=='complete':
            print('Already complete.');return
        with (ROOT/'run.log').open('a') as log:
            child=subprocess.Popen([sys.executable,'-m','rician_pilot_screen.run','--run'],stdout=log,
                stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
        write(p,dict(pid=child.pid,started_unix=time.time()))
        print(json.dumps(dict(pid=child.pid,log=str(ROOT/'run.log'))),flush=True)

def run():
    ROOT.mkdir(parents=True,exist_ok=True);torch.set_num_threads(1);torch.set_num_interop_threads(1)
    guard=None;start=time.monotonic()
    with (ROOT/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            write(ROOT/'pid.json',dict(pid=os.getpid(),started_unix=time.time()))
            if Path('/usr/bin/caffeinate').exists():guard=subprocess.Popen(['/usr/bin/caffeinate','-i','-w',str(os.getpid())])
            freeze()
            write(ROOT/'status.json',dict(state='preparing',pid=os.getpid(),updated_unix=time.time()))
            deadline=start+CONFIG['max_seconds']
            prepare('train',CONFIG['train_count'],CONFIG['train_seed'],CELLS,deadline=deadline)
            prepare('val',CONFIG['val_count'],CONFIG['val_seed'],CELLS,deadline=deadline)
            train(deadline);freeze();evaluate(deadline);freeze()
            write(ROOT/'status.json',dict(state='complete',seconds=time.monotonic()-start,updated_unix=time.time()))
        except BaseException:
            write(ROOT/'status.json',dict(state='failed',error=traceback.format_exc(),seconds=time.monotonic()-start,updated_unix=time.time()))
            raise
        finally:
            if guard is not None and guard.poll() is None:guard.terminate()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--launch',action='store_true');p.add_argument('--run',action='store_true')
    args=p.parse_args()
    if args.launch:launch()
    elif args.run:run()
    else:p.error('use --launch or --run')
