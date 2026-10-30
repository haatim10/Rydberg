"""Single-controller, budget-capped, resumable submission checks."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key]='1'
import argparse
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import torch
from sahure.data import load_observations,split_observations
from sahure.train import heldout_loss
from sahure.model import AdaptiveHankelEstimator
from sahure_expansion.protocol import write_json,sha
from .protocol import ROOT,OLD,CONFIG,jobs,cells,initialize
from .train import model_config,channel_loss,checked_step

def benchmark():
    torch.set_num_threads(1)
    data=load_observations(OLD/'data/train_observations.npz').batch(slice(0,128))
    a,b=split_observations(data,9101+10000)
    with np.load(OLD/'data/train_truth.npz') as f:truth=torch.tensor(f['g'][:128].copy(),dtype=torch.complex64)
    measurements={}
    for arm in ['supervised','final_only','gate_one','adaptive','global','none']:
        torch.manual_seed(9101);model=AdaptiveHankelEstimator(model_config(arm))
        opt=torch.optim.Adam(model.parameters(),lr=.001);durations=[]
        for i in range(4):
            ids=slice(i*32,(i+1)*32);tick=time.perf_counter();opt.zero_grad(set_to_none=True)
            loss=channel_loss(model(*a.batch(ids).args()),truth[ids]).mean() if arm=='supervised' else heldout_loss(model,a.batch(ids),b.batch(ids))
            checked_step(loss,model,opt);durations.append(time.perf_counter()-tick)
        # Include measured forward-validation cost; the full epoch validates 2,048 worlds.
        tick=time.perf_counter()
        with torch.no_grad():
            for i in range(4):heldout_loss(model,a.batch(slice(i*32,(i+1)*32)),b.batch(slice(i*32,(i+1)*32)))
        vb=(time.perf_counter()-tick)/4
        train_batch=float(np.median(durations[1:]))
        epoch=train_batch*512+vb*64*(2 if arm=='supervised' else 1)
        measurements[arm]=dict(batch_seconds=train_batch,validation_batch_seconds=vb,
            epoch_seconds=epoch,parameters=sum(p.numel() for p in model.parameters()))
    serial=sum(measurements[j['arm']]['epoch_seconds']*20 for j in jobs())
    result=dict(measurements=measurements,estimated_serial_training_hours=serial/3600,
        estimated_two_worker_training_hours=serial/7200,
        caveat='Small-batch estimate before thermal effects/contention; add generation, testing and audit time.',
        hardware=dict(platform=sys.platform,python=sys.version,torch=torch.__version__),timestamp=time.time())
    write_json(ROOT/'benchmark.json',result);print(json.dumps(result),flush=True)

def generate_training():
    import sahure_expansion.data as generator
    previous=generator.ROOT
    try:
        for rep in CONFIG['replicates']:
            generator.ROOT=ROOT/f'rep{rep}'
            for split,count,seed in [('train',16384,2026100001+rep*1000),('val',2048,2026100002+rep*1000)]:
                print(f'Generating replica {rep} {split}: {count}',flush=True)
                generator.generate_set(split,count,seed,generator.allocation(count,seed))
    finally:generator.ROOT=previous
    write_json(ROOT/'replicate_data_hashes.json',{str(p):sha(p) for rep in CONFIG['replicates']
        for p in (ROOT/f'rep{rep}'/'data').iterdir() if p.is_file()})

def status(phase,**extra):
    histories={}
    for job in jobs():
        path=ROOT/'checkpoints'/job['id']/'status.json'
        histories[job['id']]=json.loads(path.read_text()) if path.exists() else dict(state='queued',epoch=0)
    state=dict(state='running',phase=phase,pid=os.getpid(),updated_unix=time.time(),jobs=histories)
    state.update(extra)
    write_json(ROOT/'status.json',state)
    lines=['# Paper 2 submission checks','',f'State: {state["state"]}. Phase: {phase}.',
        f'Completed epochs: {sum(v.get("epoch",0) for v in histories.values())}/300.',
        '', '| Comparison | Completed epochs | State |','|---|---:|---|']
    for name,row in histories.items():lines.append(f'| {name} | {row.get("epoch",0)}/20 | {row["state"]} |')
    lines+=['','Latest worker messages are saved in the matching job logs.',
        'No inference results are final before frozen selection, fresh testing and audit.',
        'This run is local: system sleep pauses work and shutdown stops it. Completed epochs are resumable.']
    (ROOT/'STATUS.md').write_text('\n'.join(lines)+'\n')

def execute(commands,phase,deadline,workers=2):
    pending=list(commands);active=[]
    try:
        while pending or active:
            if time.monotonic()>=deadline:raise TimeoutError('Twelve-hour compute cap reached; saved progress retained')
            while pending and len(active)<workers:
                name,command=pending.pop(0)
                if 'sahure_checks.train' in command:command=command+['--seconds',str(max(1,deadline-time.monotonic()-15))]
                log=(ROOT/f'{name}.log').open('a')
                process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
                active.append((name,process,log));print(f'Started {name}',flush=True)
            for entry in active.copy():
                name,process,log=entry;code=process.poll()
                if code is None:continue
                log.close();active.remove(entry)
                if code==75:raise TimeoutError(f'{name}: compute cap reached')
                if code:raise RuntimeError(f'{name} exited {code}; inspect {name}.log')
                print(f'Completed {name}',flush=True)
            status(phase,active=[x[0] for x in active],pending=len(pending))
            if active:time.sleep(5)
    finally:
        for _,process,log in active:
            process.terminate()
            try:process.wait(timeout=10)
            except subprocess.TimeoutExpired:process.kill();process.wait()
            log.close()

def campaign():
    ROOT.mkdir(parents=True,exist_ok=True)
    with (ROOT/'controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        initialize();start=time.monotonic();deadline=start+CONFIG['maximum_hours']*3600
        write_json(ROOT/'pid.json',dict(pid=os.getpid(),started_unix=time.time(),command='python -m sahure_checks.run campaign'))
        py=[sys.executable,'-u','-m']
        try:
            # Mixed-data arms start first; replication generation cannot delay the first training epoch.
            matched=[j for j in jobs() if j['cohort']=='matched']
            execute([(j['id'],py+['sahure_checks.train','--job',j['id']]) for j in matched],
                    'matched_controls',deadline)
            execute([('generate_training',py+['sahure_checks.run','generate'])],'independent_data',deadline,1)
            execute([(j['id'],py+['sahure_checks.train','--job',j['id']]) for j in jobs() if j['cohort']!='matched'],
                    'independent_replication',deadline)
            for phase in ['freeze','prepare']:
                execute([(phase,py+['sahure_checks.evaluate',phase])],phase,deadline,1)
            execute([(f'eval_bank{rep}_{c["id"]}',py+['sahure_checks.evaluate','cell','--rep',str(rep),'--cell',c['id']])
                for rep in CONFIG['replicates'] for c in cells()],'fresh_test_evaluation',deadline)
            for phase in ['report','timing','audit']:
                execute([(phase,py+['sahure_checks.evaluate',phase])],phase,deadline,1)
            status('complete',state='complete',elapsed_seconds=time.monotonic()-start)
        except TimeoutError as exc:
            status('budget_stop',state='checkpointed',error=str(exc),elapsed_seconds=time.monotonic()-start)
        except BaseException as exc:
            status('failed',state='failed',error=repr(exc),elapsed_seconds=time.monotonic()-start);raise

def launch():
    ROOT.mkdir(parents=True,exist_ok=True)
    with (ROOT/'tests.log').open('w') as stream:
        subprocess.run([sys.executable,'-m','unittest','sahure.test_numerics','sahure.test_pipeline',
            'sahure_expansion.test_expansion','sahure_checks.test_checks'],stdout=stream,stderr=subprocess.STDOUT,check=True)
    if not (ROOT/'benchmark.json').exists():raise RuntimeError('Run the device benchmark before launch')
    initialize()
    lock=(ROOT/'controller.lock').open('a')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise SystemExit('Existing controller holds lock; not launching a duplicate')
    lock.close()
    if (ROOT/'status.json').exists() and json.loads((ROOT/'status.json').read_text())['state']=='complete':
        raise SystemExit('Checks already complete')
    with (ROOT/'run.log').open('a') as log:
        process=subprocess.Popen([sys.executable,'-u','-m','sahure_checks.run','campaign'],
            stdout=log,stderr=subprocess.STDOUT,start_new_session=True,stdin=subprocess.DEVNULL)
    time.sleep(1)
    if process.poll() is not None:raise RuntimeError('Controller exited during launch; inspect run.log')
    # Keep local CPU work awake while this specific process exists, if supported.
    if Path('/usr/bin/caffeinate').exists():
        subprocess.Popen(['/usr/bin/caffeinate','-i','-w',str(process.pid)],
            stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session=True)
    print(json.dumps(dict(pid=process.pid,log=str(ROOT/'run.log'),status=str(ROOT/'STATUS.md'))),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['benchmark','generate','campaign','launch']);a=p.parse_args()
    {'benchmark':benchmark,'generate':generate_training,'campaign':campaign,'launch':launch}[a.phase]()
