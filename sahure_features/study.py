"""Bounded, locked local runner. No automatic expansion of the experiment."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key]='1'
import argparse
import fcntl
import json
import platform
from pathlib import Path
import subprocess
import sys
import time
import traceback
import numpy as np
import torch
from sahure.data import split_observations
from sahure.train import heldout_loss
from .model import FeatureEstimator
from .protocol import ROOT,CONFIG,jobs,initialize,source_records,write_json

def preflight():
    from .train import data
    train,a,b,_=data();measurements={}
    ca,cb=split_observations(train.batch(slice(0,32)),987654)
    for arm in CONFIG['arms']:
        torch.manual_seed(345);model=FeatureEstimator(arm)
        opt=torch.optim.Adam(model.parameters(),lr=CONFIG['lr']);times=[]
        for repeat in range(4):
            tick=time.perf_counter();opt.zero_grad(set_to_none=True)
            loss=heldout_loss(model,ca,cb);loss.backward()
            assert torch.isfinite(loss) and all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
            opt.step()
            if repeat: times.append(time.perf_counter()-tick)
        with torch.no_grad():
            tick=time.perf_counter();value=heldout_loss(model,a.batch(slice(0,32)),b.batch(slice(0,32)))
            validation_batch=time.perf_counter()-tick
        measurements[arm]=dict(train_batch_seconds=float(np.median(times)),validation_batch_seconds=validation_batch,
            loss=float(value),parameters=sum(p.numel() for p in model.parameters()))
    seconds=sum(v['train_batch_seconds']*CONFIG['train_cases']/32+v['validation_batch_seconds']*CONFIG['val_cases']/32
                for v in measurements.values())*len(CONFIG['seeds'])*CONFIG['epochs']
    result=dict(measurements=measurements,estimated_training_validation_seconds=seconds,
        estimated_total_seconds_with_allowance=seconds*1.25+300,maximum_seconds=CONFIG['maximum_seconds'],
        device=platform.platform(),torch=torch.__version__,cpu_threads=torch.get_num_threads(),sources=source_records())
    write_json(ROOT/'preflight.json',result);print(json.dumps(result,indent=2))

def launch():
    ROOT.mkdir(parents=True,exist_ok=True)
    with (ROOT/'run.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            print('Existing feature study owns the lock; no duplicate launched.');return
        status=ROOT/'status.json'
        if status.exists() and json.loads(status.read_text()).get('state') in ('complete','failed','budget_reached'):
            raise RuntimeError('Study is complete, failed or at budget: review its state before a new launch')
        pre=json.loads((ROOT/'preflight.json').read_text())
        if pre['sources']!=source_records():raise RuntimeError('Source changed since preflight')
        initialize()
    with (ROOT/'run.log').open('a',buffering=1) as log:
        child=subprocess.Popen([sys.executable,'-u','-m','sahure_features.study','run'],stdin=subprocess.DEVNULL,
            stdout=log,stderr=subprocess.STDOUT,start_new_session=True,cwd=Path.cwd())
    write_json(ROOT/'pid.json',dict(pid=child.pid,launched_unix=time.time(),command='sahure_features.study run'))
    print(json.dumps(dict(pid=child.pid,status=str(ROOT/'status.json'),log=str(ROOT/'run.log'))))

def run():
    from . import train,evaluate
    ROOT.mkdir(parents=True,exist_ok=True)
    with (ROOT/'run.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            print('Duplicate worker refused.',flush=True);return
        tick=time.monotonic()
        budget=ROOT/'budget.json'
        used=json.loads(budget.read_text())['used_seconds'] if budget.exists() else 0.
        remaining=max(0.,CONFIG['maximum_seconds']-used)
        deadline=tick+remaining
        try:
            initialize()
            write_json(ROOT/'status.json',dict(state='starting',pid=os.getpid(),maximum_seconds=remaining,updated_unix=time.time()))
            trained=train.run(deadline)
            evaluated=evaluate.run(deadline) if trained and time.monotonic()<deadline else False
            if trained and evaluated and time.monotonic()<deadline:
                evaluate.audit();evaluate.summarize()
                state='complete'
            else:state='budget_reached'
            write_json(ROOT/'status.json',dict(state=state,completed_epochs=train.completed_epochs(),
                total_epochs=len(jobs())*CONFIG['epochs'],trained=trained,evaluated=evaluated,
                seconds_this_run=time.monotonic()-tick,updated_unix=time.time()))
        except Exception as exc:
            write_json(ROOT/'status.json',dict(state='failed',error=repr(exc),traceback=traceback.format_exc(),updated_unix=time.time()))
            raise
        finally:write_json(budget,dict(used_seconds=used+time.monotonic()-tick,maximum_seconds=CONFIG['maximum_seconds']))

def status():
    path=ROOT/'status.json'
    print(path.read_text() if path.exists() else 'Not launched')
    with (ROOT/'run.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);print('Process lock: not held')
        except BlockingIOError:print('Process lock: held by a worker')

if __name__=='__main__':
    torch.set_num_threads(1)
    p=argparse.ArgumentParser();p.add_argument('command',choices=['preflight','launch','run','status']);args=p.parse_args()
    globals()[args.command]()
