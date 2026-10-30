"""One locked campaign, three one-thread workers, frozen paired test and audit."""
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
import traceback
import unittest
from .protocol import ROOT,CONFIG,write,initialize,freeze_test

def alive(pid):
    try:os.kill(int(pid),0);return True
    except ProcessLookupError:return False

def launch():
    ROOT.mkdir(parents=True,exist_ok=True)
    with (ROOT/'launch.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        paths=[ROOT/'pid.json']+list((ROOT/'checkpoints').glob('seed_*/status.json'))
        for path in paths:
            if path.exists():
                state=json.loads(path.read_text())
                if state.get('pid') and state.get('state') not in ('complete','failed','time_limit') and alive(state['pid']):
                    print('Existing campaign/worker is alive; no duplicate launch.');return
        status=ROOT/'status.json'
        if status.exists() and json.loads(status.read_text()).get('state')=='complete':
            print('Already complete.');return
        with (ROOT/'run.log').open('a') as log:
            child=subprocess.Popen([sys.executable,'-m','rician_full_comparison.run','--run'],stdout=log,
                stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
        write(ROOT/'pid.json',dict(pid=child.pid,started_unix=time.time()))
        print(json.dumps(dict(pid=child.pid,log=str(ROOT/'run.log'))),flush=True)

def run():
    ROOT.mkdir(parents=True,exist_ok=True);children=[];streams=[];guard=None
    with (ROOT/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        start=time.monotonic();last=0.
        def check(state,**info):
            nonlocal last
            if time.monotonic()-start>CONFIG['max_seconds']:raise TimeoutError('four-hour cap; completed progress retained')
            if time.monotonic()-last>=2:
                write(ROOT/'status.json',dict(state=state,pid=os.getpid(),seconds=time.monotonic()-start,updated_unix=time.time(),**info));last=time.monotonic()
        write(ROOT/'pid.json',dict(pid=os.getpid(),started_unix=time.time()))
        try:
            if Path('/usr/bin/caffeinate').exists():guard=subprocess.Popen(['/usr/bin/caffeinate','-i','-w',str(os.getpid())])
            check('preflight')
            suite=unittest.defaultTestLoader.loadTestsFromName('rician_pilot_screen.test_noise')
            if not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful():raise RuntimeError('Rician view checks failed')
            replay=ROOT/'baseline_replay.json'
            if not replay.exists():
                from .checks import baseline_replay
                baseline_replay()
            if not json.loads(replay.read_text()).get('passed'):raise RuntimeError('baseline replay missing/failed')
            initialize()
            for seed in CONFIG['seeds']:
                stream=(ROOT/f'train_{seed}.log').open('a');streams.append(stream)
                children.append(subprocess.Popen([sys.executable,'-m','rician_full_comparison.train','--seed',str(seed)],stdout=stream,stderr=subprocess.STDOUT))
            while any(child.poll() is None for child in children):
                if any(child.poll() not in (None,0) for child in children):raise RuntimeError('training worker failed; inspect per-seed logs')
                workers={}
                for seed in CONFIG['seeds']:
                    path=ROOT/'checkpoints'/f'seed_{seed}'/'status.json'
                    if path.exists():workers[str(seed)]=json.loads(path.read_text())
                check('training',workers=workers,total_epochs_per_seed=CONFIG['epochs']);time.sleep(5)
            if any(child.returncode!=0 for child in children):raise RuntimeError('training worker failed')
            initialize();freeze_test()
            import torch
            torch.set_num_threads(1)
            from .evaluate import run as evaluate
            from .analyze import analyze
            evaluate(check);analyze(check);initialize();freeze_test()
            write(ROOT/'status.json',dict(state='complete',seconds=time.monotonic()-start,updated_unix=time.time()))
        except BaseException:
            for child in children:
                if child.poll() is None:child.terminate()
            for child in children:
                if child.poll() is None:
                    try:child.wait(timeout=5)
                    except subprocess.TimeoutExpired:child.kill()
            write(ROOT/'status.json',dict(state='failed',error=traceback.format_exc(),seconds=time.monotonic()-start,updated_unix=time.time()))
            raise
        finally:
            for stream in streams:stream.close()
            if guard is not None and guard.poll() is None:guard.terminate()

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--launch',action='store_true');parser.add_argument('--run',action='store_true')
    args=parser.parse_args()
    if args.launch:launch()
    elif args.run:run()
    else:parser.error('use --launch or --run')
