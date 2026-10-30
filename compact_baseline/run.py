"""One locked local campaign, resumable at complete epochs/test cells."""
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
    try: os.kill(int(pid),0);return True
    except ProcessLookupError: return False


def launch():
    ROOT.mkdir(parents=True,exist_ok=True)
    p=ROOT/'pid.json'
    if p.exists() and alive(json.loads(p.read_text())['pid']):
        print('Existing campaign is alive; no duplicate launch.');return
    with (ROOT/'run.log').open('a') as stream:
        child=subprocess.Popen([sys.executable,'-m','compact_baseline.run','--run'],stdout=stream,stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,start_new_session=True)
    write(p,dict(pid=child.pid,started_unix=time.time(),command=[sys.executable,'-m','compact_baseline.run','--run']))
    print(json.dumps(dict(pid=child.pid,log=str(ROOT/'run.log'))),flush=True)


def run():
    ROOT.mkdir(parents=True,exist_ok=True)
    children=[];streams=[];sleep_guard=None
    with (ROOT/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        start=time.monotonic();last=0.
        def check(state,**kw):
            nonlocal last
            if time.monotonic()-start>CONFIG['max_seconds']:
                raise TimeoutError('three-hour compute cap reached; progress saved')
            if time.monotonic()-last>2:
                write(ROOT/'status.json',dict(state=state,pid=os.getpid(),seconds=time.monotonic()-start,updated_unix=time.time(),**kw));last=time.monotonic()
        write(ROOT/'pid.json',dict(pid=os.getpid(),started_unix=time.time()))
        try:
            if Path('/usr/bin/caffeinate').exists():
                sleep_guard=subprocess.Popen(['/usr/bin/caffeinate','-i','-w',str(os.getpid())])
            suite=unittest.defaultTestLoader.loadTestsFromName('compact_baseline.test_checks')
            if not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful(): raise RuntimeError('numerical checks failed')
            initialize()
            for seed in CONFIG['seeds']:
                stream=(ROOT/f'train_{seed}.log').open('a');streams.append(stream)
                children.append(subprocess.Popen([sys.executable,'-m','compact_baseline.train','--seed',str(seed)],stdout=stream,stderr=subprocess.STDOUT))
            while any(child.poll() is None for child in children):
                failed=[child.returncode for child in children if child.poll() not in (None,0)]
                if failed: raise RuntimeError('a training worker failed; see seed log')
                epochs={}
                for seed in CONFIG['seeds']:
                    path=ROOT/'checkpoints'/f'seed_{seed}'/'status.json'
                    if path.exists(): epochs[str(seed)]=json.loads(path.read_text()).get('completed_epochs',json.loads(path.read_text()).get('epochs',0))
                check('training',completed_epochs=epochs,total_epochs_per_seed=20)
                time.sleep(5)
            if any(child.returncode!=0 for child in children): raise RuntimeError('training worker failed')
            initialize();freeze_test()
            import torch
            torch.set_num_threads(1)
            from .evaluate import run as evaluate
            from .analyze import analyze
            evaluate(check);analyze(check);initialize();freeze_test()
            write(ROOT/'status.json',dict(state='complete',seconds=time.monotonic()-start,updated_unix=time.time()))
        except BaseException:
            for child in children:
                if child.poll() is None: child.terminate()
            for child in children:
                if child.poll() is None:
                    try: child.wait(timeout=5)
                    except subprocess.TimeoutExpired: child.kill()
            write(ROOT/'status.json',dict(state='failed',error=traceback.format_exc(),seconds=time.monotonic()-start,updated_unix=time.time()))
            raise
        finally:
            for stream in streams: stream.close()
            if sleep_guard is not None and sleep_guard.poll() is None: sleep_guard.terminate()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--launch',action='store_true');parser.add_argument('--run',action='store_true');args=parser.parse_args()
    if args.launch: launch()
    elif args.run: run()
    else: parser.error('use --launch or --run')
