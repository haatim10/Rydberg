"""Launch one resumable mixed-training screen; never expand the experiment."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[key] = '1'
import argparse
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback
from .protocol import ROOT, CONFIG, write, sha, initialize, freeze_selection


def alive(pid):
    try:
        os.kill(int(pid), 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def launch():
    ROOT.mkdir(parents=True, exist_ok=True)
    with (ROOT / 'launch.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        path = ROOT / 'status.json'
        state = json.loads(path.read_text()) if path.exists() else {}
        if state.get('state') == 'complete':
            print('Already complete; no launch.')
            return
        if state.get('state') == 'failed':
            raise RuntimeError('prior failure needs inspection; refusing automatic restart')
        pid_path = ROOT / 'pid.json'
        if pid_path.exists() and alive(json.loads(pid_path.read_text())['pid']):
            print('Existing process is alive; no duplicate launch.')
            return
        initialize()
        preflight = ROOT / 'preflight.json'
        if not preflight.exists() or not json.loads(preflight.read_text())['passed']:
            raise RuntimeError('run and pass preflight checks before launch')
        if json.loads(preflight.read_text())['manifest_sha256'] != sha(ROOT / 'manifest.json'):
            raise RuntimeError('preflight was run with different frozen sources')
        with (ROOT / 'run.log').open('a') as log:
            child = subprocess.Popen([sys.executable, '-m', 'rician_mixed_screen.run', '--run'],
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True)
        write(pid_path, dict(pid=child.pid, started_unix=time.time()))
        print(json.dumps(dict(pid=child.pid, log=str(ROOT / 'run.log'))), flush=True)


def run():
    ROOT.mkdir(parents=True, exist_ok=True)
    guard = None
    with (ROOT / 'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        start = time.monotonic()
        last = 0.
        def check(state, **info):
            nonlocal last
            seconds = time.monotonic() - start
            if seconds > CONFIG['max_seconds']:
                raise TimeoutError('two-hour safety cap; complete progress retained')
            if time.monotonic() - last >= 2:
                write(ROOT / 'status.json', dict(state=state, pid=os.getpid(), seconds=seconds,
                      updated_unix=time.time(), **info))
                last = time.monotonic()
        write(ROOT / 'pid.json', dict(pid=os.getpid(), started_unix=time.time()))
        try:
            initialize()
            if Path('/usr/bin/caffeinate').exists():
                guard = subprocess.Popen(['/usr/bin/caffeinate', '-i', '-w', str(os.getpid())])
            check('training', seed=CONFIG['seed'], total_epochs=CONFIG['epochs'])
            from .train import run as train
            if train(CONFIG['seed']) is False:
                raise TimeoutError('training time limit; last complete epoch retained')
            from .checks import audit_training
            audit_training()
            freeze_selection()
            from .evaluate import run as evaluate, analyze
            check('evaluating')
            evaluate(check)
            analyze(check)
            initialize()
            freeze_selection()
            write(ROOT / 'status.json', dict(state='complete', seconds=time.monotonic()-start,
                  updated_unix=time.time()))
        except BaseException as exc:
            resumable = isinstance(exc, (TimeoutError, KeyboardInterrupt)) or type(exc).__name__ == 'WorkerTimeLimit'
            write(ROOT / 'status.json', dict(state='interrupted' if resumable else 'failed',
                  error=traceback.format_exc(), seconds=time.monotonic()-start,
                  updated_unix=time.time()))
            raise
        finally:
            if guard is not None and guard.poll() is None:
                guard.terminate()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--launch', action='store_true')
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    if args.launch:
        launch()
    elif args.run:
        run()
    else:
        parser.error('use --launch or --run')
