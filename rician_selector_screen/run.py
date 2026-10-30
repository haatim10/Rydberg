"""Small locked routing study; no training or automatic follow-on experiment."""
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
import unittest
import torch
from .protocol import ROOT, CONFIG, write, sha, initialize, freeze_selection


def prepare():
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    suite = unittest.defaultTestLoader.loadTestsFromName('rician_selector_screen.tests')
    tests = unittest.TextTestRunner(verbosity=2).run(suite)
    if not tests.wasSuccessful():
        raise RuntimeError('selector tests failed')
    tick = time.monotonic()
    initialize()
    frozen = freeze_selection()
    write(ROOT / 'preflight.json', dict(passed=True, tests=tests.testsRun,
          manifest_sha256=sha(ROOT / 'manifest.json'),
          selection_sha256=sha(ROOT / 'frozen_selection.json'),
          development_seconds=time.monotonic()-tick))
    print(json.dumps(dict(preflight='passed', tests=tests.testsRun,
                         threshold_db=frozen['threshold_db'])), flush=True)


def launch():
    ROOT.mkdir(parents=True, exist_ok=True)
    with (ROOT / 'launch.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state_path, pid_path = ROOT / 'status.json', ROOT / 'pid.json'
        state = json.loads(state_path.read_text()) if state_path.exists() else {}
        if state.get('state') == 'complete':
            print('Already complete; no duplicate experiment.')
            return
        if state.get('state') == 'failed':
            raise RuntimeError('previous failure requires inspection, not automatic restart')
        if pid_path.exists():
            try:
                os.kill(int(json.loads(pid_path.read_text())['pid']), 0)
                print('Existing process alive; no duplicate launch.')
                return
            except ProcessLookupError:
                pass
        initialize()
        preflight = json.loads((ROOT / 'preflight.json').read_text())
        if (not preflight['passed'] or preflight['manifest_sha256'] != sha(ROOT / 'manifest.json')
                or preflight['selection_sha256'] != sha(ROOT / 'frozen_selection.json')):
            raise RuntimeError('missing or stale preflight')
        with (ROOT / 'run.log').open('a') as log:
            child = subprocess.Popen([sys.executable, '-m', 'rician_selector_screen.run', '--run'],
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        write(pid_path, dict(pid=child.pid, started_unix=time.time()))
        print(json.dumps(dict(pid=child.pid, log=str(ROOT / 'run.log'))), flush=True)


def run():
    ROOT.mkdir(parents=True, exist_ok=True)
    with (ROOT / 'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        started, last, guard = time.monotonic(), 0., None
        def check(state, **info):
            nonlocal last
            seconds = time.monotonic()-started
            if seconds > CONFIG['max_seconds']:
                raise TimeoutError('one-hour safety cap; completed cells retained')
            if time.monotonic()-last >= 1:
                write(ROOT / 'status.json', dict(state=state, pid=os.getpid(),
                      seconds=seconds, updated_unix=time.time(), **info))
                last = time.monotonic()
        write(ROOT / 'pid.json', dict(pid=os.getpid(), started_unix=time.time()))
        try:
            torch.set_num_threads(1)
            torch.set_num_interop_threads(1)
            initialize()
            if Path('/usr/bin/caffeinate').exists():
                guard = subprocess.Popen(['/usr/bin/caffeinate', '-i', '-w', str(os.getpid())])
            from .evaluate import run as evaluate, analyze
            check('testing')
            evaluate(check)
            analyze(check)
            initialize()
            write(ROOT / 'status.json', dict(state='complete', seconds=time.monotonic()-started,
                  updated_unix=time.time()))
        except BaseException as exc:
            write(ROOT / 'status.json', dict(state='interrupted' if isinstance(exc, (TimeoutError, KeyboardInterrupt)) else 'failed',
                  error=traceback.format_exc(), seconds=time.monotonic()-started,
                  updated_unix=time.time()))
            raise
        finally:
            if guard is not None and guard.poll() is None:
                guard.terminate()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--launch', action='store_true')
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    if args.prepare:
        prepare()
    elif args.launch:
        launch()
    elif args.run:
        run()
    else:
        parser.error('use --prepare, --launch or --run')
