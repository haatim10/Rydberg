"""Small CPU-only queue with exclusive locks and complete-case resumption."""
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
from .protocol import ROOT, CONFIG, initialize, tune, write, sha


def threads():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)


def prepare():
    threads()
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromName('lrhd_screen.tests'))
    if not result.wasSuccessful():
        raise RuntimeError('LRHD tests failed')
    initialize()
    write(ROOT/'preflight.json', dict(passed=True, tests=result.testsRun,
          manifest_sha256=sha(ROOT/'manifest.json'), checked_unix=time.time()))
    print('Preflight passed; no scientific queue launched.', flush=True)


def launch():
    ROOT.mkdir(parents=True, exist_ok=True)
    with (ROOT/'launch.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = json.loads((ROOT/'status.json').read_text()) if (ROOT/'status.json').exists() else {}
        if state.get('state') == 'complete':
            print('Already complete; no duplicate run.'); return
        if state.get('state') == 'failed':
            raise RuntimeError('previous failure needs inspection')
        pidfile = ROOT/'pid.json'
        if pidfile.exists():
            try:
                os.kill(int(json.loads(pidfile.read_text())['pid']), 0)
                print('Existing process alive; no duplicate launch.'); return
            except ProcessLookupError:
                pass
        initialize()
        preflight = json.loads((ROOT/'preflight.json').read_text())
        if not preflight['passed'] or preflight['manifest_sha256'] != sha(ROOT/'manifest.json'):
            raise RuntimeError('missing/stale preflight')
        with (ROOT/'run.log').open('a') as log:
            child = subprocess.Popen([sys.executable, '-m', 'lrhd_screen.run', '--run'],
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        write(pidfile, dict(pid=child.pid, started_unix=time.time()))
        print(json.dumps(dict(pid=child.pid, log=str(ROOT/'run.log'))), flush=True)


def run():
    ROOT.mkdir(parents=True, exist_ok=True)
    with (ROOT/'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        started, last, guard = time.monotonic(), 0., None
        def check(state, **info):
            nonlocal last
            seconds = time.monotonic()-started
            if seconds > CONFIG['max_seconds']:
                raise TimeoutError('two-hour safety cap; complete cases retained')
            if time.monotonic()-last > 1:
                write(ROOT/'status.json', dict(state=state, pid=os.getpid(), seconds=seconds,
                    updated_unix=time.time(), **info)); last=time.monotonic()
        write(ROOT/'pid.json', dict(pid=os.getpid(), started_unix=time.time()))
        try:
            threads(); initialize()
            if Path('/usr/bin/caffeinate').exists():
                guard = subprocess.Popen(['/usr/bin/caffeinate', '-i', '-w', str(os.getpid())])
            tune(check)
            from .evaluate import run as evaluate
            from .analyze import run as analyze
            evaluate(check)
            analyze(check)
            initialize()
            write(ROOT/'status.json', dict(state='complete', seconds=time.monotonic()-started,
                updated_unix=time.time()))
        except BaseException as exc:
            write(ROOT/'status.json', dict(state='interrupted' if isinstance(exc, (TimeoutError, KeyboardInterrupt)) else 'failed',
                error=traceback.format_exc(), seconds=time.monotonic()-started, updated_unix=time.time()))
            raise
        finally:
            if guard is not None and guard.poll() is None:
                guard.terminate()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    for name in ('prepare', 'launch', 'run'):
        group.add_argument('--'+name, action='store_true')
    args = parser.parse_args()
    if args.prepare: prepare()
    elif args.launch: launch()
    else: run()
