"""Disjoint-reference execution only; all scientific definitions are in study."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

REPO=Path(__file__).resolve().parents[1]
OUT=REPO/'results/tcom_runtime_repair_20261020'


def write(path,record):
    temporary=path.with_suffix('.tmp.json')
    temporary.write_text(json.dumps(record,indent=2)+'\n')
    temporary.replace(path)


def run(module,*args):
    subprocess.run([sys.executable,'-u','-m',f'tcom_runtime_repair_20261020.{module}',*args],cwd=REPO,check=True)


def worker(reference):
    run('study','test','--reference',reference)
    if reference!='5':
        run('study','boundary','--reference',reference)
    run('ber','--reference',reference)


def schedule():
    from . import study
    policy,digest=study.check_selection()
    assert (OUT/'timing_window1.json').exists()
    records=[]
    jobs=[]
    started=time.time()
    for ref in ('5','12','20'):
        logfile=OUT/f'execution_rsr{ref}.log'
        stream=logfile.open('a',buffering=1)
        process=subprocess.Popen([sys.executable,'-u','-m','tcom_runtime_repair_20261020.execution',
                                  '--reference',ref],cwd=REPO,stdout=stream,stderr=subprocess.STDOUT)
        jobs.append((ref,process,stream))
        records.append(dict(reference=ref,pid=process.pid,log=logfile.name))
    while any(p.poll() is None for _,p,_ in jobs):
        counts={label:len([p for p in (OUT/folder).glob('*.npz') if '.tmp.' not in p.name])
                for label,folder in [('test_cells','test/cells'),('boundary_cells','boundary/cells'),('ber_cells','ber/cells')]}
        state=dict(phase='fresh_accuracy',selection_sha256=digest,started_unix=started,updated_unix=time.time(),
                   jobs=[dict(**r,exit_code=p.poll()) for r,(_,p,_) in zip(records,jobs)],**counts)
        write(OUT/'status.json',state)
        print(json.dumps({key:state[key] for key in ('phase','test_cells','boundary_cells','ber_cells')}),flush=True)
        time.sleep(5)
    failures=[]
    for ref,process,stream in jobs:
        stream.close()
        if process.returncode:
            failures.append(dict(reference=ref,exit_code=process.returncode))
    if failures:
        write(OUT/'status.json',dict(phase='execution_failure',failures=failures,selection_sha256=digest))
        raise RuntimeError(f'Failed reference workers: {failures}')
    write(OUT/'execution_manifest.json',dict(passed=True,selection_sha256=digest,started_unix=started,
          completed_unix=time.time(),jobs=records,all_owned_workers_exited=True,disjoint_reference_ownership=True))
    write(OUT/'status.json',dict(phase='accuracy_generation_complete',test_cells=153,boundary_cells=54,ber_cells=153,
          selection_sha256=digest,updated_unix=time.time()))
    print('All fresh channel and BER workers exited successfully.',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--reference',choices=('5','12','20'))
    args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    worker(args.reference) if args.reference else schedule()
