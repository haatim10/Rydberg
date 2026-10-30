"""Disjoint reference shards; never run runtime measurement in these workers."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from . import study
from tcom_followups_20261017.common import write_json


def main():
    selection,digest=study.check_selection()
    processes=[];started=time.time()
    for ref in study.RSRS:
        logpath=study.OUT/f'execution_rsr{ref:g}.log'
        stream=logpath.open('a')
        p=subprocess.Popen([sys.executable,'-u','-m',__package__+'.execution','--reference',str(ref)],stdout=stream,stderr=subprocess.STDOUT)
        processes.append((ref,p,stream,logpath))
    try:
        while any(p.poll() is None for _,p,_,_ in processes):
            count=lambda p:sum(not f.name.endswith('.tmp.npz') for f in p.glob('*.npz'))
            state=dict(phase='accuracy_generation',test_cells=count(study.OUT/'test/cells'),
                       ber_cells=count(study.OUT/'ber/cells'),selection_sha256=digest,
                       updated_unix=time.time())
            write_json(study.OUT/'status.json',state)
            for ref,p,_,_ in processes:
                if p.poll() not in (None,0):raise RuntimeError(f'Reference worker {ref} exited {p.returncode}')
            time.sleep(5)
        assert all(p.returncode==0 for _,p,_,_ in processes)
        result=dict(phase='accuracy_generation_complete',selection_sha256=digest,started_unix=started,completed_unix=time.time(),
                    workers=[dict(reference=r,pid=p.pid,returncode=p.returncode,log=str(l.relative_to(study.OUT))) for r,p,_,l in processes])
        write_json(study.OUT/'execution_manifest.json',result)
        write_json(study.OUT/'status.json',result)
        print(json.dumps(result),flush=True)
    finally:
        for _,p,s,_ in processes:
            if p.poll() is None:p.terminate()
            s.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--reference',type=float);args=parser.parse_args()
    if args.reference is None:main()
    else:
        for module,extra in ((study.__name__,['test']),(__package__+'.ber',[])):
            subprocess.run([sys.executable,'-u','-m',module,*extra,'--reference',str(args.reference)],check=True)
