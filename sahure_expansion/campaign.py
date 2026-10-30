"""Resumable main follow-up: three training processes, then frozen evaluation."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from .protocol import ROOT,CONFIG,initialize,write_json,cells


def parallel(jobs,workers,phase):
    pending=list(jobs);active=[];done=0
    while pending or active:
        while pending and len(active)<workers:
            name,command=pending.pop(0)
            log=(ROOT/f'{name}.log').open('a')
            process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
            active.append((name,process,log))
            print(f'Started {name}',flush=True)
        for item in active.copy():
            name,process,log=item;code=process.poll()
            if code is None: continue
            log.close();active.remove(item)
            if code:
                for _,other,handle in active: other.terminate();handle.close()
                raise RuntimeError(f'{name} failed with exit code {code}; see its log')
            done+=1;print(f'Completed {name}',flush=True)
        write_json(ROOT/'campaign_status.json',dict(phase=phase,state='running',completed_jobs=done,
            active=[x[0] for x in active],pending=len(pending),updated_unix=time.time()))
        if active: time.sleep(5)


def serial(module,*args):
    subprocess.run([sys.executable,'-u','-m',module,*args],check=True)


def main(workers,phase):
    initialize();tick=time.monotonic()
    try:
        if phase in ('all','train'):
            parallel([(f'train_seed_{s}',[sys.executable,'-u','-m','sahure_expansion.train','--seed',str(s)])
                      for s in CONFIG['seeds']],workers,'training')
        if phase in ('all','evaluate'):
            serial('sahure_expansion.evaluate','freeze')
            serial('sahure_expansion.evaluate','prepare')
            parallel([(f'evaluate_{c["id"]}',[sys.executable,'-u','-m','sahure_expansion.evaluate','cell','--cell',c['id']])
                      for c in cells()[0]],workers,'accuracy_evaluation')
            # All accuracy subprocesses have exited before latency measurement.
            write_json(ROOT/'campaign_status.json',dict(phase='serial_timing',state='running',updated_unix=time.time()))
            serial('sahure_expansion.evaluate','timing')
            serial('sahure_expansion.report')
            serial('sahure_expansion.audit')
        write_json(ROOT/'campaign_status.json',dict(state='complete',phase=phase,seconds=time.monotonic()-tick,updated_unix=time.time()))
    except Exception as exc:
        write_json(ROOT/'campaign_status.json',dict(state='failed',phase=phase,error=repr(exc),seconds=time.monotonic()-tick,updated_unix=time.time()))
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=3);p.add_argument('--phase',choices=['all','train','evaluate'],default='all')
    args=p.parse_args();main(args.workers,args.phase)
