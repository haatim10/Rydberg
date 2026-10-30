"""Execution only; scientific configuration lives in the frozen study module."""
import fcntl, json, subprocess, sys, time
from .study import OUT, REPO, SEEDS, write, verify_protocol


def run():
    lock = (OUT/'execution.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
    verify_protocol(); jobs=[]
    for seed in SEEDS:
        log=(OUT/f'training_{seed}.log').open('a')
        cmd=[sys.executable,'-m','urformer_reproduction_20261022.study','train','--seed',str(seed)]
        p=subprocess.Popen(cmd,cwd=REPO,stdout=log,stderr=subprocess.STDOUT)
        jobs.append((seed,p,log))
    write(OUT/'execution.json',dict(state='training',started_unix=time.time(),workers=[dict(seed=s,pid=p.pid) for s,p,_ in jobs]))
    codes={str(seed):p.wait() for seed,p,_ in jobs}
    for _,_,log in jobs:log.close()
    write(OUT/'worker_exits.json',codes)
    if any(codes.values()):
        write(OUT/'execution.json',dict(state='failed',codes=codes));raise SystemExit(1)
    write(OUT/'execution.json',dict(state='evaluation',workers_complete=True))
    with (OUT/'evaluation.log').open('a') as log:
        subprocess.run([sys.executable,'-m','urformer_reproduction_20261022.study','evaluate'],cwd=REPO,stdout=log,stderr=subprocess.STDOUT,check=True)
    write(OUT/'execution.json',dict(state='accuracy_complete',finished_unix=time.time()))
    print('Three full50epoch fits and both figure sweeps complete.',flush=True)


if __name__=='__main__':run()
