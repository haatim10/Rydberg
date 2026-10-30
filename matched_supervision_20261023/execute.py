"""Six one-thread workers: three seeds, two supervision modes; both architectures per worker."""
import fcntl,subprocess,sys,time
from .common import *


def phase(module,args,logname):
    write(OUT/'execution.json',dict(state=logname,updated_unix=time.time()))
    with (OUT/(logname+'.log')).open('a') as log:
        subprocess.run([sys.executable,'-m','matched_supervision_20261023.'+module,*args],cwd=REPO,stdout=log,stderr=subprocess.STDOUT,check=True)


def run():
    lock=(OUT/'execution.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    frozen();assert json.loads((OUT/'preflight.json').read_text())['passed']
    jobs=[]
    for mode in MODES:
        for seed in SEEDS:
            log=(OUT/f'training_{mode}_{seed}.log').open('a')
            proc=subprocess.Popen([sys.executable,'-m','matched_supervision_20261023.training',mode,str(seed)],cwd=REPO,stdout=log,stderr=subprocess.STDOUT)
            jobs.append((mode,seed,proc,log))
    write(OUT/'execution.json',dict(state='training',started_unix=time.time(),workers=[dict(mode=m,seed=s,pid=p.pid) for m,s,p,_ in jobs],fits=12))
    while any(p.poll() is None for _,_,p,_ in jobs):
        if any(p.poll() is not None and p.returncode!=0 for _,_,p,_ in jobs):
            for _,_,p,_ in jobs:
                if p.poll() is None:p.terminate()
            break
        time.sleep(5)
    codes={f'{m}_{s}':p.wait() for m,s,p,_ in jobs}
    for _,_,_,log in jobs:log.close()
    write(OUT/'worker_exits.json',codes)
    if any(codes.values()):
        write(OUT/'execution.json',dict(state='failed',codes=codes));raise SystemExit(1)
    phase('checks',['audit'],'training_audit')
    phase('evaluation',['accuracy'],'accuracy')
    phase('evaluation',['ber'],'ber')
    phase('evaluation',['timing'],'timing')
    phase('report',[],'report')
    write(OUT/'execution.json',dict(state='complete',finished_unix=time.time()))
    print('All12 matched fits,accuracy,BER,timing and reports complete.',flush=True)


if __name__=='__main__':run()
