"""Bounded pilot CLI. Broader campaigns are deliberately not launched here."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key]='1'
import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import time
import torch
from .data import generate,load_observations
from .train import train_models,preflight

PILOT=dict(N=32,K=3,P=30,snr=5.,rsr=12.,stages=10,train_cases=4096,
           val_cases=512,test_cases=400,batch_size=32,epochs=20,lr=.001,seed=8101,
           train_seed=2026092101,val_seed=2026092102,test_seed=2026092103,
           compute_cap_seconds=14400,bootstrap_resamples=4000)


def source_hashes():
    paths=list(Path('sahure').glob('*.py'))+list(Path('rydberg_sim').glob('*.py'))+[Path('codex_verification/solver.py')]
    return {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--phase',choices=('train','evaluate','all'),default='all')
    parser.add_argument('--output',default='results/sahure_pilot')
    args=parser.parse_args();out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    config=PILOT.copy()
    cp=out/'config.json'
    if cp.exists() and json.loads(cp.read_text())!=config: raise ValueError('configuration changed; use a new output directory')
    cp.write_text(json.dumps(config,indent=2)+'\n')
    budget_path=out/'budget.json';used=json.loads(budget_path.read_text())['used_seconds'] if budget_path.exists() else 0.
    start=time.monotonic();deadline=start+max(0,config['compute_cap_seconds']-used)
    manifest=dict(config=config,source_hashes=source_hashes(),python=platform.python_version(),
        torch=torch.__version__,platform=platform.platform(),threads=1,
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        single_user_rsr_db=config['rsr'],expected_aggregate_rsr_db=config['rsr']-10*__import__('math').log10(config['K']))
    (out/f'{args.phase}_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    status=dict(phase=args.phase,state='running')
    try:
        if time.monotonic()>=deadline:
            status['state']='compute_cap_reached';return
        if args.phase in ('train','all'):
            for split in ('train','val'):
                print(f'Generating {split} observations',flush=True)
                generate(out/'data',split,config[f'{split}_cases'],seed=config[f'{split}_seed'])
            train=load_observations(out/'data/train_observations.npz');val=load_observations(out/'data/val_observations.npz')
            timing=preflight(train,out/'preflight.json',config)
            print(json.dumps(dict(phase='preflight',timings=timing)),flush=True)
            complete=train_models(train,val,out/'checkpoints',config,deadline)
            if not complete:
                status['state']='compute_cap_reached';return
        if args.phase in ('evaluate','all'):
            from .evaluate import evaluate
            complete=evaluate(out,config,deadline)
            if not complete:
                status['state']='compute_cap_reached';return
        status['state']='complete'
    except Exception as exc:
        status.update(state='failed',error=repr(exc));raise
    finally:
        budget_path.write_text(json.dumps(dict(used_seconds=used+time.monotonic()-start,cap_seconds=config['compute_cap_seconds']),indent=2)+'\n')
        (out/'status.json').write_text(json.dumps(status,indent=2)+'\n')

if __name__=='__main__':main()
