"""Checkpointed reviewer experiments; protocol freezes before test outcomes.

Run with single-thread BLAS settings and at most a few process workers.
Raw times in a parallel campaign are workload timings, NOT serial benchmarks.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import hashlib
import json
import os
from pathlib import Path
import platform
import time
import numpy as np
from p1_revision.scenarios import configurations, make, estimate_pair
from codex_verification.solver import effective_ranks

ROOT=Path(__file__).resolve().parents[1]
DEFAULT=ROOT/'results'/'p1_revision'/'campaign'


def source_files():
    return ([ROOT/'p1_revision'/n for n in ('__init__.py','run.py','scenarios.py','baselines.py')]
            +[ROOT/'codex_verification'/n for n in ('worlds.py','solver.py')]
            +[ROOT/'sahure'/'classical.py']
            +sorted((ROOT/'rydberg_sim').glob('*.py')))


def hashes():
    return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in source_files()}


def fingerprint():
    return hashlib.sha256(json.dumps(hashes(),sort_keys=True).encode()).hexdigest()


def atomic_json(path, obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp.json')
    tmp.write_text(json.dumps(obj,indent=2,allow_nan=False))
    tmp.replace(path)


def evaluate(c,t):
    w=make(c,t)
    if c['kind']=='baselines':
        from p1_revision.baselines import estimate_comparators
        estimates,diag=estimate_comparators(w,iterations=c['iterations'],
            select_iterations=c['select_iterations'],val_fraction=.3,sweeps=c['sweeps'])
    else:
        estimates,diag=estimate_pair(w,c)
    row=dict(trial=c['seed_base']+t,den=float(np.sum(abs(w.G)**2)),
             rho_true=effective_ranks(w.G)/((c['N']+1)//2),L_k=np.asarray(w.L_k))
    for method,g in estimates.items():
        row['num_'+method]=float(np.sum(abs(g-w.G)**2))
    row['rho_em']=effective_ranks(estimates['em'])/((c['N']+1)//2)
    if c['kind']!='baselines':
        row.update({key:diag[key] for key in ('rank','seconds_select','seconds_em','seconds_hs_fit','seconds_hs_total')})
        row['scores']=diag['scores']
    if not all(np.all(np.isfinite(v)) for v in row.values()):
        raise FloatingPointError(f'nonfinite result in {c["tag"]} trial {t}')
    return row,diag


def chunk(c,start,stop,out,fp):
    if fingerprint()!=fp:
        raise RuntimeError('simulation source changed after freeze')
    dest=Path(out)/'raw'/c['tag']/f'{start:05d}.npz'
    if dest.exists():
        with np.load(dest,allow_pickle=False) as f:
            if str(f['fingerprint'])!=fp or int(f['count'])!=stop-start:
                raise ValueError('incompatible checkpoint')
        return c['tag'],stop-start,0.
    begun=time.perf_counter();rows=[];diagnostics=[]
    for t in range(start,stop):
        row,diag=evaluate(c,t);rows.append(row);diagnostics.append(diag)
    arrays={key:np.asarray([r[key] for r in rows]) for key in rows[0]}
    dest.parent.mkdir(parents=True,exist_ok=True)
    tmp=dest.with_suffix('.tmp.npz')
    np.savez_compressed(tmp,**arrays,fingerprint=fp,count=stop-start,
        diagnostics_json=json.dumps(diagnostics,allow_nan=False))
    tmp.replace(dest)
    return c['tag'],stop-start,time.perf_counter()-begun


def freeze(out,cs,workers,chunk_size):
    import scipy
    fp=fingerprint()
    p=Path(out)/'manifest.json'
    manifest=dict(source_fingerprint=fp,source_sha256=hashes(),cells=cs,
        created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        numpy=np.__version__,scipy=scipy.__version__,python=platform.python_version(),
        platform=platform.platform(),workers=workers,chunk_size=chunk_size,
        reviewed_manuscript='/Users/haatim/Downloads/ThesisProj-7.pdf',
        manuscript_sha256=hashlib.sha256(Path('/Users/haatim/Downloads/ThesisProj-7.pdf').read_bytes()).hexdigest(),
        stopping='400 paired trials per cell fixed before test outcomes; checkpoints resumable',
        pairing='identical world for all methods; shared worlds across split fractions and mismatch strengths',
        measurement='Z=abs(G S+B+W), W complex Gaussian total variance sigma2',
        snr='aggregate statistical channel power / total complex noise variance',
        rsr='reference power / single-user statistical channel power',
        split='nested final pilot columns; final estimate refits all P pilots',
        mismatch='Gaussian position offsets std in wavelengths, calibrated B, actual G target, same W across strengths; no power renormalization',
        selection='heldout magnitude residual, no channel labels; one shared selected rank across per-user Hankel matrices',
        timings='parallel campaign workload times; serial timing must be reported separately',
        bootstrap='4000 paired trial resamples; pointwise percentile95; seed20261007',
        limitations=['fixed iteration budgets do not establish global convergence',
          'position-error strengths are declared stress tests, not measured hardware tolerances',
          'no new ML training or changes to Paper1/Paper2 historical experiments'])
    if p.exists():
        old=json.loads(p.read_text())
        if old['source_fingerprint']!=fp or old['cells']!=cs:
            raise RuntimeError('refuse incompatible resume; use another output directory')
    else:
        atomic_json(p,manifest)
        for src in source_files():
            dst=Path(out)/'source_snapshot'/src.relative_to(ROOT)
            dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes(src.read_bytes())
    return fp


def preflight(out):
    measurements=[]
    for kind in ('baselines','holdout','mismatch'):
        c=next(dict(c) for c in configurations() if c['kind']==kind)
        c['seed_base']=79_100_000+10000*len(measurements)
        secs=[]
        for t in range(3):
            start=time.perf_counter();evaluate(c,t);secs.append(time.perf_counter()-start)
        measurements.append(dict(kind=kind,seconds=secs,median_seconds=float(np.median(secs))))
    pred=sum(x['median_seconds']*sum(c['trials'] for c in configurations() if c['kind']==x['kind']) for x in measurements)
    result=dict(measurements=measurements,predicted_serial_seconds=pred,
        note='three development trials per group; no evaluation errors inspected or saved')
    atomic_json(Path(out)/'preflight.json',result)
    print(json.dumps(result,indent=2),flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,default=DEFAULT)
    ap.add_argument('--workers',type=int,default=3);ap.add_argument('--chunk',type=int,default=5)
    ap.add_argument('--preflight',action='store_true');ap.add_argument('--freeze-only',action='store_true')
    ap.add_argument('--kind',choices=('all','baselines','holdout','mismatch'),default='all')
    ap.add_argument('--smoke-trials',type=int,default=0)
    ap.add_argument('--max-hours',type=float,default=4.)
    args=ap.parse_args()
    if args.preflight:
        preflight(args.out);return
    cs=[dict(c) for c in configurations() if args.kind=='all' or c['kind']==args.kind]
    if args.smoke_trials:
        if args.out==DEFAULT:
            raise ValueError('smoke must use a separate output directory')
        for c in cs:c['trials']=args.smoke_trials
    fp=freeze(args.out,cs,args.workers,args.chunk)
    if args.freeze_only:
        print(f'Frozen {len(cs)} cells and {sum(c["trials"] for c in cs)} cell trials',flush=True);return
    jobs=[(c,t,min(t+args.chunk,c['trials'])) for c in cs for t in range(0,c['trials'],args.chunk)]
    jobs.sort(key=lambda x:x[1])
    total=sum(c['trials'] for c in cs);done=0;begun=time.perf_counter()
    # Completed files are counted on resume without repeating a result.
    pending=[]
    for c,a,b in jobs:
        p=args.out/'raw'/c['tag']/f'{a:05d}.npz'
        if p.exists():
            with np.load(p,allow_pickle=False) as d:
                assert str(d['fingerprint'])==fp and int(d['count'])==b-a
            done+=b-a
        else:pending.append((c,a,b))
    print(f'START {done}/{total} complete, {args.workers} workers, fingerprint {fp}',flush=True)
    jobs=iter(pending);active={};exhausted=False;deadline=False
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        while active or not exhausted:
            if time.perf_counter()-begun>=args.max_hours*3600:
                deadline=True;exhausted=True
            while not exhausted and len(active)<2*args.workers:
                job=next(jobs,None)
                if job is None:exhausted=True;break
                f=pool.submit(chunk,*job,str(args.out),fp);active[f]=job[0]['tag']
            if not active:break
            ready,_=wait(active,return_when=FIRST_COMPLETED,timeout=30)
            for f in ready:
                tag,n,sec=f.result();del active[f];done+=n
                status=dict(completed_cell_trials=done,total_cell_trials=total,
                    elapsed_seconds=time.perf_counter()-begun,last_cell=tag,source_fingerprint=fp)
                atomic_json(args.out/'progress.json',status)
                print(f'{done}/{total} {tag} chunk_seconds={sec:.2f}',flush=True)
    status=dict(completed_cell_trials=done,total_cell_trials=total,
        elapsed_seconds=time.perf_counter()-begun,source_fingerprint=fp,
        complete=done==total,compute_cap_reached=deadline)
    atomic_json(args.out/('COMPLETE.json' if done==total else 'INCOMPLETE.json'),status)
    print(json.dumps(status),flush=True)


if __name__=='__main__':
    main()
