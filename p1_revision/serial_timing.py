"""Serial full-cost CPU timing protocol. Run after compute campaigns finish."""
import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import time
from types import SimpleNamespace
import numpy as np
from codex_verification.worlds import make_world
from .baselines import estimate_comparators
from .budget_check import ROOT, write_json

DEFAULT=ROOT/'results'/'p1_revision'/'serial_timing'
METHOD_ORDER=('hs','rician_hankel','em_smooth','em','rician_gd','xu_linearized')


def method_seconds(diagnostics):
    """Include candidate initialization/selection and final initialization/fit."""
    if set(diagnostics['methods'])!=set(METHOD_ORDER):
        raise ValueError('unexpected method set in timing diagnostics')
    out={}
    for method in METHOD_ORDER:
        details=diagnostics['methods'][method]
        a,b=float(details['seconds_select']),float(details['seconds_final_fit'])
        if not np.isfinite(a+b) or a<0 or b<0:
            raise ValueError('invalid method timing')
        out[method]=dict(seconds_select=a,seconds_final_fit=b,seconds_total=a+b)
    return out


def time_observations(w, estimator=estimate_comparators):
    """Timed estimator receives observations only, even if a world has truth."""
    observations=SimpleNamespace(S=w.S,Z=w.Z,B=w.B,sigma2=w.sigma2)
    start=time.perf_counter()
    _,diag=estimator(observations,iterations=100,select_iterations=25,val_fraction=.3,sweeps=4)
    elapsed=time.perf_counter()-start
    times=method_seconds(diag)
    reported=sum(x['seconds_total'] for x in times.values())
    if reported>elapsed+1e-8:
        raise ValueError('method times exceed estimator wrapper time')
    return times, dict(seconds_wrapper=elapsed,seconds_unattributed=elapsed-reported)


def _sources():
    return ([ROOT/'p1_revision'/x for x in ('__init__.py','serial_timing.py','budget_check.py','baselines.py')]
        +[ROOT/'codex_verification'/x for x in ('solver.py','worlds.py')]
        +[ROOT/'sahure'/'__init__.py',ROOT/'sahure'/'classical.py']
        +sorted((ROOT/'rydberg_sim').glob('*.py')))


def source_hashes():
    return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in _sources()}


def freeze(out=DEFAULT):
    import scipy
    path=Path(out)/'protocol.json'
    protocol=dict(N=32,K=3,P=30,L=7,snr_db=5.,rsr_db=12.,trial_seed_base=75_000_000,
        worlds=20,iterations=100,select_iterations=25,sweeps=4,val_fraction=.3,
        execution='All six methods on one world, in the fixed estimator order, then advance to the next world; no parallel workers.',
        method_order=list(METHOD_ORDER),precision='float64/complex128',device='CPU',threads_requested=1,
        measurement='Selection plus final-fit times returned by the frozen estimator; both include their spectral initializers. Dataset generation is excluded.',
        interpretation='One implementation and machine at one operating point. No neural models and no training costs. Fixed order is disclosed; no randomized-order claim.',
        launch_requirement='Run only after the main campaign and budget-check workers have stopped; avoid other heavy compute.',
        warmup='No discarded warmup worlds; all twenty measured worlds are included.',
        stopping='Fixed twenty worlds and one call per world; no result-dependent repetition or removal.',
        platform=platform.platform(),processor=platform.processor(),python=platform.python_version(),
        numpy=np.__version__,scipy=scipy.__version__,source_sha256=source_hashes())
    if path.exists():
        old=json.loads(path.read_text())
        for key in ('source_sha256','worlds','trial_seed_base','method_order','N','K','P','L','iterations','select_iterations'):
            if old[key]!=protocol[key]:
                raise RuntimeError('serial timing protocol changed; use another output directory')
        return old
    protocol['created_utc']=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
    write_json(path,protocol)
    for src in _sources():
        dest=Path(out)/'source_snapshot'/src.relative_to(ROOT)
        dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(src.read_bytes())
    return protocol


def run(out=DEFAULT):
    out=Path(out);protocol=freeze(out)
    if (out/'COMPLETE.json').exists():
        raise RuntimeError('timing already complete; preserve the original session')
    if (out/'raw.json').exists():
        raise RuntimeError('partial timing session exists; use a fresh predeclared directory for a new complete session')
    settings={k:os.environ.get(k) for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS')}
    if any(value!='1' for value in settings.values()):
        raise RuntimeError('set all BLAS/thread environment settings to1 before importing NumPy')
    # Automatic guard covers the two declared local campaign result trees.
    # Human/operator must also ensure their workers have actually exited.
    for name in ('campaign','budget_check'):
        if not (ROOT/'results'/'p1_revision'/name/'COMPLETE.json').exists():
            raise RuntimeError('wait for '+name+' completion before serial timing')
    rows=[];wrappers=[]
    started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
    for i in range(protocol['worlds']):
        if source_hashes()!=protocol['source_sha256']:
            raise RuntimeError('timing sources changed after protocol freeze')
        trial=protocol['trial_seed_base']+i
        w=make_world(trial,N=32,K=3,P=30,L=7,snr=5.,rsr=12.)
        times,wrapper=time_observations(w)
        wrappers.append(dict(trial=trial,**wrapper))
        rows.extend(dict(trial=trial,method=m,**times[m]) for m in METHOD_ORDER)
        write_json(out/'raw.json',dict(rows=rows,wrappers=wrappers,thread_settings=settings,started_utc=started_utc))
        print(f'{i+1}/20 serial timing worlds complete',flush=True)
    summary=[]
    for method in METHOD_ORDER:
        subset=[r for r in rows if r['method']==method]
        totals=np.asarray([r['seconds_total'] for r in subset])
        summary.append(dict(method=method,worlds=20,median_total_seconds=float(np.median(totals)),
            q25_total_seconds=float(np.quantile(totals,.25)),q75_total_seconds=float(np.quantile(totals,.75)),
            median_select_seconds=float(np.median([r['seconds_select'] for r in subset])),
            median_final_fit_seconds=float(np.median([r['seconds_final_fit'] for r in subset])),
            median_selection_fraction=float(np.median([r['seconds_select']/r['seconds_total'] for r in subset]))))
    write_json(out/'summary.json',dict(rows=summary,protocol=protocol,
        note='Median total is computed per world; it need not equal the sum of the component medians. No portable complexity claim.'))
    text=io.StringIO();writer=csv.DictWriter(text,fieldnames=list(summary[0]));writer.writeheader();writer.writerows(summary)
    (out/'summary.csv').write_text(text.getvalue())
    lines=['Serial CPU timing','',
        'Twenty fresh worlds at N=32, K=3, P=30, L=7, SNR=5 dB and RSR=12 dB.',
        'Every method includes initialization and all required parameter-selection work.',
        'Methods run in a fixed, disclosed order for each world. Training and dataset generation are excluded.','',
        'Method | Median total ms | Middle 50% ms | Median selection share']
    for r in summary:
        lines.append(f"{r['method']:14s} | {1000*r['median_total_seconds']:.3f} | "
                     f"{1000*r['q25_total_seconds']:.3f}–{1000*r['q75_total_seconds']:.3f} | "
                     f"{100*r['median_selection_fraction']:.1f}%")
    lines.extend(['','These are measured timings for this machine, implementation and operating point.'])
    (out/'REPORT.txt').write_text('\n'.join(lines)+'\n')
    write_json(out/'COMPLETE.json',dict(worlds=20,method_evaluations=120,started_utc=started_utc,
        completed_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),source_sha256=protocol['source_sha256']))
    return summary


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,default=DEFAULT)
    parser.add_argument('--freeze-only',action='store_true');args=parser.parse_args()
    if args.freeze_only:
        freeze(args.out);print('Frozen twenty-world serial timing protocol. No timings run.');return
    run(args.out)


if __name__=='__main__':
    main()
