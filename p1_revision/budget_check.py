"""Independent fixed-parameter 100-to-300 update sensitivity check.

This does not retune against the main campaign. Select ranks/thresholds using
25 context updates once, then continue a single trajectory from the original
all-pilot spectral initialization. Default protocol: 50 fresh worlds in each
of six declared cells. Numerical steadiness is not a convergence guarantee.
No main-campaign result files are read. Workload timings are not serial
benchmarks while any other experiment is running.
"""
import argparse
from collections import deque
import hashlib
import json
from pathlib import Path
import platform
import time
import numpy as np
from codex_verification.solver import initial, project
from codex_verification.worlds import make_world
from .baselines import (THRESHOLD_FACTORS, posterior_field, rician_nll,
                        select_candidates, smooth_project, fit_candidates)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT/'results'/'p1_revision'/'budget_check'
METHODS = ('em', 'rician_gd', 'hs', 'rician_hankel', 'em_smooth')
DEFAULT_CHECKPOINTS = (100, 300)


def configurations():
    return [dict(tag=f'budget_L{l}_S{snr:g}', seed_base=74_000_000+i*10_000,
                 N=32, K=3, P=30, snr=snr, rsr=12., L=l, trials=50)
            for i, (l, snr) in enumerate((l, s) for l in (3, 7, 14) for s in (0., 10.))]


def fit_trajectory(s, z, b, sigma2, *, core, rank=None, factor=None,
                   checkpoints=DEFAULT_CHECKPOINTS, sweeps=4, g0=None):
    """Observation-only trajectory; snapshot arrays and numerical diagnostics."""
    checkpoints = tuple(int(x) for x in checkpoints)
    if not checkpoints or list(checkpoints) != sorted(set(checkpoints)) or checkpoints[0] < 10:
        raise ValueError('checkpoints must be increasing distinct integers >=10')
    if core not in ('em', 'gd') or (rank is not None and factor is not None):
        raise ValueError('invalid core or overlapping structural choices')
    cap = (z.shape[0]+1)//2
    rank = cap if rank is None else rank
    g = initial(s, z, b) if g0 is None else np.array(g0, copy=True)
    sh = s.conj().T
    norm2 = np.linalg.norm(s, 2)**2
    op = np.linalg.solve((s@sh).T, sh.T).T if core == 'em' else None
    snapshots, diagnostics = {}, {}
    last_ten = deque([g.copy()], maxlen=11)
    step_ten = deque(maxlen=10)
    previous_checkpoint = None
    started = time.perf_counter()
    for step in range(1, checkpoints[-1]+1):
        mu = g@s+b
        q = posterior_field(z, mu, sigma2)
        updated = (q-b)@op if core == 'em' else g-((mu-q)@sh)/norm2
        updated = (project(updated[None], [rank], sweeps)[0] if factor is None
                   else smooth_project(updated[None], [factor], sweeps)[0])
        if not np.all(np.isfinite(updated)):
            raise FloatingPointError('nonfinite budget trajectory')
        scale = max(float(np.linalg.norm(updated)), 1e-15)
        relative_step = float(np.linalg.norm(updated-g)/scale)
        g = updated
        last_ten.append(g.copy()); step_ten.append(relative_step)
        if step in checkpoints:
            drift10 = float(np.linalg.norm(g-last_ten[0])/scale)
            objective = float(np.mean(rician_nll(z, g@s+b, sigma2)))
            change = None if previous_checkpoint is None else float(np.linalg.norm(g-previous_checkpoint)/scale)
            snapshots[step] = g.copy()
            diagnostics[step] = dict(updates=step, nll_per_measurement=objective,
                relative_step=relative_step, max_relative_step_last10=float(max(step_ten)),
                relative_drift_last10=drift10, relative_change_since_previous_checkpoint=change,
                numerically_steady=bool(drift10<1e-6 and max(step_ten)<1e-6),
                seconds_since_fit_start=time.perf_counter()-started)
            previous_checkpoint = g.copy()
    return snapshots, diagnostics


def estimate_budget(w, *, checkpoints=DEFAULT_CHECKPOINTS):
    """Parameters selected once from measurements; never inspect w.G."""
    s, z, b, sigma2 = w.S, w.Z, w.B, float(w.sigma2)
    cap = (z.shape[0]+1)//2
    selected = {}
    for method, core, smooth in [('hs','em',False), ('rician_hankel','gd',False), ('em_smooth','em',True)]:
        values = np.asarray(THRESHOLD_FACTORS) if smooth else np.arange(1,cap+1)
        started = time.perf_counter()
        index, scores = select_candidates(s,z,b,sigma2,core=core,iterations=25,
            val_fraction=.3,sweeps=4,**({'factors':values} if smooth else {'ranks':values}))
        selected[method] = dict(value=float(values[index]) if smooth else int(values[index]),
                                scores=scores.tolist(), seconds=time.perf_counter()-started)
    started = time.perf_counter(); g0=initial(s,z,b)
    initial_seconds=time.perf_counter()-started
    output, details = {}, {}
    for method in METHODS:
        core = 'gd' if method in ('rician_gd','rician_hankel') else 'em'
        kw = ({'factor':selected[method]['value']} if method=='em_smooth' else
              {'rank':selected[method]['value']} if method in selected else {'rank':cap})
        output[method], details[method] = fit_trajectory(s,z,b,sigma2,core=core,
                                                       checkpoints=checkpoints,g0=g0,**kw)
    return output, dict(selection=selected,initial_seconds=initial_seconds,trajectories=details)


def _source_paths():
    return ([ROOT/'p1_revision'/x for x in ('__init__.py','budget_check.py','baselines.py')]
            +[ROOT/'codex_verification'/x for x in ('solver.py','worlds.py')]
            +[ROOT/'sahure'/'__init__.py', ROOT/'sahure'/'classical.py']
            +sorted((ROOT/'rydberg_sim').glob('*.py')))


def _hashes():
    return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in _source_paths()}


def write_json(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp.json');temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temp.replace(path)


def freeze(out=DEFAULT, checkpoints=DEFAULT_CHECKPOINTS):
    import scipy
    path=Path(out)/'protocol.json'
    payload=dict(version=1,cells=configurations(),methods=list(METHODS),checkpoints=list(checkpoints),
        total_worlds=300,paired_methods=True,select_updates=25,sweeps=4,val_fraction=.3,
        selection='same trailing30percent pilot split and heldout amplitude residual; independent parameter per method; selected once and fixed across100/300',
        initialization='same all-pilot spectral initializer for all methods and checkpoints; one continuous forward trajectory',
        diagnostic='nll per measurement omits data-only terms; normalized step and ten-update drift relative to current estimate norm',
        steady='both maximum one-step change over last10 and ten-step drift <1e-6; numerical steadiness only, not stationarity or global convergence',
        stopping='fixed50newworlds per cell; no model/rank/threshold retuning from evaluation errors',
        preregistration='declared before inspection of main-campaign NMSE; no campaign result files are read by this module',
        timing='workload timings only; shared spectral initializer timed once and reported separately',
        numpy=np.__version__,scipy=scipy.__version__,python=platform.python_version(),
        platform=platform.platform(),source_sha256=_hashes())
    if path.exists():
        old=json.loads(path.read_text())
        for key in ('cells','methods','checkpoints','source_sha256'):
            if old[key]!=payload[key]:
                raise RuntimeError('budget protocol changed; use a fresh directory rather than mixing outcomes')
        return old
    payload['created_utc']=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
    write_json(path,payload)
    for src in _source_paths():
        dest=Path(out)/'source_snapshot'/src.relative_to(ROOT)
        dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(src.read_bytes())
    return payload


def preflight(out=DEFAULT, checkpoints=DEFAULT_CHECKPOINTS):
    """Two development worlds; save runtime/finite checks, not their NMSE."""
    rows=[]
    for trial,l,snr in [(69_100_201,3,0.),(69_100_202,14,10.)]:
        w=make_world(trial,N=32,K=3,P=30,L=l,snr=snr,rsr=12.)
        started=time.perf_counter();est,diag=estimate_budget(w,checkpoints=checkpoints)
        elapsed=time.perf_counter()-started
        # Check the100update snapshots against the already tested independent
        # fixed-budget fitter, without consuming any main campaign outcome.
        for method in METHODS:
            core='gd' if method in ('rician_gd','rician_hankel') else 'em'
            choice=diag['selection'].get(method,{}).get('value',16)
            options={'factors':[choice]} if method=='em_smooth' else {'ranks':[choice]}
            expected=fit_candidates(w.S,w.Z,w.B,w.sigma2,core=core,iterations=100,sweeps=4,**options)[0]
            np.testing.assert_allclose(est[method][100],expected,rtol=1e-11,atol=1e-11)
        rows.append(dict(trial=trial,L=l,snr=snr,seconds=elapsed,finite=True,snapshot100_parity=True))
        print(json.dumps(rows[-1]),flush=True)
    result=dict(purpose='timing and snapshot parity only; no development NMSE saved',checkpoints=list(checkpoints),
                rows=rows,median_seconds=float(np.median([x['seconds'] for x in rows])),
                estimated300world_serial_minutes=float(np.median([x['seconds'] for x in rows])*5),
                timing='workload measurements, may overlap main campaign',source_sha256=_hashes())
    write_json(Path(out)/'preflight.json',result)
    return result


def run(out=DEFAULT, checkpoints=DEFAULT_CHECKPOINTS):
    protocol=freeze(out,checkpoints)
    fingerprint=hashlib.sha256(json.dumps(protocol['source_sha256'],sort_keys=True).encode()).hexdigest()
    count=0
    for c in protocol['cells']:
        for index in range(c['trials']):
            if _hashes()!=protocol['source_sha256']:
                raise RuntimeError('budget-check sources changed after freeze')
            path=Path(out)/'raw'/c['tag']/f'{index:05d}.npz'
            if path.exists():
                with np.load(path,allow_pickle=False) as saved:
                    if str(saved['fingerprint'])!=fingerprint:
                        raise RuntimeError('incompatible budget checkpoint')
                count+=1;continue
            trial=c['seed_base']+index
            w=make_world(trial,**{k:c[k] for k in ('N','K','P','snr','rsr','L')})
            estimates,details=estimate_budget(w,checkpoints=checkpoints)
            # Channel truth is read only here, after every estimate is fixed.
            arrays=dict(trial=trial,den=float(np.sum(abs(w.G)**2)),checkpoints=np.asarray(checkpoints),fingerprint=fingerprint)
            for method in METHODS:
                arrays['num_'+method]=np.asarray([np.sum(abs(estimates[method][step]-w.G)**2) for step in checkpoints])
                for metric in ('nll_per_measurement','relative_step','max_relative_step_last10',
                               'relative_drift_last10','numerically_steady','seconds_since_fit_start'):
                    arrays[method+'_'+metric]=np.asarray([details['trajectories'][method][step][metric] for step in checkpoints])
            arrays['diagnostics_json']=json.dumps(details,allow_nan=False)
            path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix('.tmp.npz')
            np.savez_compressed(temp,**arrays);temp.replace(path)
            count+=1
            if count%10==0:
                print(f'{count}/300 budget worlds complete',flush=True)
                write_json(Path(out)/'progress.json',dict(completed=count,total=300))
    write_json(Path(out)/'COMPLETE.json',dict(completed=count,total=300,fingerprint=fingerprint))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,default=DEFAULT)
    parser.add_argument('--preflight',action='store_true')
    parser.add_argument('--freeze-only',action='store_true')
    parser.add_argument('--include-1000',action='store_true',help='Must be separately declared before any outcome inspection')
    args=parser.parse_args()
    checkpoints=(100,300,1000) if args.include_1000 else DEFAULT_CHECKPOINTS
    if args.preflight:
        print(json.dumps(preflight(args.out,checkpoints),indent=2));return
    if args.freeze_only:
        freeze(args.out,checkpoints);print('Frozen300world budget protocol');return
    run(args.out,checkpoints)


if __name__=='__main__':
    main()
