"""Audit and summarize the frozen update-budget check after completion only."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import numpy as np
from .budget_check import DEFAULT, ROOT, estimate_budget, make_world, write_json

BOOTSTRAP_SEED = 2026100801
BOOTSTRAP_RESAMPLES = 4000


def paired_gain(error100, error300, counts):
    """Positive dB gain means lower total error at 300 updates."""
    a,b=np.asarray(error100),np.asarray(error300)
    if np.sum(a)<=0 or np.sum(b)<=0:
        raise ValueError('cannot summarize an identically zero total error in dB')
    gain=float(10*np.log10(a.sum()/b.sum()))
    boot=10*np.log10((counts@a)/(counts@b))
    return gain, np.quantile(boot,[.025,.975]).tolist()


def audit_and_summarize(root=DEFAULT, *, replay=True):
    root=Path(root)
    if not (root/'COMPLETE.json').exists():
        raise RuntimeError('budget analysis requires COMPLETE.json; partial results are not summarized')
    protocol=json.loads((root/'protocol.json').read_text())
    completion=json.loads((root/'COMPLETE.json').read_text())
    if completion['completed']!=300 or completion['total']!=300:
        raise RuntimeError('budget completion does not cover all 300 declared worlds')
    if protocol['checkpoints']!=[100,300]:
        raise ValueError('this summary implements the frozen 100-versus-300 protocol')
    if len(protocol['cells'])!=6 or any(c['trials']!=50 for c in protocol['cells']):
        raise ValueError('expected six cells of 50 trials')
    hashes=protocol['source_sha256']
    fingerprint=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()
    if completion['fingerprint']!=fingerprint:
        raise RuntimeError('completion fingerprint differs from protocol')
    for relative, expected in hashes.items():
        for path in (ROOT/relative, root/'source_snapshot'/relative):
            if hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
                raise RuntimeError('frozen source hash mismatch: '+str(path))
    methods=protocol['methods'];rng=np.random.default_rng(BOOTSTRAP_SEED)
    rows=[]; all_trials=set(); artifact_hashes={}; replays=[]
    numeric_metrics=('nll_per_measurement','relative_step','max_relative_step_last10',
                     'relative_drift_last10','seconds_since_fit_start')
    for cell in protocol['cells']:
        directory=root/'raw'/cell['tag']
        paths=sorted(directory.glob('*.npz'))
        if [p.name for p in paths]!=[f'{i:05d}.npz' for i in range(50)]:
            raise RuntimeError('missing, duplicate or unexpected trial files in '+cell['tag'])
        collected=[]
        for i,path in enumerate(paths):
            artifact_hashes[str(path.relative_to(root))]=hashlib.sha256(path.read_bytes()).hexdigest()
            with np.load(path,allow_pickle=False) as raw:
                values={k:raw[k].copy() for k in raw.files}
            trial=int(values['trial'])
            if trial!=cell['seed_base']+i or trial in all_trials:
                raise RuntimeError('trial identity mismatch or cross-cell reuse')
            all_trials.add(trial)
            if str(values['fingerprint'])!=fingerprint:
                raise RuntimeError('raw fingerprint mismatch')
            np.testing.assert_array_equal(values['checkpoints'],[100,300])
            if not np.isfinite(values['den']) or values['den']<=0:
                raise ValueError('invalid channel energy')
            details=json.loads(str(values['diagnostics_json']))
            for method in methods:
                error=values['num_'+method]
                if error.shape!=(2,) or not np.isfinite(error).all() or np.any(error<0):
                    raise ValueError('invalid squared errors')
                for metric in numeric_metrics:
                    value=values[method+'_'+metric]
                    if value.shape!=(2,) or not np.isfinite(value).all():
                        raise ValueError('invalid trajectory diagnostics')
                    if metric!='nll_per_measurement' and np.any(value<0):
                        raise ValueError('negative norm or timing diagnostic')
                    expected=[details['trajectories'][method][str(t)][metric] for t in (100,300)]
                    np.testing.assert_array_equal(value,expected)
                expected_steady=(values[method+'_relative_drift_last10']<1e-6)&(values[method+'_max_relative_step_last10']<1e-6)
                np.testing.assert_array_equal(values[method+'_numerically_steady'],expected_steady)
                change=details['trajectories'][method]['300']['relative_change_since_previous_checkpoint']
                if change is None or not np.isfinite(change) or change<0:
                    raise ValueError('invalid 100-to-300 estimate change')
            values['details']=details;collected.append(values)
        # One predeclared replay per cell verifies actual errors, not only
        # copied summaries. Replay is outside any reported timing benchmark.
        if replay:
            example=collected[17]
            world=make_world(int(example['trial']),**{k:cell[k] for k in ('N','K','P','snr','rsr','L')})
            estimates,diagnostic=estimate_budget(world)
            np.testing.assert_allclose(example['den'],np.sum(abs(world.G)**2),rtol=1e-12,atol=1e-12)
            for method in methods:
                errors=[np.sum(abs(estimates[method][t]-world.G)**2) for t in (100,300)]
                np.testing.assert_allclose(example['num_'+method],errors,rtol=1e-10,atol=1e-10)
                for metric in ('nll_per_measurement','relative_step','relative_drift_last10'):
                    expected=[diagnostic['trajectories'][method][t][metric] for t in (100,300)]
                    np.testing.assert_allclose(example[method+'_'+metric],expected,rtol=1e-8,atol=1e-11)
                if method in diagnostic['selection']:
                    if example['details']['selection'][method]['value']!=diagnostic['selection'][method]['value']:
                        raise RuntimeError('replayed parameter selection differs')
            replays.append(int(example['trial']))
        energy=np.asarray([x['den'] for x in collected])
        counts=rng.multinomial(50,np.full(50,1/50),size=BOOTSTRAP_RESAMPLES)
        for method in methods:
            errors=np.asarray([x['num_'+method] for x in collected])
            nmse=10*np.log10(errors.sum(0)/energy.sum())
            gain,ci=paired_gain(errors[:,0],errors[:,1],counts)
            steady=np.asarray([x[method+'_numerically_steady'] for x in collected])
            nll=np.asarray([x[method+'_nll_per_measurement'] for x in collected])
            drift=np.asarray([x[method+'_relative_drift_last10'] for x in collected])
            change=np.asarray([x['details']['trajectories'][method]['300']['relative_change_since_previous_checkpoint'] for x in collected])
            rows.append(dict(cell=cell['tag'],L=cell['L'],snr_db=cell['snr'],method=method,trials=50,
                nmse100_db=float(nmse[0]),nmse300_db=float(nmse[1]),gain100_to300_db=gain,
                gain_ci_low_db=ci[0],gain_ci_high_db=ci[1],
                steady_fraction100=float(steady[:,0].mean()),steady_fraction300=float(steady[:,1].mean()),
                median_relative_change100_to300=float(np.median(change)),
                median_relative_drift_last10_at100=float(np.median(drift[:,0])),
                median_relative_drift_last10_at300=float(np.median(drift[:,1])),
                median_nll100=float(np.median(nll[:,0])),median_nll300=float(np.median(nll[:,1])),
                median_nll_decrease100_to300=float(np.median(nll[:,0]-nll[:,1])),
                fraction_nll_increased=float(np.mean(nll[:,1]>nll[:,0]+1e-10))))
    if len(all_trials)!=300:
        raise RuntimeError('wrong unique-world total')
    report=dict(rows=rows,bootstrap_resamples=BOOTSTRAP_RESAMPLES,bootstrap_seed=BOOTSTRAP_SEED,
        bootstrap_unit='paired world, separately within each cell; same resample counts across methods',
        interval='pointwise95percent percentile intervals; no multiplicity correction',
        gain='10log10(sum squared errors at100 / sum squared errors at300); positive favors300',
        likelihood='Rician NLL per measurement up to observation-only terms; never expressed in dB',
        numerical_steadiness='last10step maximum relative change and10step drift both below1e-6; not a convergence guarantee',
        scope='parameters fixed after25update selection; this tests final-update budget only, not rank-selection budget or globally optimal solutions')
    analysis=root/'analysis';analysis.mkdir(parents=True,exist_ok=True)
    write_json(analysis/'summary.json',report)
    text=io.StringIO();writer=csv.DictWriter(text,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (analysis/'summary.csv').write_text(text.getvalue())
    lines=['Update-budget sensitivity check','',
        'All 300 fresh worlds were evaluated at 100 and 300 updates on one continuous trajectory.',
        'Each structured method selected its rank or threshold once using 25 context updates.',
        'Positive gain means that continuing to 300 updates reduced the total squared channel error.',
        'Intervals use 4,000 paired bootstrap resamples within each 50-world cell.',
        'Numerical steadiness is a change diagnostic, not proof of convergence.','',
        'L | SNR dB | Method | Gain dB [95% interval] | Steady at 100 / 300 | Median relative estimate change']
    for r in rows:
        lines.append(f"{r['L']:2d} | {r['snr_db']:6.1f} | {r['method']:14s} | "
            f"{r['gain100_to300_db']:+.3f} [{r['gain_ci_low_db']:+.3f}, {r['gain_ci_high_db']:+.3f}] | "
            f"{100*r['steady_fraction100']:.0f}% / {100*r['steady_fraction300']:.0f}% | "
            f"{r['median_relative_change100_to300']:.4g}")
    lines.extend(['','Cells with a nonzero fraction of estimates still changing at 300 updates cannot be described as fully numerically steady.',
        'The JSON and CSV files also report likelihood changes. Lower likelihood loss does not guarantee lower channel error.',
        'These checks do not justify a global-convergence claim or a new choice of rank based on test errors.',
        'Workload timings are omitted from this accuracy summary because experiments may have run concurrently.'])
    (analysis/'REPORT.txt').write_text('\n'.join(lines)+'\n')
    write_json(analysis/'audit.json',dict(passed=True,worlds=300,cells=6,methods=methods,
        replayed_trials=replays,replay_enabled=replay,source_fingerprint=fingerprint,
        analysis_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),raw_sha256=artifact_hashes))
    return report


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=DEFAULT)
    args=parser.parse_args();result=audit_and_summarize(args.root)
    print(f"Audited 300 worlds and wrote {len(result['rows'])} method/cell summaries.")


if __name__=='__main__':
    main()
