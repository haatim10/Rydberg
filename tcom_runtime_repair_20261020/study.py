"""Output-preserving implementation comparison; all scientific choices frozen."""
import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_key] = '1'
import argparse
import json
from pathlib import Path
import platform
import time
import numpy as np
import scipy
import torch
from tcom_followups_20261017.common import REPO, sha, write_json, frozen_optshrink
from tcom_filter_continuation_20261019 import study as previous
from tcom_incumbent_guard_20261019 import risk_data
from tcom_runtime_bypass_20261017 import sparse_fast
from tcom_retrained_bypass_20261018 import study as base
from . import inference

OUT = REPO / 'results/tcom_runtime_repair_20261020'
SMOKE_SEED = 2026102000
TEST_SEED = 2026102002
DATA_SEED = 2026102003
TIMING_SEED = 2026102004
MASK_SEED = 2026102005
BOOT_SEED = 2026102009
TEST_COUNT = 128
SNRS, RSRS, SEEDS = previous.SNRS, previous.RSRS, previous.SEEDS
REGIMES = previous.REGIMES
BOUNDARY_SNRS = previous.BOUNDARY_SNRS
ALL_SNRS = previous.ALL_SNRS
NAMES, GROUPS, CONTRASTS = previous.NAMES, previous.GROUPS, previous.CONTRASTS
PRIOR_REPAIR = previous.PRIOR_REPAIR
CURRENT_REPAIR_PATH = previous.OUT / 'policy.json'
CURRENT_REPAIR = json.loads(CURRENT_REPAIR_PATH.read_text())
PRIOR_COMPACT = CURRENT_REPAIR
OPT_LOOKUP = previous.OPT_LOOKUP
OLD_POLICY = previous.OLD_POLICY
CAP_MS = 10.74
ATOL = 1e-10
RTOL = 1e-10
save, load, cellname, observed = previous.save, previous.load, previous.cellname, risk_data.observed
selected_repair_choice = previous.selected_repair_choice
compact_config = previous.compact_config
stats = previous.stats


def make_anchor(*args, **kwargs):
    kwargs.setdefault('holdout_seed', MASK_SEED)
    kwargs.setdefault('snrs', ALL_SNRS)
    return risk_data.make_anchor(*args, **kwargs)


def engine():
    """Shared loaded weights; both implementations retain their own execution."""
    canonical = inference.canonical_engine()
    optimized = inference.optimized_engine(models=canonical.models)
    return {'canonical': canonical, 'optimized': optimized}


@torch.no_grad()
def estimate_method(obs, name, selection, est):
    native, canonical = est['optimized'], est['canonical']
    if name == 'em100':
        return native.em100(obs)
    if name == 'optshrink':
        return frozen_optshrink(obs, OPT_LOOKUP)
    if name == 'prior_sparse_full':
        return sparse_fast.estimate(obs, base.FULL)
    if name in ('prior_compact', 'compact_sparse'):
        # No sparse choices, kernels or budgets changed in this study.
        return sparse_fast.estimate(obs, compact_config(obs, selection))
    seed = int(name.rsplit('_', 1)[1])
    model = f'learned_{seed}'
    if name.startswith('original_'):
        return native.original(obs, model)
    if name.startswith('old_bypass_'):
        return native.estimate(obs, model, OLD_POLICY, fallback='em100')
    if name.startswith('prior_repair_'):
        return native.repair(obs, model, PRIOR_REPAIR['repair_choices'])
    if name.startswith('current_repair_'):
        return canonical.repair(obs, model, selection['repair_choices'])
    if name.startswith('repair_'):
        return native.repair(obs, model, selection['repair_choices'])
    raise KeyError(name)


def freeze_protocol():
    previous.verify_protocol()
    previous.check_selection()
    parity = json.loads((OUT / 'smoke_parity.json').read_text())
    assert parity['passed']
    for key in ('source_hashes', 'checkpoint_hashes'):
        for name,digest in parity[key].items():
            assert sha(REPO/name)==digest, name
    assert parity['latest_policy_sha256']==sha(CURRENT_REPAIR_PATH)
    from . import ber
    ber_digest = ber.freeze()
    source_paths = [Path(__file__), Path(inference.__file__), Path(__file__).with_name('native.py'),
                    Path(__file__).with_name('analysis.py'),
                    Path(__file__).with_name('smoke_parity.py'), Path(__file__).with_name('execution.py')]
    source_paths += [REPO / p for p in json.loads((previous.OUT / 'protocol.json').read_text())['source_hashes']]
    spec = dict(
        version=1, name='Numerically equivalent native execution of the frozen learned estimator',
        N=32, K=3, deployment_P=30, snrs=SNRS, rsrs=RSRS, regimes=REGIMES, model_seeds=SEEDS,
        test_cases_per_reference_regime=TEST_COUNT, smoke_seed=SMOKE_SEED, test_seed=TEST_SEED,
        data_seed=DATA_SEED, timing_seed=TIMING_SEED, holdout_seed=MASK_SEED, bootstrap_seed=BOOT_SEED,
        methods=NAMES, groups=GROUPS, contrasts=CONTRASTS,
        implementation='NumPy/SciPy float64/complex128 executes the same ten stages, frozen weights, features, posterior, solves and shrinkage. All EM budgets, routing and final filters unchanged.',
        invariants='No retraining, policy fitting, sparse reselection, shorter iteration budget, lower precision, approximate posterior or cached observation answers.',
        parity=dict(atol=ATOL, rtol=RTOL, criterion='All channel coefficients satisfy abs(native-canonical)<=atol+rtol*abs(canonical). Numerical equivalence, not bitwise equality or population noninferiority. Independently decode every nonidentical estimate.'),
        primary=dict(scopes=['all153', 'weak_warm10018'], contrast='repair_vs_current',
                     family='Two scopes times NMSE and BER: four outcomes, Bonferroni4 intervals. No claim of equivalence from nonsignificance or zero errors.'),
        bootstrap=dict(samples=20000, seed=BOOT_SEED, unit='128 local-index clusters per regime coupled across references/SNR/method; three fits average metrics and do not triple sample size.'),
        boundary=dict(extra_snrs=BOUNDARY_SNRS, extra_cells=54, rule='Same anchors, separately weighted diagnostic; not part of main153.'),
        timing=dict(cap_ms=CAP_MS, windows=2, window1_indices=[200,201,202], window2_indices=[300,301,302],
                    repeats_per_index=1, grid='All153 conditions at each of three independent timing indices per window',
                    order='Window1 before fresh accuracy generation; window2 after all own accuracy and BER workers have exited.',
                    window1_methods=NAMES,
                    window2_methods=[n for n in NAMES if n.startswith(('current_repair_', 'repair_')) or n in ('em100','compact_sparse')],
                    acceptance='Observed whole-grid pooled repair p90 <=10.74ms in BOTH prespecified windows. Separately report weak warm100 p90, everyfit and all51 reference/SNR p90s. No sentinel normalization or retries until passing.',
                    scope='Complete batch-one call includes conversion, setup, dispatch, initialization, ten learned stages where used, unchanged EM and filters. Preloaded fixed weights; oneCPUthread; other own experiment jobs idle.'),
        baseline_fairness='Only learned inference changed; original, old bypass and prior repair all use the same native learned implementation. EM100, compact sparse/EM and full sparse kernels unchanged. Canonical latest and canonical compact explicitly retained.',
        previous_protocol_sha256=sha(previous.OUT / 'protocol.json'),
        previous_policy_sha256=sha(CURRENT_REPAIR_PATH),
        source_hashes={str(p.relative_to(REPO)):sha(p) for p in sorted(set(source_paths))},
        checkpoint_hashes={str(p.relative_to(REPO)):sha(p) for p in inference.model_paths().values()},
        ber_protocol_sha256=ber_digest,
        smoke_parity_sha256=sha(OUT / 'smoke_parity.json'),
        thread_environment={key:os.environ[key] for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS')})
    path = OUT / 'protocol.json'
    if path.exists():
        old = json.loads(path.read_text())
        old.pop('frozen_unix')
        assert old == json.loads(json.dumps(spec))
    else:
        write_json(path, dict(**spec, frozen_unix=time.time()))
    policy = dict(protocol_sha256=sha(path), previous_policy_sha256=sha(CURRENT_REPAIR_PATH),
                  repair_choices=CURRENT_REPAIR['repair_choices'],
                  compact_choices=CURRENT_REPAIR['compact_choices'],
                  ber_protocol_sha256=ber_digest,
                  ber_source_hashes=ber.protocol_spec()['source_hashes'],
                  selection='Inherited unchanged; implementation comparison performs no new statistical selection.')
    destination = OUT / 'policy.json'
    if destination.exists():
        old = json.loads(destination.read_text())
        old.pop('frozen_unix')
        assert old == policy
    else:
        write_json(destination, dict(**policy, frozen_unix=time.time()))
    print('Frozen', sha(path), sha(destination), flush=True)
    return sha(path)


def verify_protocol():
    p = json.loads((OUT / 'protocol.json').read_text())
    for key in ('source_hashes', 'checkpoint_hashes'):
        for name, digest in p[key].items():
            assert sha(REPO / name) == digest, name
    assert sha(CURRENT_REPAIR_PATH) == p['previous_policy_sha256']
    assert previous.verify_protocol() == p['previous_protocol_sha256']
    assert sha(OUT / 'smoke_parity.json') == p['smoke_parity_sha256']
    return sha(OUT / 'protocol.json')


def check_selection():
    digest = verify_protocol()
    path = OUT / 'policy.json'
    p = json.loads(path.read_text())
    assert p['protocol_sha256'] == digest
    assert p['repair_choices'] == CURRENT_REPAIR['repair_choices']
    assert p['compact_choices'] == CURRENT_REPAIR['compact_choices']
    assert p['ber_protocol_sha256'] == sha(OUT / 'ber/protocol.json')
    for name, value in p['ber_source_hashes'].items():
        assert sha(REPO / name) == value
    return p, sha(path)


def anchors(regime, L, rsr, digest, selection):
    path = OUT / 'test/anchors' / f'{regime}_rsr{rsr:g}.npz'
    if path.exists():
        a = load(path)
        assert str(a['protocol_sha256']) == digest and str(a['selection_sha256']) == selection
        return a
    a = make_anchor(TEST_SEED, range(TEST_COUNT), L, rsr, truth=True)
    a.update(protocol_sha256=np.array(digest), selection_sha256=np.array(selection), regime=np.array(regime))
    save(path, **a)
    return a


@torch.no_grad()
def predictions(est, obs, selection):
    native, canonical = est['optimized'], est['canonical']
    original = torch.stack([native.original(obs, f'learned_{s}') for s in SEEDS])
    em = native.em100(obs)
    oldmask = native.masks(obs, OLD_POLICY)['bypass']
    old = torch.where(oldmask[None,:,None,None], em[None], original)
    priorchoice = selected_repair_choice(obs, PRIOR_REPAIR)
    choice = selected_repair_choice(obs, selection)
    cache = {'original': original, 'spectral100': em[None].expand(len(SEEDS),-1,-1,-1)}
    missing = tuple(n for n in dict.fromkeys((priorchoice, choice)) if n not in cache)
    if missing:
        values = [native.values(obs, f'learned_{s}', missing, initial=original[i], spectral=em)
                  for i, s in enumerate(SEEDS)]
        cache.update({n:torch.stack([v[n] for v in values]) for n in missing})
    current = torch.stack([canonical.repair(obs, f'learned_{s}', selection['repair_choices']) for s in SEEDS])
    compact = sparse_fast.estimate(obs, compact_config(obs, selection))
    outputs = [*original, *old, *cache[priorchoice], *current, *cache[choice],
               em, frozen_optshrink(obs, OPT_LOOKUP), compact, compact, sparse_fast.estimate(obs, base.FULL)]
    return torch.stack(outputs), oldmask, priorchoice, choice


@torch.no_grad()
def test(reference=None, boundary=False):
    selection, freeze = check_selection()
    digest, est = selection['protocol_sha256'], engine()
    for rsr in ((12.,20.) if boundary else RSRS):
        if reference is not None and reference != rsr:
            continue
        for regime, L in REGIMES:
            a = anchors(regime, L, rsr, digest, freeze)
            for snr in (BOUNDARY_SNRS if boundary else SNRS):
                path = OUT / ('boundary/cells' if boundary else 'test/cells') / (cellname(regime,rsr,snr)+'.npz')
                if path.exists():
                    assert str(load(path)['selection_sha256']) == freeze
                    continue
                obs = observed(a, snr)
                pred, oldmask, priorchoice, choice = predictions(est, obs, selection)
                arr = pred.numpy()
                assert np.isfinite(arr).all()
                delta = np.abs(arr[12:15] - arr[9:12])
                bound = ATOL + RTOL*np.abs(arr[9:12])
                assert np.all(delta <= bound), (regime,rsr,snr,float(delta.max()))
                maximum = 0.
                for i, name in enumerate(NAMES):
                    if name.startswith(('original_','old_bypass_','prior_repair_','current_repair_','repair_')):
                        actual = estimate_method(obs.batch(slice(0,1)), name, selection, est)
                        error = float((actual-pred[i,0:1]).abs().max())
                        assert error < ATOL
                        maximum = max(maximum,error)
                errors = (np.abs(arr-a['g'][None])**2).sum((-2,-1))
                energy = (np.abs(a['g'])**2).sum((-2,-1))
                save(path, **{k:v.numpy() for k,v in zip(('z','s','b','sigma2'),obs.args())},
                     truth_g=a['g'], paths=a['paths'], indices=a['indices'],local_indices=a['local_indices'],
                     method_names=np.array(NAMES), estimates=arr, errors=errors, truth_energy=energy,nmse=errors/energy[None],
                     old_bypass=oldmask.numpy(),prior_repair_choice=np.array(priorchoice),repair_choice=np.array(choice),
                     snr=np.array(snr),rsr=np.array(rsr),regime=np.array(regime),seed=np.array(TEST_SEED),
                     selection_sha256=np.array(freeze),protocol_sha256=np.array(digest),
                     actual_dispatch_max_abs_error=np.array(maximum),native_canonical_max_abs_error=np.array(delta.max()),
                     native_canonical_bitwise_equal=np.array(np.array_equal(arr[12:15],arr[9:12])))
                print('Test complete',path.stem,'max difference',float(delta.max()),flush=True)
    write_json(OUT/f'worker_{reference}_{boundary}.json',dict(phase='test_complete',selection_sha256=freeze,completed_unix=time.time()))


def timing_rows(indices):
    rows = []
    for index in indices:
        for rsr in RSRS:
            for regime,L in REGIMES:
                a = make_anchor(TIMING_SEED,[index],L,rsr,truth=False)
                for snr in SNRS:
                    rows.append((dict(snr=snr,rsr=rsr,regime=regime,index=index),observed(a,snr)))
    return rows


@torch.no_grad()
def timing(window):
    selection, freeze = check_selection()
    destination = OUT/f'timing_window{window}.json'
    assert not destination.exists(), 'Timing window already recorded; no retry-until-pass'
    assert window in (1,2)
    indices = (200,201,202) if window==1 else (300,301,302)
    names = NAMES if window==1 else tuple(n for n in NAMES if n.startswith(('current_repair_','repair_')) or n in ('em100','compact_sparse'))
    rows,est = timing_rows(indices),engine()
    for name in names:
        estimate_method(rows[0][1],name,selection,est)
    rng = np.random.default_rng(np.random.SeedSequence([TIMING_SEED,window]))
    samples = {name:[] for name in names}
    started = time.time()
    for wi in rng.permutation(len(rows)):
        meta,obs = rows[wi]
        choice = selected_repair_choice(obs,selection)
        priorchoice = selected_repair_choice(obs,PRIOR_REPAIR)
        oldmask = bool(est['optimized'].masks(obs,OLD_POLICY)['bypass'][0])
        for ni in rng.permutation(len(names)):
            name = names[ni]
            start = time.perf_counter_ns()
            g = estimate_method(obs,name,selection,est)
            ms = (time.perf_counter_ns()-start)/1e6
            assert torch.isfinite(g).all()
            branch = (choice if name.startswith(('repair_','current_repair_')) else priorchoice
                      if name.startswith('prior_repair_') else ('spectral100' if oldmask else 'original')
                      if name.startswith('old_bypass_') else 'original' if name.startswith('original_') else name)
            samples[name].append(dict(ms=ms,repeat=0,branch=branch,**meta))
        if len(samples[names[0]])%25==0:
            print('Timing',window,len(samples[names[0]]),'/',len(rows),flush=True)
    values = dict(samples)
    for name,parts in GROUPS.items():
        if all(n in samples for n in parts):
            values[name] = [x for n in parts for x in samples[n]]
    methods = {}
    for name,values_ in values.items():
        methods[name] = dict(**stats([v['ms'] for v in values_]),
            by_branch={b:stats([v['ms'] for v in values_ if v['branch']==b]) for b in sorted(set(v['branch'] for v in values_))},
            by_condition={f'rsr{r:g}_snr{s:g}':stats([v['ms'] for v in values_ if v['rsr']==r and v['snr']==s]) for r in RSRS for s in SNRS},
            weak_warm100=stats([v['ms'] for v in values_ if v['rsr']==5 and v['snr']>=37.5]),
            strong_high=stats([v['ms'] for v in values_ if v['rsr'] in (12,20) and v['snr']>=40]))
    result = dict(window=window,selection_sha256=freeze,protocol_sha256=selection['protocol_sha256'],
                  started_unix=started,completed_unix=time.time(),methods=methods,raw_samples=samples,
                  indices=indices,worlds=len(rows),repeats=1,
                  cap_verification=dict(cap_ms=CAP_MS,repair_p90_ms=methods['repair_mean']['p90_ms'],
                       passed=methods['repair_mean']['p90_ms']<=CAP_MS,
                       weak_warm100_p90_ms=methods['repair_mean']['weak_warm100']['p90_ms']),
                  environment=dict(platform=platform.platform(),machine=platform.machine(),python=platform.python_version(),
                       numpy=np.__version__,scipy=scipy.__version__,torch=torch.__version__,torch_threads=torch.get_num_threads()))
    write_json(destination,result)
    print('Timing complete',window,result['cap_verification'],flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('phase',choices=['prepare','test','boundary','timing1','timing2'])
    p.add_argument('--reference',type=float,choices=RSRS)
    args=p.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    {'prepare':freeze_protocol,'test':lambda:test(args.reference),'boundary':lambda:test(args.reference,True),
     'timing1':lambda:timing(1),'timing2':lambda:timing(2)}[args.phase]()


if __name__=='__main__':
    main()
