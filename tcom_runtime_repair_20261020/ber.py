"""Fixed full-alphabet BER evaluation for fresh native-solve estimates.

This module only evaluates cached channels. Checkpoint, policy, sparse-control,
scope and sampling decisions must be frozen before any test cell is generated.
"""
import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_key] = '1'
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from tcom_followups_20261017.ber_detector import detect, hypotheses, verify_detector
from tcom_pilot_em_20261018.training_protocol import sha, write_json
from . import study

REPO = Path(__file__).resolve().parents[1]
SOURCE = study.OUT
OUT = SOURCE / 'ber'
SNRS = tuple(study.SNRS)
RSRS = tuple(study.RSRS)
REGIMES = tuple(r for r, _ in study.REGIMES)
SEEDS = tuple(study.SEEDS)
CASES = study.TEST_COUNT
TEST_SEED = study.TEST_SEED
DATA_SEED = study.DATA_SEED
BOOTSTRAPS = 20000
BOOTSTRAP_SEED = study.BOOT_SEED
VECTORS = 64
BITS_PER_CASE = 3 * 2 * VECTORS
USED = tuple(study.NAMES)
NAMES = ('perfect_csi',) + USED
GROUPS = {f'{family}_mean': tuple(f'{family}_{seed}' for seed in SEEDS)
          for family in ('original', 'old_bypass', 'prior_repair', 'current_repair', 'repair')}
CONTRASTS = {
    'repair_vs_old': ('old_bypass_mean', 'repair_mean'),
    'repair_vs_prior': ('prior_repair_mean', 'repair_mean'),
    'repair_vs_current': ('current_repair_mean', 'repair_mean'),
    'repair_vs_original': ('original_mean', 'repair_mean'),
    'repair_vs_em100': ('em100', 'repair_mean'),
    'repair_vs_compact': ('compact_sparse', 'repair_mean'),
}
PRIMARY_CONTRASTS = ('repair_vs_current',)
PRIMARY_REGIONS = ('all', 'weak_warm100')
PARITY_ATOL = 1e-10
PARITY_RTOL = 1e-10
REGIONS = ('all', 'weak_warm100', 'strong_high', 'strong_transition', 'ordinary', 'weak',
           'rsr5', 'rsr12', 'rsr20', 'all_L2', 'all_L4', 'all_random3to7',
           'weak_warm100_L2', 'weak_warm100_L4', 'weak_warm100_random3to7')


def belongs(region, regime, rsr, snr):
    weak_warm = rsr == 5 and snr >= 37.5
    if region.startswith('all_'):
        return regime == region[4:]
    if region.startswith('weak_warm100_'):
        return weak_warm and regime == region[len('weak_warm100_'):]
    return dict(all=True, weak_warm100=weak_warm,
                strong_high=rsr in (12, 20) and snr >= 40,
                strong_transition=rsr in (12, 20) and 32.5 <= snr <= 37.5,
                ordinary=snr <= 20, weak=rsr == 5,
                rsr5=rsr == 5, rsr12=rsr == 12, rsr20=rsr == 20)[region]


def cells():
    return [(r, ref, s) for r in REGIMES for ref in RSRS for s in SNRS]


def label(regime, rsr, snr):
    return f'{regime}_rsr{rsr:g}_snr{snr:g}'


def load(path):
    with np.load(path, allow_pickle=False) as arrays:
        return {k: arrays[k] for k in arrays.files}


def save(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp.npz')
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def protocol_spec():
    expected = tuple(f'{family}_{seed}' for family in ('original', 'old_bypass', 'prior_repair', 'current_repair', 'repair') for seed in SEEDS)
    expected += ('em100', 'optshrink', 'prior_compact', 'compact_sparse', 'prior_sparse_full')
    assert USED == expected
    assert {key: tuple(value) for key, value in study.GROUPS.items()} == GROUPS
    assert SNRS == (0., 5., 10., 15., 20., 22.5, 25., 27.5, 30., 32.5, 35., 37.5, 40., 42.5, 45., 50., 60.)
    assert RSRS == (5., 12., 20.) and REGIMES == ('L2', 'L4', 'random3to7')
    assert SEEDS == (9101, 9102, 9103)
    assert CASES == 128 and TEST_SEED == 2026102002 and DATA_SEED == 2026102003
    assert BOOTSTRAP_SEED == 2026102009
    assert len(cells()) == 153
    assert sum(belongs('weak_warm100', *cell) for cell in cells()) == 18
    assert sum(belongs('strong_high', *cell) for cell in cells()) == 30
    assert sum(belongs('strong_transition', *cell) for cell in cells()) == 18
    audit = Path(__file__).with_name('ber_audit.py')
    assert audit.exists(), 'Independent audit must be ready before freeze'
    detector = REPO / 'tcom_followups_20261017/ber_detector.py'
    checkpoint_record = json.loads((REPO / 'results/tcom_pilot_em_20261018/evaluation/checkpoints.json').read_text())
    selected = {name: item for name, item in checkpoint_record['checkpoints'].items() if name.startswith('context30_')}
    assert len(selected) == 3
    for item in selected.values():
        assert sha(REPO / item['path']) == item['sha256']
    spec = dict(
        version=1, name='Native-solve runtime repair: fresh fixed-alphabet BER',
        N=32, K=3, P=30, snrs=SNRS, rsrs=RSRS, regimes=REGIMES, cases_per_cell=CASES,
        test_seed=TEST_SEED, data_seed=DATA_SEED, source_methods=USED,
        additional_reference='perfect_csi', checkpoint_record=selected, neural_groups=GROUPS,
        modulation='All 64 equally likely joint Gray QPSK vectors once per channel, unit power per user and six bits per vector. This stratifies the uniform alphabet; it is not 64 independent random symbol selections.',
        data_noise='Fresh proper complex Gaussian noise independent of pilot noise. SeedSequence([data_seed, regime_index, reference_index, local_index]) generates independent antenna/vector noise. Reuse the standardized noise across SNR and rescale by sqrt(sigma2).',
        identical_inputs='Every method sees identical new data magnitudes, known reference/variance, and the same exact 64-hypothesis Rician ML detector. The detector receives no coherent observation.',
        inference='Channel truth only generates data and the explicitly labeled perfect-CSI reference. No BER-dependent model, policy, sparse choice, cell selection or additional bit allocation.',
        sampling='Fixed 128 anchors per reference/path setting, 64 vectors per channel, 384 bits per channel and 49152 bits per cell/fit. No adaptive extra bits.',
        bounds='Zero observed errors and a collapsed empirical bootstrap interval do not establish zero BER, a rare-event bound, equivalence or population noninferiority. No independent-bit binomial intervals.',
        duplicates='Bitwise-identical full channel arrays may reuse detector counts. Every reuse is recorded and audited; detector reuse is not estimator timing.',
        primary_scope='Two scopes: all153 conditions and the18 weak-reference warm100 conditions (RSR5, SNR37.5/40/42.5/45/50/60); native repair versus canonical latest only. NMSE and BER form four outcomes.',
        numerical_parity=dict(atol=PARITY_ATOL, rtol=PARITY_RTOL,
            rule='Elementwise absolute error <= atol + rtol*abs(canonical channel), assessed separately from statistical inference. Non-bitwise native estimates are independently decoded.'),
        preservation_scope='Unchanged weights, ten learned stages, EM100 counts, routing, and final noise-scaled filters. Native solve changes numerical implementation only; numerical parity is not a population BER-equivalence claim.',
        primary_contrasts={name: CONTRASTS[name] for name in PRIMARY_CONTRASTS},
        descriptive_contrasts={name: pair for name, pair in CONTRASTS.items() if name not in PRIMARY_CONTRASTS},
        improvement_sign='reference BER minus candidate BER',
        descriptive_scopes=REGIONS,
        bootstrap=dict(resamples=BOOTSTRAPS, seed=BOOTSTRAP_SEED,
                       rule='Multinomial 128-case weights in regime order L2,L4,random3to7, shared across every reference/SNR/method/fit within regime; regimes independent.',
                       pointwise_quantiles=[.025, .975], primary_bonferroni4_quantiles=[.05 / 8, 1 - .05 / 8],
                       interpretation='Conditional on fixed weights/policies. One primary contrast in two scopes times NMSE/BER forms four outcomes. Compact and older-method comparisons are descriptive. Three fits average individual metrics, never predictions, and do not triple the number of channels.'),
        source_hashes={str(Path(__file__).relative_to(REPO)): sha(__file__), str(audit.relative_to(REPO)): sha(audit), str(detector.relative_to(REPO)): sha(detector)},
    )
    return json.loads(json.dumps(spec))


def freeze():
    spec = protocol_spec()
    path = OUT / 'protocol.json'
    if path.exists():
        old = json.loads(path.read_text())
        old.pop('frozen_unix', None)
        if old != spec:
            raise RuntimeError('BER source/protocol changed after freeze')
    else:
        assert not list((SOURCE / 'test').glob('**/*.npz')), 'Freeze before any test anchors or cells'
        write_json(path, dict(**spec, frozen_unix=time.time()))
    return sha(path)


def selection_hash():
    _, digest = study.check_selection()
    return digest


def standard_noise(regime, rsr, local):
    rng = np.random.default_rng(np.random.SeedSequence([DATA_SEED, REGIMES.index(regime), RSRS.index(rsr), int(local)]))
    return (rng.standard_normal((32, VECTORS)) + 1j * rng.standard_normal((32, VECTORS))) / np.sqrt(2)


def data_for(channel, reference, sigma2, regime, rsr, local):
    x, bits = hypotheses(3)
    z = np.abs(channel @ x.T + np.asarray(reference).reshape(32, 1) + np.sqrt(sigma2) * standard_noise(regime, rsr, local))
    return z, bits


def evaluate(digest, available_only=False, reference=None):
    selected = selection_hash()
    complete = True
    if reference is not None and reference not in RSRS:
        raise ValueError('reference must be one of the frozen reference levels')
    selected_cells = [cell for cell in cells() if reference is None or cell[1] == reference]
    assert len(selected_cells) == (153 if reference is None else 51)
    for regime, rsr, snr in selected_cells:
        name = label(regime, rsr, snr)
        source = SOURCE / 'test/cells' / (name + '.npz')
        destination = OUT / 'cells' / (name + '.npz')
        if not source.exists() and available_only:
            complete = False
            continue
        if destination.exists():
            old = load(destination)
            assert str(old['protocol_sha256']) == digest and str(old['source_sha256']) == sha(source)
            assert str(old['selection_sha256']) == selected
            print('BER cached', name, flush=True)
            continue
        a = load(source)
        assert str(a['selection_sha256']) == selected and int(a['seed']) == TEST_SEED
        assert str(a['protocol_sha256']) == sha(SOURCE / 'protocol.json')
        assert float(a['snr']) == snr and float(a['rsr']) == rsr and str(a['regime']) == regime
        assert tuple(map(str, a['method_names'])) == USED
        assert a['z'].shape == (CASES, 32, 30) and a['s'].shape == (CASES, 3, 30)
        offset = {'L2': 100000, 'L4': 200000, 'random3to7': 0}[regime]
        np.testing.assert_array_equal(a['indices'], np.arange(CASES) + offset)
        np.testing.assert_array_equal(a['local_indices'], np.arange(CASES))
        truth = a['truth_g']
        assert truth.shape == (CASES, 32, 3)
        channels = np.concatenate((truth[None], a['estimates']))
        assert channels.shape == (len(NAMES), CASES, 32, 3) and np.isfinite(channels).all()
        reference, variance = a['b'][..., :1], a['sigma2']
        assert np.array_equal(a['b'], np.broadcast_to(reference, a['b'].shape))
        assert variance.shape == (CASES,) and np.all(variance > 0)
        assert np.isfinite(reference).all() and np.isfinite(variance).all()
        data = np.stack([data_for(truth[i], reference[i], float(variance[i]), regime, rsr, i)[0] for i in range(CASES)])
        bits = hypotheses(3)[1]
        datapath = OUT / 'data' / (name + '.npz')
        save(datapath, z=data, b=reference, sigma2=variance, indices=a['indices'], local_indices=a['local_indices'],
             transmitted_bits=bits, hypothesis_symbols=hypotheses(3)[0], protocol_sha256=digest)
        errors = np.zeros((len(NAMES), CASES, 3), dtype=np.int64)
        seconds = np.zeros((len(NAMES), CASES))
        reuse, seen = [], {}
        for mi, method in enumerate(NAMES):
            fingerprint = hashlib.sha256(channels[mi].tobytes()).hexdigest()
            previous = seen.get(fingerprint)
            if previous is not None:
                assert np.array_equal(channels[mi], channels[previous])
                errors[mi] = errors[previous]
                reuse.append(NAMES[previous])
                continue
            seen[fingerprint] = mi
            reuse.append('')
            for i in range(CASES):
                start = time.perf_counter()
                decoded = detect(data[i], channels[mi, i], reference[i], float(variance[i]))
                seconds[mi, i] = time.perf_counter() - start
                errors[mi, i] = np.count_nonzero(decoded != bits, axis=(0, 2))
        save(destination, method_names=np.array(NAMES), errors_per_user=errors,
             bits_per_user=np.full((CASES, 3), 2 * VECTORS, dtype=np.int64), duplicate_of=np.array(reuse),
             detector_wall_seconds=seconds, indices=a['indices'], local_indices=a['local_indices'], source_sha256=sha(source),
             data_sha256=sha(datapath), protocol_sha256=digest, selection_sha256=selected)
        print(json.dumps(dict(cell=name, unique_detector_methods=len(seen), methods=len(NAMES),
                              total_errors={method: int(errors[mi].sum()) for mi, method in enumerate(NAMES)})), flush=True)
    return complete


def interval(samples, adjusted=False):
    return np.quantile(samples, [.05 / 8, 1 - .05 / 8] if adjusted else [.025, .975]).tolist()


def scope_cells():
    allcells = cells()
    scopes = {region: [c for c in allcells if belongs(region, *c)] for region in REGIONS}
    scopes.update({f'snr:{snr:g}': [c for c in allcells if c[2] == snr] for snr in SNRS})
    return scopes


def analyze(digest):
    selected = selection_hash()
    names = NAMES + tuple(GROUPS)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    weights = {r: rng.multinomial(CASES, np.full(CASES, 1 / CASES), size=BOOTSTRAPS).astype(float) for r in REGIMES}
    scopes = scope_cells()
    memberships = {cell: [key for key, members in scopes.items() if cell in members] for cell in cells()}
    accum, rows, cell_contrasts = {}, [], []
    for regime, rsr, snr in cells():
        a = load(OUT / 'cells' / (label(regime, rsr, snr) + '.npz'))
        assert tuple(map(str, a['method_names'])) == NAMES
        assert str(a['protocol_sha256']) == digest and str(a['selection_sha256']) == selected
        counts = a['errors_per_user'].sum(-1)
        rate = counts / BITS_PER_CASE
        rate = np.concatenate((rate, np.stack([rate[[NAMES.index(n) for n in members]].mean(0) for members in GROUPS.values()])))
        samples, point = rate @ weights[regime].T / CASES, rate.mean(-1)
        for contrast, (reference, candidate) in CONTRASTS.items():
            ri, ci = names.index(reference), names.index(candidate)
            cell_contrasts.append(dict(regime=regime, rsr_db=rsr, snr_db=snr, contrast=contrast,
                                       reference=reference, candidate=candidate, ber_reduction=float(point[ri] - point[ci]),
                                       bootstrap95=interval(samples[ri] - samples[ci]), interpretation='descriptive pointwise'))
        for key in memberships[(regime, rsr, snr)]:
            if key not in accum:
                accum[key] = dict(cells=0, rates=np.zeros(len(names)), samples=np.zeros_like(samples),
                                  errors=np.zeros(len(NAMES), dtype=np.int64), case_errors=np.zeros(len(NAMES), dtype=np.int64), regimes=set())
            store = accum[key]
            store['cells'] += 1
            store['rates'] += point
            store['samples'] += samples
            store['errors'] += counts.sum(-1)
            store['case_errors'] += (counts > 0).sum(-1)
            store['regimes'].add(regime)
        for mi, method in enumerate(names):
            row = dict(regime=regime, rsr_db=rsr, snr_db=snr, method=method, ber=float(point[mi]),
                       bootstrap95=interval(samples[mi]), bits_per_fit=CASES * BITS_PER_CASE, channel_anchors=CASES)
            if method in GROUPS:
                row['errors_per_training_seed'] = {n: int(counts[NAMES.index(n)].sum()) for n in GROUPS[method]}
                row['channels_with_errors_per_fit'] = {n: int((counts[NAMES.index(n)] > 0).sum()) for n in GROUPS[method]}
            else:
                row.update(bit_errors=int(counts[mi].sum()), channels_with_errors=int((counts[mi] > 0).sum()))
            rows.append(row)

    def finish(key):
        store = accum[key]
        ncell = store['cells']
        point, samples = store['rates'] / ncell, store['samples'] / ncell
        methods, contrasts = {}, {}
        for mi, name in enumerate(names):
            row = dict(ber=float(point[mi]), bootstrap95=interval(samples[mi]), bits_per_fit=ncell * CASES * BITS_PER_CASE)
            if name in GROUPS:
                row['errors_per_training_seed'] = {n: int(store['errors'][NAMES.index(n)]) for n in GROUPS[name]}
                row['operating_cases_with_errors_per_fit'] = {n: int(store['case_errors'][NAMES.index(n)]) for n in GROUPS[name]}
                seed_rates = [float(point[NAMES.index(n)]) for n in GROUPS[name]]
                row['training_seed_ber_range'] = [min(seed_rates), max(seed_rates)]
            else:
                row.update(bit_errors=int(store['errors'][mi]), operating_cases_with_errors=int(store['case_errors'][mi]))
            methods[name] = row
        for contrast, (reference, candidate) in CONTRASTS.items():
            ri, ci = names.index(reference), names.index(candidate)
            rep = samples[ri] - samples[ci]
            record = dict(reference=reference, candidate=candidate, ber_reduction=float(point[ri] - point[ci]), bootstrap95=interval(rep))
            record['both_methods_zero_errors'] = bool(point[ri] == 0 and point[ci] == 0)
            if record['both_methods_zero_errors']:
                record['interpretation'] = 'Zero observed errors in both methods; a degenerate empirical interval does not establish equivalence or population noninferiority.'
            if key in PRIMARY_REGIONS and contrast in PRIMARY_CONTRASTS:
                adjusted = interval(rep, True)
                record.update(bonferroni4_interval=adjusted, positive_reduction_supported_familywise=adjusted[0] > 0)
            contrasts[contrast] = record
        return dict(cells=ncell, operating_cases=ncell * CASES, coupled_clusters=len(store['regimes']) * CASES,
                    methods=methods, contrasts=contrasts)

    regions = {region: finish(region) for region in REGIONS}
    summary = dict(protocol_sha256=digest, selection_sha256=selected,
                   regions=regions, overall=regions['all'], weak_warm100=regions['weak_warm100'],
                   primary={region: regions[region] for region in PRIMARY_REGIONS},
                   by_snr={f'{s:g}': finish(f'snr:{s:g}') for s in SNRS},
                   by_rsr={f'{r:g}': regions[f'rsr{r:g}'] for r in RSRS},
                   by_regime={r: regions[f'all_{r}'] for r in REGIMES},
                   cells=rows, cell_contrasts=cell_contrasts,
                   confidence='20000 paired coupled-block resamples;128 local indices shared across references/SNRs within each regime,384 clusters. Conditional on fixed models/policies. Bonferroni4 only for native-versus-canonical in whole-grid and weak-warm100 scopes across NMSE/BER.',
                   zero_errors='Zero observed errors and degenerate empirical intervals do not establish rare-event bounds, equivalence or population noninferiority. Bit counts are not independent channel trials.',
                   coverage='64 joint QPSK vectors once per channel,128 anchors per reference/regime;1152 physical anchors reused across17 SNRs. No dense-boundary BER or adaptive extra bits.')
    write_json(OUT / 'summary.json', summary)
    plot(rows)
    print(json.dumps({region: regions[region]['contrasts']['repair_vs_current'] for region in PRIMARY_REGIONS}), flush=True)


def plot(rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    styles = {
        'perfect_csi': ('Perfect CSI', '#777777', '--'),
        'original_mean': ('Retrained filtered', '#333333', '-'),
        'old_bypass_mean': ('Previous bypass', '#009e73', ':'),
        'prior_repair_mean': ('Prior repair', '#8b5a2b', '--'),
        'current_repair_mean': ('Canonical latest', '#e69f00', '-.'),
        'repair_mean': ('Native solve', '#0072b2', '-'),
        'em100': ('Direct EM100', '#cc79a7', '--'),
        'compact_sparse': ('Frozen compact sparse', '#d55e00', '-'),
    }
    fig, axes = plt.subplots(3, 3, figsize=(15, 10), sharex=True, sharey=True)
    floor = 1e-6
    for ri, rsr in enumerate(RSRS):
        for ci, regime in enumerate(REGIMES):
            ax = axes[ri, ci]
            if rsr == 5:
                ax.axvspan(37.5, 60, color='#0072b2', alpha=.055)
            for method, (label_text, color, style) in styles.items():
                selected = [r for r in rows if r['regime'] == regime and r['rsr_db'] == rsr and r['method'] == method]
                values = np.array([r['ber'] for r in selected])
                positive = values > 0
                ax.plot(SNRS, np.where(positive, values, np.nan), marker='o', markersize=2.6, color=color, linestyle=style, label=label_text)
                ax.scatter(np.array(SNRS)[~positive], np.full((~positive).sum(), floor), marker='v', s=13, color=color)
            ax.set_yscale('log')
            ax.set_ylim(.65e-6, .6)
            ax.set_xticks([0, 20, 30, 40, 50, 60])
            ax.grid(alpha=.18, which='both')
            if ri == 0:
                ax.set_title({'L2': '2 paths/user', 'L4': '4 paths/user', 'random3to7': '3–7 paths/user'}[regime])
            if ci == 0:
                ax.set_ylabel(f'RSR {rsr:g} dB\nUncoded BER')
            if ri == 2:
                ax.set_xlabel('SNR (dB)')
    handles, labels = axes[0, 0].get_legend_handles_labels()
    handles.append(Line2D([0], [0], color='#555555', marker='v', linestyle='none'))
    labels.append('Zero observed errors (marker only)')
    fig.legend(handles, labels, loc='lower center', ncol=3, fontsize=8, bbox_to_anchor=(.5, .023))
    fig.suptitle('Native solve: identical magnitude-domain data and detector', fontsize=12)
    fig.text(.5, .009, 'Shading: weak warm100 scope; whole grid is also primary. Zero-error markers at 10⁻⁶ are display markers, not estimates or bounds.', ha='center', fontsize=8)
    fig.tight_layout(rect=(0, .115, 1, .97))
    fig.savefig(OUT / 'ber.png', dpi=180, bbox_inches='tight')
    fig.savefig(OUT / 'ber.pdf', bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reference', type=float, choices=RSRS, help='Evaluate only this reference partition; never writes a shared summary.')
    parser.add_argument('--preflight', action='store_true', help='Validate the interface and detector without writing a protocol or generating data.')
    parser.add_argument('--freeze-only', action='store_true')
    parser.add_argument('--available-only', action='store_true')
    parser.add_argument('--analyze-only', action='store_true')
    args = parser.parse_args()
    if args.reference is not None and args.analyze_only:
        parser.error('--reference cannot be combined with --analyze-only')
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if args.preflight:
        spec = protocol_spec()
        print(json.dumps(dict(passed=True, source_methods=len(USED), neural_groups=len(GROUPS),
                              cells=len(cells()), primary_cells={r: sum(belongs(r, *cell) for cell in cells()) for r in PRIMARY_REGIONS},
                              reference_partition_cells=51, detector_checks=verify_detector(),
                              source_hashes=spec['source_hashes'], protocol_written=False)), flush=True)
        return
    digest = freeze()
    if args.freeze_only:
        return
    checks_name = 'detector_checks.json' if args.reference is None else f'detector_checks_rsr{args.reference:g}.json'
    write_json(OUT / checks_name, verify_detector())
    if args.reference is not None:
        complete = evaluate(digest, args.available_only, args.reference)
        write_json(OUT / f'reference_status_rsr{args.reference:g}.json', dict(reference=args.reference, complete=complete, protocol_sha256=digest, selection_sha256=selection_hash()))
        return
    if not args.analyze_only and not evaluate(digest, args.available_only):
        return
    analyze(digest)


if __name__ == '__main__':
    main()
