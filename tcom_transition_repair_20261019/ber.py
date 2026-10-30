"""Fixed full-alphabet BER evaluation for fresh transition-repair estimates.

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
          for family in ('original', 'old_bypass', 'repair')}
CONTRASTS = {
    'repair_vs_old': ('old_bypass_mean', 'repair_mean'),
    'repair_vs_original': ('original_mean', 'repair_mean'),
    'repair_vs_em100': ('em100', 'repair_mean'),
    'repair_vs_compact': ('compact_sparse', 'repair_mean'),
}
PRIMARY_CONTRASTS = ('repair_vs_old', 'repair_vs_original')


def primary(regime, rsr, snr):
    return rsr == 5 and 32.5 <= snr <= 45


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


def freeze():
    expected = tuple(f'{family}_{seed}' for family in ('original', 'old_bypass', 'repair') for seed in SEEDS)
    expected += ('em100', 'optshrink', 'prior_compact', 'compact_sparse', 'prior_sparse_full')
    assert USED == expected
    assert CASES == 128 and TEST_SEED == 2026101922 and DATA_SEED == 2026101923
    assert len(cells()) == 153 and sum(primary(*cell) for cell in cells()) == 18
    detector = REPO / 'tcom_followups_20261017/ber_detector.py'
    checkpoint_record = json.loads((REPO / 'results/tcom_pilot_em_20261018/evaluation/checkpoints.json').read_text())
    selected = {name: item for name, item in checkpoint_record['checkpoints'].items() if name.startswith('context30_')}
    assert len(selected) == 3
    for item in selected.values():
        assert sha(REPO / item['path']) == item['sha256']
    spec = dict(
        version=1, name='Weak-reference transition repair: fresh fixed-alphabet BER',
        N=32, K=3, P=30, snrs=SNRS, rsrs=RSRS, regimes=REGIMES, cases_per_cell=CASES,
        test_seed=TEST_SEED, data_seed=DATA_SEED, source_methods=USED,
        additional_reference='perfect_csi', checkpoint_record=selected, neural_groups=GROUPS,
        modulation='All 64 equally likely joint Gray QPSK vectors once per channel, unit power per user and six bits per vector. This stratifies the uniform alphabet; it is not 64 independent random symbol selections.',
        data_noise='Fresh proper complex Gaussian noise independent of pilot noise. SeedSequence([data_seed, regime_index, reference_index, local_index]) generates independent antenna/vector noise. Reuse the standardized noise across SNR and rescale by sqrt(sigma2).',
        identical_inputs='Every method sees identical new data magnitudes, known reference/variance, and the same exact 64-hypothesis Rician ML detector. The detector receives no coherent observation.',
        inference='Channel truth only generates data and the explicitly labeled perfect-CSI reference. No BER-dependent model, policy, sparse choice, cell selection or additional bit allocation.',
        sampling='Fixed 128 anchors per reference/path setting, 64 vectors per channel, 384 bits per channel and 49152 bits per cell/fit. No adaptive extra bits.',
        bounds='Zero observed errors and a collapsed empirical bootstrap interval do not establish zero BER or a rare-event bound. No independent-bit binomial intervals.',
        duplicates='Bitwise-identical full channel arrays may reuse detector counts. Every reuse is recorded and audited; detector reuse is not estimator timing.',
        primary_scope='RSR=5 dB and SNR in [32.5,45] dB, all three path regimes: 18 equally weighted cells.',
        primary_contrasts={name: CONTRASTS[name] for name in PRIMARY_CONTRASTS},
        descriptive_contrasts={name: pair for name, pair in CONTRASTS.items() if name not in PRIMARY_CONTRASTS},
        improvement_sign='reference BER minus candidate BER',
        descriptive_scopes=dict(ordinary='SNR<=20', transition='20<SNR<=45', stress='SNR in {50,60}', high_snr='SNR>20', overall='All 153 cells'),
        bootstrap=dict(resamples=BOOTSTRAPS, seed=BOOTSTRAP_SEED,
                       rule='Multinomial 128-case weights in regime order L2,L4,random3to7, shared across every reference/SNR/method/fit within regime; regimes independent.',
                       pointwise_quantiles=[.025, .975], primary_bonferroni4_quantiles=[.05 / 8, 1 - .05 / 8],
                       interpretation='Conditional on fixed weights/policies. Two primary contrasts times NMSE/BER form four outcomes. Three fits average individual metrics, never predictions, and do not triple the number of channels.'),
        source_hashes={str(Path(__file__).relative_to(REPO)): sha(__file__), str(detector.relative_to(REPO)): sha(detector)},
    )
    spec = json.loads(json.dumps(spec))
    path = OUT / 'protocol.json'
    if path.exists():
        old = json.loads(path.read_text())
        old.pop('frozen_unix', None)
        if old != spec:
            raise RuntimeError('BER source/protocol changed after freeze')
    else:
        assert not list((SOURCE / 'test/cells').glob('*.npz')), 'Freeze before any test cells'
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


def evaluate(digest, available_only=False):
    selected = selection_hash()
    complete = True
    for regime, rsr, snr in cells():
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
    scopes = {'overall': allcells, 'primary': [c for c in allcells if primary(*c)],
              'ordinary': [c for c in allcells if c[2] <= 20], 'high_snr': [c for c in allcells if c[2] > 20],
              'transition': [c for c in allcells if 20 < c[2] <= 45], 'stress': [c for c in allcells if c[2] in (50, 60)]}
    for snr in SNRS:
        scopes[f'snr:{snr:g}'] = [c for c in allcells if c[2] == snr]
    for rsr in RSRS:
        scopes[f'rsr:{rsr:g}'] = [c for c in allcells if c[1] == rsr]
        scopes[f'high_rsr:{rsr:g}'] = [c for c in allcells if c[1] == rsr and c[2] > 20]
    for regime in REGIMES:
        scopes[f'regime:{regime}'] = [c for c in allcells if c[0] == regime]
        scopes[f'high_regime:{regime}'] = [c for c in allcells if c[0] == regime and c[2] > 20]
        scopes[f'primary_regime:{regime}'] = [c for c in allcells if c[0] == regime and primary(*c)]
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
            if key == 'primary' and contrast in PRIMARY_CONTRASTS:
                adjusted = interval(rep, True)
                record.update(bonferroni4_interval=adjusted, positive_reduction_supported_familywise=adjusted[0] > 0)
            contrasts[contrast] = record
        return dict(cells=ncell, operating_cases=ncell * CASES, coupled_clusters=len(store['regimes']) * CASES,
                    methods=methods, contrasts=contrasts)

    summary = {key: finish(key) for key in ('primary', 'overall', 'ordinary', 'high_snr', 'transition', 'stress')}
    summary.update(protocol_sha256=digest, selection_sha256=selected,
                   by_snr={f'{s:g}': finish(f'snr:{s:g}') for s in SNRS},
                   by_rsr={f'{r:g}': finish(f'rsr:{r:g}') for r in RSRS},
                   by_regime={r: finish(f'regime:{r}') for r in REGIMES},
                   high_snr_by_rsr={f'{r:g}': finish(f'high_rsr:{r:g}') for r in RSRS},
                   high_snr_by_regime={r: finish(f'high_regime:{r}') for r in REGIMES},
                   primary_by_regime={r: finish(f'primary_regime:{r}') for r in REGIMES},
                   cells=rows, cell_contrasts=cell_contrasts,
                   confidence='20000 paired coupled-block resamples; 128 local indices shared across all references/SNRs within regime, 384 resampling clusters. Conditional on fixed models/policies. Bonferroni4 only for the two primary-window contrasts across NMSE/BER cooutcomes.',
                   zero_errors='Zero observed errors and degenerate empirical intervals are not zero-BER or rare-event guarantees. One error per cell/fit is 1/49152; bits within a channel are not independent channel trials.',
                   coverage='All 64 joint QPSK vectors once per channel with independent antenna/vector data noise. 1152 physical anchors reused across 17 SNRs, 19584 operating cases. No adaptive extra bits.')
    summary['regions'] = {key: summary[key] for key in ('ordinary', 'transition', 'stress', 'high_snr')}
    summary['regions'].update(all=summary['overall'], weak_transition=summary['primary'])
    summary['regions'].update({f'weak_transition_{regime}': summary['primary_by_regime'][regime] for regime in REGIMES})
    write_json(OUT / 'summary.json', summary)
    plot(rows)
    print(json.dumps(summary['primary']['contrasts']), flush=True)


def plot(rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    styles = {
        'perfect_csi': ('Perfect CSI', '#777777', '--'),
        'original_mean': ('Retrained filtered', '#333333', '-'),
        'old_bypass_mean': ('Previous bypass', '#009e73', ':'),
        'repair_mean': ('Transition repair', '#0072b2', '-'),
        'em100': ('Direct EM100', '#cc79a7', '--'),
        'compact_sparse': ('Fresh compact sparse', '#d55e00', '-'),
    }
    fig, axes = plt.subplots(3, 3, figsize=(15, 10), sharex=True, sharey=True)
    floor = 1e-6
    for ri, rsr in enumerate(RSRS):
        for ci, regime in enumerate(REGIMES):
            ax = axes[ri, ci]
            if rsr == 5:
                ax.axvspan(32.5, 45, color='#0072b2', alpha=.055)
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
    fig.legend(handles, labels, loc='lower center', ncol=4, fontsize=8, bbox_to_anchor=(.5, .023))
    fig.suptitle('Weak-reference transition repair: identical magnitude-domain data and detector', fontsize=12)
    fig.text(.5, .009, 'Shading: prespecified primary window. Zero-error markers at 10⁻⁶ are display markers, not estimates or bounds.', ha='center', fontsize=8)
    fig.tight_layout(rect=(0, .09, 1, .97))
    fig.savefig(OUT / 'ber.png', dpi=180, bbox_inches='tight')
    fig.savefig(OUT / 'ber.pdf', bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--freeze-only', action='store_true')
    parser.add_argument('--available-only', action='store_true')
    parser.add_argument('--analyze-only', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    digest = freeze()
    if args.freeze_only:
        return
    write_json(OUT / 'detector_checks.json', verify_detector())
    if not args.analyze_only and not evaluate(digest, args.available_only):
        return
    analyze(digest)


if __name__ == '__main__':
    main()
