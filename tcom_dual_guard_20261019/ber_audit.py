"""Independent checks of the fresh dual-guard BER artifacts."""
import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_key] = '1'
import json
import time
import numpy as np
from . import ber
from tcom_followups_20261017.ber_detector import detect, hypotheses, verify_detector
from tcom_pilot_em_20261018.training_protocol import sha, write_json


def close(actual, expected, tolerance=1e-15):
    error = float(np.max(np.abs(np.asarray(actual) - np.asarray(expected))))
    assert error < tolerance, (actual, expected, error)
    return error


def run():
    out = ber.OUT
    summary = json.loads((out / 'summary.json').read_text())
    protocol = json.loads((out / 'protocol.json').read_text())
    policy, selected = ber.study.check_selection()
    digest = sha(out / 'protocol.json')
    assert summary['protocol_sha256'] == digest and summary['selection_sha256'] == selected
    assert policy['ber_protocol_sha256'] == digest
    assert policy['frozen_unix'] >= protocol['frozen_unix']
    for name, source_hash in protocol['source_hashes'].items():
        assert sha(ber.REPO / name) == source_hash
    assert policy['ber_source_hashes'] == protocol['source_hashes']
    for checkpoint in protocol['checkpoint_record'].values():
        assert sha(ber.REPO / checkpoint['path']) == checkpoint['sha256']
    assert ber.CASES == 128 and ber.TEST_SEED == 2026101932 and ber.DATA_SEED == 2026101933
    assert ber.BOOTSTRAP_SEED == 2026101939 and ber.BOOTSTRAPS == 20000
    x, bits = hypotheses(3)
    assert x.shape == (64, 3) and len(np.unique(x, axis=0)) == 64
    np.testing.assert_allclose(np.abs(x) ** 2, 1, rtol=0, atol=1e-15)
    np.testing.assert_array_equal(bits.sum(0), np.full((3, 2), 32))
    names = ber.NAMES + tuple(ber.GROUPS)
    lookup = {(r['regime'], r['rsr_db'], r['snr_db'], r['method']): r for r in summary['cells']}
    contrast_lookup = {(r['regime'], r['rsr_db'], r['snr_db'], r['contrast']): r for r in summary['cell_contrasts']}
    assert len(lookup) == 153 * len(names) and len(contrast_lookup) == 153 * 5
    records, anchors, cell_hashes = [], {}, {}
    decodes = reuses = data_cases = 0
    maximum_data_error = 0.
    primary_deltas = {r: np.zeros((2, 128)) for r in ('L2', 'L4', 'random3to7')}
    for regime, rsr, snr in ber.cells():
        label = ber.label(regime, rsr, snr)
        path = out / 'cells' / (label + '.npz')
        a = ber.load(path)
        sourcepath = ber.SOURCE / 'test/cells' / (label + '.npz')
        source = ber.load(sourcepath)
        assert sourcepath.stat().st_mtime >= policy['frozen_unix']
        assert str(source['selection_sha256']) == selected and int(source['seed']) == 2026101932
        assert str(a['protocol_sha256']) == digest and str(a['selection_sha256']) == selected
        assert str(a['source_sha256']) == sha(sourcepath)
        datapath = out / 'data' / (label + '.npz')
        assert str(a['data_sha256']) == sha(datapath)
        data = ber.load(datapath)
        assert str(data['protocol_sha256']) == digest
        np.testing.assert_array_equal(data['transmitted_bits'], bits)
        np.testing.assert_array_equal(data['hypothesis_symbols'], x)
        np.testing.assert_array_equal(data['b'], source['b'][..., :1])
        np.testing.assert_array_equal(data['sigma2'], source['sigma2'])
        assert data['z'].shape == (128, 32, 64) and np.isfinite(data['z']).all() and np.all(data['z'] >= 0)
        for key in ('indices', 'local_indices'):
            np.testing.assert_array_equal(a[key], source[key])
            np.testing.assert_array_equal(data[key], source[key])
        np.testing.assert_array_equal(a['local_indices'], np.arange(128))
        assert tuple(map(str, a['method_names'])) == ber.NAMES
        source_names = list(map(str, source['method_names']))
        assert tuple(source_names) == ber.USED
        np.testing.assert_array_equal(a['bits_per_user'], np.full((128, 3), 128))
        assert a['errors_per_user'].shape == (len(ber.NAMES), 128, 3)
        assert np.issubdtype(a['errors_per_user'].dtype, np.integer)
        assert np.all(a['errors_per_user'] >= 0) and np.all(a['errors_per_user'] <= 128)
        anchor_key = (regime, rsr)
        current = {k: source[k] for k in ('truth_g', 'b', 'indices', 'local_indices', 'paths')}
        if anchor_key in anchors:
            for key, value in current.items():
                np.testing.assert_array_equal(value, anchors[anchor_key][key])
        else:
            anchors[anchor_key] = current
        for case in (0, 127):
            # Independently reconstruct the documented stream without calling
            # the production data_for or standard_noise helper.
            rng = np.random.default_rng(np.random.SeedSequence([
                2026101933, ('L2', 'L4', 'random3to7').index(regime), (5., 12., 20.).index(rsr), case]))
            unit = (rng.standard_normal((32, 64)) + 1j * rng.standard_normal((32, 64))) / np.sqrt(2)
            z = np.abs(source['truth_g'][case] @ x.T + source['b'][case, :, :1] + np.sqrt(source['sigma2'][case]) * unit)
            np.testing.assert_array_equal(z, data['z'][case])
            maximum_data_error = max(maximum_data_error, float(np.max(abs(z - data['z'][case]))))
            data_cases += 1
            for name in ('perfect_csi', 'original_9101', 'old_bypass_9102', 'prior_repair_9101', 'repair_9103', 'em100', 'compact_sparse'):
                channel = source['truth_g'][case] if name == 'perfect_csi' else source['estimates'][source_names.index(name), case]
                decoded = detect(data['z'][case], channel, data['b'][case], float(data['sigma2'][case]))
                errors = np.count_nonzero(decoded != bits, axis=(0, 2))
                np.testing.assert_array_equal(errors, a['errors_per_user'][ber.NAMES.index(name), case])
                decodes += 1
        for mi, name in enumerate(ber.NAMES):
            other = str(a['duplicate_of'][mi])
            seconds = a['detector_wall_seconds'][mi]
            if other:
                assert ber.NAMES.index(other) < mi
                channel = source['truth_g'] if name == 'perfect_csi' else source['estimates'][source_names.index(name)]
                other_channel = source['truth_g'] if other == 'perfect_csi' else source['estimates'][source_names.index(other)]
                np.testing.assert_array_equal(channel, other_channel)
                np.testing.assert_array_equal(a['errors_per_user'][mi], a['errors_per_user'][ber.NAMES.index(other)])
                np.testing.assert_array_equal(seconds, np.zeros(128))
                reuses += 1
            else:
                assert np.all(np.isfinite(seconds)) and np.all(seconds > 0)
        counts = a['errors_per_user'].sum(-1)
        rates = counts / 384
        rates = np.concatenate((rates, np.stack([rates[[ber.NAMES.index(n) for n in members]].mean(0) for members in ber.GROUPS.values()])))
        if snr <= 20:
            for seed in ber.SEEDS:
                for branch in ('old_bypass', 'prior_repair', 'repair'):
                    np.testing.assert_array_equal(source['estimates'][source_names.index(f'original_{seed}')],
                                                  source['estimates'][source_names.index(f'{branch}_{seed}')])
                    np.testing.assert_array_equal(counts[ber.NAMES.index(f'original_{seed}')], counts[ber.NAMES.index(f'{branch}_{seed}')])
        if rsr == 5:
            for seed in ber.SEEDS:
                np.testing.assert_array_equal(source['estimates'][source_names.index(f'prior_repair_{seed}')],
                                              source['estimates'][source_names.index(f'repair_{seed}')])
                np.testing.assert_array_equal(counts[ber.NAMES.index(f'prior_repair_{seed}')], counts[ber.NAMES.index(f'repair_{seed}')])
        for mi, name in enumerate(names):
            row = lookup[(regime, rsr, snr, name)]
            close(row['ber'], float(rates[mi].mean()))
            assert row['bits_per_fit'] == 49152 and row['channel_anchors'] == 128
            if name in ber.GROUPS:
                assert row['errors_per_training_seed'] == {n: int(counts[ber.NAMES.index(n)].sum()) for n in ber.GROUPS[name]}
            else:
                assert row['bit_errors'] == int(counts[mi].sum())
        for contrast, (reference, candidate) in ber.CONTRASTS.items():
            close(contrast_lookup[(regime, rsr, snr, contrast)]['ber_reduction'],
                  float((rates[names.index(reference)] - rates[names.index(candidate)]).mean()))
        # Define the primary window independently from production scope code.
        if rsr in (12, 20) and snr in (32.5, 35., 37.5):
            for ci, (reference, candidate) in enumerate((('old_bypass_mean', 'repair_mean'), ('prior_repair_mean', 'repair_mean'))):
                primary_deltas[regime][ci] += rates[names.index(reference)] - rates[names.index(candidate)]
        records.append(dict(cell=label, regime=regime, rsr=rsr, snr=snr, rates=rates.mean(-1), errors=counts.sum(-1)))
        cell_hashes[label] = sha(path)

    scopes = {
        'overall': (summary['overall'], lambda r: True),
        'primary': (summary['primary'], lambda r: r['rsr'] in (12, 20) and 32.5 <= r['snr'] <= 37.5),
        'ordinary': (summary['ordinary'], lambda r: r['snr'] <= 20),
        'high_snr': (summary['high_snr'], lambda r: r['snr'] > 20),
        'transition': (summary['transition'], lambda r: 20 < r['snr'] <= 45),
        'stress': (summary['stress'], lambda r: r['snr'] in (50, 60)),
        'weak_transition': (summary['weak_transition'], lambda r: r['rsr'] == 5 and 32.5 <= r['snr'] <= 45),
        'historical_flagged': (summary['historical_flagged'], lambda r: r['cell'] in ber.HISTORICAL_FLAGGED),
    }
    for snr in ber.SNRS:
        scopes[f'snr{snr}'] = (summary['by_snr'][f'{snr:g}'], lambda r, s=snr: r['snr'] == s)
    for snr in (32.5, 35., 37.5):
        scopes[f'primary_snr{snr:g}'] = (summary['primary_by_snr'][f'{snr:g}'], lambda r, s=snr: r['snr'] == s and r['rsr'] in (12, 20))
    for rsr in ber.RSRS:
        scopes[f'rsr{rsr}'] = (summary['by_rsr'][f'{rsr:g}'], lambda r, s=rsr: r['rsr'] == s)
        scopes[f'high_rsr{rsr}'] = (summary['high_snr_by_rsr'][f'{rsr:g}'], lambda r, s=rsr: r['rsr'] == s and r['snr'] > 20)
    for regime in ber.REGIMES:
        scopes[regime] = (summary['by_regime'][regime], lambda r, s=regime: r['regime'] == s)
        scopes[f'high_{regime}'] = (summary['high_snr_by_regime'][regime], lambda r, s=regime: r['regime'] == s and r['snr'] > 20)
        scopes[f'primary_{regime}'] = (summary['primary_by_regime'][regime], lambda r, s=regime: r['regime'] == s and r['rsr'] in (12, 20) and 32.5 <= r['snr'] <= 37.5)
        scopes[f'weak_{regime}'] = (summary['weak_transition_by_regime'][regime], lambda r, s=regime: r['regime'] == s and r['rsr'] == 5 and 32.5 <= r['snr'] <= 45)
        scopes[f'historical_{regime}'] = (summary['historical_by_regime'][regime], lambda r, s=regime: r['regime'] == s and r['cell'] in ber.HISTORICAL_FLAGGED)
    for key, (saved, include) in scopes.items():
        subset = [r for r in records if include(r)]
        assert saved['cells'] == len(subset)
        assert saved['operating_cases'] == len(subset) * 128
        assert saved['coupled_clusters'] == len({r['regime'] for r in subset}) * 128
        point = np.mean([r['rates'] for r in subset], axis=0)
        counts = np.sum([r['errors'] for r in subset], axis=0)
        for mi, name in enumerate(names):
            actual = saved['methods'][name]
            close(actual['ber'], float(point[mi]))
            assert actual['bits_per_fit'] == len(subset) * 49152
            if name in ber.GROUPS:
                assert actual['errors_per_training_seed'] == {n: int(counts[ber.NAMES.index(n)]) for n in ber.GROUPS[name]}
            else:
                assert actual['bit_errors'] == int(counts[mi])
        for contrast, (reference, candidate) in ber.CONTRASTS.items():
            close(saved['contrasts'][contrast]['ber_reduction'], point[names.index(reference)] - point[names.index(candidate)])
            adjusted = 'bonferroni4_interval' in saved['contrasts'][contrast]
            assert adjusted == (key == 'primary' and contrast in ('repair_vs_old', 'repair_vs_prior'))
            both_zero = point[names.index(reference)] == 0 and point[names.index(candidate)] == 0
            assert saved['contrasts'][contrast]['both_methods_zero_errors'] == bool(both_zero)
            if both_zero:
                assert 'noninferiority' in saved['contrasts'][contrast]['interpretation']
    for region in ('ordinary', 'transition', 'stress', 'high_snr'):
        assert summary['regions'][region] == summary[region]
    assert summary['regions']['all'] == summary['overall']
    assert summary['regions']['strong_transition'] == summary['primary']
    assert summary['regions']['weak_transition'] == summary['weak_transition']
    assert summary['regions']['historical_flagged'] == summary['historical_flagged']
    for regime in ber.REGIMES:
        assert summary['regions'][f'strong_transition_{regime}'] == summary['primary_by_regime'][regime]
        assert summary['regions'][f'weak_transition_{regime}'] == summary['weak_transition_by_regime'][regime]
        assert summary['regions'][f'historical_flagged_{regime}'] == summary['historical_by_regime'][regime]
    for snr in ('32.5', '35', '37.5'):
        assert summary['regions'][f'strong_transition_snr{snr}'] == summary['primary_by_snr'][snr]
        assert summary['primary_by_snr'][snr]['cells'] == 6
    assert summary['primary']['cells'] == 18 and summary['ordinary']['cells'] == 45
    assert summary['weak_transition']['cells'] == 18 and summary['historical_flagged']['cells'] == 10
    assert summary['high_snr']['cells'] == 108 and summary['transition']['cells'] == 90 and summary['stress']['cells'] == 18
    # Aggregate each anchor's effects before resampling: a different arithmetic
    # route from production's per-cell bootstrap accumulation.
    rng = np.random.default_rng(2026101939)
    replicates = np.zeros((2, 20000))
    for regime in ('L2', 'L4', 'random3to7'):
        weights = rng.multinomial(128, np.full(128, 1 / 128), size=20000)
        replicates += primary_deltas[regime] @ weights.T / (128 * 18)
    maximum_interval_error = 0.
    for ci, contrast in enumerate(('repair_vs_old', 'repair_vs_prior')):
        saved = summary['primary']['contrasts'][contrast]
        for key, q in [('bootstrap95', [.025, .975]), ('bonferroni4_interval', [.05 / 8, 1 - .05 / 8])]:
            actual = np.quantile(replicates[ci], q)
            maximum_interval_error = max(maximum_interval_error, close(actual, saved[key]))
        assert saved['positive_reduction_supported_familywise'] == (saved['bonferroni4_interval'][0] > 0)
    result = dict(
        passed=True, completed_unix=time.time(), protocol_sha256=digest, selection_sha256=selected,
        source_hashes_match=True, selection_precedes_tests=True, cells=153, physical_anchor_sets=9,
        ordinary_cells_exact=45, weak_reference_cells_exact_to_prior=51, weak_transition_cells_exact_to_prior=18,
        regenerated_data_cases=data_cases, maximum_data_reconstruction_error=maximum_data_error,
        independently_redecoded_case_methods=decodes, exact_prediction_reuses_verified=reuses,
        aggregate_scopes_verified=len(scopes), region_aliases_verified=True, primary_coupled_intervals_recomputed=True,
        maximum_primary_interval_discrepancy=maximum_interval_error, detector_checks=verify_detector(),
        cell_hashes=cell_hashes, audit_source_sha256=sha(__file__),
        zero_error_interpretation='No independent-bit rare-event bound is asserted.',
    )
    write_json(out / 'audit.json', result)
    print(json.dumps({k: v for k, v in result.items() if k != 'cell_hashes'}), flush=True)


if __name__ == '__main__':
    run()
