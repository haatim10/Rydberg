"""Independent checks of the fresh native-solve BER artifacts."""
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
    assert ber.CASES == 128 and ber.TEST_SEED == 2026102002 and ber.DATA_SEED == 2026102003
    assert ber.BOOTSTRAP_SEED == 2026102009 and ber.BOOTSTRAPS == 20000
    assert ber.PARITY_ATOL == ber.PARITY_RTOL == 1e-10
    assert ber.PRIMARY_CONTRASTS == ('repair_vs_current',)
    assert ber.PRIMARY_REGIONS == ('all', 'weak_warm100')
    x, bits = hypotheses(3)
    assert x.shape == (64, 3) and len(np.unique(x, axis=0)) == 64
    np.testing.assert_allclose(np.abs(x) ** 2, 1, rtol=0, atol=1e-15)
    np.testing.assert_array_equal(bits.sum(0), np.full((3, 2), 32))
    names = ber.NAMES + tuple(ber.GROUPS)
    lookup = {(r['regime'], r['rsr_db'], r['snr_db'], r['method']): r for r in summary['cells']}
    contrast_lookup = {(r['regime'], r['rsr_db'], r['snr_db'], r['contrast']): r for r in summary['cell_contrasts']}
    assert len(lookup) == 153 * len(names) and len(contrast_lookup) == 153 * 6
    records, anchors, cell_hashes = [], {}, {}
    decodes = reuses = data_cases = 0
    maximum_data_error = 0.
    maximum_channel_absolute_error = maximum_channel_relative_frobenius = 0.
    bitwise_native_cell_fits = 0
    primary_deltas = {r: np.zeros((2, 128)) for r in ('L2', 'L4', 'random3to7')}
    for regime, rsr, snr in ber.cells():
        label = ber.label(regime, rsr, snr)
        path = out / 'cells' / (label + '.npz')
        a = ber.load(path)
        sourcepath = ber.SOURCE / 'test/cells' / (label + '.npz')
        source = ber.load(sourcepath)
        assert sourcepath.stat().st_mtime >= policy['frozen_unix']
        assert str(source['selection_sha256']) == selected and int(source['seed']) == 2026102002
        assert str(source['protocol_sha256']) == sha(ber.SOURCE / 'protocol.json')
        assert str(source['regime']) == regime and float(source['rsr']) == rsr and float(source['snr']) == snr
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
                2026102003, ('L2', 'L4', 'random3to7').index(regime), (5., 12., 20.).index(rsr), case]))
            unit = (rng.standard_normal((32, 64)) + 1j * rng.standard_normal((32, 64))) / np.sqrt(2)
            z = np.abs(source['truth_g'][case] @ x.T + source['b'][case, :, :1] + np.sqrt(source['sigma2'][case]) * unit)
            np.testing.assert_array_equal(z, data['z'][case])
            maximum_data_error = max(maximum_data_error, float(np.max(abs(z - data['z'][case]))))
            data_cases += 1
            for name in ('perfect_csi', 'original_9101', 'old_bypass_9102', 'prior_repair_9101', 'current_repair_9102', 'repair_9103', 'em100', 'compact_sparse'):
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
        for seed in ber.SEEDS:
            canonical = source['estimates'][source_names.index(f'current_repair_{seed}')]
            native = source['estimates'][source_names.index(f'repair_{seed}')]
            np.testing.assert_allclose(native, canonical, atol=1e-10, rtol=1e-10)
            difference = native - canonical
            maximum_channel_absolute_error = max(maximum_channel_absolute_error, float(np.max(abs(difference))))
            relative = np.sqrt(np.sum(abs(difference)**2, axis=(-2, -1)) / np.maximum(np.sum(abs(canonical)**2, axis=(-2, -1)), 1e-30))
            maximum_channel_relative_frobenius = max(maximum_channel_relative_frobenius, float(relative.max()))
            bitwise_native_cell_fits += int(np.array_equal(native, canonical))
        if snr <= 20:
            for seed in ber.SEEDS:
                # These arms all use the same native learned ordinary branch;
                # canonical current_repair is deliberately excluded.
                for branch in ('old_bypass', 'prior_repair', 'repair'):
                    np.testing.assert_array_equal(source['estimates'][source_names.index(f'original_{seed}')],
                                                  source['estimates'][source_names.index(f'{branch}_{seed}')])
                    np.testing.assert_array_equal(counts[ber.NAMES.index(f'original_{seed}')], counts[ber.NAMES.index(f'{branch}_{seed}')])
        np.testing.assert_array_equal(source['estimates'][source_names.index('prior_compact')],
                                      source['estimates'][source_names.index('compact_sparse')])
        np.testing.assert_array_equal(counts[ber.NAMES.index('prior_compact')], counts[ber.NAMES.index('compact_sparse')])
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
        # Independently specified primary scopes: whole grid and weak warm100.
        delta = rates[names.index('current_repair_mean')] - rates[names.index('repair_mean')]
        primary_deltas[regime][0] += delta
        if rsr == 5 and snr in (37.5, 40., 42.5, 45., 50., 60.):
            primary_deltas[regime][1] += delta
        records.append(dict(cell=label, regime=regime, rsr=rsr, snr=snr, rates=rates.mean(-1), errors=counts.sum(-1)))
        cell_hashes[label] = sha(path)

    scopes = {
        'all': (summary['regions']['all'], lambda r: True),
        'weak_warm100': (summary['regions']['weak_warm100'], lambda r: r['rsr'] == 5 and r['snr'] in (37.5, 40., 42.5, 45., 50., 60.)),
        'strong_high': (summary['regions']['strong_high'], lambda r: r['rsr'] in (12, 20) and r['snr'] >= 40),
        'strong_transition': (summary['regions']['strong_transition'], lambda r: r['rsr'] in (12, 20) and 32.5 <= r['snr'] <= 37.5),
        'ordinary': (summary['regions']['ordinary'], lambda r: r['snr'] <= 20),
        'weak': (summary['regions']['weak'], lambda r: r['rsr'] == 5),
    }
    for rsr in (5., 12., 20.):
        scopes[f'rsr{rsr:g}'] = (summary['regions'][f'rsr{rsr:g}'], lambda r, ref=rsr: r['rsr'] == ref)
    for regime in ('L2', 'L4', 'random3to7'):
        scopes[f'all_{regime}'] = (summary['regions'][f'all_{regime}'], lambda r, path=regime: r['regime'] == path)
        scopes[f'weak_warm100_{regime}'] = (summary['regions'][f'weak_warm100_{regime}'],
            lambda r, path=regime: r['regime'] == path and r['rsr'] == 5 and r['snr'] in (37.5, 40., 42.5, 45., 50., 60.))
    assert set(scopes) == set(summary['regions']) == set(ber.REGIONS)
    for snr in ber.SNRS:
        scopes[f'snr{snr:g}'] = (summary['by_snr'][f'{snr:g}'], lambda r, s=snr: r['snr'] == s)
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
            assert adjusted == (key in ('all', 'weak_warm100') and contrast == 'repair_vs_current')
            both_zero = point[names.index(reference)] == 0 and point[names.index(candidate)] == 0
            assert saved['contrasts'][contrast]['both_methods_zero_errors'] == bool(both_zero)
            if both_zero:
                assert 'noninferiority' in saved['contrasts'][contrast]['interpretation']
    assert summary['overall'] == summary['regions']['all']
    assert summary['weak_warm100'] == summary['regions']['weak_warm100']
    assert summary['primary'] == {r: summary['regions'][r] for r in ('all', 'weak_warm100')}
    for rsr in (5., 12., 20.):
        assert summary['by_rsr'][f'{rsr:g}'] == summary['regions'][f'rsr{rsr:g}']
    for regime in ('L2', 'L4', 'random3to7'):
        assert summary['by_regime'][regime] == summary['regions'][f'all_{regime}']
    for region, number in (('all', 153), ('weak_warm100', 18), ('strong_high', 30), ('strong_transition', 18), ('ordinary', 45), ('weak', 51)):
        assert summary['regions'][region]['cells'] == number
    # Sum each anchor's paired effect before resampling, independently from
    # production's per-cell bootstrap accumulation.
    rng = np.random.default_rng(2026102009)
    replicates = np.zeros((2, 20000))
    for regime in ('L2', 'L4', 'random3to7'):
        weights = rng.multinomial(128, np.full(128, 1 / 128), size=20000)
        replicates += (primary_deltas[regime] @ weights.T) / (128 * np.array([153, 18])[:, None])
    maximum_interval_error = 0.
    for ci, region in enumerate(('all', 'weak_warm100')):
        saved = summary['regions'][region]['contrasts']['repair_vs_current']
        for key, q in [('bootstrap95', [.025, .975]), ('bonferroni4_interval', [.05 / 8, 1 - .05 / 8])]:
            actual = np.quantile(replicates[ci], q)
            maximum_interval_error = max(maximum_interval_error, close(actual, saved[key]))
        assert saved['positive_reduction_supported_familywise'] == (saved['bonferroni4_interval'][0] > 0)
    result = dict(
        passed=True, completed_unix=time.time(), protocol_sha256=digest, selection_sha256=selected,
        source_hashes_match=True, selection_precedes_tests=True, cells=153, physical_anchor_sets=9,
        native_ordinary_cells_exact=45, frozen_compact_cells_exact=153,
        canonical_native_parity_cell_fits=459, canonical_native_bitwise_cell_fits=bitwise_native_cell_fits,
        numerical_parity_atol=1e-10, numerical_parity_rtol=1e-10,
        maximum_channel_absolute_error=maximum_channel_absolute_error,
        maximum_channel_relative_frobenius=maximum_channel_relative_frobenius,
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
