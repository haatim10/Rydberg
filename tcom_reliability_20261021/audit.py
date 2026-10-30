"""Independent verification of the reference-aware reliability receiver.

An audit PASS verifies provenance, implementation and recorded arithmetic.
It does not establish superiority, confirm a policy, or imply a latency bound.
No audit result is used to choose or revise the frozen receiver.
"""
import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
             'VECLIB_MAXIMUM_THREADS'):
    os.environ[_key] = '1'

import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from scipy.special import i0e
from tcom_followups_20261017.common import sha, write_json

REGIMES = ('L2', 'L4', 'random3to7')
SNRS = (0., 5., 10., 15., 20., 22.5, 25., 27.5, 30., 32.5, 35.,
        37.5, 40., 42.5, 45., 50., 60.)
RSRS = (5., 12., 20.)
SEEDS = (9101, 9102, 9103)
REGIONS = ('all', 'high_all', 'transition_all', 'weak_warm100', 'strong_high',
           'strong_transition', 'ordinary', 'weak', 'weak_target', 'rsr5', 'rsr12', 'rsr20',
           'all_L2', 'all_L4', 'all_random3to7', 'weak_warm100_L2',
           'weak_warm100_L4', 'weak_warm100_random3to7')
PRIMARY = ('receiver_vs_prior', 'receiver_vs_latest')
CONTRASTS = {f'receiver_vs_{key}': (reference, 'receiver_mean') for key, reference in
             (('prior', 'prior_mean'), ('latest', 'latest_mean'), ('compact', 'compact_sparse'),
              ('adaptive40', 'adaptive40_mean'), ('global40', 'global40_mean'),
              ('classical40', 'classical40'), ('em100', 'em100'), ('algebraic20', 'algebraic20'))}
TEST_COUNT, DEV_COUNT, BOOT, BOOT_SEED = 128, 256, 20000, 2026102108
CONDITION_LIMIT, NLL_TOL = 1e8, 1e-12


def module():
    from . import study
    return study


def close(actual, expected, atol=1e-11, rtol=0):
    np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol)


def load(path):
    with np.load(path, allow_pickle=False) as arrays:
        return {key: arrays[key] for key in arrays.files}


def completed(directory):
    return sorted(path for path in Path(directory).glob('*.npz')
                  if '.tmp.' not in path.name and not path.name.endswith('.tmp.npz'))


def record(phase, **values):
    study = module()
    policy, digest = study.check_selection()
    result = dict(passed=True, audit_source_sha256=sha(__file__),
                  selection_sha256=digest, protocol_sha256=policy['protocol_sha256'],
                  completed_unix=time.time(), **values)
    destination = study.OUT / ('ber/audit.json' if phase == 'ber' else f'audit/{phase}.json')
    write_json(destination, result)
    return result


def belongs(scope, snr, ref, regime):
    core = dict(all=True, high_all=snr >= 40, transition_all=20 < snr < 40,
                weak_warm100=ref == 5 and snr >= 37.5,
                strong_high=ref in (12, 20) and snr >= 40,
                strong_transition=ref in (12, 20) and 32.5 <= snr <= 37.5,
                ordinary=snr <= 20, weak=ref == 5, weak_target=ref == 5 and snr > 30)
    if scope in core:
        return core[scope]
    if scope in ('rsr5', 'rsr12', 'rsr20'):
        return ref == int(scope[3:])
    for prefix in ('all', 'weak_warm100'):
        if scope.startswith(prefix + '_'):
            return core[prefix] and regime == scope[len(prefix) + 1:]
    raise KeyError(scope)


def algebraic_numpy(z, s, b, sigma2):
    """Independently solve all antenna RHS; no production inference call.

    The coefficient order is Re(c), Im(c), diagonal Hermitian entries, then
    Re(c_i conj(c_j)), Im(c_i conj(c_j)) for every lexicographic i<j pair.
    Invalid cases return a zero placeholder together with a false valid flag;
    only the receiver's separately audited fallback determines their output.
    """
    z = np.asarray(z, dtype=np.float64)
    s, b = np.asarray(s, dtype=np.complex128), np.asarray(b, dtype=np.complex128)
    sigma2 = np.asarray(sigma2, dtype=np.float64)
    batch, n, p = z.shape
    k = s.shape[1]
    assert s.shape == (batch, k, p) and b.shape == z.shape
    assert sigma2.shape == (batch,)
    prediction = np.zeros((batch, n, k), dtype=np.complex128)
    valid = np.zeros(batch, dtype=bool)
    ranks = np.zeros(batch, dtype=np.int64)
    conditions = np.full(batch, np.inf)
    for case in range(batch):
        if not (np.isfinite(z[case]).all() and np.isfinite(s[case]).all()
                and np.isfinite(b[case]).all() and np.isfinite(sigma2[case])
                and sigma2[case] >= 0 and (z[case] >= 0).all()):
            continue
        reference = b[case, :, 0]
        if (np.abs(reference) == 0).any() or not np.array_equal(
                b[case], np.repeat(reference[:, None], p, axis=1)):
            continue
        pilots = s[case]
        columns = [2 * pilots[i].real for i in range(k)]
        columns += [-2 * pilots[i].imag for i in range(k)]
        columns += [np.abs(pilots[i]) ** 2 for i in range(k)]
        for i in range(k):
            for j in range(i + 1, k):
                product = pilots[i] * pilots[j].conj()
                columns.extend((2 * product.real, -2 * product.imag))
        design = np.stack(columns, axis=1)
        response = ((z[case] ** 2 - sigma2[case]) /
                    np.abs(reference[:, None]) ** 2 - 1).T
        try:
            coefficients, _, rank, singular = np.linalg.lstsq(design, response, rcond=None)
        except np.linalg.LinAlgError:
            continue
        ranks[case] = rank
        condition = np.inf if singular[-1] <= 0 else singular[0] / singular[-1]
        conditions[case] = condition
        if rank != k * k + 2 * k or not np.isfinite(condition) or condition > CONDITION_LIMIT:
            continue
        value = reference[:, None] * (coefficients[:k].T + 1j * coefficients[k:2*k].T)
        if np.isfinite(value).all():
            prediction[case] = value
            valid[case] = True
    return prediction, valid, ranks, conditions


def centered_nll(z, s, b, sigma2, estimate):
    """Independent candidate-dependent Rician NLL, stable at high SNR."""
    amplitude = np.abs(np.asarray(estimate) @ np.asarray(s) + np.asarray(b))
    z, variance = np.asarray(z), np.asarray(sigma2)[:, None, None]
    argument = 2 * z * amplitude / variance
    return (((z - amplitude) ** 2 / variance) - np.log(i0e(argument))).mean(axis=(-2, -1))


def residual_scores(z, s, b, sigma2, estimate):
    amplitude = np.abs(np.asarray(estimate) @ np.asarray(s) + np.asarray(b))
    by_antenna = 2 * np.mean((np.asarray(z) - amplitude) ** 2, axis=-1) / np.asarray(sigma2)[:, None]
    return {'mean': by_antenna.mean(axis=-1), 'max': by_antenna.max(axis=-1)}


def gate_numpy(z, s, b, sigma2, base, fallback, valid, metric, threshold, eligible):
    scores = residual_scores(z, s, b, sigma2, base)[metric]
    trigger = np.asarray(eligible, dtype=bool) & (scores > threshold)
    base_nll = centered_nll(z, s, b, sigma2, base)
    fallback_nll = centered_nll(z, s, b, sigma2, fallback)
    accept = trigger & np.asarray(valid, dtype=bool) & (fallback_nll < base_nll - NLL_TOL)
    result = np.where(accept[:, None, None], fallback, base)
    return dict(estimates=result, score=scores, trigger=trigger, valid=np.asarray(valid),
                accept=accept, base_nll=base_nll, fallback_nll=fallback_nll)


def paired_bootstrap(delta, seed=BOOT_SEED, draws=BOOT):
    """delta[regime,SNR,fit,anchor]; couple SNR/fit within each anchor."""
    delta = np.asarray(delta, dtype=float)
    assert delta.ndim == 4 and delta.shape[0] == 3
    count = delta.shape[-1]
    rng = np.random.default_rng(seed)
    per_regime = []
    for regime in range(3):
        weights = rng.multinomial(count, np.full(count, 1 / count), size=draws)
        per_anchor = delta[regime].mean(axis=(0, 1))
        per_regime.append(weights @ per_anchor / count)
    return np.stack(per_regime)


def stats(rows):
    values = np.array([row['ms'] for row in rows])
    assert len(values) and np.isfinite(values).all() and (values > 0).all()
    return dict(p90_ms=float(np.quantile(values, .9)), median_ms=float(np.median(values)),
                mean_ms=float(values.mean()), calls=len(values))


def verify_stats(rows, expected):
    for key, value in stats(rows).items():
        close(expected[key], value)


@torch.no_grad()
def ber():
    """Reconstruct independent data endpoints and decode every cached method."""
    from tcom_followups_20261017.ber_detector import detect, hypotheses
    study = module()
    policy, digest = study.check_selection()
    paths = completed(study.OUT / 'ber/cells')
    assert len(paths) == 153
    x, bits = hypotheses(3)
    assert x.shape == (64, 3) and len(np.unique(x, axis=0)) == 64
    np.testing.assert_array_equal(bits.sum(0), np.full((3, 2), 32))
    names = ('perfect_csi',) + tuple(study.NAMES)
    assert len(names) == 20
    hashes, data_hashes, seen = {}, {}, set()
    decodes = regenerated = reuses = 0
    for path in paths:
        a = load(path)
        source_path = study.OUT / 'test/cells' / path.name
        data_path = study.OUT / 'ber/data' / path.name
        source, data = load(source_path), load(data_path)
        regime, ref, snr = str(source['regime']), float(source['rsr']), float(source['snr'])
        seen.add((regime, ref, snr))
        assert str(a['selection_sha256']) == str(source['selection_sha256']) == digest
        assert str(a['protocol_sha256']) == str(data['protocol_sha256']) == policy['protocol_sha256']
        assert str(a['source_sha256']) == sha(source_path)
        assert str(a['data_sha256']) == sha(data_path)
        assert int(source['seed']) == 2026102104
        np.testing.assert_array_equal(source['method_names'], study.NAMES)
        np.testing.assert_array_equal(a['method_names'], names)
        np.testing.assert_array_equal(data['hypothesis_symbols'], x)
        np.testing.assert_array_equal(data['transmitted_bits'], bits)
        np.testing.assert_array_equal(data['b'], source['b'][..., :1])
        np.testing.assert_array_equal(data['sigma2'], source['sigma2'])
        for key in ('indices', 'local_indices'):
            np.testing.assert_array_equal(a[key], source[key])
            np.testing.assert_array_equal(data[key], source[key])
        np.testing.assert_array_equal(source['local_indices'], np.arange(TEST_COUNT))
        np.testing.assert_array_equal(a['bits_per_user'], np.full((TEST_COUNT, 3), 128))
        counts = a['errors_per_user']
        assert counts.shape == (len(names), TEST_COUNT, 3)
        assert np.issubdtype(counts.dtype, np.integer) and (counts >= 0).all() and (counts <= 128).all()
        predictions = np.concatenate((source['truth_g'][None], source['estimates']))
        for case in (0, TEST_COUNT - 1):
            rng = np.random.default_rng(np.random.SeedSequence([
                2026102105, REGIMES.index(regime), RSRS.index(ref), case]))
            noise = (rng.standard_normal((32, 64)) + 1j * rng.standard_normal((32, 64))) / np.sqrt(2)
            expected = np.abs(source['truth_g'][case] @ x.T + source['b'][case, :, :1]
                              + np.sqrt(source['sigma2'][case]) * noise)
            np.testing.assert_array_equal(data['z'][case], expected)
            regenerated += 1
            for mi in range(len(names)):
                decoded = detect(data['z'][case], predictions[mi, case], data['b'][case],
                                 float(data['sigma2'][case]))
                np.testing.assert_array_equal(np.count_nonzero(decoded != bits, axis=(0, 2)),
                                              counts[mi, case])
                decodes += 1
        for mi in range(len(names)):
            other = str(a['duplicate_of'][mi])
            seconds = a['detector_wall_seconds'][mi]
            if other:
                oi = names.index(other)
                assert oi < mi
                np.testing.assert_array_equal(predictions[mi], predictions[oi])
                np.testing.assert_array_equal(counts[mi], counts[oi])
                np.testing.assert_array_equal(seconds, np.zeros(TEST_COUNT))
                reuses += 1
            else:
                assert np.isfinite(seconds).all() and (seconds > 0).all()
        hashes[path.name] = sha(path)
        data_hashes[path.name] = sha(data_path)
    assert seen == {(r, ref, snr) for r in REGIMES for ref in RSRS for snr in SNRS}
    return record('ber', cells=153, cell_hashes=hashes, data_hashes=data_hashes,
                  independently_redecoded_predictions=decodes,
                  independently_regenerated_data_cases=regenerated,
                  exact_prediction_reuses_verified=reuses,
                  exact_common_data_for_all_methods=True)


def summary():
    """Audit both metrics, including independent nine-block bootstrap draws."""
    study = module()
    policy, digest = study.check_selection()
    npth, bpth = study.OUT/'nmse_summary.json', study.OUT/'ber/summary.json'
    n, b = json.loads(npth.read_text()), json.loads(bpth.read_text())
    nraw = list(study.NAMES)
    braw = ['perfect_csi'] + nraw
    groups = {name: list(members) for name, members in study.GROUPS.items()}
    nnames, bnames = nraw + list(groups), braw + list(groups)
    ni, bi = ({name: i for i, name in enumerate(names)} for names in (nnames, bnames))
    assert len(nraw) == 19 and len(groups) == 5
    for report_, names in ((n, nnames), (b, bnames)):
        assert report_['selection_sha256'] == digest
        assert report_['protocol_sha256'] == policy['protocol_sha256']
        for key in ('confirmation', 'candidate_rule', 'operational_rule'):
            assert report_[key] == policy[key]
        assert report_['method_names'] == names and report_['groups'] == groups
        assert report_['contrasts'] == {k: list(v) for k, v in CONTRASTS.items()}
        assert report_['primary'] == dict(contrasts=list(PRIMARY), region='all', outcomes=4)
        assert report_['source_sha256'] == sha(Path(__file__).with_name('analysis.py'))
        assert report_['bootstrap']['samples'] == BOOT and report_['bootstrap']['seed'] == BOOT_SEED
        assert report_['unique_test_anchors'] == report_['bootstrap_index_clusters'] == 1152
        assert report_['condition_cases'] == 153 * TEST_COUNT
    assert n['test_hashes'] == b['test_hashes'] and n['numerical_tie_tolerance_db'] == 1e-10
    assert len(n['cells']) == 153 and len(b['cells']) == 153 * len(bnames)
    assert len(b['cell_contrasts']) == 153 * len(CONTRASTS)
    brows = {(r['regime'], r['rsr_db'], r['snr_db'], r['method']): r for r in b['cells']}
    bcomp = {(r['cell'], r['contrast']): r for r in b['cell_contrasts']}
    assert len(brows) == len(b['cells']) and len(bcomp) == len(b['cell_contrasts'])
    def expand(values, raw):
        return np.concatenate([values, np.stack([
            values[[raw.index(x) for x in members]].mean(0) for members in groups.values()])])
    rng = np.random.default_rng(BOOT_SEED)
    weights = {(r, ref): rng.multinomial(TEST_COUNT, np.full(TEST_COUNT, 1/TEST_COUNT),
               size=BOOT).T.astype(float) for r in REGIMES for ref in RSRS}
    errors, energies, counts, gates = [], [], [], []
    npoints, bpoints, nboots, bboots, full_cell_checks = [], [], [], [], []
    for row in n['cells']:
        label, regime, ref, snr = row['cell'], row['regime'], row['rsr'], row['snr']
        assert label == study.cellname(regime, ref, snr)
        paths = {'prediction': study.OUT/'test/cells'/f'{label}.npz',
                 'ber': study.OUT/'ber/cells'/f'{label}.npz',
                 'data': study.OUT/'ber/data'/f'{label}.npz'}
        assert n['test_hashes'][label] == {key: sha(path) for key, path in paths.items()}
        a, ab = load(paths['prediction']), load(paths['ber'])
        assert str(a['selection_sha256']) == str(ab['selection_sha256']) == digest
        np.testing.assert_array_equal(a['method_names'], nraw)
        np.testing.assert_array_equal(ab['method_names'], braw)
        err = np.sum(np.abs(a['estimates'] - a['truth_g'][None])**2, axis=(-2, -1))
        energy = np.sum(np.abs(a['truth_g'])**2, axis=(-2, -1))
        np.testing.assert_array_equal(err, a['errors'])
        np.testing.assert_array_equal(energy, a['truth_energy'])
        ct = ab['errors_per_user'].sum(-1)
        np.testing.assert_array_equal(ab['bits_per_user'], np.full((TEST_COUNT, 3), 128))
        w = weights[regime, ref]
        point = expand(10*np.log10(err.sum(1)/energy.sum()), nraw)
        boot = expand(10*np.log10((err@w[:, :32])/(energy@w[:, :32])[None]), nraw)
        rate = expand(ct/384, braw)
        bp, bb = rate.mean(1), rate@w[:, :32]/TEST_COUNT
        nm = err/energy[None]
        metrics = {'nmse_db': point, 'case_nmse_p95': expand(np.quantile(nm, .95, axis=1), nraw),
                   'case_nmse_p99': expand(np.quantile(nm, .99, axis=1), nraw),
                   'severe_rate': expand((nm > 1).mean(1), nraw),
                   'case_nmse_max': expand(nm.max(1), nraw)}
        for name, i in ni.items():
            for key, value in metrics.items():
                close(row['methods'][name][key], value[i])
        for name, i in bi.items():
            item = brows[regime, ref, snr, name]
            close(item['ber'], bp[i])
            assert item['bits_per_fit'] == TEST_COUNT*384 and item['channel_anchors'] == TEST_COUNT
            if name in groups:
                assert item['errors_per_training_seed'] == {m: int(ct[braw.index(m)].sum()) for m in groups[name]}
            else:
                assert item['bit_errors'] == int(ct[i].sum())
        for key, (left, right) in CONTRASTS.items():
            close(row['contrasts'][key]['gain_db'], point[ni[left]]-point[ni[right]])
            close(bcomp[label, key]['ber_reduction'], bp[bi[left]]-bp[bi[right]])
        if (ref, snr) in ((5., 0.), (12., 40.)):
            nf = expand(10*np.log10((err@w)/(energy@w)[None]), nraw)
            bf = rate@w/TEST_COUNT
            for name, i in ni.items():
                close(row['methods'][name]['conditional_ci95_db'], np.quantile(nf[i], [.025, .975]))
            for name, i in bi.items():
                close(brows[regime, ref, snr, name]['bootstrap95'], np.quantile(bf[i], [.025, .975]))
            for key, (left, right) in CONTRASTS.items():
                close(row['contrasts'][key]['pointwise_ci95_db'], np.quantile(nf[ni[left]]-nf[ni[right]], [.025, .975]))
                close(bcomp[label, key]['bootstrap95'], np.quantile(bf[bi[left]]-bf[bi[right]], [.025, .975]))
            full_cell_checks.append(label)
        errors.append(err); energies.append(energy); counts.append(ct)
        gates.append({k: a[k] for k in ('gate_scores', 'triggers', 'accepted', 'rank_invalid')})
        npoints.append(point); bpoints.append(bp); nboots.append(boot); bboots.append(bb)
    npoints, bpoints, nboots, bboots = map(np.stack, (npoints, bpoints, nboots, bboots))
    savedn, savedb = load(study.OUT/'nmse_bootstrap.npz'), load(study.OUT/'ber/bootstrap.npz')
    for saved, names, points in ((savedn, nnames, npoints), (savedb, bnames, bpoints)):
        np.testing.assert_array_equal(saved['method_names'], names)
        np.testing.assert_array_equal(saved['regions'], REGIONS)
        np.testing.assert_array_equal(saved['cell_names'], [r['cell'] for r in n['cells']])
        close(saved['cell_scores'], points)
        assert str(saved['selection_sha256']) == digest
    maximum, scope_counts = 0., {}
    for ri, region in enumerate(REGIONS):
        keep = np.array([belongs(region, r['snr'], r['rsr'], r['regime']) for r in n['cells']])
        total = int(keep.sum()); scope_counts[region] = total
        nr, br = n['aggregates'][region], b['regions'][region]
        assert nr['cells'] == br['cells'] == total
        assert nr['condition_cases'] == br['operating_cases'] == total*TEST_COUNT
        assert br['coupled_clusters'] == TEST_COUNT*len({(r['regime'], r['rsr']) for r, k in zip(n['cells'], keep) if k})
        point, bp = npoints[keep].mean(0), bpoints[keep].mean(0)
        nb, bb = savedn['aggregate_bootstraps'][ri], savedb['aggregate_bootstraps'][ri]
        for full, independent in ((nb, nboots[keep].mean(0)), (bb, bboots[keep].mean(0))):
            maximum = max(maximum, float(np.abs(full[:, :32]-independent).max()))
            close(full[:, :32], independent)
        err = np.concatenate([a for a, k in zip(errors, keep) if k], 1)
        energy = np.concatenate([a for a, k in zip(energies, keep) if k])
        nm = err/energy[None]
        linear = expand(err.sum(1)/energy.sum(), nraw)
        metrics = dict(nmse_db=point, pooled_ratio_of_sums_linear=linear,
                       pooled_ratio_of_sums_db=10*np.log10(linear),
                       case_nmse_p95=expand(np.quantile(nm, .95, axis=1), nraw),
                       case_nmse_p99=expand(np.quantile(nm, .99, axis=1), nraw),
                       severe_rate=expand((nm > 1).mean(1), nraw),
                       case_nmse_max=expand(nm.max(1), nraw))
        ct = sum(a.sum(1) for a, k in zip(counts, keep) if k)
        for name, i in ni.items():
            for key, value in metrics.items():
                close(nr['methods'][name][key], value[i])
            close(nr['methods'][name]['conditional_ci95_db'], np.quantile(nb[i], [.025, .975]))
        for name, i in bi.items():
            item = br['methods'][name]
            close(item['ber'], bp[i]); close(item['bootstrap95'], np.quantile(bb[i], [.025, .975]))
            assert item['bits_per_fit'] == total*TEST_COUNT*384
            if name in groups:
                assert item['errors_per_training_seed'] == {m: int(ct[braw.index(m)]) for m in groups[name]}
                close(item['training_seed_ber_range'], [min(bp[bi[m]] for m in groups[name]), max(bp[bi[m]] for m in groups[name])])
            else:
                assert item['bit_errors'] == ct[i]
        for key, (left, right) in CONTRASTS.items():
            nc, bc = nr['contrasts'][key], br['contrasts'][key]
            nd, bd = nb[ni[left]]-nb[ni[right]], bb[bi[left]]-bb[bi[right]]
            close(nc['gain_db'], point[ni[left]]-point[ni[right]])
            close(nc['pointwise_ci95_db'], np.quantile(nd, [.025, .975]))
            close(bc['ber_reduction'], bp[bi[left]]-bp[bi[right]])
            close(bc['bootstrap95'], np.quantile(bd, [.025, .975]))
            assert bc['both_methods_zero_errors'] == bool(bp[bi[left]] == bp[bi[right]] == 0)
            primary = region == 'all' and key in PRIMARY
            assert ('bonferroni4_interval_db' in nc) == ('bonferroni4_interval' in bc) == primary
            if primary:
                close(nc['bonferroni4_interval_db'], np.quantile(nd, [.05/8, 1-.05/8]))
                close(bc['bonferroni4_interval'], np.quantile(bd, [.05/8, 1-.05/8]))
            for seed in SEEDS:
                lf = f'{left[:-5]}_{seed}' if left.endswith('_mean') else left
                rf = f'{right[:-5]}_{seed}' if right.endswith('_mean') else right
                close(nc['seed_specific_gains_db'][str(seed)], point[ni[lf]]-point[ni[rf]])
        subset = [g for g, k in zip(gates, keep) if k]
        for i, seed in enumerate(SEEDS):
            ts, ac, inv = (np.concatenate([g[key][i] for g in subset]).astype(bool)
                           for key in ('triggers', 'accepted', 'rank_invalid'))
            expected = dict(cases=len(ts), triggers=int(ts.sum()), accepted=int(ac.sum()),
                            rank_invalid=int(inv.sum()), trigger_fraction=float(ts.mean()),
                            accepted_fraction=float(ac.mean()),
                            accepted_given_trigger=float(ac.sum()/ts.sum()) if ts.any() else None)
            assert nr['gate'][str(seed)] == expected
    assert scope_counts == dict(all=153, high_all=45, transition_all=63, weak_warm100=18,
        strong_high=30, strong_transition=18, ordinary=45, weak=51, weak_target=24,
        rsr5=51, rsr12=51, rsr20=51, all_L2=51, all_L4=51, all_random3to7=51,
        weak_warm100_L2=6, weak_warm100_L4=6, weak_warm100_random3to7=6)
    assert b['overall'] == b['regions']['all']
    for key in CONTRASTS:
        gains = [r['contrasts'][key]['gain_db'] for r in n['cells']]
        worst = int(np.argmin(gains)); item = n['negative_cells'][key]
        assert item['negative_cells'] == [r['cell'] for r, value in zip(n['cells'], gains) if value < -1e-10]
        assert item['pointwise95_harm_cells'] == [r['cell'] for r in n['cells'] if r['contrasts'][key]['pointwise_ci95_db'][1] < -1e-10]
        assert item['worst_cell'] == n['cells'][worst]['cell']; close(item['worst_gain_db'], gains[worst])
        rows = [r for r in b['cell_contrasts'] if r['contrast'] == key]
        worst = min(rows, key=lambda r: r['ber_reduction']); item = b['negative_cells'][key]
        assert item['negative_cells'] == [r['cell'] for r in rows if r['ber_reduction'] < -1e-15]
        assert item['pointwise95_harm_cells'] == [r['cell'] for r in rows if r['bootstrap95'][1] < -1e-15]
        assert item['worst_cell'] == worst['cell']; close(item['worst_ber_reduction'], worst['ber_reduction'])
    concentration_path = study.OUT/'weak_error_concentration.json'
    assert n['weak_error_concentration_sha256'] == sha(concentration_path)
    concentration = json.loads(concentration_path.read_text())
    assert concentration['selection_sha256'] == digest and concentration['protocol_sha256'] == policy['protocol_sha256']
    lookup = {(r['cell'], r['method']): r for r in concentration['cells']}
    assert len(lookup) == 51*19
    for row, err, energy in zip(n['cells'], errors, energies):
        if row['rsr'] != 5:
            continue
        nm = err/energy[None]
        for i, name in enumerate(nraw):
            item = lookup[row['cell'], name]; index = int(np.argmax(err[i])); total = err[i].sum()
            assert item['top1_local_index'] == index
            close(item['top1_error_share'], err[i, index]/total if total > 0 else 0.)
            close(item['top1_case_nmse'], nm[i, index])
            for key, value in dict(case_nmse_p95=np.quantile(nm[i], .95), case_nmse_p99=np.quantile(nm[i], .99), case_nmse_max=nm[i].max()).items():
                close(item[key], value)
    lookup = {(r['regime'], r['method']): r for r in concentration['weak_target_by_regime']}
    assert len(lookup) == 3*19
    for regime in REGIMES:
        ids = [i for i, r in enumerate(n['cells']) if r['regime'] == regime and r['rsr'] == 5 and r['snr'] > 30]
        assert len(ids) == 8
        total_error = np.stack([errors[i] for i in ids]).sum(0)
        nm = np.concatenate([errors[i]/energies[i][None] for i in ids], axis=1)
        for i, name in enumerate(nraw):
            item = lookup[regime, name]; index = int(np.argmax(total_error[i])); total = total_error[i].sum()
            assert item['conditions'] == 8 and item['physical_anchors'] == TEST_COUNT
            assert item['top1_local_index'] == index
            close(item['top1_error_share'], total_error[i, index]/total if total > 0 else 0.)
            for key, value in dict(case_nmse_p95=np.quantile(nm[i], .95), case_nmse_p99=np.quantile(nm[i], .99), case_nmse_max=nm[i].max()).items():
                close(item[key], value)
    return record('summary', summary_sha256=sha(npth), ber_summary_sha256=sha(bpth),
                  nmse_bootstrap_sha256=sha(study.OUT/'nmse_bootstrap.npz'),
                  ber_bootstrap_sha256=sha(study.OUT/'ber/bootstrap.npz'),
                  weak_error_concentration_sha256=sha(concentration_path), region_counts=scope_counts,
                  independent_physical_blocks=9, bootstrap_index_clusters=1152,
                  all_points_pooled_tails_fit_gains_counts_gates_adverse_lists_verified=True,
                  independently_recomputed_bootstrap_draws=32, maximum_bootstrap_error=maximum,
                  all20000_cell_intervals_recomputed=full_cell_checks,
                  all_aggregate_quantiles_and_primary_bonferroni4_verified=True)


def _independent_rule(cell, rule):
    base, errors = cell['heldout_nll_base'], cell['probe_errors_base']
    if rule is None:
        return base, errors, np.zeros(base.shape, bool), np.zeros(base.shape, bool)
    tail = (10, 20).index(rule['tail'])
    trigger = cell['score_'+rule['metric']] > rule['threshold']
    accept = trigger & cell['fallback_valid'][tail][None] & (
        cell['fit_nll_fallback'][tail][None] < cell['fit_nll_base']-NLL_TOL)
    return (np.where(accept, cell['heldout_nll_fallback'][tail][None], base),
            np.where(accept[..., None], cell['probe_errors_fallback'][tail][None], errors),
            trigger, accept)


def _development_score(bank, rule, cells, hashes):
    regimes, triggers, accepted = {}, [], []
    for regime in REGIMES:
        losses, errors, differences, base_errors = [], [], [], []
        for snr in SNRS:
            if snr <= 30:
                continue
            cell = cells[bank, regime, snr]
            loss, error, trigger, accept = _independent_rule(cell, rule)
            losses.append(loss); errors.append(error)
            differences.append(cell['heldout_nll_base']-loss)
            base_errors.append(cell['probe_errors_base'])
            triggers.append(trigger); accepted.append(accept)
        anchors = np.stack(differences).mean((0, 1))
        regimes[regime] = dict(mean_nll=float(np.mean(losses)), mean_improvement=float(anchors.mean()),
            candidate_errors=int(np.sum(errors)), base_errors=int(np.sum(base_errors)),
            anchor_improvements=anchors.tolist(), bits_per_fit=8*DEV_COUNT*384)
    return dict(regimes=regimes, mean_nll=float(np.mean([v['mean_nll'] for v in regimes.values()])),
                trigger_fraction=float(np.mean(triggers)), accept_fraction=float(np.mean(accepted)),
                cell_hashes={key: value for key, value in hashes.items() if key.startswith(f'validation/{bank}/cells/')})


def _compare_score(actual, expected):
    assert set(actual['regimes']) == set(expected['regimes'])
    for key in ('mean_nll', 'trigger_fraction', 'accept_fraction'):
        close(actual[key], expected[key])
    assert actual['cell_hashes'] == expected['cell_hashes']
    for regime, expected_row in expected['regimes'].items():
        for key, value in expected_row.items():
            if key in ('candidate_errors', 'base_errors', 'bits_per_fit'):
                assert actual['regimes'][regime][key] == value
            else:
                close(actual['regimes'][regime][key], value)


def _calibration(policy):
    study = module()
    path = study.OUT/'runtime_calibration.json'
    calibration = json.loads(path.read_text())
    assert calibration['protocol_sha256'] == policy['protocol_sha256']
    rules = list(study.RULES)
    assert rules == [dict(id=f'{metric}_t{threshold}_e{tail}', metric=metric,
                         threshold=float(threshold), tail=tail)
                     for metric in ('mean', 'max') for threshold in (2, 4, 8) for tail in (10, 20)]
    worlds = [(regime, ref, snr, 401) for ref in RSRS for regime in REGIMES for snr in SNRS]
    rng = np.random.default_rng(2026102103)
    menu = [(rule['id'], seed) for rule in rules for seed in SEEDS]
    expected = []
    for wi in rng.permutation(153):
        for mi in rng.permutation(36):
            expected.append((*worlds[wi], *menu[mi]))
    records = calibration['records']
    assert len(records) == 5508
    assert [(r['regime'], r['rsr'], r['snr'], r['index'], r['rule'], r['seed']) for r in records] == expected
    eligibility = {}
    for rule in rules:
        selected = [r for r in records if r['rule'] == rule['id']]
        item = calibration['scores'][rule['id']]
        weak = [r for r in selected if r['rsr'] == 5 and r['snr'] > 30]
        verify_stats(selected, item); verify_stats(weak, item['weak_target'])
        eligible = stats(selected)['p90_ms'] <= 10. and stats(weak)['p90_ms'] <= 10.
        assert item['eligible'] == eligible
        eligibility[rule['id']] = eligible
        for row in selected:
            assert not row['accepted'] or row['triggered']
            if row['rsr'] != 5 or row['snr'] <= 30:
                assert not row['triggered'] and not row['accepted']
    forced = calibration['forced_records']
    expected_forced = [(*w, tail, seed) for w in worlds if w[1] == 5 and w[2] > 30
                       for tail in (10, 20) for seed in SEEDS]
    assert [(r['regime'], r['rsr'], r['snr'], r['index'], r['tail'], r['seed']) for r in forced] == expected_forced
    assert len(forced) == 144 and all(r['triggered'] for r in forced)
    stats(forced)
    return calibration, eligibility


@torch.no_grad()
def selection():
    """Verify every A/C decision and sampled independently replayed observables."""
    from sahure.data import Observations
    from tcom_incumbent_guard_20261019 import risk_data
    from tcom_pilot_em_20261018 import continuation
    from tcom_followups_20261017.ber_detector import hypotheses, detect
    study = module()
    policy, digest = study.check_selection()
    protocol = json.loads((study.OUT/'protocol.json').read_text())
    assert not completed(study.OUT/'test/cells') and not completed(study.OUT/'test/anchors')
    assert protocol['seeds_by_role'] == dict(smoke=2026102100,
        development={'A':2026102101,'C':2026102102}, calibration=2026102103,
        test=2026102104, data=2026102105, timing=2026102106, holdout=2026102107,
        bootstrap=2026102108, development_probes=2026102109)
    assert policy['base_policy'] == protocol['base_policy'] == study.BASE_POLICY
    assert policy['base_policy_source_sha256'] == sha(study.SOURCE/'protocol.json')
    calibration, eligibility = _calibration(policy)
    A = json.loads((study.OUT/'selection_A.json').read_text())
    C = json.loads((study.OUT/'confirmation_C.json').read_text())
    assert A['protocol_sha256'] == C['protocol_sha256'] == policy['protocol_sha256']
    assert A['runtime_calibration_sha256'] == sha(study.OUT/'runtime_calibration.json')
    assert C['selection_A_sha256'] == sha(study.OUT/'selection_A.json')
    assert policy['candidate_rule'] == A['selected_rule'] == C['candidate_rule']
    assert policy['confirmation'] == C
    choice = A['selected_rule']
    symbols, bits = hypotheses(3)
    hashes, cells = {}, {}
    regenerated = algebraic_replays = base_replays = fallback_replays = probe_decodes = 0
    maximum_nll_error = maximum_algebraic_error = maximum_prediction_error = maximum_probe_error = 0.
    est = study.engine()
    forbidden = {'g', 'G', 'truth_g', 'unit_noise', 'paths', 'mu', 'channel', 'channel_truth'}
    for bank, role_seed in (('A', 2026102101), ('C', 2026102102)):
        anchor_paths = completed(study.OUT/'validation'/bank/'anchors')
        assert len(anchor_paths) == 3
        assert len(completed(study.OUT/'validation'/bank/'cells')) == 24
        for regime, pathcount in study.REGIMES:
            ap = study.OUT/'validation'/bank/'anchors'/f'{regime}.npz'
            anchor = load(ap)
            assert not forbidden.intersection(anchor)
            assert str(anchor['bank']) == bank and str(anchor['regime']) == regime
            assert str(anchor['protocol_sha256']) == policy['protocol_sha256']
            assert int(anchor['seed']) == role_seed and int(anchor['physical_seed']) == 1000*role_seed+50
            assert int(anchor['context_P']) == 30 and int(anchor['heldout_P']) == 9
            np.testing.assert_array_equal(anchor['local_indices'], np.arange(DEV_COUNT))
            np.testing.assert_array_equal(anchor['probe_symbols'], symbols)
            np.testing.assert_array_equal(anchor['probe_bits'], bits)
            assert anchor['s'].shape == (DEV_COUNT, 3, 39)
            assert anchor['b'].shape == (DEV_COUNT, 32, 39)
            assert anchor['probe_z_by_snr'].shape == (8, DEV_COUNT, 32, 64)
            assert ap.stat().st_mtime >= protocol['frozen_unix']
            if bank == 'C':
                assert ap.stat().st_mtime >= A['completed_unix']
            ids = [0, DEV_COUNT-1]
            # Truth stays inside this independent simulator reconstruction;
            # it is used only to regenerate observations, never a selection score.
            fresh = risk_data.make_anchor(1000*role_seed+50, ids, pathcount, 5.,
                truth=True, snrs=tuple(s for s in SNRS if s > 30), holdout_seed=2026102107)
            for key in ('s','b','anchor_variance','indices','local_indices'):
                np.testing.assert_array_equal(anchor[key][ids], fresh[key])
            np.testing.assert_array_equal(anchor['z_by_snr'][:, ids], fresh['z_by_snr'])
            for j, effective in enumerate(fresh['indices']):
                rng = np.random.default_rng(np.random.SeedSequence([2026102109, role_seed, int(effective), 0x50524f42]))
                raw = rng.standard_normal((32, 64, 2))
                noise = (raw[...,0]+1j*raw[...,1])/np.sqrt(2)
                field = fresh['g'][j]@symbols.T+fresh['b'][j,:,:1]
                for si, snr in enumerate(s for s in SNRS if s > 30):
                    expected = abs(field + np.sqrt(3.*10**(-snr/10))*noise)
                    actual_probe = anchor['probe_z_by_snr'][si, ids[j]]
                    maximum_probe_error = max(maximum_probe_error, float(abs(actual_probe-expected).max()))
                    # Independent two-case regeneration can select a different
                    # batched BLAS/vectorized path than the archived256-case run.
                    # This tolerance is solely for reconstruction roundoff;
                    # same-output preservation assertions remain bitwise exact.
                    close(actual_probe, expected, atol=1e-13, rtol=1e-13)
                regenerated += 1
            hashes[str(ap.relative_to(study.OUT))] = sha(ap)
            for si, snr in enumerate(s for s in SNRS if s > 30):
                cp = study.OUT/'validation'/bank/'cells'/f'{study.cellname(regime,5.,snr)}.npz'
                cell = load(cp); cells[bank, regime, snr] = cell
                assert not forbidden.intersection(cell)
                assert str(cell['anchor_sha256']) == sha(ap)
                assert str(cell['protocol_sha256']) == policy['protocol_sha256']
                assert str(cell['bank']) == bank and str(cell['regime']) == regime
                assert float(cell['rsr']) == 5. and float(cell['snr']) == snr
                assert cell['base_g'].shape == (3,DEV_COUNT,32,3)
                assert cell['fallback_g'].shape == (2,DEV_COUNT,32,3)
                assert cell['fallback_valid'].shape == (2,DEV_COUNT)
                full = risk_data.observed(anchor, snr, P=39)
                obs, hold = risk_data.fit_holdout(full)
                zz, ss, bb, vv = [x.numpy() for x in obs.args()]
                hz, hs, hb, hv = [x.numpy() for x in hold.args()]
                for group, estimates in (('base',cell['base_g']),('fallback',cell['fallback_g'])):
                    for i, g in enumerate(estimates):
                        for key, value in (('fit_nll_', centered_nll(zz,ss,bb,vv,g)),
                                           ('heldout_nll_', centered_nll(hz,hs,hb,hv,g))):
                            error = float(np.abs(value-cell[key+group][i]).max())
                            maximum_nll_error = max(maximum_nll_error,error)
                            close(cell[key+group][i],value,atol=1e-8,rtol=1e-11)
                for i, g in enumerate(cell['base_g']):
                    scores = residual_scores(zz,ss,bb,vv,g)
                    close(cell['score_mean'][i],scores['mean'],atol=1e-9,rtol=1e-12)
                    close(cell['score_max'][i],scores['max'],atol=1e-9,rtol=1e-12)
                for key, shape in (('probe_errors_base',(3,DEV_COUNT,3)),('probe_errors_fallback',(2,DEV_COUNT,3))):
                    assert cell[key].shape == shape and np.issubdtype(cell[key].dtype,np.integer)
                    assert (cell[key]>=0).all() and (cell[key]<=128).all()
                tails = (10,20) if bank == 'A' else (() if choice is None else (choice['tail'],))
                for tail in (10,20):
                    k=(10,20).index(tail)
                    if tail not in tails:
                        assert not cell['fallback_valid'][k].any()
                        np.testing.assert_array_equal(cell['fallback_g'][k],cell['base_g'][0])
                        np.testing.assert_array_equal(cell['probe_errors_fallback'][k],np.zeros((DEV_COUNT,3),np.int64))
                if snr in (32.5,60.):
                    sample = obs.batch(torch.tensor(ids))
                    initial, valid, ranks, conditions = algebraic_numpy(*[x.numpy() for x in sample.args()])
                    for i, seed in enumerate(SEEDS):
                        value = study.parent.inference.hybrid(sample,seed,study.BASE_POLICY,est).numpy()
                        maximum_prediction_error=max(maximum_prediction_error,float(abs(value-cell['base_g'][i,ids]).max()))
                        close(cell['base_g'][i,ids],value,atol=1e-10,rtol=1e-10); base_replays+=len(ids)
                    for tail in tails:
                        k=(10,20).index(tail)
                        np.testing.assert_array_equal(cell['fallback_valid'][k,ids],valid)
                        close(cell['fallback_condition'][k,ids],conditions,atol=1e-10,rtol=1e-12)
                        value = initial.copy()
                        if valid.any():
                            vi=torch.from_numpy(np.flatnonzero(valid)); part=sample.batch(vi)
                            value[valid]=study.inference.finish(part,continuation.estimate(part,torch.from_numpy(initial[valid]),tail)).numpy()
                        maximum_algebraic_error=max(maximum_algebraic_error,float(abs(value-cell['fallback_g'][k,ids]).max()))
                        close(cell['fallback_g'][k,ids],value,atol=1e-10,rtol=1e-10)
                        algebraic_replays+=len(ids); fallback_replays+=int(valid.sum())
                    for case in ids:
                        probe=anchor['probe_z_by_snr'][si,case]
                        for i,g in enumerate(cell['base_g']):
                            decoded=detect(probe,g[case],bb[case,:,:1],float(vv[case]))
                            np.testing.assert_array_equal((decoded!=bits).sum((0,2)),cell['probe_errors_base'][i,case]);probe_decodes+=1
                        for tail in tails:
                            k=(10,20).index(tail)
                            if cell['fallback_valid'][k,case]:
                                decoded=detect(probe,cell['fallback_g'][k,case],bb[case,:,:1],float(vv[case]))
                                np.testing.assert_array_equal((decoded!=bits).sum((0,2)),cell['probe_errors_fallback'][k,case]);probe_decodes+=1
                hashes[str(cp.relative_to(study.OUT))]=sha(cp)
    scores={}
    for rule in (None,*study.RULES):
        key='incumbent' if rule is None else rule['id']
        value=_development_score('A',rule,cells,hashes)
        _compare_score(A['scores'][key],value)
        eligible=rule is None or (eligibility[key] and all(
            r['mean_improvement']>NLL_TOL and r['candidate_errors']<=r['base_errors'] for r in value['regimes'].values()))
        assert A['scores'][key]['rule']==rule and A['scores'][key]['eligible']==eligible
        scores[key]=dict(rule=rule,eligible=eligible,**value)
    feasible=[r for r in scores.values() if r['eligible']]
    minimum=min(r['mean_nll'] for r in feasible)
    tied=[r for r in feasible if r['mean_nll']<=minimum+NLL_TOL]
    def tie_key(r):
        rule=r['rule']
        return (-1,0,0,0) if rule is None else (r['trigger_fraction'],rule['tail'],-rule['threshold'],int(rule['metric']=='max'))
    assert min(tied,key=tie_key)['rule']==choice
    confirm_score=_development_score('C',choice,cells,hashes)
    _compare_score(C['score'],confirm_score)
    rng=np.random.default_rng(np.random.SeedSequence([BOOT_SEED,0x434f4e46]))
    boot=[]
    for regime in REGIMES:
        values=np.array(confirm_score['regimes'][regime]['anchor_improvements'])
        w=rng.multinomial(DEV_COUNT,np.full(DEV_COUNT,1/DEV_COUNT),size=BOOT)
        means=w@values/DEV_COUNT;boot.append(means)
        close(C['regime_nll_ci95_descriptive'][regime],np.quantile(means,[.025,.975]))
    pooled=np.mean(boot,axis=0);lower=float(np.quantile(pooled,.05))
    close(C['pooled_nll_lower_one_sided95'],lower)
    close(C['pooled_nll_ci95'],np.quantile(pooled,[.025,.975]))
    ber_ok=all(r['candidate_errors']<=r['base_errors'] for r in confirm_score['regimes'].values())
    passed=choice is not None and lower>0 and ber_ok
    assert C['empirical_ber_nonincrease_every_regime']==ber_ok and C['passed']==passed
    assert C['status']==('incumbent_selected' if choice is None else ('confirmed' if passed else 'unconfirmed'))
    assert policy['operational_rule']==(choice if passed else None)
    assert protocol['frozen_unix']<=calibration['started_unix']<=calibration['completed_unix']<=A['completed_unix']<=C['completed_unix']<=policy['frozen_unix']
    return record('selection',selection_evidence_hashes=policy['selection_evidence'],
        runtime_calibration_calls=5508,forced_fallback_calibration_calls=144,
        runtime_calibration_quantiles_and_eligibility_verified=True,
        validation_artifact_hashes=hashes,independently_regenerated_validation_anchors=regenerated,
        independently_replayed_base_predictions=base_replays,independently_replayed_algebraic_starts=algebraic_replays,
        independently_replayed_fallback_predictions=fallback_replays,independently_redecoded_validation_probes=probe_decodes,
        maximum_likelihood_replay_error=maximum_nll_error,maximum_base_replay_error=maximum_prediction_error,
        maximum_fallback_replay_error=maximum_algebraic_error,
        maximum_development_probe_reconstruction_error=maximum_probe_error,
        independent_probe_reconstruction_tolerance=dict(atol=1e-13,rtol=1e-13),
        exact_rule_menu_tie_break_and_runtime_guard_verified=True,only_one_C_proposal_no_runnerup=True,
        no_truth_exports_and_independent30plus9_pilots_verified=True,
        independent_development_anchor_clusters=768,confirmation_bootstrap_draws=BOOT,
        candidate_rule=choice,confirmation_status=C['status'],operational_rule=policy['operational_rule'],
        test_artifacts_absent_at_selection_audit=True)


@torch.no_grad()
def test():
    """Reconstruct physical anchors and independently rebuild all gate decisions."""
    from sahure.data import Observations
    from tcom_incumbent_guard_20261019 import risk_data
    from tcom_pilot_em_20261018 import continuation
    from tcom_runtime_bypass_20261017.sparse_fast import spectral_initial
    study = module()
    policy, digest = study.check_selection()
    paths = completed(study.OUT/'test/cells')
    assert len(paths) == 153 and len(completed(study.OUT/'test/anchors')) == 9
    names = tuple(study.NAMES); ni = {name:i for i,name in enumerate(names)}
    assert names == tuple(f'{family}_{seed}' for family in
        ('receiver','prior','latest','adaptive40','global40') for seed in SEEDS)+(
            'classical40','em100','compact_sparse','algebraic20')
    anchors, hashes, anchor_hashes, seen = {}, {}, {}, set()
    regenerated = replayed = gated_cases = identity_fit_cells = algebraic_cases = 0
    maximum_replay = maximum_gate = 0.
    est = study.engine(); choice = policy['candidate_rule']
    for regime, pathcount in study.REGIMES:
        for ref in RSRS:
            ap = study.OUT/'test/anchors'/f'{regime}_rsr{ref:g}.npz'; a = load(ap)
            assert str(a['selection_sha256']) == digest
            assert str(a['protocol_sha256']) == policy['protocol_sha256']
            assert int(a['seed']) == 2026102104
            derived = 1000*2026102104+round(10*ref)
            assert int(a['physical_seed']) == derived
            ids = [0,TEST_COUNT-1]
            independent = risk_data.make_anchor(derived,ids,pathcount,ref,truth=True,
                                                snrs=SNRS,holdout_seed=2026102107)
            for key in ('s','b','anchor_variance','indices','local_indices','g','paths','unit_noise'):
                np.testing.assert_array_equal(a[key][ids],independent[key])
            np.testing.assert_array_equal(a['z_by_snr'][:,ids],independent['z_by_snr'])
            np.testing.assert_array_equal(a['local_indices'],np.arange(TEST_COUNT))
            anchors[regime,ref]=a;anchor_hashes[ap.name]=sha(ap);regenerated+=len(ids)
            assert ap.stat().st_mtime>=policy['frozen_unix']
    for path in paths:
        a = load(path)
        regime, ref, snr = str(a['regime']),float(a['rsr']),float(a['snr'])
        seen.add((regime,ref,snr)); anchor=anchors[regime,ref]
        assert path.name==study.cellname(regime,ref,snr)+'.npz'
        assert str(a['selection_sha256'])==digest and str(a['protocol_sha256'])==policy['protocol_sha256']
        assert int(a['seed'])==2026102104 and path.stat().st_mtime>=policy['frozen_unix']
        np.testing.assert_array_equal(a['method_names'],names)
        pred = a['estimates']
        assert pred.shape==(19,TEST_COUNT,32,3) and pred.dtype==np.complex128 and np.isfinite(pred).all()
        for key in ('indices','local_indices','paths'):
            np.testing.assert_array_equal(a[key],anchor[key])
        np.testing.assert_array_equal(a['truth_g'],anchor['g'])
        for key,value in zip(('z','s','b','sigma2'),risk_data.observed(anchor,snr).args()):
            np.testing.assert_array_equal(a[key],value.numpy())
        energy=np.sum(abs(a['truth_g'])**2,axis=(-2,-1))
        errors=np.sum(abs(pred-a['truth_g'][None])**2,axis=(-2,-1))
        np.testing.assert_array_equal(a['errors'],errors)
        np.testing.assert_array_equal(a['truth_energy'],energy)
        np.testing.assert_array_equal(a['nmse'],errors/energy[None])
        for key in ('gate_scores','triggers','accepted','rank_invalid'):
            assert a[key].shape==(3,TEST_COUNT)
        assert a['triggers'].dtype==a['accepted'].dtype==a['rank_invalid'].dtype==bool
        assert np.all(~a['accepted']|a['triggers']) and np.all(~a['rank_invalid']|a['triggers'])
        obs=Observations(*(torch.from_numpy(a[key]) for key in ('z','s','b','sigma2')))
        sample=obs.batch(slice(TEST_COUNT-1,TEST_COUNT))
        # Every non-receiver arm independently traverses its standalone path.
        for name in names:
            if name.startswith('receiver_') or name=='algebraic20':
                continue
            value=study.estimate_method(sample,name,policy,est).numpy()
            maximum_replay=max(maximum_replay,float(abs(value-pred[ni[name],-1:]).max()))
            close(value,pred[ni[name],-1:],atol=1e-10,rtol=1e-10);replayed+=1
        initial,valid,ranks,conditions=algebraic_numpy(*[x.numpy() for x in sample.args()])
        if not valid[0]:
            initial=spectral_initial(a['z'][-1:],a['s'][-1:],a['b'][-1:])
        value=study.inference.finish(sample,continuation.estimate(sample,torch.from_numpy(initial),20)).numpy()
        close(value,pred[ni['algebraic20'],-1:],atol=1e-10,rtol=1e-10)
        maximum_replay=max(maximum_replay,float(abs(value-pred[ni['algebraic20'],-1:]).max()))
        replayed+=1;algebraic_cases+=1
        eligible = ref==5. and snr>30.
        if choice is None or not eligible:
            assert np.isnan(a['gate_scores']).all()
            assert not a['triggers'].any() and not a['accepted'].any() and not a['rank_invalid'].any()
            for seed in SEEDS:
                np.testing.assert_array_equal(pred[ni[f'receiver_{seed}']],pred[ni[f'prior_{seed}']])
                identity_fit_cells+=1
        else:
            base=np.stack([pred[ni[f'prior_{seed}']] for seed in SEEDS])
            scores=np.stack([residual_scores(a['z'],a['s'],a['b'],a['sigma2'],g)[choice['metric']] for g in base])
            close(a['gate_scores'],scores,atol=1e-9,rtol=1e-12)
            triggers=scores>choice['threshold']
            np.testing.assert_array_equal(a['triggers'],triggers)
            union=triggers.any(0); alternative=np.zeros((TEST_COUNT,32,3),np.complex128)
            full_valid=np.zeros(TEST_COUNT,bool)
            if union.any():
                ids=np.flatnonzero(union);part=obs.batch(torch.from_numpy(ids))
                initial,valid,ranks,conditions=algebraic_numpy(*[x.numpy() for x in part.args()])
                refined=initial.copy()
                if valid.any():
                    vi=np.flatnonzero(valid);vpart=part.batch(torch.from_numpy(vi))
                    value=study.inference.finish(vpart,continuation.estimate(vpart,torch.from_numpy(initial[vi]),choice['tail'])).numpy()
                    finite=np.isfinite(value).all((-2,-1));valid[vi[~finite]]=False
                    refined[vi[finite]]=value[finite]
                alternative[ids]=refined;full_valid[ids]=valid;algebraic_cases+=len(ids)
            for i,seed in enumerate(SEEDS):
                before=centered_nll(a['z'],a['s'],a['b'],a['sigma2'],base[i])
                after=centered_nll(a['z'],a['s'],a['b'],a['sigma2'],alternative)
                accept=triggers[i]&full_valid&(after<before-NLL_TOL)
                np.testing.assert_array_equal(a['accepted'][i],accept)
                np.testing.assert_array_equal(a['rank_invalid'][i],triggers[i]&~full_valid)
                expected=np.where(accept[:,None,None],alternative,base[i])
                actual=pred[ni[f'receiver_{seed}']]
                maximum_gate=max(maximum_gate,float(abs(expected-actual).max()))
                close(actual,expected,atol=1e-10,rtol=1e-10)
                np.testing.assert_array_equal(actual[~accept],base[i,~accept])
                assert np.all(after[accept]<before[accept]-NLL_TOL)
                gated_cases+=TEST_COUNT
        if snr<=20:
            for seed in SEEDS:
                np.testing.assert_array_equal(pred[ni[f'receiver_{seed}']],pred[ni[f'latest_{seed}']])
        hashes[path.name]=sha(path)
    assert seen=={(r,ref,snr) for r in REGIMES for ref in RSRS for snr in SNRS}
    return record('test',cells=153,test_prediction_hashes=hashes,test_anchor_hashes=anchor_hashes,
        independently_regenerated_anchor_cases=regenerated,independent_reference_seed_domains=3,
        independently_replayed_standalone_predictions=replayed,maximum_standalone_replay_error=maximum_replay,
        independently_replayed_algebraic_cases=algebraic_cases,independently_rebuilt_gate_fit_cases=gated_cases,
        maximum_gate_replay_error=maximum_gate,exact_outside_gate_fit_cells=identity_fit_cells,
        exact_rejected_and_invalid_incumbent_preservation=True,all_error_arrays_exact=True,
        ordinary_exact_latest_preserved=True,candidate_rule_is_frozen_A_proposal=True)


def timing():
    study=module();policy,digest=study.check_selection()
    windows,hashes={},{};previous_end=0.;total=0
    for window in (1,2):
        path=study.OUT/f'timing_window{window}.json';a=json.loads(path.read_text());raw=a['raw_samples']
        assert a['window']==window and a['selection_sha256']==digest and a['protocol_sha256']==policy['protocol_sha256']
        ids=[500,501,502] if window==1 else [600,601,602]
        assert a['indices']==ids and a['worlds']==459 and set(raw)==set(study.NAMES)
        assert a['started_unix']>=max(previous_end,policy['frozen_unix']) and a['completed_unix']>=a['started_unix']
        previous_end=a['completed_unix'];assert a['environment']['torch_threads']==1
        worlds=[(regime,ref,snr,index) for ref in RSRS for regime in REGIMES for index in ids for snr in SNRS]
        rng=np.random.default_rng(np.random.SeedSequence([2026102106,window]))
        # Each world's method permutation advances the same RNG used by the
        # production scheduler, but does not alter the already-drawn world list.
        order=[worlds[i] for i in rng.permutation(459)]
        for rows in raw.values():
            assert [(r['regime'],r['rsr'],r['snr'],r['index']) for r in rows]==order
        expanded=dict(raw)
        expanded.update({name:[r for member in members for r in raw[member]] for name,members in study.GROUPS.items()})
        assert set(expanded)==set(a['methods'])==set(a['cap_verification'])
        flags,overruns={},{}
        for name,rows in expanded.items():
            item=a['methods'][name];verify_stats(rows,item)
            weak=[r for r in rows if r['rsr']==5 and r['snr']>30]
            verify_stats(weak,item['weak_target'])
            verify_stats([r for r in rows if r['rsr']==5 and r['snr']>=37.5],item['weak_warm100'])
            triggered=[r for r in rows if r.get('gate_triggered')]
            assert item['trigger_count']==len(triggered)
            assert item['accepted_count']==sum(r.get('fallback_accepted',False) for r in rows)
            if triggered:verify_stats(triggered,item['triggered_branch'])
            else:assert item['triggered_branch'] is None
            assert len(item['by_condition'])==51
            for ref in RSRS:
                for snr in SNRS:
                    verify_stats([r for r in rows if r['rsr']==ref and r['snr']==snr],item['by_condition'][f'rsr{ref:g}_snr{snr:g}'])
            cap=a['cap_verification'][name]
            close(cap['p90_ms'],item['p90_ms']);close(cap['weak_target_p90_ms'],item['weak_target']['p90_ms'])
            passed=item['p90_ms']<=10.74 and item['weak_target']['p90_ms']<=10.74
            assert cap['passed']==passed;flags[name]=passed
            overruns[name]=[key for key,v in item['by_condition'].items() if v['p90_ms']>10.74]
            for row in rows:
                if 'gate_triggered' in row:
                    assert not row['fallback_accepted'] or row['gate_triggered']
                    assert not row['rank_invalid'] or row['gate_triggered']
                    if row['rsr']!=5 or row['snr']<=30:
                        assert not row['gate_triggered'] and not row['fallback_accepted']
        if window==1:
            assert min(p.stat().st_mtime for p in completed(study.OUT/'test/anchors'))>=a['completed_unix']
        else:
            assert max(p.stat().st_mtime for p in completed(study.OUT/'test/cells'))<=a['started_unix']
            assert (study.OUT/'ber/summary.json').stat().st_mtime<=a['started_unix']
        calls=sum(map(len,raw.values()));assert calls==a['calls']==19*459;total+=calls
        windows[f'window{window}']=dict(raw_calls=calls,cap_passed_by_method=flags,conditional_overruns=overruns,
            p90_ms={n:v['p90_ms'] for n,v in a['methods'].items()},
            weak_target_p90_ms={n:v['weak_target']['p90_ms'] for n,v in a['methods'].items()},
            triggered_branch={n:v['triggered_branch'] for n,v in a['methods'].items()})
        hashes[path.name]=sha(path)
    assert total==17442
    return record('timing',timing_hashes=hashes,windows=windows,raw_calls=total,literal_cap_ms=10.74,
        formal_cap_scopes=['whole_grid','weak_target'],
        target_met_in_both_windows_by_method={n:all(w['cap_passed_by_method'][n] for w in windows.values()) for n in windows['window1']['cap_passed_by_method']},
        every_grid_quantile_trigger_statistic_and_cap_flag_verified=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('phase',choices=('selection','test','ber','summary','timing'))
    args=parser.parse_args()
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    print(json.dumps(globals()[args.phase]()),flush=True)
