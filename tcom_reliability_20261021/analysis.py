"""Frozen paired scoring of the reference-aware reliability-check receiver."""
import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_key] = '1'
os.environ['MPLCONFIGDIR'] = '/private/tmp/tcom_reliability_matplotlib'
import argparse
import json
import numpy as np
from tcom_followups_20261017.common import sha, write_json
from . import study

OUT = study.OUT
COUNT, BOOT = study.TEST_COUNT, 20000
TOL = 1e-10
NAMES = tuple(study.NAMES)
GROUPS = {name: tuple(members) for name, members in study.GROUPS.items()}
PRIMARY = ('receiver_vs_prior', 'receiver_vs_latest')
CONTRASTS = {
    'receiver_vs_prior': ('prior_mean', 'receiver_mean'),
    'receiver_vs_latest': ('latest_mean', 'receiver_mean'),
    'receiver_vs_compact': ('compact_sparse', 'receiver_mean'),
    'receiver_vs_adaptive40': ('adaptive40_mean', 'receiver_mean'),
    'receiver_vs_global40': ('global40_mean', 'receiver_mean'),
    'receiver_vs_classical40': ('classical40', 'receiver_mean'),
    'receiver_vs_em100': ('em100', 'receiver_mean'),
    'receiver_vs_algebraic20': ('algebraic20', 'receiver_mean'),
}
REGIONS = ('all','high_all','transition_all','weak_warm100','strong_high','strong_transition',
           'ordinary','weak','weak_target','rsr5','rsr12','rsr20','all_L2','all_L4','all_random3to7',
           'weak_warm100_L2','weak_warm100_L4','weak_warm100_random3to7')


def belongs(region, snr, rsr=None, regime=None):
    scopes = dict(all=True, high_all=snr>=40, transition_all=20<snr<40,
        weak_warm100=rsr==5 and snr>=37.5, strong_high=rsr in (12,20) and snr>=40,
        strong_transition=rsr in (12,20) and 32.5<=snr<=37.5, ordinary=snr<=20, weak=rsr==5, weak_target=rsr==5 and snr>30)
    if region in scopes:
        return scopes[region]
    if region.startswith('rsr'):
        return rsr==float(region[3:])
    for prefix in ('all_', 'weak_warm100_'):
        if region.startswith(prefix):
            return scopes[prefix[:-1]] and regime==region[len(prefix):]
    raise KeyError(region)


def preflight():
    expected = tuple(f'{family}_{seed}' for family in
        ('receiver','prior','latest','adaptive40','global40') for seed in study.SEEDS)
    expected += ('classical40','em100','compact_sparse','algebraic20')
    assert NAMES == expected and len(GROUPS)==5
    assert study.BOOT_SEED==2026102108 and COUNT==128
    counts = {region: sum(belongs(region,snr,ref,r) for r,_ in study.REGIMES
                          for ref in study.RSRS for snr in study.SNRS) for region in REGIONS}
    assert counts['all']==153 and counts['ordinary']==counts['high_all']==45
    assert counts['transition_all']==63 and counts['weak_warm100']==18 and counts['weak_target']==24
    return dict(passed=True,methods=len(NAMES),groups=len(GROUPS),region_counts=counts,
                contrasts=CONTRASTS,primary=PRIMARY,primary_outcomes=4)


def expand(values, raw_names):
    return np.concatenate((values, np.stack([
        values[[raw_names.index(name) for name in members]].mean(0)
        for members in GROUPS.values()])))


def interval(values):
    return np.quantile(values, [.025, .975]).tolist()


def weights():
    rng = np.random.default_rng(study.BOOT_SEED)
    return {(r,ref): rng.multinomial(COUNT,np.full(COUNT,1/COUNT),size=BOOT).T.astype(float)
            for r,_ in study.REGIMES for ref in study.RSRS}


def cell_inputs(regime, ref, snr, record, digest):
    name = study.cellname(regime,ref,snr)
    paths = dict(prediction=OUT/'test/cells'/f'{name}.npz',
                 ber=OUT/'ber/cells'/f'{name}.npz', data=OUT/'ber/data'/f'{name}.npz')
    hashes = {key:sha(path) for key,path in paths.items()}
    a,b = study.load(paths['prediction']),study.load(paths['ber'])
    assert tuple(map(str,a['method_names']))==NAMES
    assert tuple(map(str,b['method_names']))==('perfect_csi',)+NAMES
    for item in (a,b):
        assert str(item['protocol_sha256'])==record['protocol_sha256']
        assert str(item['selection_sha256'])==digest
    assert str(b['source_sha256'])==hashes['prediction'] and str(b['data_sha256'])==hashes['data']
    for key in ('indices','local_indices'):
        np.testing.assert_array_equal(a[key],b[key])
    energy = np.sum(abs(a['truth_g'])**2,axis=(-2,-1))
    err = np.sum(abs(a['estimates']-a['truth_g'][None])**2,axis=(-2,-1))
    np.testing.assert_array_equal(a['truth_energy'],energy)
    np.testing.assert_array_equal(a['errors'],err)
    assert err.shape==(19,COUNT) and np.isfinite(err).all() and np.all(energy>0)
    counts = b['errors_per_user'].sum(-1)
    assert counts.shape==(20,COUNT) and np.all(counts>=0) and np.all(counts<=384)
    np.testing.assert_array_equal(b['bits_per_user'],np.full((COUNT,3),128))
    gate = {key:a[key] for key in ('gate_scores','triggers','accepted','rank_invalid')}
    for item in gate.values():
        assert item.shape==(3,COUNT)
    assert np.all(np.logical_or(~gate['accepted'].astype(bool),gate['triggers'].astype(bool)))
    return name,err,energy,counts,hashes,gate


def gate_summary(items):
    result = {}
    for i,seed in enumerate(study.SEEDS):
        triggers = np.concatenate([a['triggers'][i].astype(bool) for a in items])
        accepted = np.concatenate([a['accepted'][i].astype(bool) for a in items])
        invalid = np.concatenate([a['rank_invalid'][i].astype(bool) for a in items])
        result[str(seed)] = dict(cases=len(triggers),triggers=int(triggers.sum()),
            accepted=int(accepted.sum()),rank_invalid=int(invalid.sum()),
            trigger_fraction=float(triggers.mean()),accepted_fraction=float(accepted.mean()),
            accepted_given_trigger=float(accepted.sum()/triggers.sum()) if triggers.any() else None)
    return result


def weak_concentration(rows,errors,energies,record,digest):
    cell_rows, target_rows = [], []
    for row,err,energy in zip(rows,errors,energies):
        if row['rsr']!=5:
            continue
        nm=err/energy[None]
        for i,name in enumerate(NAMES):
            index=int(np.argmax(err[i]));total=float(err[i].sum())
            cell_rows.append(dict(cell=row['cell'],regime=row['regime'],snr=row['snr'],method=name,
                top1_local_index=index,top1_error_share=float(err[i,index]/total) if total>0 else 0.,
                top1_case_nmse=float(nm[i,index]),case_nmse_p95=float(np.quantile(nm[i],.95)),
                case_nmse_p99=float(np.quantile(nm[i],.99)),case_nmse_max=float(nm[i].max())))
    for regime,_ in study.REGIMES:
        chosen=[i for i,r in enumerate(rows) if r['regime']==regime and r['rsr']==5 and r['snr']>30]
        assert len(chosen)==8
        total_error=np.stack([errors[i] for i in chosen]).sum(0)
        nm=np.concatenate([errors[i]/energies[i][None] for i in chosen],axis=1)
        for i,name in enumerate(NAMES):
            index=int(np.argmax(total_error[i]));total=float(total_error[i].sum())
            target_rows.append(dict(regime=regime,method=name,conditions=8,physical_anchors=128,
                top1_local_index=index,top1_error_share=float(total_error[i,index]/total) if total>0 else 0.,
                case_nmse_p95=float(np.quantile(nm[i],.95)),case_nmse_p99=float(np.quantile(nm[i],.99)),
                case_nmse_max=float(nm[i].max())))
    return dict(selection_sha256=digest,protocol_sha256=record['protocol_sha256'],
        meaning='Descriptive numerator-error concentration. Cell top1 uses one physicalanchor. Weak-target top1 sums each anchors error across8coupledSNRs before selecting it; those repeats are not independent failures. Allrawmethods/fits retained.',
        cells=cell_rows,weak_target_by_regime=target_rows,source_sha256=sha(__file__))


def run():
    preflight()
    record, digest = study.check_selection()
    nmse_names = NAMES + tuple(GROUPS)
    ber_raw = ('perfect_csi',) + NAMES
    ber_names = ber_raw + tuple(GROUPS)
    ni, bi = ({name: i for i, name in enumerate(names)} for names in (nmse_names, ber_names))
    w = weights()
    nsums = {r: np.zeros((len(nmse_names), BOOT)) for r in REGIONS}
    bsums = {r: np.zeros((len(ber_names), BOOT)) for r in REGIONS}
    rows, brows, bcontrasts, errors, energies, counts_all, gates = [], [], [], [], [], [], []
    npoints, bpoints, hashes = [], [], {}
    for regime, _ in study.REGIMES:
        for ref in study.RSRS:
            for snr in study.SNRS:
                name, err, energy, counts, current_hashes, gate = cell_inputs(regime, ref, snr, record, digest)
                hashes[name] = current_hashes
                point = expand(10 * np.log10(err.sum(1) / energy.sum()), NAMES)
                boot = expand(10 * np.log10((err @ w[(regime,ref)]) / (energy @ w[(regime,ref)])[None]), NAMES)
                case_nmse = err / energy[None]
                p95, p99 = (expand(np.quantile(case_nmse, q, axis=1), NAMES) for q in (.95, .99))
                severe = expand((case_nmse > 1).mean(1), NAMES)
                maximum = expand(case_nmse.max(1),NAMES)
                values = {method: dict(nmse_db=float(point[i]), conditional_ci95_db=interval(boot[i]),
                                      case_nmse_p95=float(p95[i]), case_nmse_p99=float(p99[i]),
                                      severe_rate=float(severe[i]),case_nmse_max=float(maximum[i])) for method,i in ni.items()}
                comparisons = {key: dict(gain_db=float(point[ni[left]] - point[ni[right]]),
                                          pointwise_ci95_db=interval(boot[ni[left]] - boot[ni[right]]))
                               for key, (left, right) in CONTRASTS.items()}
                rows.append(dict(cell=name, regime=regime, rsr=ref, snr=snr, cases=COUNT,
                                 methods=values, contrasts=comparisons))
                rate = expand(counts / 384, ber_raw)
                bpoint, bboot = rate.mean(1), (rate @ w[(regime,ref)]) / COUNT
                for method, i in bi.items():
                    value = dict(regime=regime, rsr_db=ref, snr_db=snr, method=method,
                                 ber=float(bpoint[i]), bootstrap95=interval(bboot[i]),
                                 bits_per_fit=COUNT * 384, channel_anchors=COUNT)
                    if method in GROUPS:
                        value['errors_per_training_seed'] = {
                            n: int(counts[ber_raw.index(n)].sum()) for n in GROUPS[method]}
                    else:
                        value['bit_errors'] = int(counts[i].sum())
                    brows.append(value)
                for key, (left, right) in CONTRASTS.items():
                    bcontrasts.append(dict(cell=name, regime=regime, rsr_db=ref, snr_db=snr,
                        contrast=key, reference=left, candidate=right,
                        ber_reduction=float(bpoint[bi[left]] - bpoint[bi[right]]),
                        bootstrap95=interval(bboot[bi[left]] - bboot[bi[right]])))
                for region in REGIONS:
                    if belongs(region, snr, ref, regime):
                        nsums[region] += boot
                        bsums[region] += bboot
                gates.append(gate)
                errors.append(err)
                energies.append(energy)
                counts_all.append(counts)
                npoints.append(point)
                bpoints.append(bpoint)
    assert len(rows) == 153
    npoints, bpoints = np.stack(npoints), np.stack(bpoints)
    aggregates, bregions, nboots, bboots = {}, {}, {}, {}
    for region in REGIONS:
        keep = np.array([belongs(region, r['snr'], r['rsr'], r['regime']) for r in rows])
        total = int(keep.sum())
        point, bpoint = npoints[keep].mean(0), bpoints[keep].mean(0)
        boot, bboot = nsums[region] / total, bsums[region] / total
        nboots[region], bboots[region] = boot, bboot
        err = np.concatenate([a for a, k in zip(errors, keep) if k], axis=1)
        energy = np.concatenate([a for a, k in zip(energies, keep) if k])
        pooled_linear = expand(err.sum(1) / energy.sum(), NAMES)
        pooled = 10 * np.log10(pooled_linear)
        nm = err / energy[None]
        p95, p99 = (expand(np.quantile(nm, q, axis=1), NAMES) for q in (.95, .99))
        severe = expand((nm > 1).mean(1), NAMES)
        maximum = expand(nm.max(1),NAMES)
        values = {method: dict(nmse_db=float(point[i]), conditional_ci95_db=interval(boot[i]),
                    pooled_ratio_of_sums_linear=float(pooled_linear[i]),
                    pooled_ratio_of_sums_db=float(pooled[i]), case_nmse_p95=float(p95[i]),
                    case_nmse_p99=float(p99[i]),severe_rate=float(severe[i]),case_nmse_max=float(maximum[i]))
                  for method, i in ni.items()}
        comparisons = {}
        for key, (left, right) in CONTRASTS.items():
            seed_gains = {}
            for seed in study.SEEDS:
                lfit = f'{left[:-5]}_{seed}' if left.endswith('_mean') else left
                rfit = f'{right[:-5]}_{seed}' if right.endswith('_mean') else right
                seed_gains[str(seed)] = float(point[ni[lfit]] - point[ni[rfit]])
            comparisons[key] = dict(gain_db=float(point[ni[left]] - point[ni[right]]),
                pointwise_ci95_db=interval(boot[ni[left]] - boot[ni[right]]),
                seed_specific_gains_db=seed_gains)
            if region=='all' and key in PRIMARY:
                comparisons[key]['bonferroni4_interval_db']=np.quantile(boot[ni[left]]-boot[ni[right]],[.05/8,1-.05/8]).tolist()
        aggregates[region] = dict(cells=total, condition_cases=total * COUNT,
                                  methods=values,contrasts=comparisons,gate=gate_summary([g for g,k in zip(gates,keep) if k]))
        total_errors = np.sum([a.sum(1) for a, k in zip(counts_all, keep) if k], axis=0)
        bm = {}
        for method, i in bi.items():
            value = dict(ber=float(bpoint[i]), bootstrap95=interval(bboot[i]),
                         bits_per_fit=total * COUNT * 384)
            if method in GROUPS:
                value['errors_per_training_seed'] = {
                    n: int(total_errors[ber_raw.index(n)]) for n in GROUPS[method]}
                value['training_seed_ber_range'] = [
                    float(min(bpoint[bi[n]] for n in GROUPS[method])),
                    float(max(bpoint[bi[n]] for n in GROUPS[method]))]
            else:
                value['bit_errors'] = int(total_errors[i])
            bm[method] = value
        bc = {}
        for key, (left, right) in CONTRASTS.items():
            bc[key] = dict(reference=left, candidate=right,
                ber_reduction=float(bpoint[bi[left]] - bpoint[bi[right]]),
                bootstrap95=interval(bboot[bi[left]] - bboot[bi[right]]),
                both_methods_zero_errors=bool(bpoint[bi[left]] == 0 and bpoint[bi[right]] == 0))
            if region=='all' and key in PRIMARY:
                bc[key]['bonferroni4_interval']=np.quantile(bboot[bi[left]]-bboot[bi[right]],[.05/8,1-.05/8]).tolist()
        nblocks = len({(r['regime'],r['rsr']) for r,k in zip(rows,keep) if k})
        bregions[region] = dict(cells=total, operating_cases=total * COUNT,
                                coupled_clusters=nblocks * COUNT, methods=bm, contrasts=bc)
    negative, bnegative = {}, {}
    for key in CONTRASTS:
        worst = min(rows, key=lambda r: r['contrasts'][key]['gain_db'])
        negative[key] = dict(
            negative_cells=[r['cell'] for r in rows if r['contrasts'][key]['gain_db'] < -TOL],
            pointwise95_harm_cells=[r['cell'] for r in rows if r['contrasts'][key]['pointwise_ci95_db'][1] < -TOL],
            worst_cell=worst['cell'], worst_gain_db=worst['contrasts'][key]['gain_db'])
        bsub = [r for r in bcontrasts if r['contrast'] == key]
        bworst = min(bsub, key=lambda r: r['ber_reduction'])
        bnegative[key] = dict(
            negative_cells=[r['cell'] for r in bsub if r['ber_reduction'] < -1e-15],
            pointwise95_harm_cells=[r['cell'] for r in bsub if r['bootstrap95'][1] < -1e-15],
            worst_cell=bworst['cell'], worst_ber_reduction=bworst['ber_reduction'])
    common = dict(selection_sha256=digest, protocol_sha256=record['protocol_sha256'],
        confirmation=record['confirmation'], candidate_rule=record['candidate_rule'],
        operational_rule=record['operational_rule'], groups=GROUPS, contrasts=CONTRASTS,
        unique_test_anchors=1152, condition_cases=153*COUNT, bootstrap_index_clusters=1152,
        primary=dict(contrasts=PRIMARY,region='all',outcomes=4),
        bootstrap=dict(samples=BOOT,seed=study.BOOT_SEED,
            unit='Independent128-index multinomial draws in(regime,reference)order; shared across SNR/method/fit within each of9 blocks. Newworldseeds domain-separate references, including path counts.',
            scope='Conditional on frozen fits/policies/grid. Bonferroni4 only for two whole-grid contrasts acrossNMSE/BER; all other intervals pointwise descriptive.'),
        interpretation='receiver always means the frozen A-selected proposal. If C fails, it is unconfirmed and the operational incumbent remains prior; test outcomes never change that choice.',
        test_hashes=hashes,source_sha256=sha(__file__))
    nmse = dict(**common, method_names=nmse_names, aggregates=aggregates,
        negative_cells=negative, cells=rows, numerical_tie_tolerance_db=TOL,
        scoring='Equal-cell mean dB ratio-of-sums; group means average metrics, not channels. '
                'Pooled linear is the arithmetic mean of fit-specific pooled error/energy ratios. Group p95/p99/max average corresponding fit-specific diagnostics, not an ensemble-channel distribution.')
    ber = dict(**common, method_names=ber_names, regions=bregions, overall=bregions['all'],
        negative_cells=bnegative, cells=brows, cell_contrasts=bcontrasts,
        zero_errors='Collapsed empirical intervals and zero observed BER do not establish '
                    'equivalence, population noninferiority or rare-event bounds.')
    concentration = weak_concentration(rows,errors,energies,record,digest)
    write_json(OUT/'weak_error_concentration.json',concentration)
    nmse['weak_error_concentration_sha256']=sha(OUT/'weak_error_concentration.json')
    write_json(OUT / 'nmse_summary.json', nmse)
    write_json(OUT / 'ber/summary.json', ber)
    for destination, names, points, boots in (
        (OUT / 'nmse_bootstrap.npz', nmse_names, npoints, nboots),
        (OUT / 'ber/bootstrap.npz', ber_names, bpoints, bboots)):
        study.save(destination, method_names=np.array(names), regions=np.array(REGIONS),
            aggregate_bootstraps=np.stack([boots[r] for r in REGIONS]),
            cell_names=np.array([r['cell'] for r in rows]), cell_scores=points,
            selection_sha256=np.array(digest))
    figures(nmse, ber)
    print(json.dumps(dict(confirmation=record['confirmation'], whole_grid_nmse=aggregates['all']['contrasts'],
                          whole_grid_ber=bregions['all']['contrasts'])), flush=True)


def figures(nmse, ber):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    styles = {'receiver_mean': ('Reliability-check proposal', '#0969da', '-'),
              'prior_mean': ('Prior selective EM', '#bf3989', '--'),
              'latest_mean': ('Frozen latest', '#8a63d2', ':'),
              'compact_sparse': ('Compact sparse', '#26965c', '-.')}
    for metric in ('nmse', 'ber'):
        fig, axes = plt.subplots(3, 3, figsize=(14, 10), sharex=True, sharey=True)
        for i, (regime, _) in enumerate(study.REGIMES):
            for j, ref in enumerate(study.RSRS):
                ax = axes[i, j]
                for name, (label, color, ls) in styles.items():
                    if metric == 'nmse':
                        subset = sorted([r for r in nmse['cells'] if r['regime'] == regime and r['rsr'] == ref], key=lambda r: r['snr'])
                        x = [r['snr'] for r in subset]
                        y = np.array([r['methods'][name]['nmse_db'] for r in subset])
                    else:
                        subset = sorted([r for r in ber['cells'] if r['regime'] == regime and r['rsr_db'] == ref and r['method'] == name], key=lambda r: r['snr_db'])
                        x = [r['snr_db'] for r in subset]
                        y = np.array([r['ber'] for r in subset])
                    if metric == 'ber':
                        ax.plot(x, np.where(y > 0, y, np.nan), label=label, color=color, ls=ls, marker='.', lw=1.5)
                        ax.scatter(np.array(x)[y == 0], np.full(np.sum(y == 0), 1e-6), color=color, marker='v', s=13)
                        ax.set_yscale('log')
                        ax.set_ylim(.65e-6, .6)
                    else:
                        ax.plot(x, y, label=label, color=color, ls=ls, marker='.', lw=1.5)
                ax.set_title(f'{regime}, reference {ref:g} dB')
                ax.grid(alpha=.2)
                ax.set_xticks((0, 20, 30, 40, 50, 60))
                if i == 2:
                    ax.set_xlabel('SNR (dB)')
                if j == 0:
                    ax.set_ylabel('NMSE (dB)' if metric == 'nmse' else 'Uncoded BER')
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(.5, .035), ncol=2)
        fig.suptitle('Fresh reliability-check comparison: fixed A-selected proposal\nConfirmation status: '+str(nmse['confirmation']['status']))
        if metric == 'ber':
            fig.text(.5, .01, 'Zero-error triangles at 10⁻⁶ are display markers, not bounds or equivalence evidence.', ha='center', fontsize=9)
        fig.subplots_adjust(left=.08, right=.985, bottom=.14, top=.92, hspace=.28, wspace=.15)
        fig.savefig(OUT / f'reliability_{metric}.png', dpi=180)
        fig.savefig(OUT / f'reliability_{metric}.svg')
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    if args.preflight:
        print(json.dumps(preflight()), flush=True)
    else:
        run()


if __name__ == '__main__':
    main()
