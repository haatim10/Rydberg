"""Development-only screen of EM checkpoints and convex channel blends.

No fresh policy is selected and no test channels are loaded. All inputs are
immutable observable banks, masks, model fits and bootstrap weights from the
dual-guard study. This screen informs a later independently frozen design.
"""
import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_key] = '1'
import json
from pathlib import Path
import time
import numpy as np
import torch
from sahure.data import Observations
from sahure.math import nll
from tcom_dual_guard_20261019 import study as parent
from tcom_pilot_em_20261018 import continuation
from tcom_followups_20261017.common import REPO, sha, write_json

OUT = REPO / 'results/tcom_incumbent_guard_20261019/exploratory'
RSRS = (12., 20.)
SNRS = (32.5, 35., 37.5)
LENGTHS = (0, 1, 2, 3, 4, 5, 8, 10, 12, 20, 40, 100)
PAIRS = (('original', 'warm3'), ('warm2', 'warm3'), ('warm5', 'warm20'),
         ('warm20', 'spectral100'), ('warm5', 'spectral100'))
ALPHAS = (.25, .5, .75)
CONFIGS = {('original' if length == 0 else f'warm{length}'):
           dict(kind='continuation', steps=length) for length in LENGTHS}
CONFIGS['spectral100'] = dict(kind='spectral', steps=100)
for _left, _right in PAIRS:
    for _alpha in ALPHAS:
        CONFIGS[f'blend_{_left}_{_right}_{_alpha:g}'] = dict(
            kind='convex_channel_blend', left=_left, right=_right,
            weight_second=_alpha)
NAMES = tuple(CONFIGS)


def read_bank(path):
    """Load only persisted fitting and heldout observations; never generate."""
    a = parent.load(path)
    assert not any(key in a for key in ('truth_g', 'g', 'G', 'paths'))
    assert int(a['cases_per_regime']) == parent.DEV_COUNT == 384
    return tuple(Observations(*(torch.from_numpy(a[f'{prefix}_{name}'])
        for name in ('z', 's', 'b', 'sigma2'))) for prefix in ('context', 'heldout'))


def score(holds, references, weights):
    records = []
    for mi, name in enumerate(NAMES):
        by_reference = {}; failed = []; failed_mean = []
        for reference, refname in references.items():
            delta = (holds[:, mi] - holds[:, NAMES.index(refname)]).mean(0)
            by_regime = {}
            for pi, (regime, _) in enumerate(parent.REGIMES):
                boot = weights[regime] @ delta[pi] / parent.DEV_COUNT
                lower, upper_two = np.quantile(boot, [.025, .975])
                upper = float(np.quantile(boot, .95))
                mean = float(delta[pi].mean())
                by_regime[regime] = dict(mean_delta=mean, upper95=upper,
                    ci95=[float(lower), float(upper_two)])
                if upper > 0.: failed.append(f'{reference}:{regime}')
                if mean > 0.: failed_mean.append(f'{reference}:{regime}')
            by_reference[reference] = dict(candidate=refname,
                mean_delta=float(delta.mean()), by_regime=by_regime)
        records.append(dict(candidate=name, config=CONFIGS[name],
            mean_nll=float(holds[:, mi].mean()), references=by_reference,
            all_incumbent_upper95_admissible=not failed,
            all_incumbent_mean_admissible=not failed_mean,
            failed_constraints=failed, failed_mean_constraints=failed_mean))
    return records


@torch.no_grad()
def run():
    started = time.time()
    digest = parent.verify_protocol()
    selection = parent.check_repair()
    current = json.loads((parent.OUT / 'policy.json').read_text())
    previous = parent.PRIOR_REPAIR
    weights_path = parent.OUT / 'development/repair_bootstrap_weights.npz'
    weights = parent.load(weights_path)
    assert sha(weights_path) == selection['bootstrap_weights_sha256']
    banks = {f'rsr{r:g}_snr{s:g}.npz':
        parent.OUT / 'development/banks' / f'rsr{r:g}_snr{s:g}.npz'
        for r in RSRS for s in SNRS}
    for name, path in banks.items():
        assert sha(path) == selection['development_bank_hashes'][name]
    provenance = dict(purpose='Exploratory development-only candidate screen; no deployed policy or confirmatory test.',
        source_sha256=sha(__file__), parent_protocol_sha256=digest,
        parent_repair_selection_sha256=sha(parent.OUT / 'repair_selection.json'),
        current_policy_sha256=sha(parent.OUT / 'policy.json'),
        prior_policy_sha256=sha(parent.PRIOR_REPAIR_PATH),
        bank_hashes={name:sha(path) for name,path in banks.items()},
        checkpoint_hashes={str(path.relative_to(REPO)):sha(path) for path in parent.model_paths().values()},
        bootstrap_weights_sha256=sha(weights_path),
        configurations=CONFIGS, names=NAMES, snrs=SNRS, rsrs=RSRS,
        cases_per_regime=parent.DEV_COUNT, model_seeds=parent.SEEDS,
        regimes=parent.REGIMES, context_pilots=30, heldout_pilots=9,
        input_scope='Exact existing observable banks and masks. No generation or reading of test anchors, channel truth or path counts.',
        inference='One original estimate and one EM trajectory per fit; shared spectral100 across fits. Blends are (1-alpha)*left+alpha*right. No fitting or heldout loss enters a prediction.',
        comparison='Paired per-anchor loss differences averaged over three frozen model fits; all four incumbents (original, old bypass, previous transition repair, current dual guard).',
        bootstrap='Existing4000 multinomial384 count vectors per regime, shared over all bins and candidates. Strict upper95<=0 in each regime for every incumbent; no preference band or tolerance.',
        interpretation='Exploratory reused development evidence; no population guarantee, selection commitment or training-uncertainty claim.',
        thread_environment={key:os.environ[key] for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS')},
        started_unix=started)
    assert not (OUT / 'summary.json').exists(), 'completed exploratory screen already exists'
    write_json(OUT / 'provenance.json', provenance)
    est = parent.engine(); tables = {}; hashes = {}; parity = {}
    for rsr in RSRS:
        tables[f'{rsr:g}'] = {}
        for snr in SNRS:
            stem = f'rsr{rsr:g}_snr{snr:g}'; c, h = read_bank(banks[stem + '.npz'])
            spectral = est.em100(c); fits = []; holds = []
            for seed in parent.SEEDS:
                initial = est.original(c, f'learned_{seed}')
                trajectory, _ = continuation.trajectory(c, initial, LENGTHS)
                values = {('original' if length == 0 else f'warm{length}'):trajectory[length]
                    for length in LENGTHS}
                values['spectral100'] = spectral
                for name, cfg in CONFIGS.items():
                    if cfg['kind'] == 'convex_channel_blend':
                        alpha = cfg['weight_second']
                        values[name] = (1-alpha)*values[cfg['left']] + alpha*values[cfg['right']]
                fit = torch.stack([nll(c.z, values[name] @ c.s + c.b, c.sigma2).mean((-2,-1)) for name in NAMES])
                hold = torch.stack([nll(h.z, values[name] @ h.s + h.b, h.sigma2).mean((-2,-1)) for name in NAMES])
                assert torch.isfinite(fit).all() and torch.isfinite(hold).all()
                assert torch.all(fit[1:len(LENGTHS)] <= fit[0]+1e-10*(1+fit[0].abs()))
                fits.append(fit.numpy().reshape(len(NAMES),3,parent.DEV_COUNT))
                holds.append(hold.numpy().reshape(len(NAMES),3,parent.DEV_COUNT))
            fit = np.stack(fits); hold = np.stack(holds)
            old = parent.load(parent.OUT / 'development/repair_losses' / (stem+'.npz'))
            parity[stem] = {}
            for oi, name in enumerate(old['names']):
                ni = NAMES.index(str(name))
                pe = max(float(np.max(abs(fit[:,ni]-old['fit_nll'][:,oi]))),
                         float(np.max(abs(hold[:,ni]-old['heldout_nll'][:,oi]))))
                assert pe < 1e-10, (stem, name, pe)
                parity[stem][str(name)] = pe
            path = OUT / (stem+'.npz')
            parent.save(path,names=np.array(NAMES),seeds=np.array(parent.SEEDS),
                regimes=np.array([r for r,L in parent.REGIMES]),fit_nll=fit,heldout_nll=hold,
                snr=np.array(snr),rsr=np.array(rsr),cases_per_regime=np.array(parent.DEV_COUNT),
                parent_protocol_sha256=np.array(digest),bank_sha256=np.array(sha(banks[stem+'.npz'])))
            hashes[path.name] = sha(path)
            references = dict(original='original',old_bypass='spectral100',
                prior_repair=previous['repair_choices'][f'{rsr:g}'][f'{snr:g}']['choice'],
                current_repair=current['repair_choices'][f'{rsr:g}'][f'{snr:g}']['choice'])
            records = score(hold,references,weights); tables[f'{rsr:g}'][f'{snr:g}'] = records
            print('Screen',stem,'strict feasible', [r['candidate'] for r in records if r['all_incumbent_upper95_admissible']],
                'mean feasible',[r['candidate'] for r in records if r['all_incumbent_mean_admissible']],flush=True)
            write_json(OUT / 'status.json',dict(completed_bins=len(hashes),last_bin=stem,elapsed_seconds=time.time()-started))
    write_json(OUT / 'summary.json',dict(provenance=provenance,tables=tables,
        loss_hashes=hashes,existing_nine_parity=parity,maximum_parity_error=max(max(v.values()) for v in parity.values()),
        elapsed_seconds=time.time()-started,completed_unix=time.time()))
    print('Development-only screen complete',time.time()-started,flush=True)


if __name__ == '__main__':
    torch.set_num_threads(1);torch.set_num_interop_threads(1);run()
