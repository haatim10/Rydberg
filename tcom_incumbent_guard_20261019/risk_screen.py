"""Exploratory observable channel-risk comparison on reused development banks.

The original likelihood/blend screen remains immutable. This separate screen
uses the same context-only candidates and independent-column heldout data, but
the reused banks' full-pilot rank-rejection mechanism prevents an exact claim
of unconditional heldout independence. Fresh confirmation must generate the
heldout Gaussian pilots independently after accepting the fitting context.
"""
import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_key] = '1'
import json
import time
import numpy as np
import torch
from sahure.math import nll
from tcom_pilot_em_20261018 import continuation
from tcom_followups_20261017.common import REPO, sha, write_json
from . import exploratory as screen
from . import risk

parent = screen.parent
OUT = REPO / 'results/tcom_incumbent_guard_20261019/risk_screen'
REFERENCES = ('original', 'old_bypass', 'prior_repair', 'current_repair')


def summarize(scores, refs, weights):
    # [fit,candidate,incumbent,regime,anchor]; negative favors the candidate.
    records = []
    for mi, name in enumerate(screen.NAMES):
        by_reference = {}; failed = []; failed_mean = []
        for ri, reference in enumerate(REFERENCES):
            delta = scores[:, mi, ri].mean(0)
            by_regime = {}
            for pi, (regime, _) in enumerate(parent.REGIMES):
                boot = weights[regime] @ delta[pi] / parent.DEV_COUNT
                low, high = np.quantile(boot, [.025, .975])
                upper = float(np.quantile(boot, .95)); mean = float(delta[pi].mean())
                by_regime[regime] = dict(mean_delta=mean, upper95=upper,
                    ci95=[float(low), float(high)], sample_sd=float(np.std(delta[pi], ddof=1)))
                if upper > 0.: failed.append(f'{reference}:{regime}')
                if mean > 0.: failed_mean.append(f'{reference}:{regime}')
            by_reference[reference] = dict(candidate=refs[reference],
                mean_delta=float(delta.mean()), by_regime=by_regime)
        records.append(dict(candidate=name, config=screen.CONFIGS[name],
            objective_mean_risk_vs_current=float(scores[:, mi, REFERENCES.index('current_repair')].mean()),
            references=by_reference, all_incumbent_upper95_admissible=not failed,
            all_incumbent_mean_admissible=not failed_mean,
            failed_constraints=failed, failed_mean_constraints=failed_mean))
    return records


@torch.no_grad()
def run():
    started = time.time()
    assert (screen.OUT / 'summary.json').exists(), 'complete original NLL screen first'
    assert not (OUT / 'summary.json').exists(), 'completed risk screen already exists'
    first_screen = json.loads((screen.OUT / 'summary.json').read_text())
    assert first_screen['provenance']['source_sha256'] == sha(screen.__file__)
    checks_path = REPO / 'results/tcom_incumbent_guard_20261019/risk_checks.json'
    checks = json.loads(checks_path.read_text())
    assert checks['passed'] and checks['risk_source_sha256'] == sha(risk.__file__)
    digest = parent.verify_protocol(); selection = parent.check_repair()
    current = json.loads((parent.OUT / 'policy.json').read_text())
    previous = parent.PRIOR_REPAIR
    weights_path = parent.OUT / 'development/repair_bootstrap_weights.npz'
    weights = parent.load(weights_path)
    assert sha(weights_path) == selection['bootstrap_weights_sha256']
    banks = {f'rsr{r:g}_snr{s:g}.npz':parent.OUT / 'development/banks' / f'rsr{r:g}_snr{s:g}.npz'
        for r in screen.RSRS for s in screen.SNRS}
    for name, path in banks.items():assert sha(path) == selection['development_bank_hashes'][name]
    provenance = dict(purpose='Exploratory observable raw-channel-MSE risk screen, not selection or a confirmatory test.',
        source_sha256=sha(__file__), risk_source_sha256=sha(risk.__file__),
        risk_checks_sha256=sha(checks_path),
        candidate_screen_source_sha256=sha(screen.__file__),
        likelihood_screen_summary_sha256=sha(screen.OUT / 'summary.json'),
        parent_protocol_sha256=digest, current_policy_sha256=sha(parent.OUT / 'policy.json'),
        prior_policy_sha256=sha(parent.PRIOR_REPAIR_PATH),
        bank_hashes={name:sha(path) for name,path in banks.items()},
        bootstrap_weights_sha256=sha(weights_path), configurations=screen.CONFIGS,
        names=screen.NAMES, reference_order=REFERENCES, model_seeds=parent.SEEDS,
        snrs=screen.SNRS, rsrs=screen.RSRS, cases_per_regime=parent.DEV_COUNT,
        regimes=parent.REGIMES, context_pilots=30, heldout_pilots=9,
        score='Negative favors candidate. -2/(P*N*K) Re sum conj((Ga-Gb)S) * (Z^2-|B+(Ga+Gb)S/2|^2-sigma2) / conj(B).',
        scale='Unnormalized coefficient Frobenius squared-error difference divided only by known N*K=96; no true-channel energy or NMSE normalization.',
        assumptions='The population identity requires independent proper Gaussian heldout pilot columns with identity covariance, independent proper complex noise, and nonzero known B.',
        caveat='These reused exploratory banks were accepted with a full39-pilot rank check; formal heldout independence is not claimed. Fresh confirmation must generate nine independent heldout pilots after fitting-context acceptance.',
        candidate_scope='All candidates use context30 only. No heldout score, true channel, or path-count input enters any prediction.',
        objective='Mean pairwise risk versus fixed current correction after strict four-incumbent, each-regime upper95 guards; no0.001 band. Pairwise finite scores need not be transitive.',
        bootstrap='Exact existing4000 multinomial384 count vectors per regime shared over every bin/candidate/incumbent.',
        storage='Every per-fit/per-anchor candidate-versus-four-incumbent score retained. Mean/CI/failed constraints included for every candidate.',
        started_unix=started)
    write_json(OUT / 'provenance.json', provenance)
    est = parent.engine(); tables = {}; hashes = {}; parity = {}; feasibility = {}
    for rsr in screen.RSRS:
        tables[f'{rsr:g}'] = {}; feasibility[f'{rsr:g}'] = {}
        for snr in screen.SNRS:
            stem = f'rsr{rsr:g}_snr{snr:g}'; c,h = screen.read_bank(banks[stem+'.npz'])
            assert h.z.shape[1] == 32 and h.s.shape[1] == 3 and h.z.shape[-1] == 9
            spectral = est.em100(c); raw = []; hold_loss = []
            refs = dict(original='original', old_bypass='spectral100',
                prior_repair=previous['repair_choices'][f'{rsr:g}'][f'{snr:g}']['choice'],
                current_repair=current['repair_choices'][f'{rsr:g}'][f'{snr:g}']['choice'])
            for seed in parent.SEEDS:
                initial = est.original(c, f'learned_{seed}')
                trajectory,_ = continuation.trajectory(c,initial,screen.LENGTHS)
                values = {('original' if length==0 else f'warm{length}'):trajectory[length] for length in screen.LENGTHS}
                values['spectral100'] = spectral
                for name,cfg in screen.CONFIGS.items():
                    if cfg['kind']=='convex_channel_blend':
                        alpha=cfg['weight_second']
                        values[name]=(1-alpha)*values[cfg['left']]+alpha*values[cfg['right']]
                scores=np.stack([np.stack([risk.paired_risk(h,values[name],values[refs[reference]],normalize=True)
                    for reference in REFERENCES]) for name in screen.NAMES])
                assert np.isfinite(scores).all()
                for ri,reference in enumerate(REFERENCES):
                    identity=scores[screen.NAMES.index(refs[reference]),ri]
                    np.testing.assert_array_equal(identity,np.zeros_like(identity))
                raw.append(scores.reshape(len(screen.NAMES),len(REFERENCES),3,parent.DEV_COUNT))
                hold=torch.stack([nll(h.z,values[name]@h.s+h.b,h.sigma2).mean((-2,-1)) for name in screen.NAMES])
                hold_loss.append(hold.numpy().reshape(len(screen.NAMES),3,parent.DEV_COUNT))
            scores=np.stack(raw);holds=np.stack(hold_loss)
            old=parent.load(screen.OUT/(stem+'.npz'))
            error=float(np.max(abs(holds-old['heldout_nll'])));assert error<1e-10,(stem,error)
            parity[stem]=error
            path=OUT/(stem+'.npz')
            parent.save(path,names=np.array(screen.NAMES),reference_names=np.array(REFERENCES),
                reference_candidates=np.array([refs[r] for r in REFERENCES]),seeds=np.array(parent.SEEDS),
                regimes=np.array([r for r,L in parent.REGIMES]),paired_risk=scores,heldout_nll=holds,
                snr=np.array(snr),rsr=np.array(rsr),cases_per_regime=np.array(parent.DEV_COUNT),
                scale_divisor=np.array(96),parent_protocol_sha256=np.array(digest),bank_sha256=np.array(sha(banks[stem+'.npz'])))
            hashes[path.name]=sha(path);records=summarize(scores,refs,weights);tables[f'{rsr:g}'][f'{snr:g}']=records
            strict=[r for r in records if r['all_incumbent_upper95_admissible']]
            mean=[r['candidate'] for r in records if r['all_incumbent_mean_admissible']]
            best=min(strict,key=lambda x:(x['objective_mean_risk_vs_current'],screen.NAMES.index(x['candidate']))) if strict else None
            feasibility[f'{rsr:g}'][f'{snr:g}']=dict(strict=[r['candidate'] for r in strict],mean=mean,
                minimum_objective_feasible_candidate=None if best is None else best['candidate'])
            print('Risk screen',stem,'strict feasible',[r['candidate'] for r in strict],'mean feasible',mean,
                'best',None if best is None else best['candidate'],flush=True)
            write_json(OUT/'status.json',dict(completed_bins=len(hashes),last_bin=stem,elapsed_seconds=time.time()-started))
    write_json(OUT/'summary.json',dict(provenance=provenance,tables=tables,feasibility=feasibility,
        score_hashes=hashes,likelihood_reproduction_errors=parity,maximum_parity_error=max(parity.values()),
        elapsed_seconds=time.time()-started,completed_unix=time.time()))
    print('Risk screen complete',time.time()-started,flush=True)


if __name__=='__main__':
    torch.set_num_threads(1);torch.set_num_interop_threads(1);run()
