"""Read-only diagnostics of the frozen bypass DEVELOPMENT likelihood table.

No estimator inference, test files, policy changes, or model updates occur.
Bootstrap resamples entire coupled SNR trajectories for each development
anchor, independently within each path regime and reference bin.
"""
import hashlib
import json
import math
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]/'results/tcom_runtime_bypass_20261017/bypass'


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def run():
    policy_path=ROOT/'policy.json';policy=json.loads(policy_path.read_text())
    data=[]
    for name,expected in policy['loss_table_hashes'].items():
        path=ROOT/'development/losses'/name
        if digest(path)!=expected:raise AssertionError(name)
        with np.load(path) as archive:data.append({key:archive[key].copy() for key in archive.files})
    rng=np.random.default_rng(2026101769);repetitions=4000
    ranges={};reviews={};conditions=[];snr_conditions=[]
    for ref in ('5','12','20'):
        cells=[cell for cell in data if float(cell['rsr'])==float(ref)]
        regimes=sorted(set(str(cell['regime']) for cell in cells))
        snrs=sorted(set(float(cell['snr']) for cell in cells))
        cutoffs=cells[0]['cutoffs'];count=cells[0]['losses'].shape[1]
        # Preserve shared anchor/noise/mask coupling over all SNRs.
        bootstrap_ids=[rng.integers(count,size=(repetitions,count)) for _ in regimes]
        entire=np.concatenate([cell['losses'].ravel() for cell in cells])
        ranges[ref]=dict(minimum_NLL=float(entire.min()),maximum_NLL=float(entire.max()),
            all_finite=bool(np.isfinite(entire).all()),float64_ulp_at_largest_NLL=float(np.spacing(entire.max())))
        for kind in ('same_steps','em100'):
            selected=float(policy['policies'][kind][ref]);chosen=int(np.where(cutoffs==selected)[0][0])
            # [regime, SNR, candidate, anchor]; first average the three fixed
            # model losses rather than treating three fits as new worlds.
            values=np.empty((len(regimes),len(snrs),len(cutoffs),count))
            for cell in cells:
                pi=regimes.index(str(cell['regime']));si=snrs.index(float(cell['snr']))
                learned=cell['losses'][:3].mean(axis=0)
                fallback=cell['losses'][3:6].mean(axis=0) if kind=='same_steps' else cell['losses'][6]
                values[pi,si]=np.where(cell['bypass_masks'],fallback[None],learned[None])
                delta=fallback-learned
                conditions.append(dict(reference_db=float(ref),fallback=kind,regime=str(cell['regime']),
                    snr_db=float(cell['snr']),learned_mean_NLL=float(learned.mean()),
                    fallback_mean_NLL=float(fallback.mean()),fallback_minus_learned_mean_NLL=float(delta.mean()),
                    fraction_anchors_fallback_better=float(np.mean(delta<0))))
            for snr in snrs:
                relevant=[row for row in conditions if row['reference_db']==float(ref) and row['fallback']==kind and row['snr_db']==snr]
                snr_conditions.append(dict(reference_db=float(ref),fallback=kind,snr_db=snr,
                    fallback_minus_learned_mean_NLL=float(np.mean([r['fallback_minus_learned_mean_NLL'] for r in relevant]))))
            scores=values.mean(axis=(0,1,3))
            compensated_scores=np.array([math.fsum(values[:,:,ci,:].ravel())/values[:,:,ci,:].size
                for ci in range(len(cutoffs))])
            # Removing the selected candidate per anchor eliminates all common
            # high-SNR contributions before any summation.
            centered=values-values[:,:,chosen:chosen+1,:]
            margins=centered.mean(axis=(0,1,3))
            reduced=centered.mean(axis=1)
            bootstrap=np.zeros((repetitions,len(cutoffs)))
            for pi,ids in enumerate(bootstrap_ids):
                bootstrap+=reduced[pi][:,ids].mean(axis=-1).T/len(regimes)
            winners=len(cutoffs)-1-np.argmin(bootstrap[:,::-1],axis=1)
            saved_rows=policy['validation_tables'][kind][ref]
            saved=np.array([next(r['mean_nll'] for r in saved_rows if r['cutoff_db']==c) for c in cutoffs])
            neighboring=[]
            for ci in (chosen-1,chosen+1):
                if not 0<=ci<len(cutoffs):continue
                neighboring.append(dict(cutoff_db=float(cutoffs[ci]),mean_NLL_above_selected=float(margins[ci]),
                    paired_anchor_bootstrap_95_CI=np.quantile(bootstrap[:,ci],[.025,.975]).tolist(),
                    probability_neighbor_worse=float(np.mean(bootstrap[:,ci]>0))))
            minimum_positive=min(float(x) for x in margins if x>0)
            numerical=max(float(np.max(np.abs(scores-saved))),float(np.max(np.abs(scores-compensated_scores))),
                          float(np.max(np.abs((scores-scores[chosen])-margins))))
            reviews[kind+'_'+ref]=dict(selected_cutoff_db=selected,selected_mean_NLL=float(scores[chosen]),
                maximum_saved_score_reproduction_error=float(np.max(np.abs(scores-saved))),
                maximum_float64_vs_compensated_sum_error=float(np.max(np.abs(scores-compensated_scores))),
                maximum_direct_vs_centered_margin_error=float(np.max(np.abs((scores-scores[chosen])-margins))),
                smallest_positive_candidate_margin=minimum_positive,
                numerical_error_to_smallest_margin_ratio=numerical/minimum_positive,
                neighboring_cutoffs=neighboring,
                all_candidate_margins=[dict(cutoff_db=float(c),mean_NLL_above_selected=float(margins[i])) for i,c in enumerate(cutoffs)],
                bootstrap_winner_frequencies={f'{c:g}':float(np.mean(winners==i)) for i,c in enumerate(cutoffs)})
    return dict(policy_sha256=digest(policy_path),policies=policy['policies'],development_only=True,
        frozen_loss_hashes_verified=True,bootstrap_repetitions=repetitions,bootstrap_seed=2026101769,
        summation_check='Compare ordinary float64 means, math.fsum compensated sums, and differences centered per anchor before summation. NumPy longdouble offers no additional precision on this host and is not used as an independent precision check.',
        bootstrap_scope='Fixed models and losses; resample 24 development anchors within each path regime/reference, preserving each complete SNR trajectory. This estimates finite-development-bank sensitivity, not deployment generalization or model-training variability.',
        likelihood_implementation='Existing sahure.math.nll evaluates (z-|mu|)^2/v - log(i0e(2*z*|mu|/v)) in its regular branch. This avoids subtracting large z^2+|mu|^2 and 2*z*|mu| terms. Data-only constants are shared between candidates.',
        interpretation='Positive fallback-minus-learned NLL favors retaining filtering. Positive candidate margin favors the frozen selected cutoff. High-SNR constants shared by nearby cutoff policies cancel in centered comparisons.',
        loss_ranges=ranges,reviews=reviews,reference_SNR_branch_differences=snr_conditions,
        path_reference_SNR_branch_differences=conditions)


if __name__=='__main__':
    result=run()
    (ROOT/'validation_diagnostic.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(policies=result['policies'],loss_ranges=result['loss_ranges'],
        reviews={key:{field:value[field] for field in ('selected_cutoff_db','numerical_error_to_smallest_margin_ratio',
            'neighboring_cutoffs','bootstrap_winner_frequencies')} for key,value in result['reviews'].items()}),indent=2))
