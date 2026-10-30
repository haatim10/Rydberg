"""Independent moment recalculation, saved-output audit and plain-language report."""
import json
from pathlib import Path
import time
import numpy as np
from codex_verification.solver import fit,geometry
from paper2_optshrink_loop.run import load,write
from .core import STAGES,STARTS,MODELS,realify,predictions,jacobians
from .run import ROOT,SPECS,CONFIG,inputs,initialize,sha


def moments(samples,truth):
    error=realify(samples-truth)
    mean=error.mean(0);centered=error-mean
    cov=np.einsum('rni,rnj->nij',centered,centered)/len(error)
    mse=float(np.mean(np.sum(error**2,axis=(-2,-1))))
    variance=float(np.trace(cov,axis1=-2,axis2=-1).sum())
    bias=float(np.sum(mean**2))
    np.testing.assert_allclose(mse,variance+bias,rtol=1e-13,atol=1e-14)
    # Independent complex-coordinate calculation of all total-error terms.
    ee=samples-truth
    np.testing.assert_allclose(mse,np.mean(np.sum(abs(ee)**2,axis=(-2,-1))),rtol=1e-13,atol=1e-14)
    return dict(cov=cov,mse=mse,variance=variance,bias=bias,
        bias_corrected=max(0.,bias-variance/(len(error)-1)),mean_error=mean)


def scores(emp,pred,count):
    cp=np.asarray(pred);ce=emp['cov'];diff=cp-ce
    trace=float(np.trace(cp,axis1=-2,axis2=-1).sum())
    frob=float(np.sum(diff**2));denom=float(np.sum(ce**2))
    hankel_trace=float(np.dot(count,np.trace(cp,axis1=-2,axis2=-1)))
    hankel_emp=float(np.dot(count,np.trace(ce,axis1=-2,axis2=-1)))
    hnum=float(np.sum(count[:,None,None]**2*diff**2))
    hden=float(np.sum(count[:,None,None]**2*ce**2))
    return dict(predicted_trace=trace,empirical_trace=emp['variance'],mse=emp['mse'],
        bias_power=emp['bias'],bias_fraction=emp['bias']/emp['mse'],
        corrected_bias_fraction=emp['bias_corrected']/emp['mse'],
        trace_ratio=trace/emp['variance'],total_error_ratio=trace/emp['mse'],
        relative_covariance_error=float(np.sqrt(frob/denom)),frob_numerator=frob,frob_denominator=denom,
        hankel_predicted_trace=hankel_trace,hankel_empirical_trace=hankel_emp,
        hankel_trace_ratio=hankel_trace/hankel_emp,hankel_frob_numerator=hnum,hankel_frob_denominator=hden,
        hankel_relative_covariance_error=float(np.sqrt(hnum/hden)))


def aggregate(rows):
    def total(key):return float(sum(r[key] for r in rows))
    return dict(trace_ratio=total('predicted_trace')/total('empirical_trace'),
        relative_covariance_error=float(np.sqrt(total('frob_numerator')/total('frob_denominator'))),
        total_error_ratio=total('predicted_trace')/total('mse'),
        bias_fraction=total('bias_power')/total('mse'),
        hankel_trace_ratio=total('hankel_predicted_trace')/total('hankel_empirical_trace'),
        hankel_relative_covariance_error=float(np.sqrt(total('hankel_frob_numerator')/total('hankel_frob_denominator'))),
        per_world_trace_ratio=[r['trace_ratio'] for r in rows],
        mean_trace_predicted=total('predicted_trace')/len(rows),
        mean_trace_empirical=total('empirical_trace')/len(rows))


def collect():
    per_world={};groups={};bank={};seen=[]
    for j,spec in enumerate(SPECS):
        cases=[]
        for i in range(4):
            p=ROOT/'worlds'/spec['id']/f'{i:02d}.npz';meta=json.loads(p.with_suffix('.json').read_text())
            if sha(p)!=meta['sha256'] or meta['manifest_sha256']!=sha(ROOT/'manifest.json'):
                raise RuntimeError('data/source record mismatch')
            row=load(p);cases.append(row);seen.append(str(row['world_id']))
            if list(row['starts'])!=list(STARTS) or list(row['models'])!=list(MODELS):raise RuntimeError('order mismatch')
            _,_,count=geometry(len(row['g']))
            record={}
            for si,start in enumerate(STARTS):
                record[start]={}
                for ti,t in enumerate(STAGES):
                    emp=moments(row['estimates'][si,ti],row['g'])
                    record[start][str(t)]={m:scores(emp,row['predicted_covariances'][si,mi,ti],count)
                                            for mi,m in enumerate(MODELS)}
            per_world[str(row['world_id'])]=record
        bank[spec['id']]=cases;groups[spec['id']]={}
        for start in STARTS:
            groups[spec['id']][start]={}
            for t in STAGES:
                groups[spec['id']][start][str(t)]={m:aggregate([
                    per_world[str(r['world_id'])][start][str(t)][m] for r in cases]) for m in MODELS}
    assert len(seen)==len(set(seen))==24
    return per_world,groups,bank


def audit(bank):
    tick=time.perf_counter();initialize();checks=[];stepchecks=[];worst_anchor_checks=[]
    for j,spec in enumerate(SPECS):
        for i,row in enumerate(bank[spec['id']]):
            w,z,za=inputs(j,i)
            for a,b in [(w.G,row['g']),(w.S,row['s']),(w.B,row['b']),
                        (np.array(w.sigma2),row['variance']),(z,row['z_repeats']),(za,row['z_anchors'])]:
                np.testing.assert_array_equal(a,b)
            # Existing independent scalar-world solver against stored batched EM.
            for ti,t in enumerate(STAGES):
                other=fit(row['s'],row['z_repeats'][0],row['b'],float(row['variance']),iterations=t)[0]
                expected=row['estimates'][0,ti,0]
                np.testing.assert_allclose(other,expected,rtol=1e-9,atol=1e-10)
            # Probe the most sensitive anchor, not just an easy first anchor.
            # This selection only chooses an audit target; no estimator is retuned.
            for si,mode in enumerate(STARTS):
                anchor=int(row['anchor_prediction_traces'][si,2,3].argmax())
                chosen=za[anchor:anchor+1];oracle=row['g'] if mode=='oracle' else None
                base=jacobians(chosen,w.S,w.B,w.sigma2,mode,oracle)
                half=jacobians(chosen,w.S,w.B,w.sigma2,mode,oracle,step=5e-5)
                diffs={str(t):float(np.linalg.norm(half[t]-base[t])/max(np.linalg.norm(half[t]),1e-30)) for t in STAGES}
                worst_anchor_checks.append(dict(cell=spec['id'],world=i,start=mode,anchor=anchor,relative_changes=diffs))
            if i!=0 or spec['id'] not in ('snr_-10','snr_+5','nearly_noiseless'):continue
            for si,mode in enumerate(STARTS):
                oracle=row['g'] if mode=='oracle' else None
                pred,pp,jj=predictions(za,w.S,w.B,w.sigma2,mode,oracle)
                expected=row['predicted_covariances'][si]
                value=np.stack([pred[t].mean(1) for t in STAGES],axis=1)
                np.testing.assert_array_equal(value,expected)
                for ti,t in enumerate(STAGES):np.testing.assert_array_equal(pp[t][1],row['anchor_estimates'][si,ti])
                half=jacobians(za[:1],w.S,w.B,w.sigma2,mode,oracle,step=5e-5)
                diffs={str(t):float(np.linalg.norm(half[t]-jj[t][:1])/max(np.linalg.norm(half[t]),1e-30)) for t in STAGES}
                stepchecks.append(dict(cell=spec['id'],start=mode,relative_changes=diffs))
                checks.append(dict(cell=spec['id'],start=mode,covariance_replay_max_abs=float(abs(value-expected).max())))
    stable=all(v<.001 for row in stepchecks+worst_anchor_checks for v in row['relative_changes'].values())
    return dict(status='passed' if stable else 'derivative_step_requires_review',
        source_hashes_verified=True,all_24_worlds_and_6336_observations_replayed=True,
        scalar_solver_replays=96,covariance_replays=checks,step_halving=stepchecks,
        largest_prediction_anchor_step_halving=worst_anchor_checks,
        step_check_threshold=.001,step_check_passed=stable,seconds=time.perf_counter()-tick,
        analysis_source_sha256=sha(__file__))


def screen(groups,checked):
    rows=[]
    for spec in SPECS[:5]:
        for t in (20,100):
            m=groups[spec['id']]['spectral'][str(t)]
            f=m['full_sensitivity'];iso=m['isotropic']
            reduction=(iso['relative_covariance_error']-f['relative_covariance_error'])/iso['relative_covariance_error']
            rows.append(dict(cell=spec['id'],stage=t,trace_ratio=f['trace_ratio'],
                relative_covariance_error=f['relative_covariance_error'],isotropic_error_reduction=reduction,
                total_error_ratio=f['total_error_ratio'],bias_fraction=f['bias_fraction'],
                covariance_pass=.8<=f['trace_ratio']<=1.25 and f['relative_covariance_error']<=.35,
                total_error_pass=.5<=f['total_error_ratio']<=2 and f['bias_fraction']<=.25))
    count=sum(r['covariance_pass'] for r in rows);direct=sum(r['total_error_pass'] for r in rows)
    median=float(np.median([r['isotropic_error_reduction'] for r in rows]))
    shape=count>=8 and median>=.1 and checked['step_check_passed']
    total=shape and direct>=8
    decision=('covariance_and_total_error_screen_pass_review_before_filter_design' if total else
              'covariance_only_pass_bias_or_total_error_blocks_direct_use' if shape else
              'stop_expansion_of_current_error_model')
    return dict(decision=decision,groups=rows,covariance_groups_passed=count,total_error_groups_passed=direct,
        median_geometry_improvement=median,covariance_gate_passed=shape,direct_use_gate_passed=total)


def figure(groups):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    specs=SPECS[:5];labels=['−10 dB','5 dB','20 dB','Weak reference','16 paths']
    fig,axes=plt.subplots(1,2,figsize=(10.5,4.6),layout='constrained')
    ax=axes[0];x=np.arange(5)
    for method,label,color in [('missing_phase','Missing phase','#b65749'),
                                ('frozen_update','Frozen update','#c68e20'),
                                ('full_sensitivity','Full estimator','#2568a8')]:
        values=[groups[s['id']]['spectral']['100'][method]['trace_ratio'] for s in specs]
        ax.plot(x,values,'o-',label=label,color=color)
    ax.axhspan(.8,1.25,color='#cee9d1',alpha=.8,label='Screening range')
    ax.axhline(1,color='black',lw=.8);ax.set_yscale('log')
    ax.set_ylabel('Predicted / observed random-error power')
    ax.set_xticks(x,labels,rotation=25,ha='right');ax.grid(axis='y',alpha=.2)
    ax.set_title('Covariance size after 100 updates');ax.legend(fontsize=8)
    ax=axes[1]
    for t,offset,color in [(20,-.18,'#78a3cd'),(100,.18,'#2568a8')]:
        values=[100*groups[s['id']]['spectral'][str(t)]['full_sensitivity']['bias_fraction'] for s in specs]
        ax.bar(x+offset,values,width=.36,label=f'{t} updates',color=color)
    ax.axhline(25,color='#b65749',ls='--',label='Direct-use screening limit')
    ax.set_ylim(0,100);ax.set_ylabel('Estimated bias as % of total error')
    ax.set_xticks(x,labels,rotation=25,ha='right');ax.set_title('What covariance alone misses')
    ax.legend(fontsize=8);ax.grid(axis='y',alpha=.2)
    fig.suptitle('Error-model development check · 4 fixed worlds per condition · spectral start',fontsize=11)
    fig.savefig(ROOT/'error_model_check.png',dpi=180);plt.close(fig)


def report(summary):
    gate=summary['screen'];g=summary['groups']
    lines=['HANKEL INPUT ERROR: DEVELOPMENT RESULTS','',
        f'Decision: {gate["decision"]}',
        f'Numerical audit: {summary["audit_status"]}. See audit.json; flagged derivatives are not silently accepted.',
        '24 independent fixed channel/pilot/reference worlds; 256 repeated noise draws',
        'and 8 separate prediction anchors per world. No network training or new filter.',
        'Known reference/noise, 32 antennas, 3 users, 30 pilots. Manuscript unchanged.','',
        'Main check: full-estimator sensitivity, ordinary spectral initialization.',
        f'Covariance size/shape passed in {gate["covariance_groups_passed"]}/10 primary groups (required 8).',
        f'Median geometry improvement over equal-trace isotropic control: {100*gate["median_geometry_improvement"]:.2f}% (required 10%).',
        f'Total-error/bias requirement passed in {gate["total_error_groups_passed"]}/10 groups (required 8).',
        'Bias fractions use the repeated-noise sample mean; finite-sample-corrected',
        'values are also preserved in per-world results. They are not exact population bias.','',
        'Cell | updates | predicted/observed variance | covariance relative error | bias % | predicted/total MSE']
    for row in gate['groups']:
        lines.append(f'{row["cell"]} | {row["stage"]} | {row["trace_ratio"]:.3f} | {row["relative_covariance_error"]:.3f} | {100*row["bias_fraction"]:.1f} | {row["total_error_ratio"]:.3f}')
    lines+=['','Separately labelled diagnostics',
        'At one update with a true-channel start, frozen-update covariance has an',
        'exact matched-field interpretation. It is not available as an oracle at a receiver.',
        'Cell | oracle one-step variance ratio | oracle one-step covariance error']
    for spec in SPECS:
        d=g[spec['id']]['oracle']['1']['frozen_update']
        lines.append(f'{spec["id"]} | {d["trace_ratio"]:.3f} | {d["relative_covariance_error"]:.3f}')
    lines+=['','Cell | zero-start full-map variance ratio at100 | bias % | SNR60 kept separate']
    for spec in SPECS:
        d=g[spec['id']]['zero']['100']['full_sensitivity']
        lines.append(f'{spec["id"]} | {d["trace_ratio"]:.3f} | {100*d["bias_fraction"]:.1f}')
    lines+=['','Interpretation',
        'Missing phase and estimation error are different random quantities.',
        'The verified total-variance identity separates them; equal power does',
        'not imply equal directions. Treating one as the other is unjustified.',
        'The full sensitivity approximation includes repeated use of the same',
        'observations. It is still local and does not predict systematic bias.',
        'Predicting covariance is not predicting all channel error or proving',
        'Gaussian errors, confidence coverage, improved NMSE, or safe filtering.',
        'Hankel entries share samples. Lifting a 32-sample complex sequence to',
        '17x16 duplicates information: covariance in the 272-complex-entry space',
        'has rank at most 32 complex dimensions, not 272 independent entries.',
        'This study examines unfiltered row-separable EM only. Hankel interleaving',
        'couples rows and needs its own error-model check; no transfer is assumed.',
        'Four fixed worlds per condition is a screen, not population confirmation.',
        'Noise repetitions are not independent channel/pilot replicates.','',
        'Files: summary.json for all starts/stages/models and per-world outcomes;',
        'audit.json for replays and numerical checks; error_model_check.png for the',
        'two main diagnostics; THEORY.txt and PROTOCOL.txt in hankel_error_check/.',
        'No novelty, convergence, hardware or improvement claim is established.']
    if summary.get('derivative_review_sha256'):
        review=json.loads((ROOT/'derivative_review.json').read_text())
        lines+=['','Post-screen numerical diagnosis (does not change frozen predictions)',
            'Two supplementary checks exceeded the original derivative-step tolerance.',
            'They concern the oracle start at -10 dB and zero start with a weak reference.',
            'Independent automatic differentiation checks smaller finite-difference steps.',
            'The largest ordinary-start prediction in each failing condition is also checked.',
            'Cell | start | original-step error vs autograd | smaller-step error vs autograd']
        for item in review['targets']:
            target=item['target'];first=item['steps'][0];last=item['steps'][-1]
            lines.append(f'{target["cell"]} | {target["start"]} | {first["relative_error_to_autograd"]:.7f} | {last["relative_error_to_autograd"]:.7f}')
        lines+=['These are relative Jacobian errors, not channel NMSE.',
            'The primary failure is not explained by these small derivative errors.',
            'The original audit flag and failed scientific screen remain recorded.',
            'See derivative_review.json for exact targets, steps, and source hash.']
    (ROOT/'RESULTS.txt').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))


def main():
    if json.loads((ROOT/'status.json').read_text())['state']!='complete':raise RuntimeError('run incomplete')
    worlds,groups,bank=collect()
    checked=audit(bank);write(ROOT/'audit.json',checked)
    result=dict(screen=screen(groups,checked),groups=groups,per_world=worlds,
                number_fixed_worlds=24,noise_repetitions=6144,prediction_anchors=192,
                audit_status=checked['status'],analysis_source_sha256=sha(__file__))
    if (ROOT/'derivative_review.json').exists():
        result['derivative_review_sha256']=sha(ROOT/'derivative_review.json')
    write(ROOT/'summary.json',result);figure(groups);report(result)


if __name__=='__main__':main()
