"""Recalculate saved estimates, audit pairing, and report predeclared contrasts."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key]='1'
from pathlib import Path
import json
import numpy as np
from codex_verification.solver import initial,ratio,geometry
from paper2_review_checks.optshrink_control import opt_weights
from .run import ROOT,CONFIG,SPECS,load,write,sha,sources,configs_for


def reference_estimate(z,s,b,v,cfg):
    """Slow scalar-weight replay, independent of the new vectorized filter."""
    g=initial(s,z,b)
    sh=s.conj().T;op=np.linalg.solve((s@sh).T,sh.T).T
    idx,gather,count=geometry(g.shape[0])
    for t in range(cfg['iterations']):
        mu=g@s+b
        g=(z*np.exp(1j*np.angle(mu))*ratio(2*z*abs(mu)/v)-b)@op
        active=cfg['placement']=='loop' or t==cfg['iterations']-1
        if active and cfg['strength']:
            channels=[]
            for column in g.T:
                h=column[idx];u,sv,vh=np.linalg.svd(h,full_matrices=False)
                ww=opt_weights(sv,*h.shape,cfg['rank'])
                hh=(u*ww[None,:])@vh
                flat=np.r_[hh.ravel(),0j]
                channels.append(flat[gather].sum(-1)/count)
            filtered=np.stack(channels,axis=1)
            g+=cfg['strength']*(filtered-g)
    return g


def analyze():
    frozen=json.loads((ROOT/'frozen_selection.json').read_text())
    manifest=json.loads((ROOT/'manifest.json').read_text())
    if sources()!=manifest['sources'] or frozen['source_hashes']!=manifest['sources']:
        raise RuntimeError('analysis source hashes do not match frozen protocol')
    for entry in manifest['models'].values():
        if sha(entry['path'])!=entry['sha256']: raise RuntimeError('checkpoint changed')
    for path,digest in frozen['development_hashes'].items():
        if sha(path)!=digest: raise RuntimeError('development artifact changed')
    # Independently rebuild every selection score from the saved case scores.
    from .run import choose
    all_scores=[]
    for spec in SPECS[:7]:
        rows=[load(ROOT/'development'/spec['id']/f'trial_{i:03d}.npz') for i in range(CONFIG['development_cases'])]
        for name in ('loop','final'):
            score=np.mean([r[f'{name}_scores'] for r in rows],axis=0)
            np.testing.assert_allclose(score,np.array(frozen['scores'][name])[CONFIG['snrs'].index(spec['snr'])],rtol=0,atol=0)
            assert choose(score,name)==frozen['noise_aware'][str(spec['snr'])][name]
            if name=='loop': all_scores.append(score)
    assert choose(np.mean(all_scores,axis=0),'loop')==frozen['global_loop']
    rng=np.random.default_rng(CONFIG['bootstrap_seed'])
    cells={};boot={};all_ids=set();replay=[];hashes={}
    for j,spec in enumerate(SPECS):
        directory=ROOT/'test_data'/spec['id']
        meta=json.loads((directory/'manifest.json').read_text())
        assert meta['generated_unix']>=frozen['frozen_unix']
        assert meta['frozen_selection_sha256']==sha(ROOT/'frozen_selection.json')
        assert meta['seed']==CONFIG['test_seed']+100*j
        for p,digest in meta['hashes'].items():
            assert sha(p)==digest
            hashes[p]=digest
        data=load(directory/'observations.npz');truth=load(directory/'truth.npz')
        np.testing.assert_array_equal(data['trial_ids'],truth['trial_ids'])
        energy=np.sum(abs(truth['g'])**2,axis=(-2,-1));errors=[]
        for i in range(CONFIG['test_cases']):
            path=ROOT/'test_outputs'/spec['id']/f'trial_{i:03d}.npz'
            out=load(path);hashes[str(path)]=sha(path)
            identity=str(out['trial_id'])
            assert identity==data['trial_ids'][i] and identity not in all_ids
            all_ids.add(identity)
            assert str(out['selection_sha256'])==sha(ROOT/'frozen_selection.json')
            err=np.sum(abs(out['estimates']-truth['g'][i])**2,axis=(-2,-1))
            np.testing.assert_allclose(err,out['errors'],rtol=1e-13,atol=1e-13)
            np.testing.assert_allclose(out['truth_energy'],energy[i],rtol=1e-13)
            if not np.isfinite(err).all(): raise FloatingPointError('invalid error array')
            errors.append(err)
            if i==0:
                names=out['names'].tolist()
                cfgs=configs_for(float(data['variance'][i]),frozen)
                for method in ('loop_selected','final_selected'):
                    ref=reference_estimate(data['z'][i],data['s'][i],data['b'][i],float(data['variance'][i]),cfgs[method])
                    actual=out['estimates'][names.index(method)]
                    delta=float(np.max(abs(ref-actual)))
                    np.testing.assert_allclose(ref,actual,rtol=1e-8,atol=1e-8)
                    replay.append(dict(cell=spec['id'],method=method,max_absolute_error=delta))
            else: assert names==out['names'].tolist()
        errors=np.array(errors).T
        group={name:[i] for i,name in enumerate(names) if not name.startswith('adaptive10_')}
        group['adaptive10']=[i for i,name in enumerate(names) if name.startswith('adaptive10_')]
        counts=rng.multinomial(len(energy),np.full(len(energy),1/len(energy)),size=CONFIG['bootstrap_samples'])
        methods={};bs={}
        for name,ids in group.items():
            err=errors[ids]
            seed_db=10*np.log10(err.sum(axis=1)/energy.sum())
            samples=10*np.log10((err@counts.T)/(counts@energy)[None,:]).mean(axis=0)
            methods[name]=dict(nmse_db=float(seed_db.mean()),seed_nmse_db=seed_db.tolist(),
                interval=np.quantile(samples,[.025,.975]).tolist(),mean_case_nmse=float((err/energy).mean()),
                severe_fraction=float(np.mean(err/energy>1)),seed_severe_fraction=np.mean(err/energy>1,axis=1).tolist())
            bs[name]=samples
        contrasts={name:dict(learned_gain_db=methods[name]['nmse_db']-methods['adaptive10']['nmse_db'],
            interval=np.quantile(bs[name]-bs['adaptive10'],[.025,.975]).tolist())
            for name in methods if name!='adaptive10'}
        cells[spec['id']]=dict(spec=spec,cases=len(energy),methods=methods,contrasts=contrasts)
        boot[spec['id']]=bs
    main=[s['id'] for s in SPECS[:7]]
    macro={name:dict(nmse_db=float(np.mean([cells[c]['methods'][name]['nmse_db'] for c in main])),
        seed_nmse_db=np.mean([cells[c]['methods'][name]['seed_nmse_db'] for c in main],axis=0).tolist(),
        mean_case_nmse=float(np.mean([cells[c]['methods'][name]['mean_case_nmse'] for c in main])),
        severe_fraction=float(np.mean([cells[c]['methods'][name]['severe_fraction'] for c in main])))
        for name in cells[main[0]]['methods']}
    contrasts={name:dict(learned_gain_db=macro[name]['nmse_db']-macro['adaptive10']['nmse_db'],
        interval=np.quantile(np.mean([boot[c][name]-boot[c]['adaptive10'] for c in main],axis=0),[.025,.975]).tolist())
        for name in macro if name!='adaptive10'}
    timing=json.loads((ROOT/'timing.json').read_text())
    summary=dict(main_macro=macro,contrasts=contrasts,cells=cells,timing=timing,
        primary_contrast=contrasts['loop_selected'],
        uncertainty='paired case bootstrap, conditional on frozen models and selection; no new-training-dataset uncertainty',
        selection=frozen['noise_aware'],global_loop=frozen['global_loop'],
        limitations=['three models share their original training dataset','no retraining or new-dataset confirmation',
            'known noise/reference; main sparse geometric channel distribution',
            'OptShrink adapted to a nonlinear loop, not full LRHD or a transferred optimality guarantee',
            '24-case timing at SNR 5 only, device and implementation specific',
            'stress and nonprimary intervals descriptive, not multiplicity adjusted'])
    write(ROOT/'summary.json',summary)
    write(ROOT/'verification.json',dict(passed=True,test_files=len(all_ids),expected_test_files=9*CONFIG['test_cases'],
        development_files=len(frozen['development_hashes']),unique_case_ids=True,
        errors_recalculated_from_estimates=True,selection_recalculated=True,
        test_generated_after_freeze=True,independent_scalar_replays=replay,
        frozen_sources_unchanged=True,original_checkpoint_hashes_verified=True))
    write(ROOT/'artifact_hashes.json',hashes)
    print(json.dumps(dict(primary=summary['primary_contrast'],macro=macro),indent=2),flush=True)
    return summary

if __name__=='__main__': analyze()
