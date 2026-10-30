"""Exploratory continuation-length screen on prior observable development only.

This is candidate-design evidence, not a new test or a deployed policy.  The
previous scientific sources, models, banks, masks and bootstrap weights remain
unchanged.  Inference receives only the 30 fitting observations.
"""
import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[_key]='1'
import json
from pathlib import Path
import time
import numpy as np
import torch
from sahure.math import nll
from tcom_transition_repair_20261019 import study as prior
from tcom_pilot_em_20261018 import continuation
from tcom_followups_20261017.common import REPO,sha,write_json

OUT=REPO/'results/tcom_dual_guard_20261019/exploratory'
SNRS=(30.,32.5,35.,37.5)
RSRS=(12.,20.)
LENGTHS=(0,1,2,3,5,10,20,100)
NAMES=('original',)+tuple(f'warm{x}' for x in LENGTHS if x)+('spectral100',)


@torch.no_grad()
def run():
    started=time.time();digest=prior.verify_protocol();est=prior.engine()
    parent_selection=prior.check_repair()
    weightpath=prior.OUT/'development/repair_bootstrap_weights.npz'
    weights=prior.load(weightpath)
    assert sha(weightpath)==parent_selection['bootstrap_weights_sha256']
    provenance=dict(
        purpose='Development-only candidate-design screen; no new policy or test claim.',
        source_sha256=sha(__file__),parent_protocol_sha256=digest,
        parent_repair_selection_sha256=sha(prior.OUT/'repair_selection.json'),
        development_anchor_hashes=parent_selection['development_anchor_hashes'],
        bootstrap_weights_sha256=sha(weightpath),
        snrs=SNRS,rsrs=RSRS,names=NAMES,seeds=prior.SEEDS,
        regimes=prior.REGIMES,cases_per_regime=prior.DEV_COUNT,
        context_pilots=30,heldout_pilots=9,
        masks='Unchanged tcom_transition_repair_20261019.fit_holdout masks.',
        inference='Only fitting Observations and frozen learned estimates enter EM. No truth or path-count input.',
        guard='Average three model-fit paired loss deltas within each anchor; existing 4000 multinomial weights independently per regime, shared across conditions and candidates; upper one-sided95 percentile versus original and old bypass.',
        old_bypass='At RSR12/20, original through nominal30dB and spectral100 above30dB.',
        started_unix=started)
    write_json(OUT/'provenance.json',provenance)
    tables={};hashes={}
    for rsr in RSRS:
        tables[f'{rsr:g}']={}
        for snr in SNRS:
            c,h=prior.development(rsr,snr,digest)
            spectral=est.em100(c);fit_losses=[];heldout_losses=[]
            for seed in prior.SEEDS:
                initial=est.original(c,f'learned_{seed}')
                trajectory,_=continuation.trajectory(c,initial,LENGTHS)
                values=[trajectory[length] for length in LENGTHS]+[spectral]
                fit=torch.stack([nll(c.z,g@c.s+c.b,c.sigma2).mean((-2,-1)) for g in values])
                hold=torch.stack([nll(h.z,g@h.s+h.b,h.sigma2).mean((-2,-1)) for g in values])
                assert torch.isfinite(fit).all() and torch.isfinite(hold).all()
                assert torch.all(fit[1:len(LENGTHS)]<=fit[0]+1e-10*(1+fit[0].abs()))
                fit_losses.append(fit.numpy().reshape(len(NAMES),3,prior.DEV_COUNT))
                heldout_losses.append(hold.numpy().reshape(len(NAMES),3,prior.DEV_COUNT))
            fits=np.stack(fit_losses);holds=np.stack(heldout_losses)
            # Retained menu arms must reproduce the existing development losses.
            old=prior.load(prior.OUT/'development/repair_losses'/f'rsr{rsr:g}_snr{snr:g}.npz')
            for index,name in enumerate(old['names']):
                newindex=NAMES.index(str(name))
                np.testing.assert_allclose(holds[:,newindex],old['heldout_nll'][:,index],rtol=0,atol=1e-10)
                np.testing.assert_allclose(fits[:,newindex],old['fit_nll'][:,index],rtol=0,atol=1e-10)
            path=OUT/f'rsr{rsr:g}_snr{snr:g}.npz'
            prior.save(path,names=np.array(NAMES),seeds=np.array(prior.SEEDS),regimes=np.array([r for r,L in prior.REGIMES]),
                fit_nll=fits,heldout_nll=holds,snr=np.array(snr),rsr=np.array(rsr),parent_protocol_sha256=np.array(digest))
            hashes[path.name]=sha(path)
            oldindex=0 if snr<=30. else NAMES.index('spectral100')
            records=[]
            for index,name in enumerate(NAMES):
                by_reference={}
                for reference,refindex in (('original',0),('old_bypass',oldindex)):
                    delta=(holds[:,index]-holds[:,refindex]).mean(0)
                    by_regime={}
                    for pi,(regime,L) in enumerate(prior.REGIMES):
                        boot=weights[regime]@delta[pi]/prior.DEV_COUNT
                        by_regime[regime]=dict(mean_delta=float(delta[pi].mean()),upper95=float(np.quantile(boot,.95)))
                    by_reference[reference]=dict(mean_delta=float(delta.mean()),by_regime=by_regime,
                        all_regimes_upper95_nonpositive=all(v['upper95']<=0 for v in by_regime.values()))
                records.append(dict(candidate=name,mean_nll=float(holds[:,index].mean()),references=by_reference,
                    dual_guard_admissible=all(v['all_regimes_upper95_nonpositive'] for v in by_reference.values())))
            tables[f'{rsr:g}'][f'{snr:g}']=records
            print('Development screen',f'rsr{rsr:g}_snr{snr:g}',
                'dual-admissible',[r['candidate'] for r in records if r['dual_guard_admissible']],flush=True)
    write_json(OUT/'summary.json',dict(provenance=provenance,tables=tables,loss_hashes=hashes,
        elapsed_seconds=time.time()-started,completed_unix=time.time()))
    print('Development-only screen complete',flush=True)


if __name__=='__main__':
    torch.set_num_threads(1);torch.set_num_interop_threads(1);run()
