"""Small prior-development-only likelihood screen, before new protocol choice.

No channel truth, prior test arrays, NMSE or BER is loaded by this program.
The output is exploratory and cannot serve as the fresh study's test evidence.
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key]='1'
import time
from pathlib import Path
import numpy as np
import torch
from sahure.data import Observations
from sahure.math import nll
from tcom_retrained_bypass_20261018 import study as old
from tcom_pilot_em_20261018.continuation import trajectory
from tcom_followups_20261017.common import REPO,sha,write_json

OUT=REPO/'results/tcom_transition_repair_20261019/exploratory'
COUNT=16
CONDITIONS=((5.,35.),(5.,37.5),(5.,40.),(5.,45.),(5.,60.),(12.,32.5),(12.,60.))
METHODS=('original','warm5','warm20','warm100','spectral100','dual100')


@torch.no_grad()
def main():
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    est=old.engine();records=[];hashes={};source=old.OUT/'development'/'anchors'
    started=time.perf_counter()
    for rsr,snr in CONDITIONS:
        contexts=[];holdouts=[]
        for regime,L in old.REGIMES:
            path=source/f'{regime}_rsr{rsr:g}.npz';a=old.load(path);hashes[str(path.relative_to(REPO))]=sha(path)
            assert not {'g','truth_g','paths','unit_noise','mu'}.intersection(a)
            # Construct the original48-case split first, then take first16 cases.
            c,h=old.fit_holdout(old.observed(a,snr,P=39),regime,rsr)
            contexts.append(c.batch(slice(0,COUNT)));holdouts.append(h.batch(slice(0,COUNT)))
        c=Observations(*(torch.cat([p.args()[j] for p in contexts]) for j in range(4)))
        h=Observations(*(torch.cat([p.args()[j] for p in holdouts]) for j in range(4)))
        spectral=est.em100(c)
        fit_spec=nll(c.z,spectral@c.s+c.b,c.sigma2).mean((-2,-1))
        fit_losses=[];hold_losses=[];switches=[]
        for seed in old.SEEDS:
            initial=est.original(c,f'learned_{seed}')
            warm,_=trajectory(c,initial,(0,5,20,100),audit=False)
            fit_warm=nll(c.z,warm[100]@c.s+c.b,c.sigma2).mean((-2,-1))
            choose_spectral=fit_spec<fit_warm
            dual=torch.where(choose_spectral[:,None,None],spectral,warm[100])
            predictions=[initial,warm[5],warm[20],warm[100],spectral,dual]
            fit=torch.stack([nll(c.z,g@c.s+c.b,c.sigma2).mean((-2,-1)) for g in predictions])
            hold=torch.stack([nll(h.z,g@h.s+h.b,h.sigma2).mean((-2,-1)) for g in predictions])
            assert torch.isfinite(fit).all() and torch.isfinite(hold).all()
            assert torch.all(fit[1:4]<=fit[0]+1e-10*(1+fit[0].abs()))
            fit_losses.append(fit.numpy().reshape(len(METHODS),3,COUNT));hold_losses.append(hold.numpy().reshape(len(METHODS),3,COUNT))
            switches.append(choose_spectral.numpy().reshape(3,COUNT))
        fit=np.stack(fit_losses);hold=np.stack(hold_losses);switch=np.stack(switches)
        path=OUT/f'rsr{rsr:g}_snr{snr:g}.npz'
        old.save(path,methods=np.array(METHODS),seeds=np.array(old.SEEDS),regimes=np.array([r for r,L in old.REGIMES]),
            local_indices=np.arange(COUNT),fit_nll=fit,holdout_nll=hold,dual_uses_spectral=switch,snr=np.array(snr),rsr=np.array(rsr))
        rows=[]
        for pi,(regime,L) in enumerate(old.REGIMES):
            mean=hold[:,:,pi].mean((0,2));fitmean=fit[:,:,pi].mean((0,2));delta=hold[:,:,pi]-hold[:,0:1,pi]
            # The paired fit-to-fit average per anchor is the resampling unit.
            byanchor=delta.mean(0);dual_gain=hold[:,3,pi]-hold[:,5,pi]
            rows.append(dict(regime=regime,heldout_mean_nll=dict(zip(METHODS,map(float,mean))),fit_mean_nll=dict(zip(METHODS,map(float,fitmean))),
                heldout_excess_vs_original=dict(zip(METHODS,map(float,byanchor.mean(-1)))),
                dual_spectral_fraction=float(switch[:,pi].mean()),dual_heldout_improvement_vs_warm100=float(dual_gain.mean()),
                dual_improves_warm100_fraction=float((dual_gain>1e-12).mean()),dual_worsens_warm100_fraction=float((dual_gain< -1e-12).mean())))
        row=dict(rsr=rsr,snr=snr,cases_per_regime=COUNT,rows=rows,archive_sha256=sha(path));records.append(row)
        print(row,flush=True)
    write_json(OUT/'summary.json',dict(kind='exploratory_prior_development_only',source_sha256=sha(__file__),
        checkpoint_hashes={str(p.relative_to(REPO)):sha(p) for p in old.model_paths().values()},development_observation_hashes=hashes,
        methods=METHODS,conditions=CONDITIONS,cases_per_regime=COUNT,model_seeds=old.SEEDS,
        shapes='Saved fitting/heldoutNLL arrays [3fits,6methods,3pathregimes,16anchors]; no truth or test files read.',
        choice='Dual100 picks lower mean fitting RicianNLL perworld between learned-initialized100-step exactEM and independent spectral100; exactties keep learned start.',
        interpretation='Exploratory candidate-family design only, not fresh validation, NMSE evidence, BER evidence or isolated runtime benchmarking.',
        elapsed_seconds=time.perf_counter()-started,results=records))

if __name__=='__main__':main()
