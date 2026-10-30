"""Small independent numerical checks before freezing scientific evaluation."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key]='1'
import json
import numpy as np
import torch
from sahure.math import posterior_field,right_solve,spectral_initial
from tcom_followups_20261017.common import sha,write_json
from tcom_incumbent_guard_20261019 import shrink
from tcom_pilot_em_20261018 import continuation
from . import study


@torch.no_grad()
def main():
    torch.set_num_threads(1)
    est=study.engine()
    selection=dict(classical_config=dict(kind='noise',c=2.,beta=.5),
                   budget_steps={name:75 for name in ('adaptive','global','classical','spectral','em10')})
    maximum_dispatch=maximum_em=maximum_classical=maximum_spectral=0.
    checked=0
    for snr,ref,L in ((0.,5.,2),(32.5,12.,4),(37.5,5.,None),(40.,20.,2),(60.,12.,None)):
        anchor=study.make_anchor(study.SMOKE_SEED,[0,1],L,ref,truth=False)
        obs=study.observed(anchor,snr)
        raw,estimates=study.predictions(obs,selection,est)
        for i,name in enumerate(study.NAMES):
            separate=study.estimate_method(obs,name,selection,est)
            error=float((separate-estimates[i]).abs().max())
            assert error<1e-10,(name,error)
            maximum_dispatch=max(maximum_dispatch,error)
        expected=spectral_initial(*obs.args()[:3])
        actual=est['starts'].start('spectral',obs)
        error=float((actual-expected).abs().max());assert error<1e-10
        maximum_spectral=max(maximum_spectral,error)
        canonical=expected.clone()
        for _ in range(10):
            q=posterior_field(obs.z,canonical@obs.s+obs.b,obs.sigma2)
            canonical=right_solve((q-obs.b)@obs.s.mH,obs.s@obs.s.mH)
            canonical=shrink.apply(obs,canonical,2.,.5)
        actual=est['starts'].start('classical',obs,config=selection['classical_config'])
        error=float((actual-canonical).abs().max());assert error<1e-10
        maximum_classical=max(maximum_classical,error)
        for start in raw:
            values,info=continuation.trajectory(obs,start,(1,2,5),audit=True)
            canonical=start.clone()
            for step in range(1,6):
                canonical=right_solve((posterior_field(obs.z,canonical@obs.s+obs.b,obs.sigma2)-obs.b)@obs.s.mH,obs.s@obs.s.mH)
                if step in values:
                    error=float((values[step]-canonical).abs().max());assert error<1e-10
                    maximum_em=max(maximum_em,error)
            cfg=study.filter_config(ref,snr)
            assert torch.equal(study.finish(obs,start),shrink.apply(obs,start,cfg['c'],cfg['beta']))
        checked+=1
    configs={f'rsr{r:g}_snr{s:g}':study.filter_config(r,s) for r in study.RSRS for s in study.SNRS}
    assert all(c['c']==0 for k,c in configs.items() if k.startswith('rsr5_'))
    assert configs['rsr12_snr32.5']['c']==2
    assert configs['rsr20_snr40']['c']==4
    result=dict(passed=True,smoke_seed=study.SMOKE_SEED,conditions=checked,worlds_per_condition=2,
                actual_dispatch_max_abs_error=maximum_dispatch,canonical_em_max_abs_error=maximum_em,
                canonical_classical_noise_max_abs_error=maximum_classical,spectral_max_abs_error=maximum_spectral,
                common_filter_configs=configs,em_likelihood_monotonicity=True,no_truth_exported_to_estimators=True,
                source_hashes={str(p.relative_to(study.REPO)):sha(p) for p in (study.REPO/'tcom_start_budget_20261020/study.py',study.REPO/'tcom_start_ablation_20261020/starts.py',study.REPO/'tcom_start_budget_20261020/checks.py')})
    write_json(study.OUT/'checks.json',result)
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
