"""Post-hoc mechanism replay, not new independent evidence or a selected remedy."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key]='1'
import json
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from codex_verification.solver import initial,ratio
from sahure_expansion.data import world
from sahure_expansion.protocol import write_json
from .common import ROOT,freeze_phase,frozen_models,load,save,aggregate,verify_sources


@torch.no_grad()
def run():
    torch.set_num_threads(1)
    out,protocol=freeze_phase('low_noise_replay',dict(post_hoc=True,cell='reference_12_snr+60',cases=32,
        scope='same previously inspected cases; independent no-filter arithmetic and oracle initialization diagnostic'),[__file__])
    models=frozen_models(protocol);rows=[];max_replay=0.;max_numerical=0.
    for i in range(32):
        d=load(ROOT/'geometry/data/reference_12_snr+60'/f'trial_{i:03d}.npz')
        g,z,s,b,v=(d[k] for k in ('g','z','s','b','v'));v=float(v)
        w,_=world(i,202610087201+200,N=32,K=3,P=30,snr=60,rsr=12)
        np.testing.assert_allclose(z,abs(w.G@w.S+w.B+w.W),atol=1e-13,rtol=1e-13)
        old=load(ROOT/'geometry/predictions/reference_12_snr+60'/f'trial_{i:03d}.npz')
        estimates=[];corrections=[];thresholds=[]
        for j,(seed,m) in enumerate(models.items()):
            full,trace=m.estimate(z,s,b,v,diagnostics=True)
            max_replay=max(max_replay,float(np.max(abs(full.numpy()-old['estimates'][j]))))
            nofilter=m.estimate(z,s,b,v,gate_override=0).numpy()
            gg=initial(s,z,b);op=np.linalg.solve((s@s.conj().T).T,s.conj()).T
            for step in m.steps:
                mu=gg@s+b;q=z*np.exp(1j*np.angle(mu))*ratio(2*z*abs(mu)/v)
                gg+=float(2*torch.sigmoid(step))*((q-b)@op-gg)
            max_numerical=max(max_numerical,float(np.max(abs(nofilter-gg))))
            with patch('sahure.model.spectral_initial',return_value=torch.as_tensor(g.copy(),dtype=torch.complex128)[None]):
                oracle=m.estimate(z,s,b,v).numpy()
            estimates.extend([full.numpy(),nofilter,oracle]);corrections.append(np.stack([t['correction'].numpy() for t in trace]))
            thresholds.append(np.stack([t['tau'].numpy() for t in trace]))
        estimates=np.stack(estimates);error=np.sum(abs(estimates-g)**2,axis=(-2,-1));energy=np.sum(abs(g)**2)
        save(out/f'trial_{i:03d}.npz',estimates=estimates,errors=error,energy=energy,corrections=corrections,thresholds=thresholds)
        rows.append((error,energy))
    assert max_replay<1e-12
    assert max_numerical<1e-7
    ee=np.stack([r[0] for r in rows]).T;energy=np.array([r[1] for r in rows])
    summary={name:aggregate(ee[np.arange(j,9,3)],energy) for j,name in enumerate(['learned','filter_disabled_DIAGNOSTIC','truth_initial_ORACLE'])}
    verify_sources(protocol);write_json(out/'summary.json',dict(methods=summary,max_saved_replay_difference=max_replay,
        max_independent_no_filter_difference=max_numerical,source='same 32 already evaluated high-SNR worlds',
        caveat='diagnostic interventions after seeing failure; no retraining or claim of a deployable selected policy'))
    print(json.dumps(summary,indent=2))

if __name__=='__main__':run()
