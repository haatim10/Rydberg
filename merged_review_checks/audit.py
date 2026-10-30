"""Independent score aggregation and representative replay of the added run."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key]='1'
import json
from pathlib import Path
import numpy as np
import torch
from scipy.special import i0e, i1e
from merged_review_checks.run import ROOT,OLD,SEEDS,SPECS,load,sha,one_sweep,verify_sources
from sahure.train import load_model
from codex_verification.solver import initial

torch.set_num_threads(1)
manifest=verify_sources()
summary=json.loads((ROOT/'summary.json').read_text())
worst_replay=0.;worst_neural=0.;count=0;totals=[]
for spec in SPECS:
    cid=spec['id'];bank=OLD/'test_data'/cid
    obs=load(bank/'observations.npz');truth=load(bank/'truth.npz')
    err=[];energy=[]
    for i in range(128):
        row=load(ROOT/'predictions'/cid/f'trial_{i:03d}.npz')
        assert row['estimate'].shape==(32,3)
        assert np.isfinite(row['estimate']).all() and np.isfinite(row['scores']).all()
        assert int(row['rank'])==int(row['scores'].argmin()+1)
        assert str(row['trial_id'])==obs['trial_ids'][i]==truth['trial_ids'][i]
        e=np.vdot(row['estimate']-truth['g'][i],row['estimate']-truth['g'][i]).real
        den=np.vdot(truth['g'][i],truth['g'][i]).real
        err.append(e);energy.append(den);count+=1
    score=10*np.log10(sum(err)/sum(energy));totals.append(score)
    np.testing.assert_allclose(score,summary['cells'][cid]['one_sweep']['nmse_db'],atol=1e-12)
    rate=100*float(np.mean(np.array(err)>np.array(energy)))
    np.testing.assert_allclose(rate,summary['cells'][cid]['one_sweep']['tail']['1.0']['percent'],atol=1e-12)
np.testing.assert_allclose(np.mean(totals[:7]),summary['macro']['one_sweep']['nmse_db'],atol=1e-12)

with torch.no_grad():
    for cid,i in [('snr_-10',0),('snr_+20',17),('weak_reference',91)]:
        obs=load(OLD/'test_data'/cid/'observations.npz')
        z,s,b,v=(obs[k][i] for k in ('z','s','b','variance'));v=float(v)
        g,r,scores=one_sweep(z,s,b,v)
        saved=load(ROOT/'predictions'/cid/f'trial_{i:03d}.npz')
        np.testing.assert_allclose(g,saved['estimate'],atol=0,rtol=0)
        worst_replay=max(worst_replay,float(np.max(abs(g-saved['estimate']))))
        old=load(OLD/'test_outputs'/cid/f'trial_{i:03d}.npz')
        for seed in SEEDS:
            model,_=load_model(manifest['models'][str(seed)]['path'])
            out=model.estimate(z,s,b,v).numpy()
            ref=old['estimates'][list(old['names']).index(f'adaptive10_s{seed}')]
            np.testing.assert_allclose(out,ref,rtol=1e-12,atol=1e-12)
            worst_neural=max(worst_neural,float(np.max(abs(out-ref))))

# A scalar loop with explicit antidiagonal averaging independently replays
# the selected all-pilot hard-rank fit, without the vectorized projector.
obs=load(OLD/'test_data'/'snr_+5'/'observations.npz');i=3
z,s,b,v=(obs[k][i] for k in ('z','s','b','variance'));v=float(v)
saved=load(ROOT/'predictions'/'snr_+5'/f'trial_{i:03d}.npz');rank=int(saved['rank'])
g=initial(s,z,b);idx=np.arange(17)[:,None]+np.arange(16)[None,:]
for _ in range(100):
    mu=g@s+b;x=2*z*abs(mu)/v
    assert x.max()<1e4
    q=z*(i1e(x)/i0e(x))*np.exp(1j*np.angle(mu))
    g=np.linalg.lstsq(s.T,(q-b).T,rcond=None)[0].T
    if rank<16:
        for k in range(3):
            u,sv,vh=np.linalg.svd(g[:,k][idx],full_matrices=False)
            h=(u[:,:rank]*sv[:rank])@vh[:rank]
            g[:,k]=np.array([h[idx==j].mean() for j in range(32)])
np.testing.assert_allclose(g,saved['estimate'],rtol=1e-9,atol=1e-9)
result=dict(passed=True,checked_predictions=count,all_selected_ranks_match_scores=True,
            independent_cell_scores=True,old_error_max_abs=summary['saved_error_max_abs_difference'],
            added_replays=3,added_replay_max_abs=worst_replay,
            learned_replays=9,learned_replay_max_abs=worst_neural,
            scalar_svd_lstsq_replay_max_abs=float(np.max(abs(g-saved['estimate']))),
            audit_source_sha256=sha(__file__))
(ROOT/'independent_audit.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
