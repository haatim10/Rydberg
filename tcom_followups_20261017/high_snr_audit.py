"""Read-only numerical replay; writes only a new high-SNR audit record."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[key]='1'
import json
import numpy as np
import torch
from .high_snr import OUT, REGIMES, RSRS, observed, learned_diagnostics, summarize_cell
from .common import REPO, sha, write_json, load_learned_models, em_estimate, frozen_optshrink
from rydberg_sim.gs import em_gs_channel_rows

@torch.no_grad()
def main():
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    protocol=json.loads((OUT/'protocol.json').read_text());summary=json.loads((OUT/'summary.json').read_text())
    assert summary['complete']
    assert protocol['source_sha256']==sha(REPO/'tcom_followups_20261017/high_snr.py')
    for path,digest in protocol['original_inputs']['hashes'].items():assert sha(REPO/path)==digest,path
    models=load_learned_models();checks=[];anchor_hashes={};maxerror=0.;maxreplay=0.;maxold=0.
    for regime,L in REGIMES:
        for rsr in RSRS:
            anchor_path=OUT/'anchors'/f'{regime}_rsr{rsr:g}.npz';a=dict(np.load(anchor_path))
            anchor_hashes[anchor_path.name]=sha(anchor_path)
            assert anchor_path.stat().st_mtime>=protocol['frozen_unix']
            z10=observed(a,10.).z.numpy();np.testing.assert_allclose(z10,a['anchor_z'],atol=1e-13,rtol=0)
            selected=[p for p in (OUT/'cells').glob(f'{regime}_rsr{rsr:g}_*.npz')]
            for path in selected:
                d=dict(np.load(path));assert sha(path)==summary['cell_hashes'][path.name]
                assert str(d['protocol_sha256'])==summary['protocol_sha256']
                energy=np.sum(abs(a['g'])**2,axis=(-2,-1));err=np.sum(abs(d['estimates']-a['g'][None])**2,axis=(-2,-1))
                np.testing.assert_array_equal(energy,d['truth_energy']);np.testing.assert_array_equal(err,d['errors'])
                np.testing.assert_array_equal(err/energy[None],d['nmse']);assert np.isfinite(d['estimates']).all()
                np.testing.assert_array_equal(d['indices'],a['indices'])
                maxerror=max(maxerror,float(np.max(abs(err-d['errors']))))
            # Endpoint replay checks cover all nine path/reference strata.
            for snr in (10.,60.):
                path=OUT/'cells'/f'{regime}_rsr{rsr:g}_snr{snr:g}.npz';d=dict(np.load(path));obs=observed(a,snr).batch(slice(0,1));outs={}
                for name,m in models.items():
                    outs[name]=m(*obs.args()).numpy()[0]
                    outs[name.replace('learned_','off_')]=m(*obs.args(),gate_override=0).numpy()[0]
                outs['em10']=em_estimate(obs,10).numpy()[0];outs['em100']=em_estimate(obs,100).numpy()[0];outs['optshrink']=frozen_optshrink(obs).numpy()[0]
                diff=max(float(np.max(abs(outs[str(name)]-d['estimates'][i,0]))) for i,name in enumerate(d['names']))
                assert diff<2e-10,(path,diff);maxreplay=max(maxreplay,diff)
                checks.append(dict(regime=regime,rsr=rsr,snr=snr,max_estimate_difference=diff))
                if regime=='random3to7' and rsr==12.:
                    reference=em_gs_channel_rows(obs.s.numpy()[0],obs.z.numpy()[0],obs.b.numpy()[0],float(obs.sigma2[0]),max_iter=100).G_hat
                    dd=float(np.max(abs(reference-outs['em100'])));assert dd<2e-9,dd;maxold=max(maxold,dd)
    # Recompute summaries (including paired bootstrap) rather than trust headline data.
    for row in summary['cells']:
        replay=summarize_cell(OUT/'cells'/f'{row["cell"]}.npz');assert replay==row,row['cell']
    write_json(OUT/'audit.json',dict(passed=True,source_sha256=sha(__file__),common_sha256=sha(REPO/'tcom_followups_20261017/common.py'),
       frozen_input_hashes_verified=True,all_error_arrays_exact=True,all_summaries_recomputed=True,
       cells=len(summary['cells']),endpoint_replays=len(checks),model_prediction_replays=len(checks)*9,
       max_prediction_difference=maxreplay,max_original_row_em100_difference=maxold,
       observations_at_anchor_reconstructed=True,same_channel_pilot_reference_and_standard_noise_across_snr_by_construction=True,
       unique_anchor_worlds=summary['independent_worlds'],anchor_hashes=anchor_hashes,replays=checks,
       limitation='Implementation/numerical audit, not independent external replication or uncertainty over training populations.'))
    print('High-SNR audit passed',len(summary['cells']),'cells; max replay',maxreplay,'original EM100',maxold,flush=True)

if __name__=='__main__':main()
