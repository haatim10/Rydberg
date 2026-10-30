"""Independent archive, selection, and inference replay audit; no fitting."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key]='1'
import json
import numpy as np
import torch
from sahure.data import Observations
from tcom_followups_20261017.common import sha, write_json, load_learned_models, em_estimate, frozen_optshrink
from .runtime_study import OUT, SNRS, REGIMES, CAP_NAMES, FULL, check_selection, selected_config
from .sparse_fast import estimate


@torch.no_grad()
def main():
    torch.set_num_threads(1)
    selection,digest=check_selection()
    protocol=json.loads((OUT/'protocol.json').read_text())
    calibration=json.loads((OUT/'calibration.json').read_text())
    counts={}
    for snr in SNRS:
        scores=json.loads((OUT/'development'/f'scores_snr{snr:g}.json').read_text())['scores']
        for cap in CAP_NAMES:
            eligible=[s for s in scores if s['family_key'] in calibration['eligibility'][cap]]
            best=min(eligible,key=lambda x:(x['mean_nll'],x['eligibility_p90_ms'],
                x['config']['iterations'],x['config']['inner_steps'],x['config']['dictionary_size'],x['config']['factor']))
            saved=selection['selections'][f'{snr:g}'][cap]
            assert best['config']==saved['config']
            assert best['eligibility_p90_ms']<=selection['caps_ms'][cap]
            np.testing.assert_allclose(np.mean(best['per_world_nll']),best['mean_nll'],rtol=0,atol=1e-12)
            counts[f'{snr:g}:{cap}']=len(eligible)
    models=load_learned_models()
    parent=json.loads((OUT.parents[1]/'fair_optshrink/frozen_selection.json').read_text())
    hashes={};replayed=[];maximum=0.
    for snr in SNRS:
        for regime,_ in REGIMES:
            path=OUT/'test'/f'{regime}_snr{snr:g}.npz'
            with np.load(path) as a:
                assert str(a['selection_sha256'])==digest
                assert int(a['seed'])==protocol['test_seed']==2026101742
                assert path.stat().st_mtime>=selection['frozen_unix']
                names=list(map(str,a['method_names']))
                np.testing.assert_array_equal(np.sum(abs(a['estimates']-a['truth_g'][None])**2,axis=(-2,-1)),a['errors'])
                np.testing.assert_array_equal(np.sum(abs(a['truth_g'])**2,axis=(-2,-1)),a['truth_energy'])
                np.testing.assert_array_equal(a['errors']/a['truth_energy'][None],a['nmse'])
                assert np.isfinite(a['estimates']).all()
                # Single-case replay in every condition checks batch invariance.
                obs=Observations(*(torch.as_tensor(a[k][:1],dtype=torch.complex128 if k in ('s','b') else torch.float64)
                                   for k in ('z','s','b','sigma2')))
                outputs={name:m(*obs.args()).numpy()[0] for name,m in models.items()}
                outputs.update({cap:estimate(obs,selected_config(selection,obs,cap)).numpy()[0] for cap in CAP_NAMES})
                outputs.update(em100=em_estimate(obs).numpy()[0],optshrink=frozen_optshrink(obs,parent).numpy()[0],fast_full=estimate(obs,FULL).numpy()[0])
                diff=max(float(np.max(abs(outputs[name]-a['estimates'][names.index(name),0]))) for name in names)
                assert diff<2e-10,(path,diff)
                maximum=max(maximum,diff);replayed.append(dict(cell=path.stem,max_prediction_difference=diff))
            hashes[path.name]=sha(path)
    final=json.loads((OUT/'final_timing.json').read_text())
    assert final['selection_sha256']==digest and final['no_reselection_after_test']
    write_json(OUT.parent/'runtime_audit.json',dict(passed=True,
        frozen_input_hashes_verified=True,selection_optimal_among_eligible_saved_candidates=True,
        no_test_worlds_generated_before_selection=True,raw_error_arrays_exact=True,
        cells=15,prediction_replays=150,max_prediction_replay_difference=maximum,
        cap_constraints_apply_to_calibration_p90=True,fresh_timing_overshoot_report=final['cap_verification'],
        candidate_counts=counts,test_archive_hashes=hashes,replays=replayed,source_sha256=sha(__file__),
        scope='Numerical and procedural audit; fixed validation selections and fixed trained models. '
              'Latency quantiles are empirical summaries, not hard real-time guarantees.'))
    print('Runtime study audit passed: 15 cells, 150 prediction replays.',flush=True)


if __name__=='__main__':main()
