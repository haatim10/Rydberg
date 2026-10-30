"""Post-test numerical diagnosis only; no extra iterations or policy changes."""
import os
for _name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[_name]='1'
import json
import numpy as np
import torch
from sahure.data import Observations
from tcom_followups_20261017.common import em_estimate,sha,write_json
from . import study


@torch.no_grad()
def run():
    torch.set_num_threads(1)
    try:torch.set_num_interop_threads(1)
    except RuntimeError:pass
    policy,digest=study.check_selection();est=study.engine();cases=(30,18);rows=[]
    for snr in (37.5,60.):
        path=study.OUT/'test/cells'/f'L2_rsr5_snr{snr:g}.npz';cell=study.load(path)
        assert str(cell['selection_sha256'])==digest
        obs=Observations(*(torch.tensor(cell[k][list(cases)],dtype=torch.complex128 if k in ('s','b') else torch.float64) for k in ('z','s','b','sigma2')))
        saved=torch.tensor(cell['estimates'][study.NAMES.index('em100'),list(cases)],dtype=torch.complex128)
        canonical=em_estimate(obs,100);optimized=est.em100(obs)
        torch.testing.assert_close(canonical,saved,atol=1e-10,rtol=1e-10)
        torch.testing.assert_close(optimized,saved,atol=1e-10,rtol=1e-10)
        for i,local in enumerate(cases):
            truth=cell['truth_g'][local];energy=float((np.abs(truth)**2).sum())
            err=float((np.abs(canonical[i].numpy()-truth)**2).sum())
            rows.append(dict(snr=snr,rsr=5.,regime='L2',local_case=local,effective_case=int(cell['indices'][local]),
                canonical_vs_saved_max_abs_error=float((canonical[i]-saved[i]).abs().max()),
                optimized_replay_vs_saved_max_abs_error=float((optimized[i]-saved[i]).abs().max()),
                canonical_case_nmse=err/energy,saved_case_nmse=float(cell['nmse'][study.NAMES.index('em100'),local]),
                source_cell_sha256=sha(path)))
    result=dict(passed=True,selection_sha256=digest,source_sha256=sha(__file__),cases=rows,
        interpretation='Post-test implementation diagnosis on the observed adverse cases only. Canonical and optimized EM100 agree; no extra iterations, starts, tuning, model changes or selection changes were made. Agreement does not establish a global optimum.')
    write_json(study.OUT/'audit/adverse_case_parity.json',result);return result


if __name__=='__main__':print(json.dumps(run(),indent=2))
