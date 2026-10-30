"""Independent SVD reference for the solve-based final Hankel filter."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key] = '1'
import torch
from sahure.math import hankel, unhankel
from sahure.data import Observations
from tcom_followups_20261017.common import REPO, write_json, sha
from . import shrink


def run():
    torch.set_num_threads(1)
    generator = torch.Generator().manual_seed(2026101940)
    s = torch.randn(7,3,30,dtype=torch.complex128,generator=generator)
    noise = torch.tensor([1e-12,1e-9,1e-6,1e-3,.01,.1,1.],dtype=torch.float64)
    obs = Observations(torch.zeros(7,32,30),s,torch.ones(7,32,30,dtype=torch.complex128),noise)
    cases = {'full':torch.randn(7,32,3,dtype=torch.complex128,generator=generator),
             'rank_one':torch.exp(1j*torch.arange(32,dtype=torch.float64)[None,:,None]*torch.tensor([.1,.2,.3])[None,None,:]).expand(7,-1,-1),
             'zero':torch.zeros(7,32,3,dtype=torch.complex128)}
    errors = {}
    for name,g in cases.items():
        assert torch.equal(shrink.apply(obs,g,0,1),g)
        assert torch.equal(shrink.apply(obs,g,1,0),g)
        h = hankel(g.mT);u,d,vh = torch.linalg.svd(h,full_matrices=False)
        errs = []
        for cfg in shrink.CONFIGS.values():
            tau = cfg['c']*(2*noise[:,None]*torch.linalg.inv(s@s.mH).diagonal(dim1=-2,dim2=-1).real*17).sqrt()
            filt = u @ torch.diag_embed(d**3/(d*d+tau[...,None]**2)).to(g.dtype) @ vh
            target = g+cfg['beta']*(unhankel(filt).mT-g)
            actual = shrink.apply(obs,g,cfg['c'],cfg['beta'])
            assert torch.isfinite(actual).all()
            error = float((actual-target).abs().max());errs.append(error)
            assert error < 1e-9,(name,cfg,error)
        errors[name] = errs
    result = dict(passed=True,source_sha256=sha(shrink.__file__),checks_source_sha256=sha(__file__),
        identity_exact=True,svd_max_errors=errors,
        scope='Positive supplied variance only, enforced by receiver_keys. Full-rank, rank-one and zero estimates; variance1e-12 through1. Zero-noise inference is outside the protocol.')
    write_json(REPO/'results/tcom_incumbent_guard_20261019/shrink_checks.json',result)
    return result


if __name__=='__main__':
    print(run())
