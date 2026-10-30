"""Post-screen numerical diagnosis; never replaces frozen study predictions."""
import json
import time
import numpy as np
import torch
from paper2_optshrink_loop.run import load, write
from .run import ROOT, sha
from .core import jacobians, path


def automatic(z, s, b, v, start, oracle=None):
    """Six reverse passes suffice because the unfiltered map is row-separable.

    Sum each output coordinate over rows. Each input row affects only its own
    output row, so the derivative retains every row's 6-by-P Jacobian.
    The spectral initializer is differentiated, including its phase anchor.
    """
    ss=torch.tensor(s); bb=torch.tensor(b)
    def fn(zz, reduce=True):
        if start == 'zero':
            g=torch.zeros((len(b),len(s)),dtype=torch.complex128)
        elif start == 'oracle':
            g=torch.tensor(oracle)
        else:
            a=torch.cat((ss.T[None].expand(len(b),-1,-1),bb[...,None]),-1)
            matrix=a.mH@(zz[...,None]*a); matrix=(matrix+matrix.mH)/2
            _,vectors=torch.linalg.eigh(matrix); vec=vectors[...,-1]
            projected=(a@vec[...,None]).squeeze(-1).abs()
            scale=(projected*zz).sum(-1)/projected.square().sum(-1)
            g=(scale[...,None]*vec*torch.exp(-1j*torch.angle(vec[...,-1]))[...,None])[...,:len(s)]
        gram=ss@ss.mH
        for _ in range(100):
            mu=g@ss+bb; amplitude=mu.abs(); x=2*zz*amplitude/v
            r=torch.special.i1e(x)/torch.special.i0e(x)
            safe=x.clamp_min(1.)
            r=torch.where(x>1e4,1-1/(2*safe)-1/(8*safe.square()),r).clamp(0,1)
            # Algebraically identical to the NumPy phase update away from zero.
            q=zz*mu/amplitude*r
            g=torch.linalg.solve(gram.T,((q-bb)@ss.mH).T).T
        output=torch.cat((g.real,g.imag),-1)
        return output.sum(0) if reduce else output
    zz=torch.tensor(z,requires_grad=True)
    jj=torch.autograd.functional.jacobian(fn,zz).detach().numpy().transpose(1,0,2)
    out=fn(zz,False).detach().numpy()
    return jj,out


def main():
    torch.set_num_threads(1);tick=time.perf_counter()
    audit=json.loads((ROOT/'audit.json').read_text())
    flags=[r for r in audit['largest_prediction_anchor_step_halving']
           if max(r['relative_changes'].values())>=audit['step_check_threshold']]
    targets=[dict(r,reason='flagged_extra_start') for r in flags]
    # Independently check the largest ordinary-start predictions in the two
    # failing scientific conditions. This selects audit targets, not tuning.
    for cell in ('snr_-10','weak_reference'):
        choices=[]
        for i in range(4):
            row=load(ROOT/'worlds'/cell/f'{i:02d}.npz')
            traces=row['anchor_prediction_traces'][0,2,3]
            a=int(traces.argmax()); choices.append((float(traces[a]),i,a))
        _,i,a=max(choices)
        targets.append(dict(cell=cell,world=i,anchor=a,start='spectral',reason='largest_primary_prediction'))
    results=[]
    for target in targets:
        row=load(ROOT/'worlds'/target['cell']/f'{target["world"]:02d}.npz')
        z=row['z_anchors'][target['anchor']:target['anchor']+1]
        s,b,v=row['s'],row['b'],float(row['variance']); start=target['start']
        oracle=row['g'] if start=='oracle' else None
        ad,out=automatic(z[0],s,b,v,start,oracle)
        g=path(z,s,b,v,start,oracle,stages=(100,))[100][1][0]
        expected=np.concatenate((g.real,g.imag),-1)
        np.testing.assert_allclose(out,expected,rtol=1e-8,atol=1e-9)
        steps=[]
        for step in (1e-4,5e-5,2.5e-5,1e-5):
            fd=jacobians(z,s,b,v,start,oracle,step=step,stages=(100,))[100][0]
            steps.append(dict(step=step,relative_error_to_autograd=float(np.linalg.norm(fd-ad)/np.linalg.norm(ad)),
                              squared_jacobian_norm_ratio=float(np.sum(fd**2)/np.sum(ad**2))))
        results.append(dict(target=target,forward_max_absolute_error=float(abs(out-expected).max()),steps=steps))
    review=dict(original_audit_status=audit['status'],original_predictions_unchanged=True,
                note='Post-screen diagnostic, not a replacement estimator or a retrospective gate pass.',
                targets=results,seconds=time.perf_counter()-tick,source_sha256=sha(__file__))
    write(ROOT/'derivative_review.json',review)
    print(json.dumps(review,indent=2))


if __name__=='__main__':main()
