"""Numerical equivalence and descent checks, excluded from production timing."""
import json
from pathlib import Path
import numpy as np
import torch

from . import sparse_fast as fast
from tcom_followups_20261017.common import world_batch,em_estimate,write_json,sha
from tcom_followups_20261017.sparse import sparse_trajectory as reference
from sahure.math import spectral_initial as torch_initial, posterior_field as torch_posterior


def relative(a,b):
    return float(np.linalg.norm(np.asarray(a)-np.asarray(b))/max(1.,np.linalg.norm(np.asarray(b))))


def run():
    torch.set_num_threads(1)
    obs,_,_=world_batch(2026101740,range(3),snr=10.,rsr=12.,L=None)
    z,s,b,v=fast.as_numpy(obs)
    rng=np.random.default_rng(2026101740)
    checks={}
    for d in (64,128,256):
        fft=fast.AngleFrame(32,d);matrix=fast.AngleFrame(32,d,backend='matrix')
        x=rng.normal(size=(3,d,3))+1j*rng.normal(size=(3,d,3))
        y=rng.normal(size=(3,32,3))+1j*rng.normal(size=(3,32,3))
        ax=fft.forward(x);ahy=fft.adjoint(y)
        lhs=np.vdot(ax,y);rhs=np.vdot(x,ahy)
        error=abs(lhs-rhs)/max(1.,abs(lhs),abs(rhs))
        forward=relative(ax,matrix.forward(x));adjoint=relative(ahy,matrix.adjoint(y))
        tight=relative(fft.forward(fft.adjoint(y)),(d/32)*y)
        checks[f'frame_D{d}']=dict(adjoint_identity_relative_error=error,FFT_vs_matrix_forward=forward,
                                  FFT_vs_matrix_adjoint=adjoint,tight_frame_relative_error=tight)
        if max(error,forward,adjoint,tight)>1e-12:raise AssertionError(checks[f'frame_D{d}'])
    initial=fast.spectral_initial(z,s,b)
    init_error=relative(initial,torch_initial(obs.z,obs.s,obs.b).numpy())
    post_error=relative(fast.posterior_field(z,initial@s+b,v),torch_posterior(obs.z,torch.from_numpy(initial)@obs.s+obs.b,obs.sigma2).numpy())
    checks['statistics']=dict(initializer_relative_error=init_error,posterior_field_relative_error=post_error)
    if max(init_error,post_error)>1e-12:raise AssertionError(checks['statistics'])
    frame=fast.AngleFrame(32,128)
    x=rng.normal(size=(3,128,3))+1j*rng.normal(size=(3,128,3))
    direction=rng.normal(size=x.shape)+1j*rng.normal(size=x.shape)
    target=rng.normal(size=z.shape)+1j*rng.normal(size=z.shape)
    sh=s.conj().swapaxes(-1,-2);gram=s@sh;rhs=target@sh
    g=frame.forward(x)
    gradient=frame.adjoint(g@gram-rhs)
    expected=float(np.real(np.vdot(gradient,direction)))
    def value(xx):return .5*np.sum(np.abs(frame.forward(xx)@s-target)**2)
    eps=1e-5;numerical=(value(x+eps*direction)-value(x-eps*direction))/(2*eps)
    gradient_error=abs(numerical-expected)/max(1.,abs(expected))
    quadratic_error=abs(float(fast.quadratic(g,x,gram,rhs,np.zeros(3)).sum()+.5*np.sum(np.abs(target)**2))-value(x))/max(1.,value(x))
    checks['M_step']=dict(gradient_finite_difference_relative_error=gradient_error,quadratic_identity_relative_error=quadratic_error)
    if gradient_error>1e-7 or quadratic_error>1e-12:raise AssertionError(checks['M_step'])
    em,info=fast.trajectory(obs,0.,inner=1,budgets=(100,),audit=True)
    em_error=relative(em[100].numpy(),em_estimate(obs,100).numpy())
    checks['zero_penalty']=dict(relative_error_vs_torch_EM100=em_error,max_observed_objective_increase=info['max_observed_objective_increase'])
    if em_error>1e-11:raise AssertionError(checks['zero_penalty'])
    cases=[]
    for d in (64,128,256):
        for inner in (1,2,4,8):
            output,info=fast.trajectory(obs,.3,d=d,inner=inner,budgets=(12,),audit=True)
            old,old_info=reference(obs,.3,d=d,inner=inner,budgets=(12,),audit=True)
            error=relative(output[12].numpy(),old[12].numpy())
            row=dict(dictionary_size=d,inner_steps=inner,outer_steps=12,relative_channel_error_vs_old=error,
                     max_observed_objective_increase=info['max_observed_objective_increase'],
                     new_inner_restarts=info['inner_restarts'],old_inner_restarts=old_info['inner_restarts'])
            cases.append(row)
            if error>1e-9:raise AssertionError(row)
    output,info=fast.trajectory(obs,.3,d=128,inner=25,budgets=(100,),audit=True)
    old,old_info=reference(obs,.3,d=128,inner=25,budgets=(100,),audit=True)
    error=relative(output[100].numpy(),old[100].numpy())
    checks['original_full_budget']=dict(dictionary_size=128,inner_steps=25,outer_steps=100,
        relative_channel_error_vs_old=error,max_observed_objective_increase=info['max_observed_objective_increase'],
        new_inner_restarts=info['inner_restarts'],old_inner_restarts=old_info['inner_restarts'],
        max_final_relative_proximal_residual=max(info['final_relative_proximal_residual']))
    if error>1e-8:raise AssertionError(checks['original_full_budget'])
    checks['finite_budget_equivalence']=cases
    checks['solver_source_sha256']=sha(Path(fast.__file__))
    checks['smoke_master_seed']=2026101740
    checks['notes']='No development/test outcomes used. Production omits audit diagnostics but keeps all monotone-FISTA restart logic. Timings require a separate idle process.'
    return checks


if __name__=='__main__':
    checks=run()
    path=Path(__file__).resolve().parents[1]/'results/tcom_runtime_bypass_20261017/sparse_fast_checks.json'
    write_json(path,checks)
    print(json.dumps(checks,indent=2))
