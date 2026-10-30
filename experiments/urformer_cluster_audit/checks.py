"""Independent checks before the validation draws are generated."""
import importlib.util,json
from pathlib import Path
import numpy as np
from scipy.integrate import quad
import torch
from audit import ROOT,REPO,CFG,channels,rng,cn,observation,classical,genie,write,protected_snapshot,verify_protected,M,K,j0

def main():
    if not (ROOT/'protected_before.json').exists():write('protected_before.json',protected_snapshot())
    verify_protected()
    h,angles=channels('checks',0,4,1.,'laplacian');shared=dict(w=cn(rng('checks',0,11),(4,M,35)),init=cn(rng('checks',0,12),(4,M,K)))
    obs=observation(h,shared,20,5)
    R=M*j0(np.pi*(np.arange(M)[:,None]-np.arange(M)[None,:]))
    pred=genie(obs,R);A=np.kron(obs['s'].T,np.eye(M));C=np.kron(np.eye(K),R)
    direct=C@A.conj().T@np.linalg.solve(A@C@A.conj().T+obs['v']*np.eye(M*20),np.eye(M*20))
    dense=np.stack([(direct@(y-obs['b']).reshape(-1,order='F')).reshape((M,K),order='F') for y in obs['y']])
    lmmse_error=float(np.max(np.abs(dense-pred['genie_LMMSE'])))
    np.testing.assert_allclose(dense,pred['genie_LMMSE'],atol=1e-9,rtol=1e-10)
    # A vanishing noise variance must approach LS (the empirical prior is full rank).
    tiny=dict(obs,v=1e-12);np.testing.assert_allclose(genie(tiny,R)['genie_LMMSE'],pred['genie_LS'],atol=1e-10,rtol=1e-10)
    spec=importlib.util.spec_from_file_location('frozen_urformer',REPO/'urformer_reproduction_20261022/model.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);torch.set_num_threads(1);torch.set_num_interop_threads(1)
    layer=mod.Layer().double();ours=classical(obs,shared['init']);errs={}
    with torch.no_grad():
        for name,mode in [('GS100','gs'),('EM100','em')]:
            x=torch.from_numpy(shared['init'].copy());s=torch.from_numpy(obs['s']);v=torch.full((4,),obs['v'],dtype=torch.float64)
            for _ in range(100):x=layer(x,torch.from_numpy(obs['z']),s,torch.tensor(obs['b']),v,torch.linalg.pinv(s),classical=mode)
            errs[name]=float(np.max(np.abs(x.numpy()-ours[name])));assert errs[name]<1e-8
    h0,a0=channels('checks',0,4,0.,'laplacian');h1,a1=channels('checks',0,4,0.,'gaussian');assert np.array_equal(h0,h1)
    sv=np.linalg.svd(np.exp(1j*np.pi*np.arange(M)[None,None,:,None]*np.sin(a0.reshape(-1,K,40))[:,:,None,:]),compute_uv=False)
    ranks=np.sum(sv>40*np.finfo(float).eps*sv[...,:1],axis=-1);assert np.all(ranks<=4)
    h2,a2=channels('checks',0,4,None,'independent');assert np.max(np.abs(a2))<=np.pi/2
    # Exact old independent-ray formula with the new audit seed.
    old_angles=rng('checks',0,1).uniform(-np.pi/2,np.pi/2,(4,K,4,10));gains=cn(rng('checks',0,2),old_angles.shape)
    expected=np.sqrt(M/40)*np.sum(np.exp(1j*np.pi*np.arange(M)[None,:,None,None,None]*np.sin(old_angles)[:,None])*gains[:,None],axis=(-2,-1))
    assert np.array_equal(h2,expected)
    quadrature={str(l):float(quad(lambda x:np.cos(np.pi*l*np.sin(x)),-np.pi/2,np.pi/2,epsabs=1e-12)[0]/np.pi-j0(np.pi*l)) for l in (0,1,3,31)}
    assert max(map(abs,quadrature.values()))<1e-10
    v_expected=M*np.sum(np.abs(obs['s'])**2)/20/10**.5;assert obs['v']==v_expected
    # Paired +/- offsets preserve angular marginal under reflection.
    centers=np.linspace(-np.pi/2,np.pi/2,20000,endpoint=False)+np.pi/40000
    invariance=[]
    for offset in (np.deg2rad(.5),np.deg2rad(10),1.7):
        for lag in (1,3,31):
            both=np.arcsin(np.sin(centers[:,None]+np.array([-offset,offset])))
            invariance.append(abs(np.mean(np.exp(1j*np.pi*lag*np.sin(both)))-j0(np.pi*lag)))
    assert max(invariance)<1e-10
    write('preflight_checks.json',dict(passed=True,complex_LMMSE_separable_vs_dense_max_absolute_error=lmmse_error,classical_vs_frozen_torch_max_absolute_error=errs,zero_spread_rank_max=int(ranks.max()),independent_generator_matches_old_formula=True,noise_convention_preserved=True,Bessel_integral_errors=quadrature,reflected_uniform_marginal_quadrature_max_error=float(max(invariance)),protected_files_unchanged=verify_protected()))
    print('All independent preflight checks passed.',flush=True)

if __name__=='__main__':main()
