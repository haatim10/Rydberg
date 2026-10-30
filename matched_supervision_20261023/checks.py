"""Checks before fitting and independent replay after all fitting."""
import argparse,time
import numpy as np
import torch
from sahure.model import AdaptiveHankelEstimator,ModelConfig
from sahure.math import nll
from .common import *
from .models import build,ratio
from .training import parameter_sha,validate


def preflight():
    frozen();val,_=load_observations('val');sl=slice(0,4)
    args=contexts(val,sl,dtype=torch.float64)
    args=frontend(*args[1:5]);g,z,s,b,v,op=args
    proposed=build('proposed',777,torch.float64)
    canonical=AdaptiveHankelEstimator(ModelConfig()).double();canonical.load_state_dict(proposed.state_dict())
    error=float((proposed(*args)-canonical(z,s,b,v)).abs().max());assert error<1e-9,error
    forbidden=False
    try:load_labels('train','selfsupervised')
    except PermissionError:forbidden=True
    assert forbidden
    a={k:x[:32] for k,x in val.items()};target,_=load_labels('val','supervised');target=target[:32]
    records={}
    for arch in ARCHES:
        initial_hash=[]
        for mode in MODES:
            m=build(arch,777);initial_hash.append(parameter_sha(m));opt=torch.optim.Adam(m.parameters(),lr=LR)
            inputs=contexts(a);hz,hs,hb,hv=heldout(a);losses=[];times=[]
            for step in range(3):
                tick=time.monotonic();opt.zero_grad(set_to_none=True);pred=m(*inputs)
                loss=ratio(pred,target).mean() if mode=='supervised' else nll(hz,pred@hs+hb,hv).mean()
                assert torch.isfinite(loss);loss.backward()
                assert all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters())
                norm=torch.nn.utils.clip_grad_norm_(m.parameters(),1.,error_if_nonfinite=True);opt.step()
                losses.append(float(loss.detach()));times.append(time.monotonic()-tick)
            records[f'{arch}_{mode}']=dict(discarded_smoke_losses=losses,seconds_per_step=float(np.median(times)),
                parameters=sum(p.numel() for p in m.parameters()),finite_gradients=True)
        assert initial_hash[0]==initial_hash[1]
        records[arch+'_paired_initial_weights_sha256']=initial_hash[0]
    # A heldout perturbation must not affect the shared model inputs or initializer.
    changed={k:x.clone() for k,x in a.items()};changed['z'][...,20:29]+=7.
    for x,y in zip(contexts(a),contexts(changed)):assert torch.equal(x,y)
    # The likelihood omits only target/variance-dependent constants. Verify
    # candidate differences independently with scipy's full Rician density.
    from scipy.stats import rice
    rng=np.random.default_rng(2026102300);zz=rng.uniform(.1,8,(6,1));aa=rng.uniform(.1,8,(6,5));vv=1.3
    actual=nll(torch.tensor(zz),torch.tensor(aa,dtype=torch.complex128),torch.tensor(vv,dtype=torch.float64)).numpy()
    exact=-rice.logpdf(zz,b=aa/np.sqrt(vv/2),scale=np.sqrt(vv/2))
    np.testing.assert_allclose(actual-actual[:,[0]],exact-exact[:,[0]],atol=1e-11,rtol=1e-11)
    write(OUT/'preflight.json',dict(passed=True,proposed_canonical_max_error=error,
        selfsupervised_label_access_forbidden=forbidden,heldout_perturbation_leaves_inputs_bitwise_identical=True,
        shared_likelihood_scipy_check=True,records=records))
    print('All four-arm preflight checks passed.',records,flush=True)


@torch.no_grad()
def audit():
    frozen();val,_=load_observations('val');records=[]
    for arch in ARCHES:
        for seed in SEEDS:
            hashes=[]
            for mode in MODES:
                folder=OUT/'training'/f'{arch}_{mode}'/str(seed)
                state=torch.load(folder/'final.pt',map_location='cpu',weights_only=False)
                best=torch.load(folder/'best.pt',map_location='cpu',weights_only=False)
                assert state['epoch']==50 and len(state['history'])==50
                assert [e['epoch'] for e in state['history']]==list(range(1,51))
                assert state['true_channel_labels_loaded']==(mode=='supervised')
                if mode=='selfsupervised':assert not any('/labels/' in p for p in state['loaded_inputs'])
                hashes.append(state['initial_parameter_sha256'])
                m=build(arch,seed);m.load_state_dict(state['model']);score=validate(m,val)
                assert abs(score-state['history'][-1]['validation_heldout_nll'])<1e-9
                assert best['epoch']==state['best_epoch']
                m.load_state_dict(best['model']);best_score=validate(m,val)
                assert abs(best_score-state['best_score'])<1e-9
                x=contexts(val,slice(0,4));p32=m(*x)
                m=m.double();x64=tuple(t.to(torch.complex128 if t.is_complex() else torch.float64) for t in x)
                p64=m(*x64)
                rel=float(((p32-p64).abs().square().sum()/p64.abs().square().sum()).sqrt())
                assert rel<1e-3,rel
                records.append(dict(arch=arch,mode=mode,seed=seed,final_epoch=50,best_epoch=best['epoch'],
                    validation_replay_passed=True,relative_float32_float64_prediction_error=rel,
                    final_checkpoint_sha256=sha(folder/'final.pt'),best_checkpoint_sha256=sha(folder/'best.pt')))
            assert hashes[0]==hashes[1]
    write(OUT/'training_audit.json',dict(passed=True,records=records,paired_initial_weights_verified=True))
    print('All12 complete fits,validation scores,label separation and precision independently checked.',flush=True)


if __name__=='__main__':
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    p=argparse.ArgumentParser();p.add_argument('phase',choices=('preflight','audit'));a=p.parse_args();globals()[a.phase]()
