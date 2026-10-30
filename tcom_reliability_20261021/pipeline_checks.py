"""Reserved smoke-only checks of sharing, validation routing and generator."""
import numpy as np
import torch
from . import study as s
from tcom_followups_20261017.common import write_json,sha
from tcom_followups_20261017.ber_detector import detect,hypotheses
from tcom_incumbent_guard_20261019 import risk_data
from tcom_pilot_em_20261018 import continuation


def run():
    torch.set_num_threads(1);est=s.engine();maximum=0.;checks=0;rule_checks=0
    for snr,ref,L in [(10.,5.,2),(30.,5.,4),(35.,5.,None),(60.,5.,2),(40.,12.,4),(60.,20.,None)]:
        a=s.make_anchor(s.SMOKE_SEED,[10,11],L,ref,truth=True,snrs=(snr,));obs=s.observed(a,snr)
        policy={'candidate_rule':s.RULES[0]};batch,diag=s.predictions(obs,policy,est)
        for i,name in enumerate(s.NAMES):
            full=s.estimate_method(obs,name,policy,est)
            err=float((batch[i]-full).abs().max());maximum=max(maximum,err);assert err<1e-10,(name,err)
            one=s.estimate_method(obs.batch(slice(0,1)),name,policy,est)
            err=float((batch[i,:1]-one).abs().max());maximum=max(maximum,err);assert err<1e-10,(name,err);checks+=1
        full=s.observed(a,snr,P=39);fit,hold=risk_data.fit_holdout(full)
        loss=continuation.nll(hold.z.numpy(),batch[0].numpy()@hold.s.numpy()+hold.b.numpy(),hold.sigma2.numpy()).mean((-2,-1));assert np.isfinite(loss).all()
        symbols,bits=hypotheses(3);probe=abs(a['g'][0]@symbols.T+a['b'][0,:,:1]);decoded=detect(probe,a['g'][0],a['b'][0,:,:1],1e-10);assert np.array_equal(decoded,bits)
        if ref==5 and snr>30:
            base=batch[3:6].numpy();fb=[];valid=[]
            for tail in (10,20):
                g,v,_=s.inference.fallback(obs,tail);fb.append(g.numpy());valid.append(v)
            fb=np.stack(fb);z,ss,b,v=[x.numpy() for x in obs.args()]
            residual=2*np.mean((z[None]-abs(base@ss[None]+b[None]))**2,axis=-1)/v[None,:,None]
            ln=lambda g:continuation.nll(z,g@ss+b,v).mean((-2,-1))
            cell=dict(score_mean=residual.mean(-1),score_max=residual.max(-1),fallback_valid=np.stack(valid),
                fit_nll_base=np.stack([ln(g) for g in base]),fit_nll_fallback=np.stack([ln(g) for g in fb]),
                heldout_nll_base=np.zeros((3,2)),heldout_nll_fallback=np.ones((2,2)),
                probe_errors_base=np.zeros((3,2,3),int),probe_errors_fallback=np.ones((2,2,3),int))
            for rule in s.RULES:
                losses,errors,triggers,accept=s.rule_arrays(cell,rule);tail=(10,20).index(rule['tail'])
                for seed_index in range(3):
                    g,d=s.inference.from_base(obs,torch.from_numpy(base[seed_index]),rule,diagnostics=True)
                    np.testing.assert_array_equal(triggers[seed_index],d['triggered'])
                    np.testing.assert_array_equal(accept[seed_index],d['accepted'])
                    expected=np.where(accept[seed_index,:,None,None],fb[tail],base[seed_index])
                    np.testing.assert_array_equal(g.numpy(),expected);rule_checks+=1
    independent=[s.make_anchor(s.SMOKE_SEED,[0,1],None,r,truth=True,snrs=(30.,40.)) for r in s.RSRS]
    assert len({int(a['physical_seed']) for a in independent})==3
    assert all(not np.array_equal(independent[0]['g'],x['g']) for x in independent[1:])
    assert all(not np.array_equal(independent[0]['paths'],x['paths']) for x in independent[1:])
    files=[s.REPO/'tcom_reliability_20261021'/name for name in ('study.py','inference.py','pipeline_checks.py','execution.py')]
    result=dict(passed=True,checks=checks,validation_gate_checks=rule_checks,max_error=maximum,
        reference_domain_seeds=[int(a['physical_seed']) for a in independent],source_hashes={str(p.relative_to(s.REPO)):sha(p) for p in files},smoke_seed=s.SMOKE_SEED)
    write_json(s.OUT/'pipeline_checks.json',result);print('Pipeline checks PASS',result,flush=True)


if __name__=='__main__':run()
