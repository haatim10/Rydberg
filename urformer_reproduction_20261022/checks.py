"""Independent numerical checks after the frozen accuracy experiment."""
import time
import numpy as np
import torch
from scipy.special import i0e,i1e
from .model import URformer,Layer,ratios
from .study import OUT,SEEDS,sha,write,verify_protocol,data,observation,validation


@torch.no_grad()
def check():
    verify_protocol();a=data('val');a={k:v[:32] for k,v in a.items()}
    z,s,b,v,pinv=observation(a,double=True);initial=a['init'].to(torch.complex128)
    arr=[t.numpy() for t in (z,s,b,v,initial)];zz,ss,bb,vv,g=arr
    pinv_np=np.linalg.pinv(ss)
    layer=Layer().double()
    classical={}
    for name in ('gs','em'):
        ng=g.copy();tg=initial.clone()
        for _ in range(10):
            field=ng@ss+bb;direct=zz*field/np.maximum(abs(field),1e-8)
            k=2*zz*abs(field)/vv[:,None,None]
            r=1 if name=='gs' else i1e(k)/i0e(k)
            ng=(direct*r-bb)@pinv_np
            tg=layer(tg,z,s,b,v,pinv,name)
        error=float(np.max(abs(ng-tg.numpy())));assert error<1e-10
        classical[name]=error
    models={};val=data('val')
    for seed in SEEDS:
        checkpoint=OUT/'training'/str(seed)/'final.pt'
        state=torch.load(checkpoint,map_location='cpu',weights_only=False)
        m32=URformer().eval();m32.load_state_dict(state['model'])
        m64=URformer().double().eval();m64.load_state_dict(state['model'])
        z32,s32,b32,v32,p32=observation(a)
        p32out=m32(a['init'],z32,s32,b32,v32,p32)
        # Same rounded observations and inverse isolate inference arithmetic.
        p64out=m64(a['init'].to(torch.complex128),z32.double(),s32.to(torch.complex128),b32.double(),v32.double(),p32.to(torch.complex128))
        relative=float(((p64out-p32out).abs().square().sum()/p64out.abs().square().sum()).sqrt())
        assert relative<1e-3,relative
        batched=m64(initial,z,s,b,v,pinv)
        individual=torch.cat([m64(initial[i:i+1],z[i:i+1],s,b,v[i:i+1],pinv) for i in range(4)])
        batch_error=float((batched[:4]-individual).abs().max());assert batch_error<1e-9
        score=validation(m32,val);stored=state['history'][-1]['validation_nmse']
        assert abs(score-stored)<1e-10,(score,stored)
        history=state['history'];assert len(history)==50 and [x['epoch'] for x in history]==list(range(1,51))
        assert all(np.isfinite(x['train_nmse']) and np.isfinite(x['validation_nmse']) and np.isfinite(x['max_preclip_grad']) for x in history)
        models[str(seed)]=dict(final_checkpoint_sha256=sha(checkpoint),validation_recomputed=score,
            validation_stored=stored,batch_one_vs_batch32_max_error=batch_error,
            float32_vs_float64_relative_prediction_error=relative,epochs=50)
    # Exactly reproduce one stored final-test point from independent calls.
    test=data('test');small={k:v[:16] for k,v in test.items()}
    z,s,b,v,pinv=observation(small,20,5.,double=True)
    with np.load(OUT/'test/P20_snr5.npz') as stored:
        for seed in SEEDS:
            state=torch.load(OUT/'training'/str(seed)/'final.pt',map_location='cpu',weights_only=False)
            m=URformer().double().eval();m.load_state_dict(state['model'])
            pred=m(small['init'].to(torch.complex128),z,s,b,v,pinv)
            r=ratios(pred,small['h'].to(torch.complex128)).numpy()
            error=float(abs(r-stored[f'URformer_{seed}_final'][:16]).max());assert error<1e-10
            models[str(seed)]['saved_test_ratio_replay_max_error']=error
    verify_protocol();write(OUT/'independent_checks.json',dict(passed=True,classical_numpy_vs_torch_10steps=classical,models=models,finished_unix=time.time()))
    print('All independent numerical checks passed.',flush=True)


if __name__=='__main__':
    torch.set_num_threads(1);torch.set_num_interop_threads(1);check()
