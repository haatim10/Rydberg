"""Post-run convention, precision, dispatch and checkpoint checks; no tuning."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='1'
import numpy as np,torch
from .run import OUT,old,load,ob,write,SEEDS,sha,frozen
from .urformer import U,spec,ModelConfig,NumericConfig,URformer,initial,validate
from tcom_pilot_em_20261018 import continuation
from tcom_pilot_em_20261018.training_data import load_set,paired_split
from tcom_pilot_em_20261018.training_protocol import CONFIG
from tcom_runtime_bypass_20261017.sparse_fast import spectral_initial
@torch.no_grad()
def run():
 spec();frozen();results=[];maximum_degenerate=0.;maximum_batch=0.;maximum_precision=0.
 for si,seed in enumerate(SEEDS):
  state=torch.load(U/'checkpoints'/f'seed_{seed}'/'best.pt',map_location='cpu',weights_only=False);assert state['protocol_sha256']==sha(U/'protocol.json')
  m64=URformer(32,3,ModelConfig(use_hankel=False,use_transformer=True),NumericConfig(dtype='float64')).double().eval();m64.load_state_dict(state['model'])
  m32=URformer(32,3,ModelConfig(use_hankel=False,use_transformer=True),NumericConfig(dtype='float32')).eval();m32.load_state_dict(state['model'])
  for name in ('random3to7_rsr5_snr10','random3to7_rsr12_snr10','random3to7_rsr20_snr10','random3to7_rsr20_snr45'):
   a=load(old.OUT/'test/cells'/f'{name}.npz');obs=ob(a).batch(slice(0,4));g0=torch.from_numpy(spectral_initial(obs.z.numpy(),obs.s.numpy(),obs.b.numpy()));p=m64(g0,*obs.args());saved=load(U/'test/cells'/f'{name}.npz')['estimates'][si,:4]
   batch=float(abs(p.numpy()-saved).max());maximum_batch=max(maximum_batch,batch);assert batch<1e-10
   one=m64(g0[:1],*obs.batch(slice(0,1)).args());assert float(abs(one-p[:1]).max())<1e-10
   small=Observations(*(x.to(torch.complex64 if x.is_complex() else torch.float32) for x in obs.args()));f32=m32(g0.to(torch.complex64),*small.args());prec=float(abs(f32.to(torch.complex128)-p).max());maximum_precision=max(maximum_precision,prec);assert prec<1e-3
   m64._set_test_mode(filter_override='exact_bessel',alpha=1.,disable_residual=True,disable_hankel=True);classical=m64(g0,*obs.args());expected=continuation.estimate(obs,g0,10);delta=float(abs(classical-expected).max());maximum_degenerate=max(maximum_degenerate,delta);assert delta<1e-10;m64._clear_test_mode()
  val,conditions=load_set('val');_,va,vh=paired_split(val,CONFIG['validation_mask_seed']);g=torch.from_numpy(np.load(U/'data/val_truth.npy'));score,hl=validate(m32,va,vh,g,conditions);assert abs(score-state['val_score'])<1e-12;assert abs(hl-state['heldout_nll'])<1e-10
  results.append(dict(seed=seed,selected_epoch=state['epoch'],checkpoint_sha256=sha(U/'checkpoints'/f'seed_{seed}'/'best.pt'),validation_score=score,heldout_nll=hl))
 result=dict(passed=True,maximum_batch_vs_saved_difference=maximum_batch,maximum_float32_vs_float64_channel_difference=maximum_precision,maximum_exact_EM_degeneration_difference=maximum_degenerate,validation_recomputed_exactly=True,selected_checkpoints=results,interpretation='Checks implementation conventions and precision only. No hyperparameters or weights changed; not evidence of URformer convergence.')
 write(U/'independent_checks.json',result);print(result,flush=True)
if __name__=='__main__':
 from sahure.data import Observations
 torch.set_num_threads(1);torch.set_num_interop_threads(1);run()
