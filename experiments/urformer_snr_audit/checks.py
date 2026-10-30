import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='1'
import sys,json,importlib.util
from pathlib import Path
import numpy as np
import torch
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE));from run import generate,world,classical,pilots,power,write,ratio
spec=importlib.util.spec_from_file_location('frozen_urformer',HERE.parents[1]/'urformer_reproduction_20261022/model.py')
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
torch.set_num_threads(1);torch.set_num_interop_threads(1)
raw={k:v[:4] for k,v in world('oracle').items()};obs=generate(raw,20,5,'reproduction_ensemble')
ours=classical(obs,raw['init']);layer=mod.Layer().double();errors={}
with torch.no_grad():
 for name,mode in [('GS100','gs'),('EM100','em')]:
  h=torch.from_numpy(raw['init'].copy());s=torch.from_numpy(obs['s']);op=torch.linalg.pinv(s)
  for _ in range(100):h=layer(h,torch.from_numpy(obs['z']),s,torch.tensor(obs['b']),torch.from_numpy(obs['v']),op,classical=mode)
  error=float(np.max(np.abs(h.numpy()-ours[name])));assert error<1e-8;errors[name]=error
v=generate(raw,20,5,'paper_vector');assert np.array_equal(v['z'],obs['z'])
a=generate(raw,20,5,'ours');b=generate(raw,20,5,'paper_shifted');np.testing.assert_allclose(a['v']/b['v'],32,rtol=1e-14)
c=generate(raw,20,5,'array_signal_scalar_noise');np.testing.assert_allclose(c['v']/obs['v'],32,rtol=1e-14)
rng=np.random.default_rng(2026102709);x=(rng.normal(size=(10000,4,20))+1j*rng.normal(size=(10000,4,20)))/np.sqrt(2)
tr=np.trace(np.linalg.inv(x@x.conj().transpose(0,2,1)),axis1=-2,axis2=-1).real
assert abs(tr.mean()-.25)<5*tr.std()/100
write('independent_checks.json',dict(passed=True,classical_numpy_vs_frozen_torch_max_error=errors,
     paper_vector_bitwise_equal_to_existing=True,shift_variance_factor=32,scalar_noise_variance_factor=32,
     random_pilot_inverse_gram_trace_mean=float(tr.mean()),theory_inverse_gram_trace=.25,
     monte_carlo_standard_error=float(tr.std()/100),fixed_pilot_inverse_gram_trace=float(np.trace(np.linalg.inv(pilots(20)@pilots(20).conj().T)).real)))
print('Independent 100-step numerical and SNR-equivalence checks passed.',flush=True)
