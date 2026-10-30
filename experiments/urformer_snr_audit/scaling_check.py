"""Distinguish an SNR shift from a consistent rescaling of all physical fields."""
import numpy as np
from run import ROOT,generate,classical,ratio,write

def main():
    with np.load(ROOT/'data/oracle.npz') as f:
        raw={k:f[k][:4].copy() for k in f.files}
    obs=generate(raw,20,5,'reproduction_ensemble')
    scale=1/np.sqrt(32)
    scaled=dict(obs,z=obs['z']*scale,b=obs['b']*scale,v=obs['v']*scale**2,y=obs['y']*scale)
    base=classical(obs,raw['init']);changed=classical(scaled,raw['init']*scale)
    errs={k:float(np.max(np.abs(changed[k]/scale-base[k]))) for k in base}
    for k in base:
        np.testing.assert_allclose(changed[k]/scale,base[k],rtol=1e-10,atol=1e-10)
        np.testing.assert_allclose(ratio(changed[k],raw['h']*scale),ratio(base[k],raw['h']),rtol=1e-11,atol=1e-11)
    write('scaling_check.json',dict(passed=True,scale=scale,cases=4,iterations=100,max_absolute_recovered_channel_difference=errs,
      meaning='Consistent scaling of H, B, W, H0 and noise variance preserves classical normalized error and SNR. This does not establish how the authors scale neural inputs or the effect on training.'))
    print('SNR-preserving scale check passed:',errs)

if __name__=='__main__':main()
