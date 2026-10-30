"""Export observation-only training files, and fresh independent test worlds."""
import json,time
import numpy as np
import torch
from .common import *


def generate():
    frozen()
    for split,count in (('train',20000),('val',2000)):
        folder=OUT/'data'/split
        if (folder/'complete.json').exists():continue
        with np.load(OLD/'data'/f'{split}.npz') as a:
            h=a['h'].copy();w=a['w'][...,:29].copy();snr=a['snr'].copy()
        assert len(h)==count
        # Keep the physical noise/reference calibration of the20fittingpilots
        # while also generating9independent training-target observations.
        s=torch.tensor(pilots(29),dtype=torch.complex64)
        v=(power(20)/10**(torch.from_numpy(snr).double()/10)).float()
        b=torch.tensor(np.sqrt(10*power(20)),dtype=torch.float32)
        z=(torch.from_numpy(h)@s+b+v[:,None,None].sqrt()*torch.from_numpy(w)).abs()
        initials=[]
        for start in range(0,count,64):
            sl=slice(start,start+64);zz=z[sl,...,:20]
            ss=s[None,:,:20].expand(len(zz),-1,-1);bb=torch.full_like(zz,float(b)).to(torch.complex64)
            initials.append(frontend(zz,ss,bb,v[sl])[0])
        (folder/'observations').mkdir(parents=True,exist_ok=True);(folder/'labels').mkdir(exist_ok=True)
        for name,array in dict(z=z.numpy(),v=v.numpy(),initial=torch.cat(initials).numpy()).items():np.save(folder/'observations'/f'{name}.npy',array)
        np.save(folder/'labels/h.npy',h)
        write(folder/'complete.json',dict(count=count,files={str(p.relative_to(folder)):sha(p) for p in folder.rglob('*.npy')}))
        print('Shared training data exported',split,count,flush=True)
    path=OUT/'data/test_raw.npz'
    if not path.exists():
        raw=raw_worlds(TEST_MASTER,3,2000);np.savez(path,**raw)
    manifests={str(p.relative_to(OUT)):sha(p) for p in (OUT/'data').rglob('*') if p.is_file() and p.name!='manifest.json'}
    write(OUT/'data/manifest.json',dict(files=manifests,protocol_sha256=sha(OUT/'protocol.json')))
    frozen();print('Data generation complete; fresh2000channel final test frozen.',flush=True)


if __name__=='__main__':
    torch.set_num_threads(1);torch.set_num_interop_threads(1);generate()
