"""Shared, frozen design and observation-only data access."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key]='1'
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/rydberg-matched-matplotlib')
import hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from .models import ARCHES,frontend

REPO=Path(__file__).resolve().parents[1]
CODE=Path(__file__).parent
OUT=REPO/'results/matched_supervision_20261023'
PROJECT=REPO.parents[1]
OLD=REPO/'results/urformer_reproduction_20261022'
SEEDS=(2026102301,2026102302,2026102303)
MODES=('supervised','selfsupervised')
SNRS=(-10.,-5.,0.,5.,10.,15.,20.)
PILOTS=(10,15,20,25,30,35)
GRID=tuple([(20,s) for s in SNRS]+[(p,5.) for p in PILOTS if p!=20])
EPOCHS=50;BATCH=32;LR=.001
TEST_MASTER=2026102340;DATA_MASTER=2026102341;TIMING_MASTER=2026102342


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(p,a):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(a,indent=2,allow_nan=False)+'\n');tmp.replace(p)


def checkpoint(p,a):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp.pt')
    torch.save(a,tmp);tmp.replace(p)


def cn(r,shape):return (r.standard_normal(shape)+1j*r.standard_normal(shape))/np.sqrt(2)


def pilots(p):
    return cn(np.random.default_rng(np.random.SeedSequence([2026102240,77])),(4,35))[:,:p]


def power(p):return float(32*np.square(abs(pilots(p))).sum()/p)


def freeze():
    assert not (OUT/'protocol.json').exists()
    parents=[OLD/'protocol.json',OLD/'completion.json',REPO/'results/tcom_reliability_20261021/protocol.json',
             REPO/'results/tcom_reliability_20261021/completion.json']
    sources=[*CODE.glob('*.py'),REPO/'sahure/model.py',REPO/'sahure/math.py',
             REPO/'urformer_reproduction_20261022/model.py',REPO/'tcom_followups_20261017/ber_detector.py']
    inputs=[OLD/'data'/f'{split}.npz' for split in ('train','val')]+[OLD/'data/manifest.json',*parents]
    design=dict(
        question='Architecture x supervision, with all physical/training/test conditions shared.',
        arms=[f'{a}_{m}' for a in ARCHES for m in MODES],
        physical=dict(M=32,K=4,rays=40,clusters_declared=4,rays_per_cluster=10,
            channel='Literal printed unnormalized positive-sign steering, sqrt(M/40), independent uniform ray angles and CN(0,1)gains. Same disclosed assumptions as previous reproduction.',
            reference_ratio_db=10,reference_definition='Expected total multiuser signal power; known reference/variance.',
            training_snr='Uniform nominal[-10,20]dB, original20k underlying channel/noise realizations',
            spacing='lambda/2',transduction='Known scalar divided out',pilot_bank='Same fixed Gaussian4x35 bank for every arm, first20columns are fitting pilots'),
        training=dict(cases=20000,validation_cases=2000,context=20,withheld=9,total_available_pilots=29,
            data='Identical29 noisy pilot measurements available to all four arms. First20 enter each model; last9 are separate measurement targets/validation, never model inputs.',
            initialization='Same spectral initializer and pilot-only unregularized LS operator for both architectures. Spectral initialization is computed only from the20context pilots.',
            stages=10,epochs=EPOCHS,batch=BATCH,lr=LR,optimizer='Adam defaults,weight_decay0',
            scheduler='CosineAnnealingLR,Tmax50',gradient_clip=1.,precision='float32/complex64',
            seeds=SEEDS,orders='Identical epoch permutation seed+20000+epoch for all four arms. Same architecture initial weights across supervision modes.',
            supervised_loss='Mean per-example ||G-H||²/||H||² on true-channel labels; labels used only in supervised workers.',
            selfsupervised_loss='Mean exact Rician NLL on the9withheld magnitude measurements. Observation-only worker never opens true-channel files.',
            selection='Primary: final50epoch weights,identical update counts. Secondary: minimum common heldout Rician validation NLL,epoch0eligible for every arm. No test-based selection.',
            augmentation='Fixed measurement population and fixed20/9split for all arms; no arm-specific augmentation,pretraining,normalization,hyperparameter search or warmstart.',
            comparison_limit='Matched examples/updates/settings,not equal training FLOPs or a proof of best achievable convergence. True labels are training targets,never inference inputs.'),
        test=dict(master=TEST_MASTER,cases=2000,grid=GRID,
            fresh='Fresh physical channels/noise independent of every earlier test; common channels/noise across SNR,pilot count,arms,seeds.',
            primary='P20SNRsweep; each model trained forP20. Pilot-count sweep is equally a transfer diagnostic for all arms.',
            calibration='Each testedP uses power(P) for known reference/noise; same physical inputs for all methods.',
            truth='Used after inference for NMSE/BER scoring only; no phase alignment or channel-dependent calibration.',
            inference='SamePyTorchfloat64backend,common spectral initialization,10learnedstages,no extra EM/filter/recovery. This is the complete ordinary-SNR adaptive core,not the high-SNR hybrid.',
            baselines=['GS100_common_spectral','EM10_common_spectral','EM100_common_spectral']),
        ber=dict(master=DATA_MASTER,scope='P20sevenSNRpoints',physical_channels=512,data_vectors_per_channel=32,
            bits_per_channel=256,detector='Identical exact Rician likelihood over all256jointQPSK hypotheses for every estimate; known-channel detector also reported.',
            coupling='Same independent data observations,bits,reference,variance for all arms and fits. Pilot/data noise independent.',
            calibration='Data uses same physical noise/reference as its pilot case; no channel-truth power normalization.'),
        runtime=dict(master=TIMING_MASTER,windows=2,anchors_per_window=32,grid=GRID,threads=1,
            precision='float64/complex128',backend='PyTorch for ALL learned/classical methods',
            included='Completebatch1call: conversion,common spectral initializer,pilot solve,and all model stages. Weights loaded in advance.',
            excludes='Data generation,model loading,and common data detector',
            order='Both windows after all training/accuracy/BER workers exit. Prespecified random method/case order,no retry or optimization to pass.',
            budget_ms=10.74,interpretation='Host-specificp90,not per-call deadline; record maxima and overruns.'),
        statistics='Paired channel-bootstrap2000draws,conditional on3fits; preserve SNR/method/fit pairing. Per-seed results retained; fits are not an ensemble.',
        publication='Published URformer curve excluded from the measured four-arm ranking; this tests our documented implementation,not a verified author-code reproduction.',
        source_hashes={str(p.relative_to(REPO)):sha(p) for p in sources},
        input_hashes={str(p.relative_to(REPO)):sha(p) for p in inputs},created_unix=time.time())
    write(OUT/'protocol.json',design);print('Matched protocol frozen',sha(OUT/'protocol.json'),flush=True)


def frozen(check_inputs=True):
    a=json.loads((OUT/'protocol.json').read_text())
    # Observation-only training workers verify code and their exported inputs.
    # The main runner verifies legacy raw-data archives, which also contain H.
    for field in (('source_hashes','input_hashes') if check_inputs else ('source_hashes',)):
        for n,h in a[field].items():assert sha(REPO/n)==h,n
    return a


def load_observations(split):
    base=OUT/'data'/split/'observations'
    files={n:base/f'{n}.npy' for n in ('z','v','initial')}
    return {n:torch.from_numpy(np.load(p).copy()) for n,p in files.items()},list(map(str,files.values()))


def load_labels(split,mode):
    if mode!='supervised':raise PermissionError('Channel labels are forbidden to self-supervised training.')
    p=OUT/'data'/split/'labels/h.npy'
    return torch.from_numpy(np.load(p).copy()),str(p)


def contexts(a,ids=None,dtype=torch.float32):
    if ids is None:ids=slice(None)
    cdtype=torch.complex128 if dtype==torch.float64 else torch.complex64
    z=a['z'][ids,...,:20].to(dtype);v=a['v'][ids].to(dtype);g=a['initial'][ids].to(cdtype)
    s=torch.tensor(pilots(20),dtype=cdtype)[None].expand(len(z),-1,-1)
    b=torch.full_like(z,np.sqrt(10*power(20))).to(cdtype)
    gram=s@s.mH;op=torch.linalg.solve(gram.mT,s.mH.mT).mT
    return g,z,s,b,v,op


def heldout(a,ids=None,dtype=torch.float32):
    if ids is None:ids=slice(None)
    cdtype=torch.complex128 if dtype==torch.float64 else torch.complex64
    z=a['z'][ids,...,20:29].to(dtype);v=a['v'][ids].to(dtype)
    s=torch.tensor(pilots(29)[:,20:29],dtype=cdtype)[None].expand(len(z),-1,-1)
    b=torch.full_like(z,np.sqrt(10*power(20))).to(cdtype)
    return z,s,b,v


def raw_worlds(master,split,count):
    hs=[];ws=[]
    for start in range(0,count,128):
        n=min(128,count-start)
        rr=lambda stream:np.random.default_rng(np.random.SeedSequence([master,split,start,stream]))
        angles=rr(1).uniform(-np.pi/2,np.pi/2,(n,4,4,10));gains=cn(rr(2),angles.shape)
        steering=np.exp(1j*np.pi*np.arange(32)[None,:,None,None,None]*np.sin(angles)[:,None])
        hs.append((np.sqrt(32/40)*np.sum(steering*gains[:,None],(-2,-1))).astype(np.complex64))
        ws.append(cn(rr(3),(n,32,35)).astype(np.complex64))
    return dict(h=np.concatenate(hs),w=np.concatenate(ws))


def test_observations(raw,p,snr):
    h=torch.from_numpy(raw['h']).to(torch.complex128);w=torch.from_numpy(raw['w'][...,:p]).to(torch.complex128)
    s=torch.tensor(pilots(p),dtype=torch.complex128)[None].expand(len(h),-1,-1)
    v=torch.full((len(h),),power(p)/10**(snr/10),dtype=torch.float64)
    b=torch.full((len(h),32,p),np.sqrt(10*power(p)),dtype=torch.complex128)
    z=(h@s+b+v[:,None,None].sqrt()*w).abs()
    return z,s,b,v
