"""Independent, assumption-explicit reconstruction of arXiv:2509.12586v3."""
from __future__ import annotations
import argparse
import copy
import fcntl
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import tempfile
import time
import traceback
from dataclasses import replace

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
import paper2_long_training as engine
from paper2_original_channel import PaperChannelData, draw_channel, rng_for, cfg_for as channel_cfg
from rydberg_sim.forward import exact_forward
from trackD_urformer.dataset import collate
from trackD_urformer.transformer import tokenize, detokenize
from trackD_urformer.urformer import URformer
from trackD_urformer.torch_forward import gs_layer, em_gs_layer

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/paper2_reproduction'
COUNTS = {'train':20000, 'val':2000, 'test':14000}
SEEDS = {1:2026092101, 2:2026092102, 3:2026092103}
SNRS = (-10,-5,0,5,10,15,20)


class PaperEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln = nn.LayerNorm(64)
        self.attn = nn.MultiheadAttention(64,4,batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(64,256),nn.GELU(),nn.Linear(256,64))

    def forward(self,v):
        h = self.ln(v)
        a = v + self.attn(h,h,h,need_weights=False)[0]
        return a + self.ffn(a)


class PaperFormer(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Linear(64,64,bias=False)
        self.pos = nn.Parameter(torch.empty(4,64))
        nn.init.normal_(self.pos,std=.02)
        self.blocks = nn.Sequential(*(PaperEncoder() for _ in range(3)))
        self.output = nn.Sequential(nn.Linear(64,256),nn.GELU(),nn.Linear(256,64))

    def forward(self,g):
        return detokenize(self.output(self.blocks(self.embed(tokenize(g))+self.pos)),32,g.dtype)


def cfg_for(seed,hankel=False,budget=50):
    assert not hankel, 'Baseline reproduction first; no Hankel arm authorized in this queue.'
    c = channel_cfg(1,False,budget)
    return replace(c,model=replace(c.model,filter_input='kappa',gate_init='neutral'),
        train=replace(c.train,seed=SEEDS[seed],grad_clip=None),
        data=replace(c.data,n_train=COUNTS['train'],n_val=COUNTS['val'],n_test=COUNTS['test'],
          train_seed_range=(7000000,8000000),val_seed_range=(8000000,9000000),
          test_seed_range=(9000000,10000000)))


def build_model(cfg,arm=None):
    torch.manual_seed(cfg.train.seed)
    model = URformer(32,4,cfg.model,cfg.numeric)
    for layer in model.layers:
        layer.former = PaperFormer()
    return model, {'implementation':'independent_v3_equations_with_documented_assumptions',
        'params':sum(p.numel() for p in model.parameters()),
        'output_ffn':'64->256->64 GELU; default initialization',
        'extra_normalization':False,'gradient_clipping':False}


class ReproductionData(PaperChannelData):
    def __init__(self,split,count=None):
        super().__init__(split,'literal_independent',COUNTS[split] if count is None else count)
        self.start = {'train':7000000,'val':8000000,'test':9000000}[split]

    def raw(self,index):
        if not 0<=index<self.count:
            raise IndexError(index)
        # Full test uses the same 2,000 channels/noise draws at all seven SNRs.
        trial = self.start + (index % 2000 if self.split=='test' else index)
        snr = float(SNRS[index//2000]) if self.split=='test' else float(np.round(rng_for(trial,5).uniform(-10,20),3))
        sigma2 = self.signal_power/10**(snr/10)
        g = draw_channel(trial,self.mode)
        return trial,snr,sigma2,g,exact_forward(g,self.S,self.B,sigma2,rng_noise=rng_for(trial,6))


def train_step(model,opt,batch,cfg):
    opt.zero_grad(set_to_none=True)
    pred = model(batch['G0'],batch['Z'],batch['S'],batch['B'],batch['sigma2'])
    loss = engine.nmse_loss(pred,batch['G_true'])
    if not torch.isfinite(loss):
        raise RuntimeError('Non-finite loss in reconstruction; no automatic model change.')
    loss.backward()
    # Infinity threshold measures and checks gradients without modifying them.
    norm = nn.utils.clip_grad_norm_(model.parameters(),float('inf'),error_if_nonfinite=True)
    opt.step()
    return float(loss.detach()),float(norm)


def configure():
    engine.OUT=OUT
    engine.COUNTS=COUNTS
    engine.SEEDS=SEEDS
    engine.cfg_for=cfg_for
    engine.dataset=lambda cfg,split:ReproductionData(split)
    engine.build_model=build_model
    engine.train_step=train_step
    engine.summarize=summarize


def targets():
    source=OUT/'source/fig3.eps'
    eps=source.read_text()
    paths=re.findall(r'(63\.3223 [\d.]+ mo\n(?:[\d.]+ [\d.]+ li\n){6})',eps)
    paths=[p for p in paths if p.splitlines()[1].startswith('156.322 ')]
    assert len(paths)==5, f'Expected five Figure 3 curves, found {len(paths)}'
    curves={}
    for name,path in zip(('GS','EM-GS','CNN','Transformer','URformer'),paths):
        xy=np.array([[float(x) for x in line.split()[:2]] for line in path.splitlines()])
        np.testing.assert_allclose(xy[:,0],63.3223+93*np.arange(7),atol=.001)
        # Tick y=106.488 -> 0 dB, y=351.382 -> -30 dB (inspected on PDF).
        curves[name]=((106.488-xy[:,1])*30/(351.382-106.488)).tolist()
    record={'source':'https://arxiv.org/src/2509.12586v3','figure':3,'snr_db':SNRS,
        'method':'EPS polyline extraction; axis-calibrated, not author raw data',
        'conservative_digitization_tolerance_db':.02,'source_sha256':engine.hash_file(source),
        'curves_nmse_db':curves}
    engine.save_json(OUT/'figure3_targets.json',record)
    return record


def curve(records):
    result=[]
    for snr in SNRS:
        mask=records['snr_db']==snr
        if mask.any():
            e=records['numerator'][mask]/records['denominator'][mask]
            assert np.isfinite(e).all() and (e>0).all()
            result.append({'snr_db':snr,'n':int(mask.sum()),'nmse_db':float(10*np.log10(e.mean()))})
    return result


def summarize():
    target=json.loads((OUT/'figure3_targets.json').read_text())['curves_nmse_db']['URformer']
    results=[]
    for seed in SEEDS:
        folder=OUT/f'seed{seed}_U_50ep'
        if not (folder/'result.json').exists():
            continue
        for label in ('final','selected'):
            points=curve(np.load(folder/f'test_{label}.npz'))
            for p in points:
                p['paper_plot_nmse_db']=target[SNRS.index(p['snr_db'])]
                p['ours_minus_paper_db']=p['nmse_db']-p['paper_plot_nmse_db']
                p['needs_review_over_1db']=abs(p['ours_minus_paper_db'])>1
            results.append({'seed':seed,'checkpoint':label,'points':points})
    engine.save_json(OUT/'summary.json',{'label':'independent reconstruction, not certified reproduction',
        'primary_checkpoint':'final','results':results})


class Classical(nn.Module):
    def __init__(self,em):
        super().__init__()
        self.em=em

    def forward(self,g,z,s,b,sigma):
        for _ in range(100):
            g=em_gs_layer(g,z,s,b,sigma,eps=1e-8) if self.em else gs_layer(g,z,s,b,eps=1e-8)
        return g


def classical(data):
    if (OUT/'classical.json').exists():
        return
    target=json.loads((OUT/'figure3_targets.json').read_text())['curves_nmse_db']
    loader=DataLoader(data,batch_size=32,collate_fn=collate)
    result={}
    for name,em in (('GS',False),('EM-GS',True)):
        records=engine.test_model(Classical(em),loader)
        np.savez(OUT/f'classical_{name}.npz',**records)
        result[name]=curve(records)
        for p in result[name]:
            p['paper_plot_nmse_db']=target[name][SNRS.index(p['snr_db'])]
            p['ours_minus_paper_db']=p['nmse_db']-p['paper_plot_nmse_db']
    engine.save_json(OUT/'classical.json',result)


def provenance():
    files=[Path(__file__),ROOT/'scripts/paper2_original_channel.py',ROOT/'scripts/paper2_long_training.py',
           ROOT/'reports/paper2_reproduction_plan.md',OUT/'source/fig3.eps']
    files+=sorted((ROOT/'trackD_urformer').glob('*.py'))+sorted((ROOT/'rydberg_sim').glob('*.py'))
    return {str(p.relative_to(ROOT)):engine.hash_file(p) for p in files}


def smoke():
    global OUT,COUNTS
    torch.set_num_threads(1)
    configure()
    target=targets()
    cfg=cfg_for(1)
    ds=ReproductionData('train',1024)
    batches=[collate([ds[i] for i in range(j,j+32)]) for j in range(0,1024,32)]
    model,_=build_model(cfg)
    assert sum(isinstance(m,nn.LayerNorm) for m in model.modules())==30
    assert all(l.filter_net.filter_input=='kappa' and not l.use_hankel for l in model.layers)
    opt=torch.optim.Adam(model.parameters(),lr=.001)
    start=time.monotonic()
    losses=[train_step(model,opt,b,cfg)[0] for b in batches]
    elapsed=(time.monotonic()-start)/len(batches)
    state,ostate,rng=copy.deepcopy(model.state_dict()),copy.deepcopy(opt.state_dict()),torch.get_rng_state()
    train_step(model,opt,batches[0],cfg)
    expected=copy.deepcopy(model.state_dict())
    model.load_state_dict(state);opt.load_state_dict(ostate);torch.set_rng_state(rng)
    train_step(model,opt,batches[0],cfg)
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in expected.items())
    saved_out,saved_counts=OUT,COUNTS
    try:
        with tempfile.TemporaryDirectory(prefix='check-',dir=OUT) as tmp:
            OUT=Path(tmp);COUNTS={'train':64,'val':64,'test':64};configure()
            engine.save_json(OUT/'figure3_targets.json',target)
            engine.cache_data()
            data={s:engine.CachedData(s) for s in COUNTS}
            engine.train_one(1,False,2,data)
            ckpt=OUT/'seed1_U_2ep/checkpoint.pt'
            before=engine.hash_file(ckpt)
            engine.train_one(1,False,2,data)
            assert before==engine.hash_file(ckpt)
            (OUT/'seed1_U_2ep/result.json').unlink()
            engine.train_one(1,False,2,data)
            assert before==engine.hash_file(ckpt)
            classical(data['test'])
    finally:
        OUT,COUNTS=saved_out,saved_counts
        configure()
    record={'passed':True,'checks':['32 finite training updates','exact next-update resume',
        '30 LayerNorm modules (only attention pre-norm)','raw kappa, no Hankel',
        'tiny cached training/validation/test pipeline','complete-run skip and checkpoint recovery',
        'classical 100-step evaluation'], 'seconds_per_training_batch':elapsed,
        'losses':losses,'three_seed_training_hours_excluding_eval':3*50*625*elapsed/3600,
        'source_hashes':provenance()}
    engine.save_json(OUT/'smoke.json',record)
    print(json.dumps(record,indent=2),flush=True)


def run():
    configure()
    with (OUT/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        torch.set_num_threads(1);torch.set_num_interop_threads(1)
        os.nice(10)
        try:
            hashes=provenance()
            check=json.loads((OUT/'smoke.json').read_text())
            assert check['passed'] and check['source_hashes']==hashes
            manifest={'source_hashes':hashes,'python':sys.version,'torch':torch.__version__,
                'numpy':np.__version__,'platform':platform.platform(),'seeds':SEEDS,
                'counts':COUNTS,'config':cfg_for(1).to_dict()}
            if (OUT/'manifest.json').exists():
                assert json.loads((OUT/'manifest.json').read_text())==json.loads(json.dumps(manifest))
            else:
                engine.save_json(OUT/'manifest.json',manifest)
            engine.cache_data()
            data={s:engine.CachedData(s) for s in COUNTS}
            for seed in SEEDS:
                engine.train_one(seed,False,50,data)
            engine.save_json(OUT/'status.json',{'state':'classical_evaluation'})
            classical(data['test'])
            summarize()
            engine.save_json(OUT/'status.json',{'state':'complete','runs':3,'reproduction_verified':False,
                'next_step':'Review all curve discrepancies and assumptions before any Hankel comparison.'})
        except Exception:
            engine.save_json(OUT/'status.json',{'state':'failed','error':traceback.format_exc()})
            raise


def launch():
    # Serialize launchers and probe the worker lock to prevent duplicate queues.
    with (OUT/'launch.lock').open('a') as launch_lock:
        fcntl.flock(launch_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with (OUT/'run.lock').open('a') as worker_lock:
            fcntl.flock(worker_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (OUT/'pid.json').exists():
            pid=json.loads((OUT/'pid.json').read_text())['pid']
            try:os.kill(pid,0)
            except ProcessLookupError:pass
            else:raise RuntimeError(f'Process {pid} already alive; no duplicate launch.')
        with (OUT/'run.log').open('a') as log:
            child=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'run'],cwd=ROOT,
                stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True,
                env={**os.environ,'PYTHONPATH':str(ROOT),'OMP_NUM_THREADS':'1',
                     'OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'})
        engine.save_json(OUT/'pid.json',{'pid':child.pid,'started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())})
        print(f'Started independent reproduction attempt PID {child.pid}',flush=True)


if __name__=='__main__':
    os.chdir(ROOT);OUT.mkdir(parents=True,exist_ok=True)
    parser=argparse.ArgumentParser()
    parser.add_argument('command',choices=('smoke','run','launch','targets'))
    args=parser.parse_args()
    globals()[args.command]()
