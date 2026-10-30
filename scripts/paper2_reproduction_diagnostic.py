"""Isolated fixed-SNR versus mixed-SNR diagnostic; validation only, no Hankel."""
from __future__ import annotations
import argparse
from dataclasses import replace
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import numpy as np
import torch
from torch.utils.data import DataLoader
import paper2_long_training as engine
import paper2_reproduction as reproduction
from paper2_original_channel import PaperChannelData, draw_channel, rng_for
from rydberg_sim.forward import exact_forward
from trackD_urformer.dataset import collate
from trackD_urformer.stage2 import validation_per_trial, select_epoch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'results/paper2_reproduction_diagnostic'
ARMS = ('mixed', 'fixed5')
COUNTS = {'train': 20000, 'val': 2000}
SEED = 2026092201
EPOCHS = 10
SCHEDULE_EPOCHS = 50


def cfg():
    c = reproduction.cfg_for(1)
    return replace(c, train=replace(c.train, seed=SEED, epochs=EPOCHS),
                   data=replace(c.data, n_train=COUNTS['train'], n_val=COUNTS['val'],
                                train_seed_range=(10000000,11000000),
                                val_seed_range=(11000000,12000000)))


class DiagnosticData(PaperChannelData):
    def __init__(self, split, arm, count=None):
        if split not in COUNTS or arm not in ARMS:
            raise ValueError('Only train/validation and registered arms are allowed')
        super().__init__(split, 'literal_independent', COUNTS[split] if count is None else count)
        self.start = {'train':10000000, 'val':11000000}[split]
        self.arm = arm

    def raw(self, index):
        if not 0 <= index < self.count:
            raise IndexError(index)
        trial = self.start+index
        snr = (5. if self.split == 'val' or self.arm == 'fixed5' else
               float(np.round(rng_for(trial,5).uniform(-10,20),3)))
        sigma2 = self.signal_power/10**(snr/10)
        g = draw_channel(trial, self.mode)
        return trial, snr, sigma2, g, exact_forward(g,self.S,self.B,sigma2,rng_noise=rng_for(trial,6))


def provenance():
    files = [Path(__file__).resolve(), ROOT/'scripts/paper2_reproduction.py',
             ROOT/'scripts/paper2_original_channel.py', ROOT/'scripts/paper2_long_training.py']
    files += sorted((ROOT/'trackD_urformer').glob('*.py'))
    files += sorted((ROOT/'rydberg_sim').glob('*.py'))
    return {str(p.relative_to(ROOT)):engine.hash_file(p) for p in files}


def configure_cache(arm):
    engine.OUT = OUT/arm
    engine.COUNTS = COUNTS
    engine.cfg_for = lambda *args: cfg()
    engine.dataset = lambda config,split: DiagnosticData(split,arm)
    engine.OUT.mkdir(parents=True,exist_ok=True)


def weights_hash(model):
    h=hashlib.sha256()
    for key,value in model.state_dict().items():
        h.update(key.encode());h.update(value.detach().numpy().tobytes())
    return h.hexdigest()


def loader(data, epoch):
    # Dedicated generator isolates batch order from model/validation RNG use.
    generator=torch.Generator().manual_seed(SEED+epoch)
    return DataLoader(data,batch_size=32,shuffle=True,collate_fn=collate,generator=generator)


def train_arm(arm):
    folder=OUT/arm
    if (folder/'result.json').exists():
        return
    configure_cache(arm)
    engine.cache_data()
    data={split:engine.CachedData(split) for split in COUNTS}
    config=cfg()
    model,meta=reproduction.build_model(config)
    initial_hash=weights_hash(model)
    optimizer=torch.optim.Adam(model.parameters(),lr=config.train.lr)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=SCHEDULE_EPOCHS)
    val_loader=DataLoader(data['val'],batch_size=32,shuffle=False,collate_fn=collate)
    checkpoint=folder/'checkpoint.pt'
    start,history,validations=0,[],[]
    if checkpoint.exists():
        saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
        assert saved['arm']==arm and saved['config']==config.to_dict()
        assert saved['initial_hash']==initial_hash
        model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer'])
        scheduler.load_state_dict(saved['scheduler'])
        start=saved['epoch']+1;history=saved['history'];validations=saved['validations']
        torch.set_rng_state(saved['torch_rng'])
    for epoch in range(start,EPOCHS):
        model.train();begin=time.monotonic();total=0.;order=hashlib.sha256()
        batches=loader(data['train'],epoch)
        for index,batch in enumerate(batches):
            order.update(batch['trial'].numpy().tobytes())
            loss,gradient=reproduction.train_step(model,optimizer,batch,config)
            total+=loss
            if (index+1)%100==0:
                engine.save_json(OUT/'status.json',{'state':'training','arm':arm,'epoch':epoch+1,
                    'epochs_per_arm':EPOCHS,'batch':index+1,'batches':len(batches),
                    'epoch_elapsed_seconds':time.monotonic()-begin})
        scheduler.step()
        values=validation_per_trial(model,val_loader,config)
        if not np.isfinite(values).all() or (values<=0).any():
            raise RuntimeError('Invalid validation error; no silent numerical fix')
        validations.append(values)
        record={'epoch':epoch+1,'train_loss':total/len(batches),
                'validation_5db_nmse_db':float(10*np.log10(values.mean())),
                'lr':scheduler.get_last_lr()[0],'data_order_sha256':order.hexdigest(),
                'seconds':time.monotonic()-begin}
        history.append(record)
        engine.save_torch(folder/f'epoch_{epoch+1:03d}.pt',{'model':model.state_dict()})
        engine.save_torch(checkpoint,{'model':model.state_dict(),'optimizer':optimizer.state_dict(),
            'scheduler':scheduler.state_dict(),'epoch':epoch,'history':history,'validations':validations,
            'config':config.to_dict(),'arm':arm,'torch_rng':torch.get_rng_state(),'initial_hash':initial_hash})
        engine.save_json(folder/'history.json',history)
        print(arm,json.dumps(record),flush=True)
    curve=np.array([h['validation_5db_nmse_db'] for h in history])
    chosen,selection=select_epoch(curve,validations[int(curve.argmin())],'one_se')
    np.savez(folder/'validation_errors.npz',errors=np.asarray(validations),
             trial=data['val'].arrays['trial'],snr_db=data['val'].arrays['snr_db'])
    engine.save_json(folder/'result.json',{'arm':arm,'epochs':EPOCHS,'scheduler_duration':SCHEDULE_EPOCHS,
        'initial_hash':initial_hash,'chosen_epoch_1based':chosen+1,'selection':selection,
        'history':history,'model_meta':meta,'config':config.to_dict(),
        'warning':'One seed; validation diagnostic only; no published-reproduction claim or test-set scoring.'})


def summary():
    records=[json.loads((OUT/arm/'result.json').read_text()) for arm in ARMS]
    assert records[0]['initial_hash']==records[1]['initial_hash']
    assert [h['data_order_sha256'] for h in records[0]['history']]==[h['data_order_sha256'] for h in records[1]['history']]
    files=[np.load(OUT/arm/'validation_errors.npz') for arm in ARMS]
    np.testing.assert_array_equal(files[0]['trial'],files[1]['trial'])
    np.testing.assert_array_equal(files[0]['snr_db'],files[1]['snr_db'])
    for key in ('G_true','G0','S','B','trial'):
        assert engine.hash_file(OUT/ARMS[0]/'data/train'/f'{key}.npy')==engine.hash_file(OUT/ARMS[1]/'data/train'/f'{key}.npy')
    for key in ('G_true','G0','S','B','trial','Z','sigma2','snr_db'):
        assert engine.hash_file(OUT/ARMS[0]/'data/val'/f'{key}.npy')==engine.hash_file(OUT/ARMS[1]/'data/val'/f'{key}.npy')
    result={'matched_initial_weights':True,'matched_orders':True,'matched_latent_train':True,
            'identical_validation':True,'checkpoint_results':{}}
    for label in ('final','selected'):
        errors=[a['errors'][-1 if label=='final' else r['chosen_epoch_1based']-1] for a,r in zip(files,records)]
        result['checkpoint_results'][label]={'mixed_db':float(10*np.log10(errors[0].mean())),
              'fixed5_db':float(10*np.log10(errors[1].mean())),
              'fixed5_gain_db':float(10*np.log10(errors[0].mean()/errors[1].mean()))}
    result['limitations']='Validation-only single-seed 10-epoch prefix of a 50-epoch schedule. Not a reproduction, convergence proof, or independent test result.'
    engine.save_json(OUT/'summary.json',result)


def run():
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            torch.set_num_threads(1);torch.set_num_interop_threads(1);os.nice(10)
            manifest={'source_hashes':provenance(),'counts':COUNTS,'arms':ARMS,'seed':SEED,
                      'epochs':EPOCHS,'schedule_epochs':SCHEDULE_EPOCHS,'target_snr_db':5,
                      'torch':torch.__version__,'python':sys.version,'config':cfg().to_dict()}
            if (OUT/'manifest.json').exists():
                assert json.loads((OUT/'manifest.json').read_text())==json.loads(json.dumps(manifest))
            else:
                engine.save_json(OUT/'manifest.json',manifest)
            for arm in ARMS:
                engine.save_json(OUT/'status.json',{'state':'preparing','arm':arm})
                train_arm(arm)
            assert provenance()==manifest['source_hashes']
            summary()
            engine.save_json(OUT/'status.json',{'state':'complete','runs':2,'validation_only':True})
        except Exception:
            engine.save_json(OUT/'status.json',{'state':'failed','error':traceback.format_exc()})
            raise


def launch():
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'launch.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with (OUT/'run.lock').open('a') as worker:
            fcntl.flock(worker,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (OUT/'pid.json').exists():
            pid=json.loads((OUT/'pid.json').read_text())['pid']
            try:os.kill(pid,0)
            except ProcessLookupError:pass
            else:raise RuntimeError(f'Existing process {pid}; refusing duplicate')
        if (OUT/'status.json').exists() and json.loads((OUT/'status.json').read_text()).get('state')=='complete':
            raise RuntimeError('Diagnostic is already complete')
        with (OUT/'run.log').open('a') as log:
            child=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'run'],
                cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True,
                env={**os.environ,'PYTHONPATH':str(ROOT),'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'})
        engine.save_json(OUT/'pid.json',{'pid':child.pid,'started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())})
        print(f'Started baseline diagnostic PID {child.pid}')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('run','launch'))
    arguments=parser.parse_args()
    {'run':run,'launch':launch}[arguments.command]()
