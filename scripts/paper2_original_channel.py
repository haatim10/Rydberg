"""Isolated, resumable original-paper-style channel study.

Reuses the verified training engine by configuring its module-local globals.
Never edits that engine or writes to its original results directory.
"""
from __future__ import annotations
import argparse
import copy
import fcntl
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
import traceback
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from scipy.stats import t
from torch.utils.data import Dataset

import paper2_long_training as engine
from trackD_urformer.baselines import make_initial_G
from trackD_urformer.config import TrackDConfig
from trackD_urformer.dataset import make_fixed_pilots, collate
from trackD_urformer.hankel import project_G
from rydberg_sim.forward import exact_forward

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/paper2_original_channel'
MODES = ('literal_independent', 'clustered_5deg')
SEEDS = {1: 2026092001, 2: 2026092002, 3: 2026092003}
COUNTS = {'train': 20000, 'val': 2000, 'test': 2000}
BUDGET = 50
MASTER = 2026091940
ACTIVE_MODE = MODES[0]


def cfg_for(seed, hankel, budget=50):
    c = TrackDConfig()
    return replace(c,
        system=replace(c.system,K=4,N=32,P=20,beta=32.,rsr_db=10+10*np.log10(4),master_seed=MASTER),
        data=replace(c.data,n_train=20000,n_val=2000,n_test=2000,
                     train_seed_range=(4000000,5000000),
                     val_seed_range=(5000000,6000000),test_seed_range=(6000000,7000000)),
        train=replace(c.train,seed=SEEDS[seed],init='random',epochs=budget),
        model=replace(c.model,use_hankel=hankel,hankel_rank=7,hankel_mode='fixed',hankel_gate='none'))


def rng_for(trial, stream):
    return np.random.default_rng(np.random.SeedSequence([MASTER,int(trial),int(stream)]))


def draw_channel(trial, mode):
    """Eq. (1)/(2) as printed: positive sign, unnormalized steering, power M."""
    if mode not in MODES:
        raise ValueError(mode)
    if mode == MODES[0]:
        angles = rng_for(trial,1).uniform(-np.pi/2,np.pi/2,(4,4,10))
    else:
        centers = rng_for(trial,2).uniform(-np.pi/2,np.pi/2,(4,4,1))
        offsets = np.deg2rad(rng_for(trial,3).uniform(-5,5,(4,4,10)))
        angles = np.clip(centers+offsets,-np.pi/2,np.pi/2)
    gains_rng = rng_for(trial,4)
    gains = (gains_rng.standard_normal((4,4,10))+1j*gains_rng.standard_normal((4,4,10)))/np.sqrt(2)
    steering = np.exp(1j*np.pi*np.arange(32)[:,None,None,None]*np.sin(angles)[None,:,:,:])
    return np.sqrt(32/40)*np.sum(steering*gains[None,:,:,:],axis=(2,3))


class PaperChannelData(Dataset):
    def __init__(self, split, mode, count=None):
        self.mode,self.split = mode,split
        self.count = COUNTS[split] if count is None else count
        self.start = {'train':4000000,'val':5000000,'test':6000000}[split]
        self.S = make_fixed_pilots(4,20,777000001)
        self.signal_power = 32*float(np.square(np.abs(self.S)).sum())/20
        self.B = np.full((32,20), np.sqrt(10*self.signal_power),dtype=np.complex128)
        # The reused cache engine clears this after each generated sample.
        self._g0_cache = {}

    def __len__(self):
        return self.count

    def raw(self,index):
        if not 0 <= index < self.count:
            raise IndexError(index)
        trial = self.start+index
        snr = float(np.round(rng_for(trial,5).uniform(-10,20),3))
        sigma2 = self.signal_power/10**(snr/10)
        g = draw_channel(trial,self.mode)
        ex = exact_forward(g,self.S,self.B,sigma2,rng_noise=rng_for(trial,6))
        return trial,snr,sigma2,g,ex

    def __getitem__(self,index):
        trial,snr,sigma2,g,ex = self.raw(index)
        g0 = make_initial_G('random',S=self.S,Z=ex.Z,B=self.B,seed=trial)
        return {'G_true':torch.tensor(g,dtype=torch.complex64),
                'G0':torch.tensor(g0,dtype=torch.complex64),
                'Z':torch.tensor(np.array(ex.Z),dtype=torch.float32),
                'S':torch.tensor(self.S,dtype=torch.complex64),
                'B':torch.tensor(self.B,dtype=torch.complex64),
                'sigma2':torch.tensor(sigma2,dtype=torch.float32),
                'snr_db':torch.tensor(snr,dtype=torch.float32),
                'trial':torch.tensor(trial,dtype=torch.int64)}


def configure(mode):
    global ACTIVE_MODE
    ACTIVE_MODE = mode
    engine.OUT = OUT/mode
    engine.OUT.mkdir(parents=True,exist_ok=True)
    engine.COUNTS = COUNTS
    engine.SEEDS = SEEDS
    engine.BUDGETS = (BUDGET,)
    engine.cfg_for = cfg_for
    engine.dataset = lambda cfg,split: PaperChannelData(split,mode)
    engine.summarize = lambda: summarize_mode(mode)


def summarize_mode(mode):
    result = {}
    for label in ('selected','final'):
        pairs = []
        for seed in SEEDS:
            folders = [OUT/mode/f'seed{seed}_{arm}_50ep' for arm in ('U','H')]
            if not all((p/'result.json').exists() for p in folders):
                continue
            records = [json.loads((p/'result.json').read_text()) for p in folders]
            assert [r['data_order_sha256'] for r in records[0]['history']] == [r['data_order_sha256'] for r in records[1]['history']]
            u,h = [np.load(p/f'test_{label}.npz') for p in folders]
            for key in ('trial','snr_db','denominator'):
                np.testing.assert_array_equal(u[key],h[key])
            masks = {'all':np.ones(len(u['trial']),dtype=bool),'snr_ge5':u['snr_db']>=5}
            masks.update({f'[{lo},{lo+5})':(u['snr_db']>=lo)&(u['snr_db']<lo+5) for lo in range(-10,20,5)})
            bins = {}
            for group,mask in masks.items():
                a,b,d = u['numerator'][mask],h['numerator'][mask],u['denominator'][mask]
                assert np.isfinite(a).all() and np.isfinite(b).all() and (a>0).all() and (b>0).all()
                bins[group] = {'n':int(mask.sum()),'paired_median_db':float(np.median(10*np.log10(a/b))),
                    'raw_error_sum_gain_db':float(10*np.log10(a.sum()/b.sum())),
                    'normalized_error_sum_gain_db':float(10*np.log10((a/d).sum()/(b/d).sum())),
                    'U_nmse_db':float(10*np.log10(np.mean(a/d))),
                    'H_nmse_db':float(10*np.log10(np.mean(b/d)))}
            pairs.append({'seed':seed,'bins':bins})
        record = {'pairs':pairs,'n_seeds':len(pairs)}
        if pairs:
            values = np.array([p['bins']['snr_ge5']['paired_median_db'] for p in pairs])
            record['mean_gain_db'] = float(values.mean())
            if len(values)>1:
                sd = float(values.std(ddof=1)); margin = float(t.ppf(.975,len(values)-1)*sd/np.sqrt(len(values)))
                record.update(sd_db=sd,nominal_seed_ci95=[float(values.mean()-margin),float(values.mean()+margin)])
        result[label] = record
    engine.save_json(OUT/mode/'summary.json',result)


def smoke():
    torch.set_num_threads(1)
    OUT.mkdir(parents=True,exist_ok=True)
    checks = {'modes':{},'timing':{}}
    for mode in MODES:
        ds = PaperChannelData('train',mode,count=1200)
        energies=[]; powers=[]; noises=[]; retained=[]; distortion=[]
        for i in range(1000):
            trial,snr,sigma2,g,ex = ds.raw(i)
            energies.append(float(np.mean(abs(g)**2)))
            powers.append(float(np.mean(abs(g@ds.S)**2)))
            noises.append(float(np.mean(abs(ex.W)**2)/sigma2))
            if i<100:
                gt=torch.tensor(g[None],dtype=torch.complex128)
                projected=project_G(gt,rank=7).numpy()[0]
                distortion.append(float(np.sum(abs(projected-g)**2)/np.sum(abs(g)**2)))
                matrices=np.lib.stride_tricks.sliding_window_view(g.T,17,axis=1)
                sv=np.linalg.svd(matrices,compute_uv=False)
                retained.append(float(np.mean(np.sum(sv[:,7:]**2,axis=1)/np.sum(sv**2,axis=1))))
        assert abs(np.mean(energies)/32-1)<.06
        assert abs(np.mean(powers)/ds.signal_power-1)<.06
        assert abs(np.mean(noises)-1)<.01
        assert np.linalg.matrix_rank(ds.S)==4
        for i in (0,31,999):
            assert all(torch.equal(v,ds[i][k]) for k,v in ds[i].items())
        checks['modes'][mode]={'expected_channel_power':32.,'measured_channel_power':float(np.mean(energies)),
            'signal_power_for_calibration':ds.signal_power,'measured_signal_power':float(np.mean(powers)),
            'measured_RSR_db':float(10*np.log10(np.mean(abs(ds.B)**2)/np.mean(powers))),
            'mean_realized_noise_over_sigma2':float(np.mean(noises)),
            'rank7_noiseless_projection_relative_error_mean':float(np.mean(distortion)),
            'rank7_hankel_tail_energy_fraction_mean':float(np.mean(retained)),
            'deterministic_generator':True}
        batches=[collate([ds[j] for j in range(i,i+32)]) for i in range(0,640,32)]
        models=[engine.build_model(cfg_for(1,h),'arm1b_full_random')[0] for h in (False,True)]
        assert all(torch.equal(a,b) for a,b in zip(models[0].parameters(),models[1].parameters()))
        for arm,model in zip(('U','H'),models):
            cfg=cfg_for(1,arm=='H'); opt=torch.optim.Adam(model.parameters(),lr=.001)
            losses=[]
            for batch in batches[:3]:
                engine.train_step(model,opt,batch,cfg)
            start=time.monotonic()
            for batch in batches[3:]:
                losses.append(engine.train_step(model,opt,batch,cfg)[0])
            seconds=(time.monotonic()-start)/(len(batches)-3)
            # Exact next-update resume, including RNG and optimizer state.
            state,osave=copy.deepcopy(model.state_dict()),copy.deepcopy(opt.state_dict())
            rng=torch.get_rng_state()
            engine.train_step(model,opt,batches[0],cfg)
            expected=copy.deepcopy(model.state_dict())
            model.load_state_dict(state); opt.load_state_dict(osave); torch.set_rng_state(rng)
            engine.train_step(model,opt,batches[0],cfg)
            assert all(torch.equal(v,model.state_dict()[k]) for k,v in expected.items())
            start=time.monotonic()
            engine.test_model(model,batches[:3])
            val_seconds=(time.monotonic()-start)/3
            checks['timing'][f'{mode}_{arm}']={'training_seconds_per_batch':seconds,
                'validation_seconds_per_batch':val_seconds,'finite_losses':losses,
                'exact_next_update_resume':True,'matched_initial_parameters':True}
        print(mode,'calibration, matching and resume checks passed',flush=True)
    hours=sum(3*50*(625*x['training_seconds_per_batch']+63*x['validation_seconds_per_batch'])
              for x in checks['timing'].values())/3600
    checks.update(passed=True,estimated_training_validation_hours=hours,
                  conservative_hours_range=[hours*1.2,hours*1.8])
    engine.save_json(OUT/'smoke.json',checks)
    print(json.dumps(checks,indent=2),flush=True)


def provenance():
    files=[Path(__file__),ROOT/'scripts/paper2_long_training.py',ROOT/'reports/paper2_original_channel_plan.md']
    files += sorted((ROOT/'trackD_urformer').glob('*.py'))+sorted((ROOT/'rydberg_sim').glob('*.py'))
    return {str(p.relative_to(ROOT)):engine.hash_file(p) for p in files}


def pipeline_check():
    """Tiny isolated end-to-end run; temporary artifacts are not study results."""
    global OUT, COUNTS
    torch.set_num_threads(1)
    saved_out,saved_counts = OUT,COUNTS
    OUT.mkdir(parents=True,exist_ok=True)
    try:
        with tempfile.TemporaryDirectory(prefix='pipeline-check-',dir=OUT) as folder:
            OUT=Path(folder); COUNTS={'train':64,'val':64,'test':64}
            for mode in MODES:
                configure(mode)
                engine.cache_data()
                data={split:engine.CachedData(split) for split in COUNTS}
                histories=[]
                for hankel in (False,True):
                    engine.train_one(1,hankel,2,data)
                    target=OUT/mode/f'seed1_{"H" if hankel else "U"}_2ep'
                    result=json.loads((target/'result.json').read_text())
                    assert len(result['history'])==2
                    histories.append(result['history'])
                    # Re-enter from final checkpoint to recover a missing result.
                    before=engine.hash_file(target/'checkpoint.pt')
                    (target/'result.json').unlink()
                    engine.train_one(1,hankel,2,data)
                    assert before==engine.hash_file(target/'checkpoint.pt')
                    assert json.loads((target/'result.json').read_text())==result
                    for label in ('selected','final'):
                        z=np.load(target/f'test_{label}.npz')
                        assert len(z['numerator'])==64 and np.isfinite(z['numerator']).all()
                assert [x['data_order_sha256'] for x in histories[0]]==[x['data_order_sha256'] for x in histories[1]]
    finally:
        OUT,COUNTS=saved_out,saved_counts
        configure(MODES[0])
    engine.save_json(OUT/'pipeline_check.json',{'passed':True,'both_modes_tested':True,
        'cached_data_equal_generator':True,'matched_orders':True,
        'checkpoint_recovery':True,'selected_and_final_scoring':True})
    print('Both channel pipelines passed.',flush=True)


def run():
    OUT.mkdir(parents=True,exist_ok=True)
    lock=(OUT/'run.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    try:
        assert json.loads((OUT/'smoke.json').read_text())['passed']
        assert json.loads((OUT/'pipeline_check.json').read_text())['passed']
        manifest={'source_hashes':provenance(),'torch':torch.__version__,'numpy':np.__version__,
                  'python':sys.version,'platform':platform.platform(),'modes':MODES,'seeds':SEEDS,
                  'budget':BUDGET,'counts':COUNTS,'config':cfg_for(1,False).to_dict()}
        manifest=json.loads(json.dumps(manifest))
        path=OUT/'manifest.json'
        if path.exists():
            assert json.loads(path.read_text())==manifest,'Resume provenance differs; refusing.'
        else:
            engine.save_json(path,manifest)
        for mode in MODES:
            configure(mode)
            engine.save_json(OUT/'status.json',{'state':'generating_data','mode':mode})
            engine.cache_data()
        for seed in SEEDS:
            for mode in MODES:
                configure(mode)
                data={split:engine.CachedData(split) for split in COUNTS}
                for hankel in (False,True):
                    engine.save_json(OUT/'status.json',{'state':'running','mode':mode,'seed':seed,
                        'arm':'H' if hankel else 'U','active_status':str(engine.OUT/'status.json'),
                        'runs_total':12,'epochs_total':600})
                    engine.train_one(seed,hankel,50,data)
        engine.save_json(OUT/'status.json',{'state':'complete','runs':12,'epochs':600})
    except Exception:
        engine.save_json(OUT/'status.json',{'state':'failed','error':traceback.format_exc()})
        raise


def launch():
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'launch.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        # Check both PID and worker lock to avoid simultaneous duplicate launches.
        if (OUT/'pid.json').exists():
            pid=json.loads((OUT/'pid.json').read_text())['pid']
            try:
                os.kill(pid,0)
            except ProcessLookupError:
                pass
            else:
                raise RuntimeError(f'Existing process {pid} is alive; no duplicate launched.')
        with (OUT/'run.lock').open('a') as worker_lock:
            fcntl.flock(worker_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with (OUT/'run.log').open('a') as log:
            child=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'run'],
                cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,
                start_new_session=True,env={**os.environ,'PYTHONPATH':str(ROOT),
                    'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'})
        engine.save_json(OUT/'pid.json',{'pid':child.pid,'launched_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())})
        print('Started original-channel study, PID',child.pid,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('command',choices=['smoke','validate-pipeline','run','launch','status'])
    args=parser.parse_args()
    os.chdir(ROOT)
    if args.command=='smoke': smoke()
    elif args.command=='validate-pipeline': pipeline_check()
    elif args.command=='run': run()
    elif args.command=='launch': launch()
    else:
        for name in ('pid.json','status.json'):
            path=OUT/name
            if path.exists(): print(name,path.read_text())
