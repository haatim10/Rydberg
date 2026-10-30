"""Isolated classical-start diagnostic; reuses the tested epoch engine in-process."""
import argparse
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
import paper2_baseline_learning as base

OUT = ROOT/'results/paper2_classical_start'
SEED = 2026092501
ARMS = ('ordinary','classical_start')
SPEC = dict(train=4096,val=512,epochs=12,start=24000000,snr=None)
STOPS = (1,3,5,10,20,50,100)


class ClassicalStart(torch.nn.Module):
    def __init__(self, original):
        super().__init__()
        self.formers = torch.nn.ModuleList([layer.former for layer in original.layers])
        self.eps = original.layers[0].eps
        for former in self.formers:
            torch.nn.init.zeros_(former.output[-1].weight)
            torch.nn.init.zeros_(former.output[-1].bias)

    def forward(self,g,z,s,b,var):
        for former in self.formers:
            g = base.em_gs_layer(g,z,s,b,var,eps=self.eps)
            g = g+former(g)
        return g


def build(arm):
    if arm not in ARMS:
        raise ValueError(arm)
    model,meta = base.repro.build_model(base.cfg())
    if arm == 'classical_start':
        model = ClassicalStart(model)
        meta = {'implementation':'fixed_exact_EMGS10_plus_zero_initialized_PaperFormer_corrections',
                'published_reproduction':False,'fixed_bessel_filter':True}
    meta['params'] = sum(p.numel() for p in model.parameters())
    return model,meta


@contextmanager
def configured(out=None):
    # Patch only this Python process, never source files or prior result paths.
    with patch.multiple(base,OUT=OUT if out is None else out,SEED=SEED,ARMS=ARMS,
                        STUDIES={'development':SPEC},build=build):
        yield


def sources():
    result = base.provenance()
    for path in (Path(__file__),ROOT/'scripts/paper2_baseline_learning.py',
                 ROOT/'reports/paper2_classical_start_plan.md'):
        result[str(path.relative_to(ROOT))] = base.engine.hash_file(path)
    return result


def scores(errors,snr):
    masks = {'all':np.ones(len(snr),bool)}
    masks.update({f'[{lo},{lo+5})':(snr>=lo)&(snr<lo+5) for lo in range(-10,20,5)})
    return {k:{'n':int(m.sum()),'nmse_db':float(10*np.log10(errors[m].mean()))}
            for k,m in masks.items() if m.any()}


def references(data):
    folder = base.OUT/'references'; folder.mkdir(parents=True,exist_ok=True)
    tune_spec = dict(SPEC,train=1,val=256,start=25000000)
    tune = base.LearningData('development','val',tune_spec) # IDs26m, separate from validation25m.
    tune = base.Tensors(base.collate([tune[i] for i in range(len(tune))]))
    np.savez(folder/'tuning_ids.npz',trial=tune.arrays['trial'].numpy())
    assert not set(tune.arrays['trial'].tolist()) & set(data['val'].arrays['trial'].tolist())
    values = {}
    for steps in STOPS:
        errors = base.evaluate(base.Classical(steps),tune)
        values[steps] = float(errors.mean())
        np.savez(folder/f'tuning_emgs_{steps}.npz',errors=errors,trial=tune.arrays['trial'].numpy())
    chosen = min(STOPS,key=lambda step:values[step])
    snr = data['val'].arrays['snr_db'].numpy(); val = {}
    initial,_ = build('classical_start')
    init_errors = base.evaluate(initial,data['val'])
    for steps in sorted(set((10,100,chosen))):
        errors = base.evaluate(base.Classical(steps),data['val'])
        np.savez(folder/f'validation_emgs_{steps}.npz',errors=errors,
                 trial=data['val'].arrays['trial'].numpy(),snr_db=snr)
        val[str(steps)] = scores(errors,snr)
        if steps == 10:
            np.testing.assert_array_equal(errors,init_errors)
    np.savez(folder/'initial_classical_start.npz',errors=init_errors,
             trial=data['val'].arrays['trial'].numpy(),snr_db=snr)
    result = {'selected_steps':chosen,'tuning_mean_linear_nmse':values,
              'validation':val,'initial_classical_start_matches_emgs10':True}
    base.engine.save_json(folder/'summary.json',result)
    return result


def summarize(data,refs):
    folder = base.OUT/'development'; records = []; arrays = []
    for arm in ARMS:
        r = json.loads((folder/arm/'result.json').read_text()); records.append(r)
        f = np.load(folder/arm/'validation_errors.npz'); arrays.append(f['errors'])
        np.testing.assert_array_equal(f['trial'],data['val'].arrays['trial'])
        assert np.isfinite(f['errors']).all()
        model,_ = build(arm)
        for ep in set((r['chosen_epoch'],SPEC['epochs'])):
            model.load_state_dict(torch.load(folder/arm/f'epoch_{ep:03d}.pt',weights_only=True)['model'])
            np.testing.assert_array_equal(base.evaluate(model,data['val']),f['errors'][ep-1])
    assert [h['data_order_sha256'] for h in records[0]['history']] == [h['data_order_sha256'] for h in records[1]['history']]
    snr = data['val'].arrays['snr_db'].numpy()
    result = {'matched_data_order':True,'checkpoint_replays_exact':True,'validation_only':True,
              'references':refs,'scores':{},'single_seed':SEED}
    for label in ('final','selected'):
        result['scores'][label] = {arm:scores(arr[-1 if label=='final' else r['chosen_epoch']-1],snr)
                                   for arm,arr,r in zip(ARMS,arrays,records)}
    base.engine.save_json(base.OUT/'summary.json',result)
    return result


def run():
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'run.lock').open('a') as lock, configured():
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        torch.set_num_threads(1)
        manifest = {'source_hashes':sources(),'spec':SPEC,'arms':list(ARMS),'seed':SEED,
                    'torch':torch.__version__,'numpy':np.__version__}
        if (OUT/'manifest.json').exists():
            assert json.loads((OUT/'manifest.json').read_text()) == manifest
        else:
            base.engine.save_json(OUT/'manifest.json',manifest)
        base.engine.save_json(OUT/'pid.json',{'pid':os.getpid(),'started_unix':time.time()})
        try:
            base.engine.save_json(OUT/'status.json',{'state':'preparing_data'})
            data = base.cache('development',SPEC)
            refs = references(data)
            for arm in ARMS:
                base.train_arm('development',arm,SPEC,data)
            summarize(data,refs)
            assert sources() == manifest['source_hashes']
            base.engine.save_json(OUT/'status.json',{'state':'complete','runs':2,'validation_only':True})
        except Exception:
            base.engine.save_json(OUT/'status.json',{'state':'failed','error':traceback.format_exc()})
            raise


def launch():
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'run.lock').open('a') as lock:
        try:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            print('Existing worker holds the lock; no duplicate launched.'); return
        status = OUT/'status.json'
        if status.exists() and json.loads(status.read_text())['state'] in ('complete','failed'):
            print(status.read_text()); return
    with (OUT/'worker.log').open('ab') as log:
        process = subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'run'],
            cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    time.sleep(2)
    if process.poll() is not None:
        raise RuntimeError(f'Worker exited {process.returncode}; inspect {OUT}/worker.log')
    print(json.dumps({'launched_pid':process.pid,'output':str(OUT)}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('command',choices=('run','launch'))
    args = parser.parse_args()
    run() if args.command == 'run' else launch()
