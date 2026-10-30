"""Controlled objective-only ablation on the unchanged classical-start model."""
import argparse
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import sys
import time
import traceback
from unittest.mock import patch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
import paper2_classical_start as prior
base = prior.base
OUT = ROOT/'results/paper2_loss_weighting'
OLD = prior.OUT
ARMS = ('standard','high_snr_weighted')
SPEC = dict(prior.SPEC)
FACTOR = 10.
WEIGHT_MEAN = 1.
ORIGINAL_STEP = base.repro.train_step
ORIGINAL_SOURCES = prior.sources
ORIGINAL_SUMMARIZE = prior.summarize


def weights(snr,factor=FACTOR):
    return torch.where(snr>=15.,factor,1.)


def objective(errors,snr,normalizer,factor=FACTOR):
    return (errors*weights(snr,factor)/normalizer).mean()


def build(arm):
    if arm not in (*ARMS,'classical_start'):
        raise ValueError(arm)
    original,_ = base.repro.build_model(base.cfg())
    model = prior.ClassicalStart(original)
    model.loss_arm = arm
    return model, {'implementation':'unchanged_classical_start',
        'params':sum(p.numel() for p in model.parameters()),'loss_arm':arm,
        'high_snr_factor':FACTOR if arm=='high_snr_weighted' else 1.,
        'training_weight_mean':WEIGHT_MEAN,'logged_train_nmse_is_unweighted':True}


def train_step(model,opt,batch,cfg):
    if model.loss_arm != 'high_snr_weighted':
        return ORIGINAL_STEP(model,opt,batch,cfg)
    opt.zero_grad(set_to_none=True)
    pred = model(*(batch[k] for k in ('G0','Z','S','B','sigma2')))
    errors = abs(pred-batch['G_true']).square().sum((1,2))/abs(batch['G_true']).square().sum((1,2))
    loss = objective(errors,batch['snr_db'],WEIGHT_MEAN)
    if not torch.isfinite(loss):
        raise RuntimeError('Non-finite weighted loss; no automatic changes.')
    loss.backward()
    grad = torch.nn.utils.clip_grad_norm_(model.parameters(),float('inf'),error_if_nonfinite=True)
    opt.step()
    return float(errors.detach().mean()),float(grad)


@contextmanager
def configured(out=None):
    with patch.multiple(base,OUT=OUT if out is None else out,SEED=prior.SEED,
                        ARMS=ARMS,STUDIES={'development':SPEC},build=build), \
         patch.object(base.repro,'train_step',train_step):
        yield


def sources():
    result = ORIGINAL_SOURCES()
    for p in (Path(__file__),ROOT/'reports/paper2_loss_weighting_plan.md',
              ROOT/'tests/test_paper2_loss_weighting.py'):
        result[str(p.relative_to(ROOT))] = base.engine.hash_file(p)
    return result


def verify_old():
    m = json.loads((OLD/'manifest.json').read_text())
    for p,h in m['source_hashes'].items():
        assert base.engine.hash_file(ROOT/p)==h,p
    paths = [OLD/'manifest.json',OLD/'summary.json']
    for arm in ('ordinary','classical_start'):
        paths += [OLD/'development'/arm/'result.json',OLD/'development'/arm/'validation_errors.npz']
    return {str(p.relative_to(ROOT)):base.engine.hash_file(p) for p in paths}


def finish(data,refs):
    with patch.multiple(prior,OUT=OUT,ARMS=ARMS,build=build):
        result = ORIGINAL_SUMMARIZE(data,refs)
    records = [json.loads((OUT/'development'/a/'result.json').read_text()) for a in ARMS]
    assert records[0]['initial_hash']==records[1]['initial_hash']
    old = np.load(OLD/'development/classical_start/validation_errors.npz')
    current = np.load(OUT/'development/standard/validation_errors.npz')
    for k in ('trial','snr_db','errors'):
        np.testing.assert_array_equal(old[k],current[k])
    ordinary = np.load(OLD/'development/ordinary/validation_errors.npz')
    for k in ('trial','snr_db'):
        np.testing.assert_array_equal(ordinary[k],current[k])
    old_result = json.loads((OLD/'development/ordinary/result.json').read_text())
    result['control_replays_previous_run_exactly'] = True
    result['matched_initial_weights'] = True
    result['weight_mean'] = WEIGHT_MEAN
    result['screen'] = {}
    for label in ('final','selected'):
        ordinary_scores = prior.scores(ordinary['errors'][-1 if label=='final' else old_result['chosen_epoch']-1],ordinary['snr_db'])
        result['scores'][label]['ordinary_reference'] = ordinary_scores
        w = result['scores'][label]['high_snr_weighted']
        s = result['scores'][label]['standard']
        overall = ordinary_scores['all']['nmse_db']-w['all']['nmse_db']
        high = w['[15,20)']['nmse_db']-refs['validation']['10']['[15,20)']['nmse_db']
        result['screen'][label] = {'overall_gain_vs_ordinary_db':overall,
            'overall_gain_vs_standard_db':s['all']['nmse_db']-w['all']['nmse_db'],
            'high_snr_deterioration_vs_initial_db':high,
            'overall_pass':overall>=.5,'preservation_pass':high<=.5,'both_pass':overall>=.5 and high<=.5}
    base.engine.save_json(OUT/'summary.json',result)


def run():
    global WEIGHT_MEAN
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'run.lock').open('a') as lock,configured():
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        torch.set_num_threads(1)
        manifest = {'source_hashes':sources(),'prior_artifact_hashes':verify_old(),
                    'spec':SPEC,'factor':FACTOR,'seed':prior.SEED,'torch':torch.__version__,'numpy':np.__version__}
        if (OUT/'manifest.json').exists():
            assert json.loads((OUT/'manifest.json').read_text())==manifest
        else:
            base.engine.save_json(OUT/'manifest.json',manifest)
        base.engine.save_json(OUT/'pid.json',{'pid':os.getpid(),'started_unix':time.time()})
        try:
            base.engine.save_json(OUT/'status.json',{'state':'preparing_data'})
            data = base.cache('development',SPEC)
            for split in ('train','val'):
                old = torch.load(OLD/'development/data'/f'{split}.pt',weights_only=True)
                for key,value in old.items():
                    torch.testing.assert_close(value,data[split].arrays[key],rtol=0,atol=0)
            WEIGHT_MEAN = float(weights(data['train'].arrays['snr_db']).mean())
            base.engine.save_json(OUT/'weights.json',{'factor':FACTOR,'threshold_db':15.,
                'normalizer':WEIGHT_MEAN,'high_snr_training_count':int((data['train'].arrays['snr_db']>=15).sum())})
            refs = prior.references(data)
            for arm in ARMS:
                base.train_arm('development',arm,SPEC,data)
            finish(data,refs)
            assert sources()==manifest['source_hashes'] and verify_old()==manifest['prior_artifact_hashes']
            base.engine.save_json(OUT/'status.json',{'state':'complete','runs':2,'validation_only':True})
        except Exception:
            base.engine.save_json(OUT/'status.json',{'state':'failed','error':traceback.format_exc()})
            raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('command',choices=('run',)); parser.parse_args()
    run()
