"""Portable read-only replay, plus an explicitly requested clean regeneration workspace."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key]='1'
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
import torch
from sahure.train import load_model as old_model
from sahure_expansion.data import world
from .models import load_model


def check_archive_manifest(root):
    p=root/'SHA256SUMS.json'
    if not p.exists():return 0
    record=json.loads(p.read_text())
    for name,digest in record.items():
        path=root/name
        if hashlib.sha256(path.read_bytes()).hexdigest()!=digest:raise RuntimeError(f'hash failed: {name}')
    return len(record)


def replay(root):
    torch.set_num_threads(1);verified=check_archive_manifest(root);checks=[]
    frozen=json.loads((root/'results/sahure_expansion/frozen_selection.json').read_text())
    data=root/'results/paper2_review_round2/calibration/data/snr_+15_rsr_12/trial_000.npz'
    with np.load(data) as d:raw={k:d[k] for k in d.files}
    pred=root/'results/paper2_review_round2/calibration/predictions/snr_+15_rsr_12/matched/trial_000.npz'
    with np.load(pred) as d:saved=d['estimates'][:3].copy()
    with torch.no_grad():
        for j,seed in enumerate((9101,9102,9103)):
            m,_=old_model(root/frozen['models'][f'em_adaptive_s{seed}']['path'])
            got=m.estimate(raw['z'],raw['s'],raw['b'],float(raw['v'])).numpy()
            np.testing.assert_allclose(got,saved[j],rtol=1e-8,atol=1e-9);checks.append(f'original model {seed}')
        test=root/'results/paper2_review_round2/training_test'
        if (test/'summary.json').exists():
            with np.load(test/'data/cell_00_observations.npz') as d:raw={k:d[k][0] for k in d.files}
            for path in sorted((root/'results/paper2_review_round2/training').glob('*/selected.pt')):
                m,_=load_model(path);got=m.estimate(raw['z'],raw['s'],raw['b'],float(raw['sigma2'])).numpy()
                with np.load(test/'predictions'/path.parent.name/'selected/cell_00.npz') as d:saved=d['estimates'][0]
                np.testing.assert_allclose(got,saved,rtol=2e-7,atol=1e-8);checks.append(path.parent.name)
    # Check deterministic generation, independently of saved summary numbers.
    generation=root/'results/paper2_review_round2/geometry/data/pilots_P6_snr-5/trial_000.npz'
    with np.load(generation) as d:raw={k:d[k] for k in d.files}
    w,_=world(0,202610087101,N=32,K=3,P=6,snr=-5,rsr=12)
    for key,value in [('z',w.Z),('s',w.S),('b',w.B),('g',w.G)]:np.testing.assert_allclose(raw[key],value,rtol=1e-12,atol=1e-12)
    print(json.dumps(dict(manifest_files_verified=verified,checkpoint_replays=checks,world_regeneration='passed'),indent=2))


def prepare(root,destination):
    destination=Path(destination).resolve()
    if destination.exists():raise FileExistsError('Choose a new directory; existing results will not be replaced')
    destination.mkdir(parents=True)
    for module in ('paper2_review_round2','paper2_optshrink_loop','sahure','sahure_expansion','sahure_checks','sahure_features','rydberg_sim','codex_verification'):
        for source in (root/module).rglob('*'):
            if not source.is_file() or source.suffix not in ('.py','.txt','.md'):continue
            target=destination/source.relative_to(root);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
    needed=['results/sahure_expansion/frozen_selection.json','results/paper2_optshrink_loop/frozen_selection.json']
    frozen=json.loads((root/needed[0]).read_text())
    needed += [frozen['models'][f'em_adaptive_s{s}']['path'] for s in (9101,9102,9103)]
    for name in needed:
        target=destination/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(root/name,target)
    print('Clean review replication workspace:',destination)
    print('Run the numerical tests before launching a bounded study. Existing evidence was not copied over or altered.')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--prepare-regeneration');args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    prepare(root,args.prepare_regeneration) if args.prepare_regeneration else replay(root)
