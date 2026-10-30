"""Read-only reproducibility checks after the paired campaign completes."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key]='1'
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from .data import load_observations
from .train import load_model,validation_score


def audit(out):
    out=Path(out);torch.set_num_threads(1)
    c=json.loads((out/'config.json').read_text())
    frozen=json.loads((out/'frozen_selection.json').read_text())
    for name,info in frozen['checkpoints'].items():
        assert hashlib.sha256(Path(info['path']).read_bytes()).hexdigest()==info['sha256'],name
    for path,digest in frozen['estimator_source_hashes'].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest,path
    # Archive extraction can change filesystem times. Preserve the observed
    # ordering together with hashes, so this check survives copying the bundle.
    order_path=out/'generation_order.json'
    observed={str(p):dict(mtime=p.stat().st_mtime,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
              for p in (out/'frozen_selection.json',out/'data/test_observations.npz')}
    if order_path.exists():
        order=json.loads(order_path.read_text())
        assert all(order[k]['sha256']==v['sha256'] for k,v in observed.items())
    else:
        order=observed
        assert order[str(out/'frozen_selection.json')]['mtime']<order[str(out/'data/test_observations.npz')]['mtime']
        order_path.write_text(json.dumps(order,indent=2)+'\n')
    assert order[str(out/'frozen_selection.json')]['mtime']<order[str(out/'data/test_observations.npz')]['mtime']
    rows=[json.loads(x) for x in (out/'trials.jsonl').read_text().splitlines()]
    pairs={(r['trial'],r['method']) for r in rows}
    methods={r['method'] for r in rows}
    assert len(methods)==17 and len(rows)==len(pairs)==17*c['test_cases']
    with np.load(out/'data/test_truth.npz') as data:
        energy=np.einsum('bnk,bnk->b',data['g'].conj(),data['g']).real
    summary=json.loads((out/'summary.json').read_text())['methods']
    for method in methods:
        group=sorted((r for r in rows if r['method']==method),key=lambda r:r['trial'])
        assert [r['trial'] for r in group]==list(range(c['test_cases']))
        np.testing.assert_allclose([r['true_energy'] for r in group],energy,rtol=1e-12)
        e=np.array([r['error_energy'] for r in group]);assert np.isfinite(e).all() and (e>=0).all()
        assert all(len(r['times_seconds'])==3 and min(r['times_seconds'])>0 for r in group)
        nmse=10*np.log10(e.sum()/energy.sum())
        assert abs(nmse-summary[method]['nmse_db'])<1e-10
    val=load_observations(out/'data/val_observations.npz')
    decisions={}
    for name in frozen['checkpoints']:
        model,checkpoint=load_model(out/'checkpoints'/f'{name}_best.pt',torch.float32)
        score=validation_score(model,val,c['seed']+30000)
        assert abs(score-checkpoint['best_score'])<2e-6,(name,score,checkpoint['best_score'])
        decisions[name]=dict(recomputed_validation_nll=score,saved_validation_nll=checkpoint['best_score'],selected_epoch=checkpoint['epoch'])
    result=dict(passed=True,paired_trials=c['test_cases'],methods=len(methods),unique_method_trial_rows=len(rows),
        checkpoint_hashes_match=True,source_hashes_match=True,test_generated_after_freeze=True,
        independently_recomputed_nmse_matches=True,validation_decisions=decisions)
    (out/'audit.json').write_text(json.dumps(result,indent=2)+'\n')
    return result

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',default='results/sahure_pilot')
    print(json.dumps(audit(parser.parse_args().output),indent=2))
