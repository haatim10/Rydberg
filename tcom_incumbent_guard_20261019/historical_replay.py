"""Frozen postselection replay on an explicitly exposed historical bank."""
import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[_key]='1'
import argparse
import json
import numpy as np
import torch
from sahure.data import Observations
from sahure.math import posterior_field,right_solve
from tcom_followups_20261017.common import sha,write_json
from . import study

ROOT=study.OUT/'historical_replay'
SOURCE=study.REPO/'results/tcom_dual_guard_20261019'
NAMES=tuple(f'{prefix}_{seed}' for prefix in ('original','old_bypass','prior_repair','current_repair','repair') for seed in study.SEEDS)
GROUPS={f'{prefix}_mean':[f'{prefix}_{seed}' for seed in study.SEEDS] for prefix in ('original','old_bypass','prior_repair','current_repair','repair')}


def conditions():
    return [(regime,float(ref),float(snr)) for regime,_ in study.REGIMES for ref in (12,20) for snr in (32.5,35,37.5)]



def prepare():
    selection,digest=study.check_selection()
    prior=json.loads((SOURCE/'nmse_summary.json').read_text())
    from .analysis import HISTORICAL
    flags=list(HISTORICAL);assert len(flags)==8
    assert set(flags)==set(prior['negative_cells']['repair_vs_prior']['target_negative_cells'])
    cells=[study.cellname(*row) for row in conditions()];assert len(cells)==len(set(cells))==18
    assert set(flags)<=set(cells)
    record=dict(version=1,selection_sha256=digest,protocol_sha256=selection['protocol_sha256'],source_sha256=sha(__file__),
        historical_seed=2026101932,historical_policy_sha256=sha(SOURCE/'policy.json'),historical_summary_sha256=sha(SOURCE/'nmse_summary.json'),
        cells=cells,historical_flagged_cells=flags,methods=NAMES,groups=GROUPS,
        source_cell_hashes={cell:sha(SOURCE/'test/cells'/f'{cell}.npz') for cell in cells},
        rule='Only new frozen repair estimates are recomputed on archived Z/S/B/sigma2. Original, old bypass, previous repair and current correction predictions are read unchanged from their archived cells.',
        interpretation='Exposed-bank regression diagnostic. Not independent confirmation, not added to fresh-test sample counts, and never used to alter the new policy. No new historical BER.')
    path=ROOT/'protocol.json'
    if path.exists():assert json.loads(path.read_text())==json.loads(json.dumps(record))
    else:write_json(path,record)
    return selection,record,sha(path)


@torch.no_grad()
def evaluate():
    selection,protocol,digest=prepare();est=study.engine()
    for regime,ref,snr in conditions():
        cell=study.cellname(regime,ref,snr);path=ROOT/'cells'/f'{cell}.npz'
        if path.exists():assert str(study.load(path)['protocol_sha256'])==digest;continue
        old=study.load(SOURCE/'test/cells'/f'{cell}.npz');assert int(old['seed'])==2026101932
        oldnames=list(map(str,old['method_names']))
        obs=Observations(*(torch.from_numpy(old[k]) for k in ('z','s','b','sigma2')))
        cached=[old['estimates'][oldnames.index(f'{prefix}_{seed}')] for prefix in ('original','old_bypass','prior_repair','repair') for seed in study.SEEDS]
        fresh=[study.estimate_method(obs,f'repair_{seed}',selection,est).numpy() for seed in study.SEEDS]
        predictions=np.stack(cached+fresh);assert np.isfinite(predictions).all()
        error=(np.abs(predictions-old['truth_g'][None])**2).sum((-2,-1));energy=(np.abs(old['truth_g'])**2).sum((-2,-1))
        study.save(path,method_names=np.array(NAMES),estimates=predictions,errors=error,truth_energy=energy,nmse=error/energy[None],
            regime=np.array(regime),rsr=np.array(ref),snr=np.array(snr),repair_choice=np.array(study.selected_repair_choice(obs,selection)),
            source_cell_sha256=np.array(protocol['source_cell_hashes'][cell]),selection_sha256=np.array(protocol['selection_sha256']),protocol_sha256=np.array(digest))
        print('Historical diagnostic',cell,flush=True)


def analyze():
    selection,protocol,digest=prepare();rows=[];negative={key:[] for key in ('old_bypass','prior_repair','current_repair','original')}
    for regime,ref,snr in conditions():
        cell=study.cellname(regime,ref,snr);a=study.load(ROOT/'cells'/f'{cell}.npz');assert str(a['protocol_sha256'])==digest
        points=10*np.log10(a['errors'].sum(1)/a['truth_energy'].sum())
        metrics={group:dict(nmse_db=float(points[[NAMES.index(n) for n in names]].mean()),per_fit_nmse_db=points[[NAMES.index(n) for n in names]].tolist()) for group,names in GROUPS.items()}
        contrasts={}
        for reference in negative:
            ids=[NAMES.index(f'{reference}_{seed}') for seed in study.SEEDS]
            values=points[ids]-points[-3:];gain=float(values.mean())
            contrasts['repair_vs_'+reference]=dict(gain_db=gain,per_fit_gains_db=values.tolist())
            if gain < -1e-10:negative[reference].append(cell)
        rows.append(dict(cell=cell,regime=regime,rsr=ref,snr=snr,repair_choice=str(a['repair_choice']),
            historically_flagged=cell in protocol['historical_flagged_cells'],methods=metrics,contrasts=contrasts))
    result=dict(selection_sha256=protocol['selection_sha256'],protocol_sha256=digest,cells=rows,cell_count=18,
        historical_flagged_cells=protocol['historical_flagged_cells'],negative_cells=negative,numerical_tie_tolerance_db=1e-10,
        flagged_cells_remaining_negative_vs_prior=[c for c in protocol['historical_flagged_cells'] if c in negative['prior_repair']],
        flagged_cells_remaining_negative_vs_old=[c for c in protocol['historical_flagged_cells'] if c in negative['old_bypass']],
        interpretation=protocol['interpretation'],source_sha256=sha(__file__))
    write_json(ROOT/'summary.json',result);return result


@torch.no_grad()
def audit():
    selection,protocol,digest=prepare();summary=json.loads((ROOT/'summary.json').read_text());assert summary['protocol_sha256']==digest
    lookup={r['cell']:r for r in summary['cells']};assert set(lookup)==set(protocol['cells'])
    canonical=0;maximum=0.;source_reuses=0
    for regime,ref,snr in conditions():
        cell=study.cellname(regime,ref,snr);source=SOURCE/'test/cells'/f'{cell}.npz';old=study.load(source);a=study.load(ROOT/'cells'/f'{cell}.npz')
        assert sha(source)==protocol['source_cell_hashes'][cell]==str(a['source_cell_sha256'])
        assert str(a['selection_sha256'])==protocol['selection_sha256'] and str(a['protocol_sha256'])==digest
        np.testing.assert_array_equal(a['method_names'],NAMES)
        oldnames=list(map(str,old['method_names']))
        for i,(prefix,seed) in enumerate((p,s) for p in ('original','old_bypass','prior_repair','repair') for s in study.SEEDS):
            np.testing.assert_array_equal(a['estimates'][i],old['estimates'][oldnames.index(f'{prefix}_{seed}')]);source_reuses+=1
        err=(np.abs(a['estimates']-old['truth_g'][None])**2).sum((-2,-1));energy=(np.abs(old['truth_g'])**2).sum((-2,-1))
        np.testing.assert_array_equal(err,a['errors']);np.testing.assert_array_equal(energy,a['truth_energy']);np.testing.assert_array_equal(err/energy[None],a['nmse'])
        obs=Observations(*(torch.from_numpy(old[k]) for k in ('z','s','b','sigma2')))
        choice=study.selected_repair_choice(obs,selection);assert choice==str(a['repair_choice'])
        est=study.engine();small=obs.batch(slice(0,1))
        for i,seed in enumerate(study.SEEDS):
            g=study.estimate_method(small,f'repair_{seed}',selection,est).numpy()
            difference=float(np.abs(g-a['estimates'][i+12,:1]).max());maximum=max(maximum,difference)
            assert difference<1e-10;canonical+=1
        points=10*np.log10(err.sum(1)/energy.sum());row=lookup[cell]
        for name,names in GROUPS.items():np.testing.assert_allclose(row['methods'][name]['nmse_db'],points[[NAMES.index(n) for n in names]].mean(),atol=1e-12,rtol=0)
    result=dict(passed=True,cells=18,cached_prediction_reuses=source_reuses,canonical_warm_replays=canonical,maximum_canonical_error=maximum,
        selection_sha256=protocol['selection_sha256'],protocol_sha256=digest,summary_sha256=sha(ROOT/'summary.json'),source_sha256=sha(__file__),
        scope='Postselection exposed-bank regression replay only. All original artifacts remain unchanged; no result authorizes reselection.')
    write_json(ROOT/'audit.json',result);return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=('prepare','evaluate','analyze','audit'));args=parser.parse_args()
    torch.set_num_threads(1)
    try:torch.set_num_interop_threads(1)
    except RuntimeError:pass
    result=globals()[args.phase]()
    if args.phase in ('analyze','audit'):print(json.dumps({k:v for k,v in result.items() if k!='cells'},indent=2))
