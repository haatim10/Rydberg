from pathlib import Path
import json
import time
from sahure_expansion.protocol import CONFIG as PREVIOUS, write_json, sha

ROOT = Path('results/sahure_checks')
OLD = Path('results/sahure_expansion')
CONFIG = dict(version=1, core='em', epochs=20, batch_size=32, lr=.001,
    train_cases=16384, val_cases=2048, stages=10, seeds=[9101,9102,9103],
    replicates=[1,2], replicate_training_seed=9101, test_cases=200,
    workers=2, maximum_hours=12, bootstrap_resamples=4000,
    primary='equal-weight mean of seven per-SNR ratio-of-sums NMSE dB values',
    supervised_loss='mean per-world channel NMSE on the same 21 context pilots',
    supervised_selection='equal-stratum validation channel NMSE; also retain NLL-selected checkpoint',
    self_supervised_selection='equal-stratum held-out Rician NLL',
    test_status='fresh fixed-protocol follow-up; not the earlier proposed five-dataset confirmation',
    primary_contrasts=['adaptive vs supervised','adaptive vs final_only','gate_one vs adaptive',
                       'replicate adaptive vs global','replicate adaptive vs none','replicate adaptive vs hs_gs'])

def jobs():
    out=[]
    for seed in CONFIG['seeds']:
        for arm in ['supervised','final_only','gate_one']:
            out.append(dict(id=f'matched_{arm}_{seed}', cohort='matched', arm=arm, seed=seed))
    for rep in CONFIG['replicates']:
        for arm in ['adaptive','global','none']:
            out.append(dict(id=f'rep{rep}_{arm}',cohort=f'rep{rep}',arm=arm,seed=CONFIG['replicate_training_seed']))
    return out

def data_root(cohort):
    return OLD if cohort=='matched' else ROOT/cohort

def cells():
    base=dict(N=32,K=3,P=30,snr=5.,rsr=12.,L=None,channel='ula')
    out=[dict(base,id=f'snr_{s}',snr=float(s),primary=True) for s in PREVIOUS['snrs']]
    out += [dict(base,id='weak_reference',rsr=0.,primary=False),dict(base,id='two_paths',L=2,primary=False)]
    return out

def sources():
    paths=list(Path('sahure_checks').glob('*.py'))+list(Path('sahure').glob('*.py'))
    paths+=list(Path('sahure_expansion').glob('*.py'))+list(Path('rydberg_sim').glob('*.py'))
    paths+=[Path('codex_verification/solver.py')]
    return {str(p):sha(p) for p in sorted(set(paths))}

def initialize():
    ROOT.mkdir(parents=True,exist_ok=True)
    path=ROOT/'protocol.json'
    state=dict(config=CONFIG,jobs=jobs(),cells=cells(),sources=sources())
    # Lock both old inputs and old selected controls without rewriting them.
    refs=[OLD/'config.json',OLD/'frozen_selection.json',OLD/'audit.json']
    refs+=list((OLD/'data').glob('train*'))+list((OLD/'data').glob('val*'))
    for seed in CONFIG['seeds']:
        refs += [OLD/'checkpoints'/f'seed_{seed}'/f'em_{arm}_best.pt' for arm in ['adaptive','global','none']]
    state['previous_sha256']={str(p):sha(p) for p in sorted(refs)}
    if path.exists():
        previous=json.loads(path.read_text())
        if any(previous[k]!=v for k,v in state.items()): raise RuntimeError('Frozen protocol, sources or old inputs changed')
        return previous
    state['declared_unix']=time.time();write_json(path,state);return state

def check_sources():
    protocol=json.loads((ROOT/'protocol.json').read_text())
    if protocol['sources']!=sources(): raise RuntimeError('Source changed after declaration')
    return protocol
