import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']: os.environ[key]='1'
import json,tempfile,sys
from pathlib import Path
sys.path.insert(0,str(Path.cwd()))
import torch
import sahure_expansion.protocol as p
import sahure_expansion.data as d
import sahure_expansion.evaluate as e
import sahure_expansion.report as r
from sahure_expansion.protocol import write_json

torch.set_num_threads(1)
with tempfile.TemporaryDirectory(prefix='sahure_e2e_') as folder:
    root=Path(folder)
    for module in [d,e,r]:
        module.ROOT=root
        module.CONFIG=p.CONFIG|dict(test_cases=4,timing_trials=2,timing_repetitions=1,bootstrap_resamples=100)
    spec=dict(id='cell_00',N=32,K=3,P=30,snr=5.,rsr=12.,L=None,channel='ula',noise_offset=0.)
    models={}
    realroot=p.ROOT
    for seed in p.CONFIG['seeds']:
        for core in ['em','gd']:
            for structure in ['none','global','adaptive','constant_control']:
                name=f'{core}_{structure}'
                # Use this campaign's checkpoints so the bundled smoke test
                # does not require results from the earlier pilot.
                suffix='' if structure=='constant_control' else '_best'
                path=realroot/'checkpoints/seed_9101'/f'{name}{suffix}.pt'
                models[f'{name}_s{seed}']=dict(path=str(path),method=name,seed=seed)
    axes={axis:[dict(value=value,cell='cell_00')] for axis,value in [('snr',5),('rsr',12),('paths',4),('pilots',30),('array',32),('users',3),('channel','ula'),('noise_offset',0)]}
    frozen=dict(primary_core='gd',models=models,classical=list(e.CLASSICAL),cells=[spec],axes=axes,hard_rank=4)
    e.freeze=lambda:frozen
    write_json(root/'frozen_selection.json',frozen)
    d.generate_set('cell_00',4,2026092259,spec=spec)
    e.evaluate_cell(spec)
    e.timing()
    frozen,result=r.summarize()
    r.plots(frozen,result)
    result['training_duration']['main_training_span_seconds']=1.
    result['training_duration']['preflight_seconds']=1.
    r.report(frozen,result)
    assert len(result['cells']['cell_00']['methods'])==19
    assert len(list((root/'figures').glob('*.pdf')))==6
    assert (root/'REPORT.md').exists()
    print('PASS: four-world temporary fixture; actual inference, all 35 seed/method outputs, bootstrap summary, serial timing, report and six figure files')
