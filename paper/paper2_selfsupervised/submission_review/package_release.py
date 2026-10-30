"""Package existing source/evidence without changing frozen experiments."""
from pathlib import Path
import hashlib
import io
import json
import zipfile
import numpy as np

REPO=Path(__file__).resolve().parents[3]
REVIEW=Path(__file__).resolve().parent
OUT=REPO/'output/pdf'
OUT.mkdir(parents=True,exist_ok=True)
assert json.loads((REVIEW/'evidence_audit.json').read_text())['state']=='passed'
manifest={}
with zipfile.ZipFile(OUT/'paper2_reproducibility.zip','w',zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
    def add(name,data):
        assert name not in manifest,name
        archive.writestr(name,data)
        manifest[name]=hashlib.sha256(data).hexdigest()
    for module in ['sahure','sahure_expansion','sahure_checks','sahure_features','rydberg_sim','codex_verification']:
        for p in sorted((REPO/module).rglob('*')):
            if p.is_file() and p.suffix in {'.py','.md','.txt'}:
                add(str(p.relative_to(REPO)),p.read_bytes())
    for study in ['sahure_expansion','sahure_checks','sahure_features']:
        root=REPO/'results'/study
        for p in sorted(root.rglob('*')):
            if not p.is_file() or 'mpl_cache' in p.parts: continue
            name=str(p.relative_to(REPO))
            evaluation=('cells' in p.parts or 'evaluation' in p.parts)
            if p.suffix=='.npz' and evaluation:
                with np.load(p) as a:
                    keep={k:a[k] for k in a.files if k in {'error','energy','trial_ids','observation_sha256','checkpoint_sha256','batched_total_seconds'}}
                buffer=io.BytesIO();np.savez_compressed(buffer,**keep);add(name,buffer.getvalue())
            elif p.suffix in {'.json','.md','.pt'} or p.name=='timings.jsonl':
                add(name,p.read_bytes())
    # Include only the compact review calculations, not a second manuscript copy.
    for name in ['evidence_audit.json','REVIEW.md','RELATED_WORK_CHECK.md']:
        add('review/'+name,(REVIEW/name).read_bytes())
    add('README.md',(REVIEW/'REPRODUCIBILITY.md').read_bytes())
    add('reproduce_saved.py',(REVIEW/'reproduce_saved.py').read_bytes())
    root=REPO/'results/sahure_expansion'
    with np.load(root/'data/cell_03_observations.npz') as a:
        subset={k:a[k][:8] for k in a.files}
    for seed in [9101,9102,9103]:
        name=f'em_adaptive_s{seed}'
        with np.load(root/'cells/cell_03'/f'{name}.npz') as a:subset[name]=a['estimates'][:8]
    buffer=io.BytesIO();np.savez_compressed(buffer,**subset);add('replay_subset.npz',buffer.getvalue())
    archive.writestr('SHA256SUMS.json',json.dumps(manifest,indent=2)+'\n')
with zipfile.ZipFile(OUT/'paper2_reproducibility.zip') as archive:
    assert archive.testzip() is None
    for name,expected in manifest.items():assert hashlib.sha256(archive.read(name)).hexdigest()==expected
print(json.dumps({'archive':str(OUT/'paper2_reproducibility.zip'),'payload_files':len(manifest),'bytes':(OUT/'paper2_reproducibility.zip').stat().st_size}))
