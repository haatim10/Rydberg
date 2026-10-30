"""Portable results/code bundle, with compact per-world errors and diagnostics."""
import io
import json
from pathlib import Path
import zipfile
import numpy as np
from .protocol import ROOT


def package():
    audit=json.loads((ROOT/'audit.json').read_text())
    if audit['state']!='passed': raise RuntimeError('Package only audited results')
    output=ROOT.parent/'sahure_expansion_bundle.zip'
    included=[]
    with zipfile.ZipFile(output,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        for directory in ['sahure','sahure_expansion','rydberg_sim','codex_verification']:
            for path in sorted(Path(directory).rglob('*')):
                if path.is_file() and path.suffix in ['.py','.md','.txt']:
                    archive.write(path,str(path));included.append(str(path))
        for path in sorted(ROOT.rglob('*')):
            if not path.is_file() or 'mpl_cache' in path.parts or path.suffix=='.tmp': continue
            if path.parent.name=='data' and (path.name.endswith('_observations.npz') or path.name.startswith('train_truth')):
                continue
            name=str(path)
            if path.suffix=='.npz' and 'cells' in path.parts:
                with np.load(path) as f:
                    buffer=io.BytesIO();np.savez_compressed(buffer,**{k:f[k] for k in f.files if k!='estimates'})
                archive.writestr(name,buffer.getvalue());included.append(name)
            else: archive.write(path,name);included.append(name)
        archive.writestr('BUNDLE_README.md',
            '# Mixed-condition Hankel results\n\n'
            'Read results/sahure_expansion/REPORT.md and sahure_expansion/README.md.\n\n'
            'This compact bundle includes code, checkpoints, figures, summaries, per-world errors, '
            'structural diagnostics, test truths, timings and provenance. Observation arrays and full '
            'learned channel predictions are omitted to keep the download smaller; the original local '
            'result directory contains them. Datasets can be regenerated from the recorded seeds. '
            'artifact_hashes.json describes the full local run, not only this compact bundle.\n\n'
            'To reproduce from scratch, use the included code with a fresh result directory and the '
            'recorded configuration. To regenerate figures from this compact bundle, run '
            '`python -m sahure_expansion.report` from its root. The source-hash audit is intended '
            'for the original complete run and retains that machine’s training manifest paths.\n')
        archive.writestr('bundle_manifest.json',json.dumps(dict(included=included,
            omitted=['observation arrays','training truth','full learned channel predictions']),indent=2))
    print(json.dumps(dict(bundle=str(output),bytes=output.stat().st_size)),flush=True)


if __name__=='__main__':package()
