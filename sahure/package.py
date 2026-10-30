"""Bundle only this experiment and its simulator dependencies."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile


def package(out):
    out=Path(out)
    status=json.loads((out/'status.json').read_text())
    if status['state']!='complete' or status['phase'] not in ('evaluate','all'):
        raise ValueError('evaluation is not complete')
    files=list(Path('sahure').glob('*.py'))+list(Path('sahure').glob('*.md'))+[Path('sahure/requirements.txt')]
    files+=list(Path('rydberg_sim').glob('*.py'))
    files+=[p for p in (Path('codex_verification/__init__.py'),Path('codex_verification/solver.py')) if p.exists()]
    files+=[p for p in out.rglob('*') if p.is_file() and 'mpl_cache' not in p.parts
            and '__pycache__' not in p.parts and p.name!='artifact_hashes.json']
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(set(files))}
    manifest=out/'artifact_hashes.json';manifest.write_text(json.dumps(hashes,indent=2)+'\n')
    files.append(manifest)
    archive=out.parent/(out.name+'_bundle.zip')
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(set(files)):z.write(p,arcname=Path('Rydberg_sahure')/p)
    with zipfile.ZipFile(archive) as z:
        bad=z.testzip()
        if bad is not None: raise IOError(f'archive verification failed: {bad}')
    return archive

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',default='results/sahure_pilot')
    print(package(parser.parse_args().output))
