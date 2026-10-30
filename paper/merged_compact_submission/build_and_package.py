"""Compile both documents and verify the self-contained Overleaf archive."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import zipfile

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]
OUTPUT=REPO/'output/pdf'
TEX=Path('/Library/TeX/texbin')

def compile_one(folder,name):
    commands=[['pdflatex','-interaction=nonstopmode','-halt-on-error','-file-line-error',name+'.tex'],
              ['bibtex',name],
              ['pdflatex','-interaction=nonstopmode','-halt-on-error','-file-line-error',name+'.tex'],
              ['pdflatex','-interaction=nonstopmode','-halt-on-error','-file-line-error',name+'.tex']]
    for i,args in enumerate(commands):
        result=subprocess.run([str(TEX/args[0]),*args[1:]],cwd=folder,text=True,capture_output=True)
        (folder/f'{name}_build_{i}.txt').write_text(result.stdout+result.stderr)
        if result.returncode:
            raise RuntimeError(f'{name} build step {i} failed: '+(result.stdout+result.stderr)[-3000:])
    log=(folder/(name+'.log')).read_text()
    problems=[line for line in log.splitlines() if any(s in line for s in ('undefined','Overfull','multiply defined','Missing character'))]
    if problems:
        raise RuntimeError(f'{name}: unresolved layout/reference warnings: '+repr(problems))
    return folder/(name+'.pdf')

def main(package=False):
    OUTPUT.mkdir(parents=True,exist_ok=True)
    for name,dest in [('main','rydberg_tcom_compact.pdf'),('supplement','rydberg_tcom_compact_supplement.pdf')]:
        pdf=compile_one(HERE,name)
        shutil.copy2(pdf,OUTPUT/dest)
        print('Compiled '+str(OUTPUT/dest),flush=True)
    if not package:
        return
    paths=[]
    for path in HERE.rglob('*'):
        rel=path.relative_to(HERE)
        if not path.is_file() or '__pycache__' in rel.parts:
            continue
        if path.suffix in ('.tex','.bib') or (path.suffix=='.pdf' and 'figures' in rel.parts) or 'evidence' in rel.parts or path.name in ('README.txt','REVISION_NOTES.txt','make_latest_material.py','make_compact_material.py'):
            paths.append(path)
    zip_path=OUTPUT/'rydberg_tcom_compact_overleaf.zip'
    with zipfile.ZipFile(zip_path,'w',compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(paths): archive.write(path,path.relative_to(HERE))
    # The extract location is a new, task-specific directory; no broad cleanup.
    folder=Path(tempfile.mkdtemp(prefix='rydberg-overleaf-verify-'))
    with zipfile.ZipFile(zip_path) as archive:
        assert archive.testzip() is None
        archive.extractall(folder)
    for name in ('main','supplement'):
        compile_one(folder,name)
        # Compare content streams rather than PDF creation timestamps.
        from pypdf import PdfReader
        original,repacked=PdfReader(HERE/(name+'.pdf')),PdfReader(folder/(name+'.pdf'))
        assert len(original.pages)==len(repacked.pages)
        for a,b in zip(original.pages,repacked.pages):
            assert a.extract_text()==b.extract_text(),f'{name}: repack text mismatch'
    record=dict(zip_sha256=hashlib.sha256(zip_path.read_bytes()).hexdigest(),
        entry_count=len(paths),extracted_compile='passed',extracted_pdf_text='matched',
        checked_folder=str(folder),source_pdfs={n:hashlib.sha256((HERE/(n+'.pdf')).read_bytes()).hexdigest() for n in ('main','supplement')})
    (OUTPUT/'compact_package_verification.json').write_text(json.dumps(record,indent=2)+'\n')
    print('Extracted archive compiled and matched both documents: '+str(zip_path),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--package',action='store_true');main(p.parse_args().package)
