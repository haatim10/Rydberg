"""Package only a completed, manually reviewed revision. Never train or tune."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import zipfile

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
OUT=ROOT/'output/pdf'


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def main():
    required=['results/paper2_review_round2/calibration/summary.json',
              'results/paper2_review_round2/geometry/summary.json',
              'results/paper2_review_round2/training/complete.json',
              'results/paper2_review_round2/training_test/summary.json',
              'results/paper2_review_round2/training_timing.json']
    for name in required:
        if not (ROOT/name).exists():raise RuntimeError('Release blocked: missing '+name)
    approval_path=HERE/'APPROVED_FOR_RELEASE.json'
    if not approval_path.exists():raise RuntimeError('Release blocked: final numerical and rendered-page review not recorded')
    approval=json.loads(approval_path.read_text())
    assert approval['all_pages_visually_checked'] and approval['all_seven_review_questions_reconciled']
    for name,h in approval['hashes'].items():
        if digest(ROOT/name)!=h:raise RuntimeError('Approval is stale: '+name)
    response=HERE/'RESPONSE_TO_LATEST_REVIEW.txt'
    if not response.exists():raise RuntimeError('Release blocked: final point-by-point response missing')
    sources=[p for p in HERE.iterdir() if p.suffix in ('.tex','.bib','.bbl')]
    sources+=sorted((HERE/'figures').glob('*.pdf'))+[HERE/'README.txt',response]
    OUT.mkdir(parents=True,exist_ok=True)
    bundle=OUT/'paper2_round2_overleaf.zip'
    with zipfile.ZipFile(bundle,'w',zipfile.ZIP_DEFLATED) as archive:
        for p in sources:archive.write(p,str(p.relative_to(HERE)))
    with tempfile.TemporaryDirectory(prefix='paper2-round2-overleaf-') as temp:
        staging=Path(temp)
        with zipfile.ZipFile(bundle) as archive:
            assert archive.testzip() is None;archive.extractall(staging)
        for base in ('paper2_revised','paper2_supplement'):
            for command in [['/Library/TeX/texbin/pdflatex','-interaction=batchmode','-halt-on-error',base+'.tex'],
                            ['/Library/TeX/texbin/bibtex',base],
                            ['/Library/TeX/texbin/pdflatex','-interaction=batchmode','-halt-on-error',base+'.tex'],
                            ['/Library/TeX/texbin/pdflatex','-interaction=batchmode','-halt-on-error',base+'.tex']]:
                r=subprocess.run(command,cwd=staging,capture_output=True,text=True)
                if r.returncode:raise RuntimeError(r.stdout+r.stderr)
            log=(staging/(base+'.log')).read_text()
            if 'undefined' in log.lower() or 'Overfull' in log:raise RuntimeError('Archive compilation warning: '+base)
    for source,name in [('paper2_revised.pdf','paper2_round2.pdf'),('paper2_supplement.pdf','paper2_round2_supplement.pdf'),
                        ('paper2_revised.tex','paper2_round2.tex'),('RESPONSE_TO_LATEST_REVIEW.txt','paper2_round2_review_response.txt')]:
        shutil.copy2(HERE/source,OUT/name)
    payload=set()
    for directory in ('sahure','sahure_expansion','sahure_checks','sahure_features','rydberg_sim','codex_verification',
                      'paper2_review_checks','paper2_optshrink_loop','paper2_review_round2'):
        payload.update(p for p in (ROOT/directory).rglob('*') if p.is_file() and p.suffix in ('.py','.md','.txt'))
    for study in ('sahure_expansion','sahure_checks','sahure_features','paper2_review_checks','paper2_optshrink_loop','paper2_review_round2'):
        payload.update(p for p in (ROOT/'results'/study).rglob('*') if p.is_file()
            and p.suffix in ('.npz','.npy','.pt','.json','.jsonl','.md','.txt')
            and not any('cache' in part or 'preflight_20261008' in part for part in p.parts))
    for directory in ('ieee','reviewer_revision','review_round2','submission_review'):
        payload.update(p for p in (ROOT/'paper/paper2_selfsupervised'/directory).rglob('*')
            if p.is_file() and p.suffix in ('.py','.tex','.bib','.bbl','.pdf','.png','.json','.md','.txt'))
    # Repository scripts used by saved-result audits and figure provenance.
    payload.update(p for p in (ROOT/'reports').glob('*paper2_optshrink*.py'))
    payload.add(ROOT/'reports/audit_paper2_optshrink_runtime.py')
    expanded=OUT/'paper2_round2_full_reproducibility.zip';hashes={}
    with zipfile.ZipFile(expanded,'w',zipfile.ZIP_DEFLATED,compresslevel=4) as archive:
        for p in sorted(payload):
            name=str(p.relative_to(ROOT));hashes[name]=digest(p);archive.write(p,name)
        archive.writestr('SHA256SUMS.json',json.dumps(hashes,indent=2)+'\n')
    with zipfile.ZipFile(expanded) as archive:
        assert archive.testzip() is None
        for name,h in hashes.items():
            if hashlib.sha256(archive.read(name)).hexdigest()!=h:raise RuntimeError('Archive hash failed: '+name)
    artifacts=[OUT/name for name in ('paper2_round2.pdf','paper2_round2_supplement.pdf','paper2_round2.tex',
                                     'paper2_round2_review_response.txt','paper2_round2_overleaf.zip','paper2_round2_full_reproducibility.zip')]
    record={p.name:dict(bytes=p.stat().st_size,sha256=digest(p)) for p in artifacts}
    record['verification']=dict(archive_compile=True,expanded_archive_integrity=True,payload_files=len(payload),
        visual_review=approval)
    (OUT/'paper2_round2_manifest.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record,indent=2))

if __name__=='__main__':main()
