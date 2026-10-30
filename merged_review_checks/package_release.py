"""Package bounded manuscript/evidence releases and verify portable builds."""
from pathlib import Path
import hashlib
import json
import re
import shutil
import subprocess
import tempfile
import zipfile
import sys

BASE=Path.cwd();PAPER=BASE/'paper/merged_tcom';OUT=BASE/'output/pdf'
PDFTOTEXT=shutil.which('pdftotext') or '/Users/haatim/.cache/codex-runtimes/codex-primary-runtime/dependencies/native/poppler/poppler/bin/pdftotext'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def run(args,cwd):
    r=subprocess.run(args,cwd=cwd,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    if r.returncode: raise RuntimeError(r.stdout[-6000:])
    return r.stdout
def pages(p):
    return int(re.search(r'Pages:\s+(\d+)',run(['pdfinfo',str(p)],BASE))[1])

# The original PDF and exported ZIP are retained; corrected files have explicit names.
OUT.mkdir(parents=True,exist_ok=True)
for source,target in [('merged_tcom.pdf','rydberg_merged_tcom_corrected.pdf'),
 ('merged_supplement.pdf','rydberg_merged_tcom_corrected_supplement.pdf'),
 ('merged_tcom.tex','rydberg_merged_tcom_corrected.tex'),
 ('ONLINE_REVIEW_RESPONSE.txt','rydberg_merged_tcom_review_response.txt')]:
    shutil.copy2(PAPER/source,OUT/target)

main=(PAPER/'merged_tcom.tex').read_text()
assert 'Acknowledgment' not in main and 'Codex' not in main
for name in ('merged_tcom','merged_supplement'):
    log=(PAPER/f'{name}.log').read_text()
    assert not re.search(r'Overfull|LaTeX Warning:|^!',log,re.M), name
abstract=main.split(r'\begin{abstract}')[1].split(r'\end{abstract}')[0]
files=[p for p in PAPER.rglob('*') if p.is_file() and (
    (p.parent==PAPER and p.name in ('merged_tcom.tex','merged_supplement.tex','references.bib',
       'merged_tcom.bbl','merged_supplement.bbl','followup_main.tex','followup_tails.tex',
       'followup_timing.tex','README.txt','ONLINE_REVIEW_RESPONSE.txt')) or
    (p.parent==PAPER/'figures' and p.suffix=='.pdf') or
    ('support' in p.relative_to(PAPER).parts and p.suffix in ('.tex','.bib','.pdf')))]
record=dict(main_pages=pages(PAPER/'merged_tcom.pdf'),supplement_pages=pages(PAPER/'merged_supplement.pdf'),
            abstract_words=len(abstract.split()),keyword_count=5,
            input_pdf_preserved=True,new_training=False,new_comparator_worlds=1152,
            original_test_bank_reused=True,
            numerical_audit=json.loads((BASE/'results/merged_review_checks/independent_audit.json').read_text()),
            source_hashes={str(p.relative_to(PAPER)):sha(p) for p in sorted(files)},
            pdf_hashes={name:sha(PAPER/name) for name in ('merged_tcom.pdf','merged_supplement.pdf')},
            native_compiler='External project figures unsupported; full TeX Live build successful')
(PAPER/'revision_manifest.json').write_text(json.dumps(record,indent=2)+'\n')
files.append(PAPER/'revision_manifest.json')
overleaf=OUT/'rydberg_merged_tcom_corrected_overleaf.zip'
with zipfile.ZipFile(overleaf,'w',zipfile.ZIP_DEFLATED) as z:
    for p in sorted(files): z.write(p,p.relative_to(PAPER))

verify=Path(tempfile.mkdtemp(prefix='merged-overleaf-verify-',dir=BASE/'tmp/pdfs'))
with zipfile.ZipFile(overleaf) as z: z.extractall(verify)
for name in ('merged_tcom','merged_supplement'):
    for command in (['pdflatex','-interaction=nonstopmode','-halt-on-error',name+'.tex'],
                    ['bibtex',name],
                    ['pdflatex','-interaction=nonstopmode','-halt-on-error',name+'.tex'],
                    ['pdflatex','-interaction=nonstopmode','-halt-on-error',name+'.tex']):
        run(command,verify)
    assert pages(verify/f'{name}.pdf')==pages(PAPER/f'{name}.pdf')
    assert run([PDFTOTEXT,'-layout',str(verify/f'{name}.pdf'),'-'],BASE)==run(
        [PDFTOTEXT,'-layout',str(PAPER/f'{name}.pdf'),'-'],BASE)
    assert not re.search(r'Overfull|LaTeX Warning:|^!',(verify/f'{name}.log').read_text(),re.M)

old=json.loads((BASE/'results/paper2_optshrink_loop/manifest.json').read_text())
evidence=set(files)
for folder in ('merged_review_checks','rydberg_sim','codex_verification','sahure',
               'sahure_expansion','paper2_optshrink_loop','paper2_review_checks'):
    evidence.update(p for p in (BASE/folder).glob('*.py'))
evidence.update(BASE/p for p in old['sources'])
evidence.update(BASE/v['path'] for v in old['models'].values())
evidence.update((BASE/'merged_review_checks').glob('*.txt'))
for folder in ('results/paper2_optshrink_loop','results/merged_review_checks'):
    evidence.update(p for p in (BASE/folder).rglob('*') if p.is_file() and p.suffix in ('.json','.npz','.png','.txt'))
archive=OUT/'rydberg_merged_review_evidence.zip'
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
    for p in sorted(evidence): z.write(p,p.relative_to(BASE))

portable=Path(tempfile.mkdtemp(prefix='merged-evidence-verify-',dir=BASE/'tmp/pdfs'))
with zipfile.ZipFile(archive) as z: z.extractall(portable)
portable_audit=run([sys.executable,'-m','merged_review_checks.audit'],portable)
result=dict(main_pages=record['main_pages'],supplement_pages=record['supplement_pages'],
            abstract_words=record['abstract_words'],overleaf_compile='both documents successful; identical extracted text',
            no_overfull_boxes=True,portable_numerical_audit=json.loads(portable_audit),
            overleaf_sha256=sha(overleaf),evidence_sha256=sha(archive),
            overleaf_bytes=overleaf.stat().st_size,evidence_bytes=archive.stat().st_size,
            original_readonly_pdf_sha256=sha('/Users/haatim/Downloads/rydberg_merged_tcom.pdf'))
(OUT/'rydberg_merged_tcom_corrected_checks.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
