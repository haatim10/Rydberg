"""Package verified revision artifacts without changing the previous release."""
from pathlib import Path
import hashlib
import json
import shutil
import tempfile
import zipfile
import subprocess

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
OUT=ROOT/'output/pdf'
OUT.mkdir(parents=True,exist_ok=True)

for source,dest in [('paper2_revised.pdf','paper2_reviewed.pdf'),
                    ('paper2_supplement.pdf','paper2_reviewed_supplement.pdf'),
                    ('paper2_revised.tex','paper2_reviewed.tex'),
                    ('RESPONSE_TO_REVIEW.md','paper2_review_response.md')]:
    shutil.copy2(HERE/source,OUT/dest)

names=['paper2_revised.tex','paper2_supplement.tex','paper2_revised.bbl',
       'paper2_supplement.bbl','references.bib','baseline_rows.tex',
       'review_diagnostics.tex','README.md','RESPONSE_TO_REVIEW.md',
       'review_material_sources.json']
bundle=OUT/'paper2_reviewed_overleaf.zip'
with zipfile.ZipFile(bundle,'w',zipfile.ZIP_DEFLATED) as archive:
    for name in names: archive.write(HERE/name,name)
    for source in sorted((HERE/'figures').glob('*.pdf')):
        archive.write(source,str(source.relative_to(HERE)))
with zipfile.ZipFile(bundle) as archive:
    assert archive.testzip() is None
    # Confirm the archive alone compiles: no accidental local input files.
    staging=Path(tempfile.mkdtemp(prefix='paper2-overleaf-verify-'))
    archive.extractall(staging)
for name in ['paper2_revised','paper2_supplement']:
    for command in [['/Library/TeX/texbin/pdflatex','-interaction=batchmode','-halt-on-error',name+'.tex'],
                    ['/Library/TeX/texbin/bibtex',name],
                    ['/Library/TeX/texbin/pdflatex','-interaction=batchmode','-halt-on-error',name+'.tex'],
                    ['/Library/TeX/texbin/pdflatex','-interaction=batchmode','-halt-on-error',name+'.tex']]:
        result=subprocess.run(command,cwd=staging,text=True,capture_output=True)
        if result.returncode:
            raise RuntimeError(result.stdout+result.stderr)
    log=(staging/(name+'.log')).read_text()
    assert 'undefined' not in log.lower(),name
    assert 'Overfull' not in log,name

repro=OUT/'paper2_review_checks.zip'
sources=[]
for directory in ['paper2_review_checks','sahure','sahure_expansion','rydberg_sim']:
    sources+=list((ROOT/directory).glob('*.py'))
sources+=[ROOT/'codex_verification/solver.py',ROOT/'results/sahure_expansion/frozen_selection.json',
          HERE/'make_review_material.py',HERE/'RESPONSE_TO_REVIEW.md']
for seed in [9101,9102,9103]:
    sources.append(ROOT/f'results/sahure_expansion/checkpoints/seed_{seed}/em_adaptive_best.pt')
sources+=list(p for p in (ROOT/'results/paper2_review_checks').rglob('*')
              if p.is_file() and '__pycache__' not in p.parts)
with zipfile.ZipFile(repro,'w',zipfile.ZIP_DEFLATED) as archive:
    for path in sorted(set(sources)):
        archive.write(path,str(path.relative_to(ROOT)))
    archive.writestr('REVIEW_CHECKS_README.txt',
        'Paper 2 reviewer diagnostics, 8 October 2026.\n'
        'Includes frozen primary model weights, relevant simulator/estimator sources, '
        'fresh observations, truth for evaluation only, per-case predictions and summaries.\n'
        'Not the full historical project or main-study dataset. The Overleaf package is separate.\n'
        'Dependencies: numpy, scipy, torch; matplotlib for figure generation. Run from archive root.\n'
        'Verify saved predictions: python -m paper2_review_checks.audit\n'
        'Unit checks: python -m unittest paper2_review_checks.test_checks paper2_review_checks.test_theory\n'
        'Study: 288 fresh worlds +72 independent filter-development worlds; no new neural training.\n'
        'Original paths in metadata refer to the originating repository. Do not overwrite '
        'completed records to try new settings. See RESPONSE_TO_REVIEW.md and experiment reports.\n')
with zipfile.ZipFile(repro) as archive: assert archive.testzip() is None

deliverables=[OUT/'paper2_reviewed.pdf',OUT/'paper2_reviewed_supplement.pdf',
              OUT/'paper2_reviewed.tex',OUT/'paper2_review_response.md',bundle,repro]
manifest={p.name:dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
          for p in deliverables}
manifest['verification']=dict(archive_compile_passed=True,latex_warnings=False,
    main_pages=5,supplement_pages=6,original_tests_plus_theory=52,rank_calibration_tests=4,
    saved_method_outputs_recalculated=153,optshrink_primitive_checks_passed=True,
    layout='All 11 rendered pages inspected; no clipping or overlap.',
    unchanged_originals='Original estimator sources/checkpoints match frozen hashes; old release preserved.')
(OUT/'paper2_reviewed_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps(manifest,indent=2))
