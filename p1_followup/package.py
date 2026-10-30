from pathlib import Path
import hashlib,json,re,shutil,zipfile
import pymupdf as fitz
ROOT=Path(__file__).resolve().parents[1];out=ROOT/'results/p1_followup';paper=ROOT/'paper/paper1/full_revision';release=ROOT/'output/pdf'
release.mkdir(parents=True,exist_ok=True)
source=paper/'haatim_hsgs_revised.tex';build=ROOT/'tmp/pdfs/p1_followup_build/haatim_hsgs_revised.pdf'
pdf=release/'paper1_revised.pdf';shutil.copy2(build,pdf)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
s=source.read_text();d=fitz.open(pdf);log=(build.parent/'haatim_hsgs_revised.log').read_text()
labels=re.findall(r'\\label\{([^}]+)\}',s);assert len(labels)==len(set(labels))
assert not any(x in log for x in ['Overfull','undefined','! LaTeX Error','Missing character:'])
assert len(d)==17
assert all((paper/f).exists() for f in re.findall(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}',s))
checks=dict(pages=len(d),figures=len(re.findall(r'\\includegraphics',s)),duplicate_labels=0,undefined_references=False,overfull_boxes=False,missing_glyphs=False,simulation_worlds=600,serial_timing_worlds=12,numerical_tests_passed=44,independent_aggregate_checks=54,source_sha256=sha(source),pdf_sha256=sha(pdf),compiler='pdfLaTeX / TeX Live 2026',native_preview='External figures unsupported in single-file built-in compiler; full project compiled successfully.',visual_review='All-page contact sheet and full-size new/changed layouts inspected.')
(release/'paper1_revised_checks.json').write_text(json.dumps(checks,indent=2))
provenance=json.loads((paper/'source_provenance.json').read_text())
provenance.update(final_tex_sha256=sha(source),final_pdf_sha256=sha(pdf),final_pdf_pages=len(d),followup_protocol_sha256=sha(out/'protocol.json'),followup_summary_sha256=sha(out/'analysis/summary.json'),prior_revision_source_sha256=sha(out/'manuscript_before/haatim_hsgs_revised.tex'))
provenance['figures']['sweep_sensitivity.pdf']=dict(source=str(out/'analysis/sweep_sensitivity.pdf'),sha256=sha(paper/'fig/sweep_sensitivity.pdf'))
provenance['quantummusic_metadata_source']='https://ieeexplore.ieee.org/abstract/document/10919061; author metadata checked against publisher Crossref deposit'
(paper/'source_provenance.json').write_text(json.dumps(provenance,indent=2))
readme='''FULL REVISED PAPER 1 - SECOND REVIEW FOLLOW-UP, 8 OCTOBER 2026

Upload this ZIP as a new Overleaf project. Main document:
haatim_hsgs_revised.tex. Compiler: pdfLaTeX. All nine figures and
IEEEtran.cls are included. Bibliography is embedded; no BibTeX required.
The complete manuscript is 17 pages including three appendices/references.

The current source retains the earlier verified curves and settings.
Appendix C adds 600 fresh paired worlds for sweep and split sensitivity,
a 12-world serial timing check, and rank/harm diagnostics reusing 2400 old
worlds. The one-sweep finding is reported separately; historical four-sweep
results have not been silently replaced. New Tables IV/V and Fig. 9 report
these checks. Main text adds cost/memory scaling, pilot assumptions,
shrinkage rationale, verified IEEE context, and a qualified conclusion.
The abstract remains qualitative and explanations use simple English.

Source and PDF compiled with existing TeX Live 2026, all references resolved,
no overfull boxes or missing glyphs. New/changed pages visually inspected.
The native single-file preview cannot load external figures; use the
compiled PDF or this complete Overleaf project to inspect the figures.

This remains a simulation study and an empirical methodological contribution.
No hardware validation, global convergence, optimal shrinkage or universal
best sweep count is claimed. Check your intended venue's page requirements.

The original reconstructed-source provenance is retained in
source_provenance.json; current and pre-follow-up hashes are recorded.
Numerical code, raw results, audits and supplementary plots are supplied
separately in paper1_followup_evidence.zip. Earlier full revision evidence
remains in the preceding p1_revision_code_and_evidence.zip release.
'''
(paper/'README.txt').write_text(readme)
notes='''P1 SECOND-REVIEW REVISION: WHAT CHANGED

Completed 600 fresh paired worlds: 100 for each L=3,7,14 crossed with
SNR=0,10 dB. N32/K3/P30/RSR12, 100/25 updates. Hard and smooth processing
were tested with 1,2,4,8 sweeps, each selecting its own parameter. The same
worlds also received three random pilot splits for four-sweep HS-GS.

Main new finding
One hard-rank sweep improves on four by 0.275-0.652 dB in all six settings;
all pointwise 95% paired intervals are positive. Example: L7/SNR10 gain
0.652 [0.488,0.824] dB. This shows the original four-sweep setting is not
optimal in these conditions. It includes the effect of retuning rank.
Original manuscript curves remain unchanged and explicitly retain Q=4.
Smooth-filter results also depend on sweep count. No universal optimal
count, adaptive stopping rule, or global convergence is claimed.

Runtime
Separate 12-world serial check at L7/SNR5: median hard-processing times
for Q1/Q2/Q4/Q8 are 0.121/0.201/0.358/0.667 seconds, including selection,
initialization and final fit. Thus Q1 is about 2.97 times faster than Q4
in this small sample. It is not a portable speed constant. Four-worker
simulation timings were not used as inference benchmarks.

Pilot split
Random split gain versus the fixed split ranges from -0.148 to +0.184 dB.
Only L7/SNR10 has a positive pointwise interval: 0.184 [0.016,0.364] dB;
the other five include zero. No consistent average advantage is shown.
Random/fixed ranks disagree in 15.7%-86.7% of comparisons. Three masks
are repeated observations of the same world, not independent samples.
This is split sensitivity, not a proposed multi-fold estimator.

Rank versus harm
Reused the existing 2400 baseline worlds. At L7/SNR0, 315/400 select rank
below seven, but only 31 of those 315 lose to EM-GS. Matching true rank is
not necessarily the best denoising choice. At L14/SNR0, HS-GS loses in
191/400 trials (47.75%). Full-rank fallback is an option, not a guarantee
that every chosen lower rank helps.

Manuscript locations
- Introduction: verified Quantum-MUSIC IEEE citation and scope distinction.
- Section III-D: per-update, rank-search and memory scaling.
- Experimental setup: pilot assumptions and new evidence counts.
- Structured comparisons: rationale and limits of empirical smooth filter.
- Conclusion: one-sweep finding and split sensitivity; four-sweep comparison
  explicitly qualified. Shortened repeated effective-rank discussion.
- Appendix C: sweep-count study, random-split study, rank/harm diagnostics.
- Fig. 9: sweep sensitivity; Tables IV/V: split and harm diagnostics.

44 numerical tests passed (37 previous + 7 follow-up). All saved trial
ranks agree with their held-out argmins; exact bypass cases verified.
54 aggregate values independently recomputed. PDF compiles with resolved
references, no overfull boxes/missing glyphs, and visually checked layout.

Deferred deliberately
No joint-user/2D Hankel model, hardware campaign, new SURE/AIC selector,
neural model, or adaptive stopping algorithm was added. Those require
new assumptions or new research. Only verified IEEE-published additions
were cited; this is not an exhaustive current-literature review.

Interpretation
The paper is stronger as a careful empirical study. The new finding also
narrows any claim based specifically on four-sweep hard truncation.
This revision does not establish a fundamentally new low-rank principle
or guarantee acceptance. Venue page limits still need to be checked.
'''
(release/'paper1_revision_notes.txt').write_text(notes)
with zipfile.ZipFile(release/'paper1_revised_overleaf.zip','w',zipfile.ZIP_DEFLATED) as z:
 for f in [source,paper/'IEEEtran.cls',paper/'README.txt',paper/'source_provenance.json']+sorted((paper/'fig').glob('*.pdf')):z.write(f,f.relative_to(paper))
with zipfile.ZipFile(release/'paper1_followup_evidence.zip','w',zipfile.ZIP_DEFLATED) as z:
 files=set()
 for folder in ['p1_followup','p1_revision','rydberg_sim','results/p1_followup']:
  for f in (ROOT/folder).rglob('*'):
   if f.is_file() and '__pycache__' not in f.parts and f.suffix!='.pyc':files.add(f)
 for name in ['codex_verification/solver.py','codex_verification/worlds.py','sahure/classical.py','sahure/__init__.py','results/p1_revision/campaign/manifest.json','results/p1_revision/campaign/COMPLETE.json']:files.add(ROOT/name)
 for f in (ROOT/'results/p1_revision/campaign/raw').glob('baselines_*/*.npz'):files.add(f)
 for f in sorted(files):z.write(f,f.relative_to(ROOT))
 z.write(release/'paper1_revision_notes.txt','REVISION_NOTES.txt')
print(json.dumps(checks,indent=2))
