"""Build and package this manuscript; no experiment generation or source mutation."""
from pathlib import Path
import argparse, concurrent.futures, hashlib, json, os, shutil, subprocess, tempfile, zipfile
P=Path(__file__).resolve().parent
REPO=P.parents[1]
BUILD=REPO/'tmp/pdfs/tcom_reliability_merged'
OUTPUT=REPO/'output/pdf'
STEM='rydberg_merged_tcom_reliability'
LATEX='/Library/TeX/texbin/pdflatex'
BIBTEX='/Library/TeX/texbin/bibtex'

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def compile_one(source,build,name):
 build.mkdir(parents=True,exist_ok=True)
 env=dict(os.environ,BIBINPUTS=str(source)+':')
 latex=[LATEX,'-interaction=nonstopmode','-halt-on-error','-file-line-error','-output-directory='+str(build),name+'.tex']
 for i,command in enumerate((latex,[BIBTEX,name],latex,latex)):
  with (build/f'{name}_pass{i}.txt').open('w') as f:
   r=subprocess.run(command,cwd=build if i==1 else source,stdout=f,stderr=subprocess.STDOUT,env=env)
  if r.returncode:raise RuntimeError((build/f'{name}_pass{i}.txt').read_text()[-5000:])
 log=(build/(name+'.log')).read_text()
 assert 'There were undefined references' not in log
 assert 'There were multiply-defined labels' not in log
 return build/(name+'.pdf')

def build():
 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
  for q in pool.map(lambda n:compile_one(P,BUILD,n),('main','supplement')):print('Compiled',q,flush=True)

def package():
 from pypdf import PdfReader
 OUTPUT.mkdir(parents=True,exist_ok=True)
 for name,suffix in [('main',''),('supplement','_supplement')]:
  assert (BUILD/f'{name}.pdf').exists()
  shutil.copy2(BUILD/f'{name}.pdf',OUTPUT/f'{STEM}{suffix}.pdf')
 archive=OUTPUT/f'{STEM}_overleaf.zip'
 allowed={'.tex','.bib','.pdf','.png','.json','.txt','.py','.svg'}
 files=[q for q in P.rglob('*') if q.is_file() and q.suffix in allowed and '__pycache__' not in q.parts and q.name not in ('mfput.log',) and not q.name.endswith('_build.txt')]
 with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
  for q in sorted(files):z.write(q,q.relative_to(P))
 with tempfile.TemporaryDirectory(prefix='rydberg-merged-verify-') as d:
  extracted=Path(d)
  with zipfile.ZipFile(archive) as z:z.extractall(extracted)
  for name in ('main','supplement'):
   pdf=compile_one(extracted,extracted/'_verify',name)
   a='\n\f\n'.join(p.extract_text(extraction_mode='layout') for p in PdfReader(BUILD/f'{name}.pdf').pages)
   b='\n\f\n'.join(p.extract_text(extraction_mode='layout') for p in PdfReader(pdf).pages)
   assert a==b,f'Extracted PDF text differs: {name}'
 result={'status':'verified','zip':str(archive),'zip_sha256':sha(archive),'entries':len(files),
  'extracted_main_and_supplement_compile':'pass','extracted_pdf_text_matches':True,
  'outputs':{n:{'path':str(OUTPUT/f'{STEM}{s}.pdf'),'sha256':sha(OUTPUT/f'{STEM}{s}.pdf')} for n,s in [('main',''),('supplement','_supplement')]}}
 (OUTPUT/f'{STEM}_verification.json').write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['build','package']);a=parser.parse_args()
 {'build':build,'package':package}[a.phase]()
