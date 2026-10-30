"""Render both final PDFs and make labelled contact sheets for visual QA."""
from pathlib import Path
import json
import subprocess
from PIL import Image, ImageDraw
import pdfplumber

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'output/pdf'
QA=ROOT/'tmp/pdfs/tcom_10page_system_review'
BIN='/Users/haatim/.cache/codex-runtimes/codex-primary-runtime/dependencies/bin/override/pdftoppm'

def main():
    QA.mkdir(parents=True,exist_ok=True)
    report={}
    for name in ('rydberg_tcom_10page_system',):
        path=OUT/(name+'.pdf');folder=QA/name;folder.mkdir(exist_ok=True)
        subprocess.run([BIN,'-r','110','-png',str(path),str(folder/'page')],check=True,capture_output=True)
        with pdfplumber.open(path) as pdf:
            text=[];outside=[]
            for i,p in enumerate(pdf.pages,1):
                t=p.extract_text() or '';text.append(t)
                for w in p.extract_words():
                    if w['x0'] < -1 or w['x1'] > p.width+1 or w['top'] < -1 or w['bottom']>p.height+1:
                        outside.append(dict(page=i,word=w['text']))
            alltext='\n\n'.join(text)
            assert not outside,outside
            assert '??' not in alltext and '\ufffd' not in alltext
            assert 'Research Draft' not in alltext and 'TODO' not in alltext
            (folder/'extracted_text.txt').write_text(alltext)
            report[name]=dict(pages=len(pdf.pages),out_of_page_words=outside,unresolved_text_markers=False)
        pages=sorted(folder.glob('page-*.png'))[:report[name]['pages']]
        for start in range(0,len(pages),4):
            tilew,tileh=800,1060
            sheet=Image.new('RGB',(tilew*2,tileh*2),'#dddddd');draw=ImageDraw.Draw(sheet)
            for j,p in enumerate(pages[start:start+4]):
                im=Image.open(p).convert('RGB');im.thumbnail((790,1028))
                x=(j%2)*tilew+(tilew-im.width)//2;y=(j//2)*tileh+25
                sheet.paste(im,(x,y));draw.text(((j%2)*tilew+10,(j//2)*tileh+7),f'{name}: page {start+j+1}',fill='black')
            sheet.save(folder/f'contact-{start//4+1}.png')
    (QA/'text_checks.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__': main()
