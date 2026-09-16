"""Assemble a portable plot/data/source bundle, with PDF rendering checks."""
import argparse,hashlib,json,shutil,zipfile
from pathlib import Path
import pymupdf
from codex_verification.run import OUT,source_hash
from codex_verification.analyze import PUB

PLOTS=[("figure2_performance_array","Main performance and array size"),
       ("path_count","Path-count sweep"),
       ("figure3_original_design","Effective rank: original mixed design"),
       ("figure3_matched","Effective rank: matched design"),
       ("distribution_transfer","Alternative channel distributions"),
       ("users_and_forced_constraint","User count and forced projection"),
       ("projection_placement","Controlled projection placement"),
       ("runtime","Serial runtime measurements")]


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--output",required=True);a=ap.parse_args()
    target=Path(a.output).resolve();target.mkdir(parents=True,exist_ok=True)
    book=pymupdf.open();toc=[];qa=[]
    previews=OUT/"pdf_render_checks";previews.mkdir(exist_ok=True)
    for stem,title in PLOTS:
        p=PUB/(stem+".pdf");doc=pymupdf.open(p)
        assert len(doc)==1
        page=doc[0];assert len(page.get_text())>40
        # All text must lie within the exported tight bounding box.
        for block in page.get_text("blocks"):
            rect=pymupdf.Rect(block[:4])
            assert (page.rect+(-1,-1,1,1)).contains(rect),(stem,rect,page.rect)
        page.get_pixmap(matrix=pymupdf.Matrix(2,2)).save(previews/(stem+".png"))
        toc.append([1,title,len(book)+1]);book.insert_pdf(doc)
        qa.append(dict(plot=stem,pages=1,text_characters=len(page.get_text()),
                       embedded_fonts=[f[3] for f in page.get_fonts()]))
    book.set_toc(toc);book.set_metadata(dict(title="Paper 1 independent verification: all plots",author="Abdullah Haatim; independent Codex verification"))
    book.save(PUB/"all_plots.pdf",garbage=4,deflate=True)
    (PUB/"pdf_checks.json").write_text(json.dumps(qa,indent=2))
    for p in PUB.iterdir():
        if p.is_file():shutil.copy2(p,target/p.name)
    repro=target/"reproduction";cv=repro/"codex_verification";cv.mkdir(parents=True,exist_ok=True)
    root=OUT.parent;repo=root.parent
    for p in root.iterdir():
        if p.is_file() and p.suffix in (".py",".md",".txt",".json"):
            shutil.copy2(p,cv/p.name)
    for name in ("trials","bounds"):
        shutil.copytree(OUT/name,cv/"results"/name,dirs_exist_ok=True)
    for name in ("manifest.json","validation.json","stored_trial_spotchecks.json",
                 "COMPLETE.json","runtime.json","pytest.log","spotchecks.log","hardware.json"):
        if (OUT/name).exists():shutil.copy2(OUT/name,cv/"results"/name)
    for p in (repo/"rydberg_sim").glob("*.py"):
        (repro/"rydberg_sim").mkdir(exist_ok=True);shutil.copy2(p,repro/"rydberg_sim"/p.name)
    (repro/"scripts").mkdir(exist_ok=True)
    shutil.copy2(repo/"scripts/constrained_crlb.py",repro/"scripts/constrained_crlb.py")
    # Existing summaries are convenient; the reproduction scripts can rebuild
    # every figure from the portable trial files without resimulating.
    (cv/"publication").mkdir(exist_ok=True)
    for name in ("verification_results.json","supplemental_statistics.json"):
        shutil.copy2(PUB/name,cv/"publication"/name)
    (target/"START_HERE.md").write_text("# Paper 1 independent verification\n\n"
        "Read VERIFICATION_REPORT.md, then open all_plots.pdf. Individual plot files "
        "are supplied as PDF, SVG and 300-dpi PNG. suggested_replacements.tex contains "
        "integration snippets for the nine-page manuscript.\n\n"
        "The reproduction folder contains the exact estimator/generator sources, "
        "27,300 per-trial records, paired bound records, software pins and checks. "
        "From that folder, install codex_verification/requirements.txt, then run "
        "`python -m codex_verification.analyze` and `python -m codex_verification.supplement` "
        "to rebuild the analysis without rerunning simulations. See its README for full reruns.\n\n"
        "Source fingerprint: `"+source_hash()+"`. Local branch: `codex`, based on `739dba4`.\n")
    hashes={str(p.relative_to(target)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in target.rglob("*") if p.is_file() and p.name!="SHA256SUMS.json"}
    (target/"SHA256SUMS.json").write_text(json.dumps(hashes,indent=2))
    zpath=target.parent/(target.name+".zip")
    with zipfile.ZipFile(zpath,"w",zipfile.ZIP_DEFLATED) as z:
        for p in sorted(target.rglob("*")):
            if p.is_file():z.write(p,p.relative_to(target.parent))
    with zipfile.ZipFile(zpath) as z:assert z.testzip() is None
    print(json.dumps(dict(output=str(target),zip=str(zpath),files=len(hashes)+1,pdf_pages=len(book)),indent=2))

if __name__=="__main__":main()
