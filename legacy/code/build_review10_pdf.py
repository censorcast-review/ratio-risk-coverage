"""Build the current paper in isolation; packaged inputs are never overwritten."""
from pathlib import Path
import argparse,json,re,shutil,subprocess,tempfile
from pypdf import PdfReader
from r2_io import sha,write,dump
ROOT=Path(__file__).resolve().parents[1]

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=ROOT/'reproduction_outputs/paper')
    parser.add_argument('--max-main-pages',type=int,default=9)
    parser.add_argument('--max-total-pages',type=int,default=21)
    args=parser.parse_args();out=args.output_dir.resolve()
    for name in ('paper','results','provenance','inputs','code','docs'):
        protected=(ROOT/name).resolve()
        if out==protected or protected in out.parents:
            parser.error('Output must be outside packaged scientific inputs: '+str(protected))
    paper=ROOT/'paper'
    before={str(p.relative_to(paper)):sha(p) for p in paper.rglob('*') if p.is_file()}
    with tempfile.TemporaryDirectory(prefix='ratio-risk-paper-') as tmp:
        build=Path(tmp)
        for rel in before:
            p=paper/rel
            if p.suffix in {'.tex','.sty','.bst','.bib','.png'} or (p.suffix=='.pdf' and 'figures' in p.parts):
                dst=build/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dst)
        commands=[['pdflatex','-interaction=nonstopmode','-halt-on-error','main.tex'],['bibtex','main'],['pdflatex','-interaction=nonstopmode','-halt-on-error','main.tex'],['pdflatex','-interaction=nonstopmode','-halt-on-error','main.tex']]
        for command in commands:
            result=subprocess.run(command,cwd=build,capture_output=True,text=True)
            if result.returncode:
                raise RuntimeError(result.stdout[-12000:]+result.stderr[-2000:])
        log=(build/'main.log').read_text();aux=(build/'main.aux').read_text()
        m=re.search(r'newlabel\{end:main\}\{\{[^}]*\}\{(\d+)\}',aux)
        assert m,'Main-text page marker absent'
        main_pages=int(m.group(1));pages=len(PdfReader(build/'main.pdf').pages)
        problems=[line for line in log.splitlines() if 'Overfull' in line or 'undefined references' in line or 'undefined citations' in line]
        for name in ('main.pdf','main.log','main.aux','main.bbl'):
            write(out/name,(build/name).read_bytes())
    after={str(p.relative_to(paper)):sha(p) for p in paper.rglob('*') if p.is_file()}
    assert before==after,'Build modified packaged paper inputs'
    ok=not problems and main_pages<=args.max_main_pages and pages<=args.max_total_pages
    audit={'status':'PASS' if ok else 'LAYOUT_REQUIRES_ATTENTION','pages':pages,'main_pages':main_pages,'diagnostics':problems,'packaged_inputs_unchanged':True,'pdf_sha256':sha(out/'main.pdf'),'source_sha256':sha(paper/'main.tex'),'visual_review':'SEPARATE'}
    dump(out/'PDF_BUILD_AUDIT.json',audit);print(json.dumps(audit,indent=2))
    if not ok:raise SystemExit(1)
if __name__=='__main__':main()
