"""Compile manuscript without modifying source or frozen evidence."""
from pathlib import Path
import argparse,shutil,tempfile,subprocess,json,re,hashlib
from pypdf import PdfReader
R=Path(__file__).resolve().parents[1]
def main():
 a=argparse.ArgumentParser();a.add_argument('--output',type=Path,default=R/'reproduction_outputs/paper');x=a.parse_args();out=x.output.resolve();out.mkdir(parents=True,exist_ok=True)
 for folder in ['paper','legacy','evidence','forecast_archive','code']:
  p=(R/folder).resolve()
  if out==p or p in out.parents:raise ValueError('Output must not overwrite scientific inputs')
 with tempfile.TemporaryDirectory() as tmp:
  b=Path(tmp);shutil.copytree(R/'paper',b,dirs_exist_ok=True)
  for cmd in [['pdflatex','-interaction=nonstopmode','-halt-on-error','main.tex'],['bibtex','main'],['pdflatex','-interaction=nonstopmode','-halt-on-error','main.tex'],['pdflatex','-interaction=nonstopmode','-halt-on-error','main.tex']]:
   q=subprocess.run(cmd,cwd=b,capture_output=True,text=True)
   if q.returncode:raise RuntimeError(q.stdout[-5000:])
  log=(b/'main.log').read_text();aux=(b/'main.aux').read_text();n=int(re.search(r'newlabel\{end:main\}\{\{[^}]*\}\{(\d+)\}',aux).group(1));bad=[s for s in log.splitlines() if 'Overfull' in s or 'undefined references' in s or 'undefined citations' in s]
  assert n<=9 and not bad,(n,bad)
  shutil.copy2(b/'main.pdf',out/'CENSORCAST_ICLR_2027_Final.pdf');dump={'status':'PASS','main_pages':n,'total_pages':len(PdfReader(b/'main.pdf').pages),'warnings':bad,'pdf_sha256':hashlib.sha256((b/'main.pdf').read_bytes()).hexdigest()};(out/'BUILD_REPORT.json').write_text(json.dumps(dump,indent=2)+'\n');print(json.dumps(dump))
if __name__=='__main__':main()
