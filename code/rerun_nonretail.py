"""Rerun a disclosed non-retail experiment into a new directory."""
from pathlib import Path
import tempfile,shutil,subprocess,sys,argparse
R=Path(__file__).resolve().parents[1];p=argparse.ArgumentParser();p.add_argument('dataset',choices=['bike','delicious']);p.add_argument('--output',type=Path,required=True);a=p.parse_args();o=a.output.resolve()
if o.exists():raise ValueError('Output must be new')
for folder in ['code','paper','evidence','legacy','forecast_archive','data_parts']:
 if o==R/folder or R/folder in o.parents:raise ValueError('Output overlaps scientific inputs')
with tempfile.TemporaryDirectory() as td:
 t=Path(td)
 for n in ['PROTOCOL.json','run.py','bike.zip' if a.dataset=='bike' else 'delicious.bz2']:shutil.copy2(R/'evidence/nonretail'/n,t/n)
 subprocess.run([sys.executable,str(t/'run.py'),a.dataset],check=True)
 o.parent.mkdir(parents=True,exist_ok=True);shutil.copytree(t/a.dataset,o)
print('Reproduction outputs:',o)
