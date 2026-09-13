"""Optional fresh fit/sweep in a separate output directory; preserves original records."""
from pathlib import Path
import argparse,sys,tempfile,shutil,subprocess
R=Path(__file__).resolve().parents[1];p=argparse.ArgumentParser();p.add_argument('experiment',choices=['yeast','sensitivity']);p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output.resolve()
if out.exists():raise ValueError('Use a new output directory')
for n in ['paper','evidence','code','forecast_archive','legacy','data_parts']:
 if R/n==out or R/n in out.parents:raise ValueError('Cannot overwrite scientific inputs')
with tempfile.TemporaryDirectory() as td:
 t=Path(td);(t/'additional_experiments').mkdir();(t/'review_revision').mkdir();(t/'review_revision/CENSORCAST').symlink_to(R,target_is_directory=True)
 if a.experiment=='yeast':
  dest=t/'additional_experiments/yeast';dest.mkdir()
  for n in ['PROTOCOL.json','yeast-train.arff','yeast-test.arff']:shutil.copy2(R/'evidence/additional/yeast'/n,dest/n)
 else:subprocess.run([sys.executable,str(R/'code/extract_development_inputs.py'),'--output',str(t/'integration_inputs')],check=True)
 subprocess.run([sys.executable,str(R/'evidence/additional'/f'run_{a.experiment}.py')],cwd=t,check=True)
 out.parent.mkdir(parents=True,exist_ok=True);shutil.copytree(t/'additional_experiments'/a.experiment,out)
print('Fresh replay outputs:',out)
