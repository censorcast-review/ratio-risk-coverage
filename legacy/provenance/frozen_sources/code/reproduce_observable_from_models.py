"""Recreate moments/readouts in a new directory, reusing all 33 saved fits.

No original result or protocol is overwritten. This is a reproduction helper;
the completed paper run and its repair history remain in the source package.
"""
from pathlib import Path
import argparse,json,shutil,subprocess,sys
from r2_io import sha,dump
ROOT=Path(__file__).resolve().parents[1]
def main():
    a=argparse.ArgumentParser();a.add_argument('--data',type=Path,required=True);a.add_argument('--destination',type=Path,required=True);p=a.parse_args()
    target=p.destination.resolve()
    if target.exists():raise RuntimeError('Use a new reproduction directory; existing results are preserved')
    branches=['observable_extension','capacity_hit_extension'];manifest={}
    for b,n in zip(branches,[21,12]):
        ms=sorted((ROOT/'results'/b/'models').glob('*.txt'));assert len(ms)==n
        for model in ms:
            rr=model.with_suffix('.json');info=json.loads(rr.read_text());assert sha(model)==info['model_sha256']
            manifest[str(model.relative_to(ROOT))]=sha(model)
    shutil.copytree(ROOT/'code',target/'code',ignore=shutil.ignore_patterns('__pycache__'))
    point='results/point_baselines/observed_l1_s20260906.txt';q=target/point;q.parent.mkdir(parents=True);shutil.copy2(ROOT/point,q)
    shutil.copytree(ROOT/'results/review2/cache_design',target/'results/review2/cache_design')
    for b in branches:shutil.copytree(ROOT/'results'/b/'models',target/'results'/b/'models')
    dump(target/'MODEL_REUSE_MANIFEST.json',{'model_hashes':manifest,'expected_new_fit_calls':0,'source_project':str(ROOT),
      'fresh_guardian_opened':False,'external_opened':False,'scope':'REPRODUCTION_ON_CONSUMED_DESIGN_ONLY'})
    for b,runner in zip(branches,['run_observable_extension.py','run_capacity_hit_extension.py']):
        subprocess.run([sys.executable,'-u',str(target/'code'/runner),'--data',str(p.data.resolve()),'--output',str(target/'results'/b)],check=True,cwd=target/'code')
        subprocess.run([sys.executable,str(target/'code/analyze_observable_extension.py'),'--branch',b],check=True,cwd=target/'code')
    for name,h in manifest.items():assert sha(target/name)==h
    print(json.dumps({'status':'SAVED_MODEL_REPRODUCTION_COMPLETE','models_reused':len(manifest),'model_hashes_unchanged':True,'new_directory':str(target)},indent=2))
if __name__=='__main__':main()
