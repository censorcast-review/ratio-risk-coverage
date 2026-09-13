"""Run the theory, original frozen-evidence, and external-statistics checks."""
from pathlib import Path
import subprocess,json,sys
R=Path(__file__).resolve().parents[1];out=R/'reproduction_outputs/core';out.mkdir(parents=True,exist_ok=True);results=[]
for script in ['verify_public_package.py','verify_ratio_generalization.py','verify_bounded_exposure.py','verify_classification_example.py','verify_objective_geometry.py','verify_external_budget.py','verify_nb_nll_uncertainty.py','verify_review5_group_constraints.py']:
 p=subprocess.run([sys.executable,str(R/'legacy/code'/script)],capture_output=True,text=True);(out/(script+'.log')).write_text(p.stdout+p.stderr);results.append({'script':script,'returncode':p.returncode});print(script,'PASS' if p.returncode==0 else 'FAIL',flush=True)
 if p.returncode:raise SystemExit(p.returncode)
(out/'RESULTS.json').write_text(json.dumps({'status':'PASS','checks':results},indent=2)+'\n')
