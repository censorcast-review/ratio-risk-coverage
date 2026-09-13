"""Verify the new revision artifacts without reopening any sealed data."""
from pathlib import Path
import ast,base64,hashlib,io,json,zipfile,tempfile
import numpy as np
from run_selective_study import load,metric
root=Path(__file__).resolve().parents[1];d=root/'results/conditional_heads'
result=json.loads((d/'REVIEW_COMPARISON.json').read_text());assert len(result['records'])==42
v=load(d/'cache/shadow_aligned.npz')
for r in result['records']:
 mask=load(d/f"mask_{r['score']}_{r['threshold_mode']}_s{r['seed']}.npz")['shadow']
 actual=metric(v['truth'],v['proposal'],v['baseline'],mask);expected=r['metrics']['shadow']['ALL']
 assert actual==expected
f=json.loads((root/'provenance/FROZEN_REVIEW_GUARDIAN_COMPARISON.json').read_text())
for rel,h in f['immutable_files'].items():assert hashlib.sha256((root/rel).read_bytes()).hexdigest()==h
notebooks=[root/'RUN_CENSORCAST_M5_FROZEN_COMPARATIVE_GUARDIAN.ipynb',root.parent/'deliverables/RUN_CENSORCAST_ICLR_ADDITIONAL_EXPERIMENTS.ipynb']
verified=[]
for nbpath in notebooks:
 nb=json.loads(nbpath.read_text());codes=[''.join(c['source']) for c in nb['cells'] if c['cell_type']=='code']
 for source in codes:compile(source,nbpath.name,'exec')
 vars={}
 for source in codes[:2]:
  for node in ast.parse(source).body:
   if isinstance(node,ast.Assign) and isinstance(node.targets[0],ast.Name) and node.targets[0].id in ['PAYLOAD_B64','BUNDLE','BUNDLE_SHA256']:
    vars[node.targets[0].id]=ast.literal_eval(node.value)
 raw=base64.b64decode(vars.get('PAYLOAD_B64',vars.get('BUNDLE')))
 with zipfile.ZipFile(io.BytesIO(raw)) as z:
  assert z.testzip() is None
  for rel in z.namelist():assert z.read(rel)==(root/rel).read_bytes(),rel
 verified.append({'notebook':nbpath.name,'code_cells_compiled':len(codes),'embedded_sources_match':True})
 if 'GUARDIAN' in nbpath.name:
  with tempfile.TemporaryDirectory() as tmp:
   env={'OPEN_RECEIPT':Path(tmp)/'missing.json'}
   try:exec(codes[3],env)
   except RuntimeError as e:assert 'remains sealed' in str(e)
   else:raise AssertionError('Default must not authorize')
   assert not list(Path(tmp).iterdir())
receipt={'status':'REVIEW_REVISION_VERIFIED','new_reports_recomputed_from_masks':42,'additional_demand_fits':3,'notebooks':verified,'frozen_files_verified':len(f['immutable_files']),'default_guardian_opening_cell_writes_zero_files':True,'guardian_opened':False,'external_opened':False,'guardian_runner_executed':False,'colab_live_mount_tested':False,'statistical_scope':'consumed_design_revision_only','theory_human_verification':'not asserted'}
(root/'provenance/REVIEW_VERIFICATION_RECEIPT.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt,indent=2))
