"""Verify numerical reporting, checkpoint consistency, and notebook sources."""
from pathlib import Path
import hashlib,json,zipfile,base64,io,ast,tempfile
import numpy as np
from run_selective_study import load,metric
root=Path(__file__).resolve().parents[1]
manifest=json.loads((root/'provenance/input_manifest.json').read_text())
for name,v in manifest.items():
    with (root/'inputs/m5'/name).open('rb') as f:h=hashlib.file_digest(f,'sha256').hexdigest()
    assert h==v['sha256'],name
checked=0
for study in ['selective','selective_strong']:
    folder=root/'results'/study
    doc=json.loads((folder/'SELECTIVE_RESULTS.json').read_text())
    data=load(folder/'cache/shadow_aligned.npz');meta=load(folder/'cache/metadata.npz')
    assert data['truth'].shape==(18290,114)
    for r in doc['records']:
        if r['score'] not in ['contract_excess','relative_error','mean_error','quantile_error','paired_excess']:continue
        m=load(folder/f"mask_{r['score']}_{r['threshold_mode']}_s{r['seed']}.npz")['shadow']
        actual=metric(data['truth'],data['proposal'],data['baseline'],m)
        expected=r['metrics']['shadow']
        assert actual['accepted_n']==expected['accepted_n']
        assert actual['wape']==expected['wape']
        source='mean_error' if r['score']=='relative_error' else r['score']
        s=load(folder/f"{source}_s{r['seed']}_scores.npz")['shadow']
        if r['score']=='relative_error':s=np.maximum(s,0)/np.maximum(data['proposal'],.25)
        mask=np.zeros(m.shape,bool)
        for group,t in r['thresholds'].items():
            rows=np.ones(len(m),bool) if group=='ALL' else meta['cat_id']==group
            if t is not None:mask[rows]=s[rows]<=t
        np.testing.assert_array_equal(m,mask)
        checked+=1
    assert not doc['fresh_guardian_opened'] and not doc['external_opened'] and not doc['certificate_issued']
npzs=0
for folder in ['point_baselines','adapter_v2']:
    for p in (root/'results'/folder).glob('*.npz'):
        with zipfile.ZipFile(p) as z:assert z.testzip() is None,p
        npzs+=1
nbpath=root.parent/'deliverables/RUN_CENSORCAST_ICLR_ADDITIONAL_EXPERIMENTS.ipynb'
if not nbpath.exists():nbpath=root/'RUN_CENSORCAST_ICLR_ADDITIONAL_EXPERIMENTS.ipynb'
nb=json.loads(nbpath.read_text());codes=[''.join(c['source']) for c in nb['cells'] if c['cell_type']=='code']
for c in codes:compile(c,nbpath.name,'exec')
tree=ast.parse(codes[1]);assign={n.targets[0].id:ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id in ['BUNDLE','BUNDLE_SHA256']}
payload=base64.b64decode(assign['BUNDLE']);assert hashlib.sha256(payload).hexdigest()==assign['BUNDLE_SHA256']
with zipfile.ZipFile(io.BytesIO(payload)) as z:
    assert z.testzip() is None
    for name in z.namelist():
        assert z.read(name)==(root/name).read_bytes(),name
with tempfile.TemporaryDirectory() as td:
    p=Path(td);(p/'prepared').mkdir();(p/'source').mkdir();(p/'run/checkpoints').mkdir(parents=True);(p/'run/results').mkdir()
    # Exercise the notebook's actual source path resolver using local links.
    for name in manifest:
        if name=='design_outcomes_v0_5.npz':dest=p/'prepared'/name
        elif name in ['calendar.csv','sell_prices.csv']:dest=p/'source'/name
        elif name=='DEVELOPMENT_DECISION_v0_5.json':dest=p/'run/results'/name
        else:dest=p/'run/checkpoints'/name
        dest.symlink_to((root/'inputs/m5'/name).resolve())
    fn=next(n for n in ast.parse(codes[3]).body if isinstance(n,ast.FunctionDef) and n.name=='input_source')
    env={'DATA_ROOT':p,'OLD_RUN':p/'run'}
    exec(compile(ast.Module(body=[fn],type_ignores=[]),'notebook-resolver','exec'),env)
    for name in manifest:assert env['input_source'](name).resolve()==(root/'inputs/m5'/name).resolve()
receipt={'status':'MANUSCRIPT_NUMERICAL_AND_NOTEBOOK_AUDIT_PASS',
    'verified_input_files':len(manifest),'selective_reports_recomputed':checked,
    'point_and_adapter_archives_crc_checked':npzs,'notebook_code_cells_compiled':len(codes),
    'notebook_embedded_sources_match':True,'notebook_source_paths_exercised':True,
    'outcome_shards_used':['previously_consumed_M5_design'],'fresh_guardian_opened':False,'external_opened':False,
    'certificate_issued':False,'statistical_scope':'POST_HOC_RETAIL_DEVELOPMENT_AND_SEPARATE_SYNTHETIC_EXPERIMENTS',
    'colab_live_mount_tested':False,'colab_note':'Compute scripts executed locally; authenticated Colab mounting not executed in this session.'}
(root/'provenance/VERIFICATION_RECEIPT.json').write_text(json.dumps(receipt,indent=2))
print(json.dumps(receipt,indent=2))
