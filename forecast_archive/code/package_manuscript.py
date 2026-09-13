"""Create a compact, neutral-named manuscript/evidence archive."""
from pathlib import Path
import hashlib,json,zipfile
root=Path(__file__).resolve().parents[1]
excluded_dirs={'qa','tmp','__pycache__'}
excluded_suffixes={'.aux','.log','.out','.blg','.fls','.fdb_latexmk','.synctex.gz'}
excluded_names={'MANIFEST.json','iclr-style.zip','CENSORCAST_ICLR_2027_Accuracy_Manuscript.pdf'}
files=[]
for p in sorted(root.rglob('*')):
    rel=p.relative_to(root)
    if not p.is_file() or any(x in excluded_dirs for x in rel.parts):continue
    if p.suffix in excluded_suffixes or p.name in excluded_names:continue
    files.append(p)
manifest={'schema':'censorcast-risk-calibration-revision-3','files':{str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},'historical_frozen_evidence_modified':False,'raw_uci_source':'public download; checksum in STUDY_PROTOCOL.json','m5_foundation_forecast_arrays_included':False,'risk_calibration_three_seed_models_included':True,'m5_observable_first_seed_head_arrays_included':True,'m5_development_inputs_included':True,'separate_selection_companion_included':True}
(root/'MANIFEST.json').write_text(json.dumps(manifest,indent=2))
target=root.parent/'CENSORCAST_RC_ICLR_2027_Source_and_Evidence.zip'
with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
    for p in files+[root/'MANIFEST.json']:z.write(p,Path('CENSORCAST')/p.relative_to(root))
with zipfile.ZipFile(target) as z:
    assert z.testzip() is None
    for name,digest in manifest['files'].items():assert hashlib.sha256(z.read('CENSORCAST/'+name)).hexdigest()==digest
print(json.dumps({'archive':str(target),'bytes':target.stat().st_size,'sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'verified_files':len(files)}))
