"""Package revision v12 while checking preservation of the preceding release."""
from pathlib import Path
import argparse,hashlib,json,zipfile,shutil,subprocess,datetime
R=Path(__file__).resolve().parents[1]
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(p):return json.loads(p.read_text())
def write(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
def main():
    p=argparse.ArgumentParser();p.add_argument('--prior-archive',type=Path,required=True);p.add_argument('--output',type=Path,required=True);x=p.parse_args();D=x.output.resolve();D.mkdir(exist_ok=True,parents=True)
    b=R/'reproduction_outputs/revision_v12_build';report=read(b/'BUILD_REPORT.json');qa=read(R/'docs/revision_v12/VISUAL_QA.json')
    assert report['status']=='PASS' and report['main_pages']==9 and report['total_pages']==48
    assert qa['all_pages_visually_verified'] and qa['pdf_sha256']==report['pdf_sha256']==sha(b/'CENSORCAST_ICLR_2027_Final.pdf')
    checks={}
    for part in ['SEED','EXTERNAL','THEORY']:
        q=read(R/f'docs/revision_v12/{part}_VERIFICATION.json');assert q['status']=='PASS' and q['verifier_sha256']==sha(R/'code/verify_revision_v12.py');checks[part]=q['checks']
    master=R/'evidence/revision_v12_seeds/PROTOCOL.json';mp=read(master);assert sha(master)==master.with_suffix('.sha256').read_text().split()[0]
    fit_calls=0
    for seed in mp['additional_seeds']:
        o=master.parent/f'seed_{seed}';q=read(o/'PROTOCOL.json');c=read(o/'COMPLETE.json');assert q['master_protocol_sha256']==sha(master) and mp['utc']<q['utc']<c['utc'];fit_calls+=c['fit_calls']
        for rel,h in c['files'].items():assert sha(o/rel)==h,rel
    o=R/'evidence/revision_v12_external';c=read(o/'COMPLETE.json');fit_calls+=c['fit_calls'];assert fit_calls==23
    for rel,h in c['files_sha256'].items():assert sha(o/rel)==h,rel
    assert not read(o/'RESULTS.json')['primary_success']
    old_recovery=R/'evidence/revision_v12_seeds/archive_recovery';bad=old_recovery/'seed_20260914_censored_CALIBRATION.npz'
    write(R/'docs/revision_v12/ARCHIVE_RECOVERY_STATUS.json',dict(first_replay_status='FAILED_PERSISTED_ARCHIVE_CHECK',recorded_in_process_sha256=read(old_recovery/'COMPLETE.json')['rebuilt_sha256'],later_observed_sha256=sha(bad),original_and_failed_replay_preserved=True,supported_recovery='evidence/revision_v12_seeds/array_recovery/arrays/',supported_recovery_verified=True,new_fit_calls=0,cause='unresolved storage/serialization failure'))
    shutil.copy2(b/'CENSORCAST_ICLR_2027_Final.pdf',R/'paper/CENSORCAST_ICLR_2027_Final.pdf');shutil.copy2(b/'BUILD_REPORT.json',R/'docs/revision_v12/BUILD_REPORT.json')
    base=read(R/'docs/revision_v12/BASELINE_MANIFEST_v11.json');changed=[]
    for rel,rec in base['files'].items():
        assert (R/rel).is_file(),rel
        if sha(R/rel)!=rec['sha256']:changed.append(rel)
    allowed={'README.md','REVIEW_LINK.json','paper/main.tex','paper/matched_main_table.tex','paper/references.bib','paper/review5_ratio_theory.tex','paper/revision_v9_main_results.tex','paper/CENSORCAST_ICLR_2027_Final.pdf'}
    assert set(changed)==allowed,changed
    write(R/'docs/revision_v12/ORIGINAL_CONTENT_AUDIT.json',dict(status='PASS',baseline_files=len(base['files']),unchanged_files=len(base['files'])-len(changed),changed_authoring_files=changed,all_preexisting_scientific_records_unchanged=True,new_training_fits=fit_calls,additional_seeds=2,new_external_datasets=1,external_primary_success=False,independent_checks=checks,public_mutations=0))
    names={'CENSORCAST_ICLR_2027_Final.pdf':R/'paper/CENSORCAST_ICLR_2027_Final.pdf','CENSORCAST_Revision_Notes_JA.md':R/'docs/revision_v12/REVISION_NOTES_JA.md','CENSORCAST_Evidence_Plan_and_Rebuttal.md':R/'docs/revision_v12/RESEARCH_RESPONSE_AND_REBUTTAL.md'}
    for name,src in names.items():shutil.copy2(src,D/name)
    with zipfile.ZipFile(x.prior_archive) as z:distributed={i.filename.split('/',1)[1] for i in z.infolist() if not i.is_dir()}
    new={str(p.relative_to(R)) for folder in ['docs/revision_v12','evidence/revision_v12_seeds','evidence/revision_v12_external'] for p in (R/folder).rglob('*') if p.is_file() and '__pycache__' not in str(p)}
    new|={str(p.relative_to(R)) for p in (R/'code').glob('*v12.py')}
    new|={str(p.relative_to(R)) for p in (R/'paper').glob('revision_v12*.tex')}
    new.add('requirements-v12.txt');files=set(base['files'])|new
    additional_assets={rel for rel in new if rel.startswith(('evidence/revision_v12_seeds/','evidence/revision_v12_external/')) and Path(rel).suffix in ['.npz','.npy','.txt']}
    manifest=dict(format='censorcast-integrated-1',revision='v12',date='2026-09-12',files={})
    for rel in sorted(files):
        p=R/rel;manifest['files'][rel]=dict(bytes=p.stat().st_size,sha256=sha(p),numerical_asset=rel in additional_assets or base['files'].get(rel,{}).get('numerical_asset',False))
    write(R/'MANIFEST.json',manifest);distributed|=new|{'MANIFEST.json'}
    assert all(manifest['files'][p]['numerical_asset'] for p in files-distributed)
    archive=D/'CENSORCAST_Anonymous_GitHub_Ready.zip';addon=D/'CENSORCAST_v12_Additional_Evidence.zip'
    sets={archive:distributed-additional_assets,addon:additional_assets}
    for target,subset in sets.items():
        with zipfile.ZipFile(target,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6,allowZip64=True) as z:
            for rel in sorted(subset):
                p=R/rel;compress=zipfile.ZIP_STORED if p.suffix.lower() in ['.npz','.gz','.zip','.pdf','.png','.jpg'] else zipfile.ZIP_DEFLATED
                z.write(p,'CENSORCAST/'+rel,compress_type=compress)
        assert target.stat().st_size<512*1024*1024
    with zipfile.ZipFile(archive) as z:
        for name,src in names.items():assert z.read('CENSORCAST/'+str(src.relative_to(R)))==(D/name).read_bytes()
    for target,subset in sets.items():
        with zipfile.ZipFile(target) as z:
            assert z.testzip() is None
            for rel in subset-{'MANIFEST.json'}:assert hashlib.sha256(z.read('CENSORCAST/'+rel)).hexdigest()==manifest['files'][rel]['sha256'],rel
    result=json.loads(subprocess.run(['python',str(R/'code/verify_release.py')],check=True,capture_output=True,text=True).stdout)
    names['CENSORCAST_Anonymous_GitHub_Ready.zip']=archive
    names['CENSORCAST_v12_Additional_Evidence.zip']=addon
    check=dict(status='PASS',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),manifest_check=result,zip_members=len(distributed),zip_crc='PASS',zip_members_match_manifest=True,pdf_notes_and_rebuttal_match_archive=True,all_preexisting_scientific_records_unchanged=True,deliverables={name:dict(bytes=(D/name).stat().st_size,sha256=sha(D/name)) for name in names})
    write(D/'DELIVERY_CHECKS.json',check);print(json.dumps(check,indent=2),flush=True)
if __name__=='__main__':main()
