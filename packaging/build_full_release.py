"""Update the full source attachment without duplicating restored large assets."""
from pathlib import Path
import hashlib,json,zipfile,shutil

work=Path(__file__).resolve().parent
root=work/'CENSORCAST'
original=work.parent/'v12_downloads/CENSORCAST_Anonymous_GitHub_Ready.zip'
target=work/'delivery/CENSORCAST_Anonymous_GitHub_Ready.zip'
def digest(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

# Preserve reproducible packaging source, outside frozen experimental code.
pkg=root/'packaging';pkg.mkdir(exist_ok=True)
shutil.copy2(work/'build_submission_bundle.py',pkg/'build_submission_bundle.py')
shutil.copy2(Path(__file__),pkg/'build_full_release.py')
shutil.copytree(work/'submission_tools',pkg/'submission_tools',dirs_exist_ok=True,
                ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
(pkg/'README.md').write_text('''# Release packaging source

These scripts are editorial packaging utilities, not frozen experiment code.
`build_submission_bundle.py` accepts explicit `--source`, `--stage`, `--cache`
and `--output` paths. Set `--source` to the extracted CENSORCAST directory and
all three output paths to a separate workspace outside that directory. Its
`submission_tools/` siblings derive and independently replay the compact M5
statistics. Never package or regenerate inside a frozen evidence directory.

`build_full_release.py` records the original integration workspace layout and
expects both original download attachments in its sibling v12_downloads/.
The submitted supplement has its own inventory and verification entrypoints.
''')
with zipfile.ZipFile(original) as z:
    shipped={n.removeprefix('CENSORCAST/') for n in z.namelist() if not n.endswith('/')}
baseline=json.loads((root/'docs/revision_v13/BASELINE_MANIFEST_v12.json').read_text())
logical=set(baseline['files'])
added=set()
for top in ['docs/revision_v13','evidence/revision_v13_nonwape','packaging']:
    for p in (root/top).rglob('*'):
        if p.is_file() and not any(x in {'visual','__pycache__'} for x in p.parts):
            added.add(p.relative_to(root).as_posix())
added|={'paper/appendix_n_v13.tex','paper/external_v13_main_table.tex','paper/fancyhdr.sty'}
manifest_paths=(logical|shipped|added)-{'MANIFEST.json'}
manifest={'format':'censorcast-integrated-1','revision':'v13-integrated','date':'2026-09-13','files':{}}
for rel in sorted(manifest_paths):
    p=root/rel;assert p.is_file(),rel
    manifest['files'][rel]={'bytes':p.stat().st_size,'sha256':digest(p),
      'numerical_asset':baseline['files'].get(rel,{}).get('numerical_asset',p.suffix in {'.npz','.npy','.joblib'})}
(root/'MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
archive_paths=shipped|added|{'MANIFEST.json'}
target.parent.mkdir(exist_ok=True)
with zipfile.ZipFile(target,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
    for rel in sorted(archive_paths):z.write(root/rel,'CENSORCAST/'+rel)
with zipfile.ZipFile(target) as z:
    assert z.testzip() is None
    assert z.read('CENSORCAST/paper/CENSORCAST_ICLR_2027_Final.pdf')==(root/'paper/CENSORCAST_ICLR_2027_Final.pdf').read_bytes()
    for rel in sorted(archive_paths-{'MANIFEST.json'}):
        assert hashlib.sha256(z.read('CENSORCAST/'+rel)).hexdigest()==manifest['files'][rel]['sha256'],rel
report={'status':'PASS','file':target.name,'bytes':target.stat().st_size,'sha256':digest(target),
        'members':len(archive_paths),'logical_files_after_merge_and_restore':len(manifest['files']),
        'requires_unchanged_attachment':'CENSORCAST_v12_Additional_Evidence.zip',
        'embedded_pdf_matches_standalone':True,'raw_external_archives_included':False}
(work/'delivery/FULL_RELEASE_CHECK.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report))
