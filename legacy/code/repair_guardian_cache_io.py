"""Regenerate a derived cache with durable writes, without changing the analysis.

Used after a truncated archive was found during verification. The frozen feature
constructor is unchanged. Every readable original array must match exactly.
"""
from pathlib import Path
import io, os, json, hashlib, zipfile, datetime, shutil
import numpy as np
from run_selective_study import construct
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'results/guardian_comparison'
OLD=BASE/'evaluation/cache'
NEW=BASE/'evaluation/durable_repaired_cache'
NEW.mkdir(exist_ok=True)
freeze=ROOT/'provenance/FROZEN_REVIEW_GUARDIAN_COMPARISON.json'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(freeze)=='3d94aec80d993b92684a61f058ed4e3ceb2351233f07fb8eea3f9f542fe2b156'
assert json.loads((BASE/'predictions/GUARDIAN_OPENED.json').read_text())['freeze_sha256']==sha(freeze)
for rel,h in json.loads(freeze.read_text())['immutable_files'].items():assert sha(ROOT/rel)==h

def durable(path,blob):
    path=Path(path);tmp=path.with_name(path.name+'.durable_tmp')
    with tmp.open('wb') as f:
        for start in range(0,len(blob),1024*1024):f.write(blob[start:start+1024*1024])
        f.flush();os.fsync(f.fileno())
    assert tmp.stat().st_size==len(blob)
    assert hashlib.sha256(tmp.read_bytes()).digest()==hashlib.sha256(blob).digest()
    os.replace(tmp,path)

original_save=np.save;original_savez=np.savez_compressed
def npy(path,*args,**kwargs):
    if not isinstance(path,(str,Path)):return original_save(path,*args,**kwargs)
    stream=io.BytesIO();original_save(stream,*args,**kwargs);durable(path,stream.getvalue())
def npz(path,*args,**kwargs):
    if not isinstance(path,(str,Path)):return original_savez(path,*args,**kwargs)
    stream=io.BytesIO();original_savez(stream,*args,**kwargs);durable(path,stream.getvalue())
np.save=npy;np.savez_compressed=npz
try:construct(BASE/'predictions',NEW)
finally:np.save=original_save;np.savez_compressed=original_savez

matched=[];repairs=[]
for target in sorted(NEW.glob('*')):
    original=OLD/target.name
    if target.suffix not in ['.npy','.npz']:continue
    if target.suffix=='.npy':
        assert np.array_equal(np.load(original,mmap_mode='r'),np.load(target,mmap_mode='r'))
        matched.append(target.name)
        continue
    with zipfile.ZipFile(target) as z:assert z.testzip() is None
    try:
        with zipfile.ZipFile(original) as z:
            if z.testzip() is not None:raise zipfile.BadZipFile('CRC mismatch')
        with np.load(original,allow_pickle=False) as a,np.load(target,allow_pickle=False) as b:
            assert a.files==b.files
            for k in a.files:assert np.array_equal(a[k],b[k]),(target.name,k)
        matched.append(target.name)
    except zipfile.BadZipFile:
        assert target.name=='shadow_aligned.npz'
        record={'file':target.name,'incomplete_sha256':sha(original),'repaired_sha256':sha(target)}
        quarantine=ROOT.parent/'guardian_control/incomplete_shadow_aligned.npz'
        shutil.copy2(original,quarantine)
        durable(original,target.read_bytes());repairs.append(record)
record={'status':'DERIVED_CACHE_IO_REPAIR_VERIFIED',
    'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'cause_observed':'Truncated derived ZIP after the computation completed; a first regeneration also produced one truncated NPY. Durable buffered writes repair serialization only.',
    'repairs':repairs,'original_arrays_exactly_reproduced':matched,
    'training_calls':0,'threshold_selection_calls':0,'new_data_access':False,
    'frozen_source_changed':False,'result_json_changed':False,
    'metric_verification':'Performed separately by verify_guardian_results.py'}
durable(ROOT/'provenance/GUARDIAN_CACHE_IO_REPAIR.json',(json.dumps(record,indent=2)+'\n').encode())
print(json.dumps(record,indent=2))
