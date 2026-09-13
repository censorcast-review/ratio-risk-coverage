"""Replay frozen predictions after an incomplete NPZ serialization; never fit."""
from pathlib import Path
import sys, json, hashlib, datetime, struct, zlib, io, gc, os
import numpy as np
import lightgbm as lgb
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'forecast_archive/code'))
from replay_m5 import DesignPanel, FeatureBuilder, HierarchyFeatures
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def write(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
def main():
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);a=p.parse_args()
    out=ROOT/'evidence/revision_v12_seeds/array_recovery'
    proto=json.loads((out/'PROTOCOL.json').read_text())
    assert not (out/'COMPLETE.json').exists()
    for rel,h in proto['source_sha256'].items():assert sha(ROOT/rel)==h,rel
    for name,h in proto['input_sha256'].items():assert sha(a.data/'data'/name)==h,name
    dest=ROOT/'evidence/revision_v12_seeds/seed_20260914/censored'
    for rel,h in proto['frozen_sha256'].items():assert sha(ROOT/rel)==h,rel
    write(out/'START.json',dict(utc=now(),fit_calls=0,protocol_sha256=sha(out/'PROTOCOL.json')))
    # Retain and validate every complete local ZIP member before replaying.
    raw=(dest/'CALIBRATION.npz').read_bytes();i=0;retained={}
    while raw[i:i+4]==b'PK\x03\x04':
        h=struct.unpack_from('<4s5H3L2H',raw,i);fn,ex=h[-2:]
        name=raw[i+30:i+30+fn].decode();extra=raw[i+30+fn:i+30+fn+ex]
        usize,csize=struct.unpack_from('<QQ',extra,4);start=i+30+fn+ex;end=start+csize
        assert end<=len(raw)
        body=zlib.decompress(raw[start:end],-15)
        assert len(body)==usize and zlib.crc32(body)==h[6]
        retained[name[:-4]]=np.load(io.BytesIO(body));i=end
    assert set(retained)=={'f','e220','e660','w220'} and i==len(raw)
    panel=DesignPanel.load(a.data/'data/design_outcomes_v0_5.npz',a.data/'data/calendar.csv',a.data/'data/sell_prices.csv')
    builder=FeatureBuilder(panel);hier=HierarchyFeatures(panel);n=panel.n_series
    def features(ss,dd):
        xx=builder.make_features(ss,dd,include_censor=False)
        origin=dd-FeatureBuilder.horizon_for_day(dd);week=panel.day_to_week_index[origin-1].astype(int)
        price=panel.price_matrix[ss,week];xx[:,13]=price
        xx[:,14]=price/np.maximum(panel.price_matrix[ss,np.maximum(week-1,0)],1e-3)-1
        xx[:,15]=price/np.maximum(panel.price_matrix[ss,np.maximum(week-4,0)],1e-3)-1
        return np.column_stack([xx,hier.make(ss,dd)[:,:6]]).astype(np.float32)
    model=lgb.Booster(model_file=str(dest/'POINT.txt'))
    heads={k:lgb.Booster(model_file=str(dest/(k+'.txt'))) for k in ['e220','e660','w220','w660']}
    days=np.arange(1611,1674);ans={k:np.empty((n,len(days)),np.float32) for k in ['f',*heads]}
    for start in range(0,len(days),7):
        ds=days[start:start+7];ss=np.repeat(np.arange(n),len(ds));dd=np.tile(ds,n)
        xx=features(ss,dd);ff=np.maximum(model.predict(xx,num_threads=4),0)
        ans['f'][:,start:start+len(ds)]=ff.reshape(n,-1);ex=np.column_stack([xx,ff])
        for key,h in heads.items():ans[key][:,start:start+len(ds)]=np.maximum(h.predict(ex if key[0]=='e' else xx,num_threads=4),0).reshape(n,-1)
        print('REPLAYED days',int(ds[0]),int(ds[-1]),flush=True)
    cal={k:v.ravel().astype(float) for k,v in ans.items()}
    cal['y']=panel.truth[:,days-1].ravel().astype(float);cal['blocks']=np.tile((days>=1646).astype(int),n)
    for k,v in retained.items():assert np.array_equal(v,cal[k]),k
    arrays=out/'arrays';arrays.mkdir(exist_ok=False);saved={}
    for k,v in cal.items():
        destarray=arrays/(k+'.npy');temporary=arrays/(k+'.tmp')
        with temporary.open('xb') as f:
            np.save(f,v);f.flush();os.fsync(f.fileno())
        os.replace(temporary,destarray)
        assert np.array_equal(v,np.load(destarray))
        saved[k]=dict(sha256=sha(destarray),bytes=destarray.stat().st_size)
    for k,v in cal.items():assert np.array_equal(v,np.load(arrays/(k+'.npy')))
    write(out/'COMPLETE.json',dict(utc=now(),fit_calls=0,source_sha256=sha(__file__),protocol_sha256=sha(out/'PROTOCOL.json'),original_sha256=sha(dest/'CALIBRATION.npz'),arrays=saved,exact_retained_array_matches=sorted(retained),rows=len(cal['y']),interpretation='Seven individual NPY arrays replayed from frozen models, atomically written and reread. Original incomplete archives preserved; no fitting, selection, or outcome changes.'))
    print('RECOVERY COMPLETE',flush=True)
if __name__=='__main__':main()
