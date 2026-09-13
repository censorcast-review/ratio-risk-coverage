"""Durable serialization for new post-hoc diagnostics; frozen files are untouched."""
from pathlib import Path
import io, os, hashlib, json
import numpy as np

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for x in iter(lambda:f.read(1048576),b''): h.update(x)
    return h.hexdigest()

def write(path, blob):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.tmp')
    with tmp.open('wb') as f:
        for i in range(0,len(blob),1048576):f.write(blob[i:i+1048576])
        f.flush();os.fsync(f.fileno())
    assert tmp.stat().st_size==len(blob)
    os.replace(tmp,p)

def dump(path,x):write(path,(json.dumps(x,indent=2,allow_nan=False)+'\n').encode())

def install_numpy_writers():
    for name in ['save','savez_compressed']:
        original=getattr(np,name)
        def wrapped(path,*a,_original=original,**kw):
            stream=io.BytesIO();_original(stream,*a,**kw);blob=stream.getvalue()
            if hasattr(path,'write'):
                for i in range(0,len(blob),1048576):path.write(blob[i:i+1048576])
            else:write(path,blob)
        setattr(np,name,wrapped)
