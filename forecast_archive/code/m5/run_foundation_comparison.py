"""Resumable, inference-only comparison on consumed M5 design items.

No guardian/external outcome input is accepted. All model revisions, input
hashes, dates and numerical settings are persisted before forecasting.
This script was delivered without downloaded weights in the authoring runtime.
"""
from pathlib import Path
import argparse, gc, hashlib, importlib.metadata, io, json, os, time, zipfile
import numpy as np
from legacy_features.features import DesignPanel

DATA_HASHES={
 'design_outcomes_v0_5.npz':'5d63b6aff7854ed1811f6b5bffd2f397a09d671c6754e0f1345e2313b35e48fd',
 'calendar.csv':'d12b5914ef03e66649adf5dd9e996e6602251c22b7a6af8f1f7e3aa12f8860f5',
 'sell_prices.csv':'9da3ad1f8b8ccacdbdc70612191dd375ec24a4ac6625c24b75b3bc60b0bed2ef'}
MODEL_SPECS=[
 {'name':'chronos_bolt_small','repo':'amazon/chronos-bolt-small','revision':'772f3d25d38aec6d914c8949dab4462e2d46f5d8','covariates':False},
 {'name':'chronos2_univariate','repo':'amazon/chronos-2','revision':None,'covariates':False},
 {'name':'chronos2_covariates','repo':'amazon/chronos-2','revision':None,'covariates':True}]
BLOCKS=['calibration_a','calibration_b','shadow']
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
def write(p,b):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+'.partial')
    with open(t,'wb') as f:
        for start in range(0,len(b),1024*1024):f.write(b[start:start+1024*1024])
        f.flush();os.fsync(f.fileno())
    if t.stat().st_size!=len(b):raise RuntimeError('Incomplete output write')
    os.replace(t,p)
def dump(p,x):write(p,(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n').encode())
def save_npz(p,**kw):
    b=io.BytesIO();np.savez_compressed(b,**kw);write(p,b.getvalue())
def load_npz(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def origin(days):
    d=np.asarray(days,dtype=int);return d-(1+(d-1)%7)
def make_inputs(panel,rows,o,prediction_length,context,covariates):
    """Only observed history ending at o, never target-day observations."""
    rows=np.asarray(rows);start=max(0,int(o)-context)
    target=panel.observed[rows,start:o].astype(np.float32,copy=True)
    if not covariates:return target
    inputs=[]
    for j,s in enumerate(rows):
        past={'capacity_hit':(panel.observed[s,start:o]>=panel.capacity[s,start:o]).astype(np.float32),
              'capacity':panel.capacity[s,start:o].astype(np.float32)}
        future={}
        for key in ['wday','month','event_any','event_type']:
            v=panel.calendar_arrays[key]
            past[key]=v[start:o].copy();future[key]=v[o:o+prediction_length].copy()
        state=panel.metadata['state_id'][s];snap=panel.calendar_arrays['snap_'+state]
        past['snap']=snap[start:o].copy();future['snap']=snap[o:o+prediction_length].copy()
        price=panel.price_matrix[s,panel.day_to_week_index]
        past['price']=price[start:o].copy();future['price']=price[o:o+prediction_length].copy()
        inputs.append({'target':target[j],'past_covariates':past,'future_covariates':future})
    return inputs
def median_array(q,series_count,horizon):
    # Chronos-2 returns a list of (1, horizon, quantile) tensors;
    # Bolt returns a (batch, horizon, quantile) tensor.
    if isinstance(q,list):a=np.stack([v.detach().cpu().float().numpy()[0,:,1] for v in q])
    else:a=q.detach().cpu().float().numpy()[:,:,1]
    if a.shape!=(series_count,horizon) or not np.isfinite(a).all():raise RuntimeError('Invalid forecast shape or nonfinite median')
    return np.maximum(a,0).astype(np.float32)
def metric(y,p):
    y=np.asarray(y,dtype=float);p=np.asarray(p,dtype=float);mass=float(y.sum())
    return {'n':int(y.size),'mae':float(np.mean(np.abs(y-p))),
            'wape':float(np.abs(y-p).sum()/mass) if mass>0 else None,
            'wpe':float((p-y).sum()/mass) if mass>0 else None}
def freeze(args,panel,cache):
    import torch
    from huggingface_hub import HfApi
    out=args.output;out.mkdir(parents=True,exist_ok=True)
    inputs={str(p.relative_to(args.data)):sha(p) for p in [args.data/n for n in DATA_HASHES]}
    for n,h in DATA_HASHES.items():
        if inputs[n]!=h:raise RuntimeError('Pinned design input mismatch: '+n)
    cache_hashes={p.name:sha(p) for p in sorted(args.cache.glob('*.npz'))}
    days=sorted(set(int(d) for k in BLOCKS for d in cache[k]['target_days']))
    base={'schema_version':'foundation-1','scope':'POST_HOC_CONSUMED_DESIGN_ONLY',
      'input_hashes':inputs,'comparison_cache_hashes':cache_hashes,
      'runner_sha256':sha(Path(__file__)),'feature_loader_sha256':sha(Path(__file__).parent/'legacy_features/features.py'),
      'context':args.context,'batch_size':args.batch_size,'internal_chronos2_batch':256,
      'cross_learning':False,'point_statistic':'nonnegative median','quantile_levels':[.1,.5,.9],
      'dtype':'float32','horizon_max':7,'seed':20260906,'target_days':days,
      'origins':sorted(set(origin(days).tolist())),
      'series_count':panel.n_series,'item_count':len(set(panel.metadata['item_id'])),
      'versions':{n:importlib.metadata.version(n) for n in ['torch','chronos-forecasting','transformers','numpy','pandas','huggingface-hub']},
      'device':torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu',
      'cuda':torch.version.cuda,'finetuning_calls':0,'fresh_guardian_opened':False,'external_opened':False,
      'known_covariates_assumption':'Calendar and published weekly price schedule known at origin; same price filling as original tabular comparison; past capacity-hit indicators derived from sales/capacity, not the strict benchmark flag',
      'pretraining_overlap':'Not asserted absent; this is inference without task-specific fitting, not a dataset-unseen claim'}
    path=out/'FOUNDATION_PROTOCOL.json'
    if path.exists():
        old=json.loads(path.read_text())
        for k,v in base.items():
            if old.get(k)!=v:raise RuntimeError('Resume configuration changed: '+k+'; keep the prior run and use a new output directory for a different experiment')
        return old
    revisions={};specs=[]
    for spec in MODEL_SPECS:
        spec=spec.copy();repo=spec['repo']
        if repo not in revisions:revisions[repo]=spec['revision'] or HfApi().model_info(repo).sha
        spec['revision']=revisions[repo];specs.append(spec)
    base.update(models=specs,status='FROZEN_BEFORE_ANY_FORECAST',created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
    dump(path,base);return base
def predictions(args,panel,protocol):
    import torch
    from chronos import BaseChronosPipeline
    from huggingface_hub import snapshot_download
    torch.manual_seed(protocol['seed']);np.random.seed(protocol['seed']);torch.set_num_threads(4)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    device='cuda' if torch.cuda.is_available() else 'cpu'
    if args.require_gpu and device!='cuda':raise RuntimeError('Select a GPU runtime, then resume; no GPU is available')
    ph=sha(args.output/'FOUNDATION_PROTOCOL.json');files=[]
    for spec in protocol['models']:
        snapshot=Path(snapshot_download(repo_id=spec['repo'],revision=spec['revision'],
          allow_patterns=['config.json','*.safetensors','*.safetensors.index.json','README.md']))
        weights={p.name:sha(p) for p in sorted(snapshot.iterdir()) if p.is_file()}
        wp=args.output/(spec['name']+'_WEIGHTS.json')
        wr={'repo':spec['repo'],'revision':spec['revision'],'files':weights,'protocol_sha256':ph}
        if wp.exists() and json.loads(wp.read_text())!=wr:raise RuntimeError('Model weights changed')
        if not wp.exists():dump(wp,wr)
        pipeline=BaseChronosPipeline.from_pretrained(str(snapshot),device_map=device,torch_dtype=torch.float32)
        for o in protocol['origins']:
            dest=args.output/'predictions'/spec['name']/f'origin_{o}.npz';receipt=dest.with_suffix('.json')
            if receipt.exists():
                rr=json.loads(receipt.read_text())
                if rr['protocol_sha256']!=ph or rr['sha256']!=sha(dest):raise RuntimeError('Prediction checkpoint hash mismatch')
                files.append(rr);continue
            h=min(7,panel.n_days-o);pp=np.empty((panel.n_series,h),np.float32);start=time.time()
            for st in range(0,panel.n_series,args.batch_size):
                rows=np.arange(st,min(st+args.batch_size,panel.n_series))
                xx=make_inputs(panel,rows,o,h,args.context,spec['covariates'])
                if spec['name'].startswith('chronos2'):
                    if not spec['covariates']:xx=xx[:,None,:]
                    q,_=pipeline.predict_quantiles(xx,prediction_length=h,quantile_levels=[.1,.5,.9],
                      context_length=args.context,batch_size=256,cross_learning=False)
                else:q,_=pipeline.predict_quantiles(torch.from_numpy(xx),prediction_length=h,quantile_levels=[.1,.5,.9])
                pp[rows]=median_array(q,len(rows),h)
            save_npz(dest,forecast=pp,days=np.arange(o+1,o+h+1))
            rr={'model':spec['name'],'origin':o,'sha256':sha(dest),'protocol_sha256':ph,'seconds':time.time()-start}
            dump(receipt,rr);files.append(rr)
            print(json.dumps({'phase':'FORECAST','model':spec['name'],'origin':o,'completed_origins':len(files),'total_origins':len(protocol['origins'])*len(protocol['models'])}),flush=True)
        del pipeline;gc.collect()
        if device=='cuda':torch.cuda.empty_cache()
    dump(args.output/'PREDICTION_RECEIPTS.json',files)
def analyze(args,panel,cache,protocol):
    # Outcomes are used only after every model forecast has completed.
    rows=[];uncertainty=[];seed=20260906
    items,inv=np.unique(panel.metadata['item_id'],return_inverse=True)
    rng=np.random.default_rng(seed);weights=rng.multinomial(len(items),np.ones(len(items))/len(items),size=2000).astype(float)
    for name in [s['name'] for s in protocol['models']]+['observed_l1','original_censor_adapter']:
        for key in BLOCKS:
            c=cache[key];days=c['target_days'];truth=c['truth'].astype(np.float64)
            if not np.array_equal(truth,panel.truth[:,days-1]):raise RuntimeError('Comparison truth alignment differs')
            if name in ['observed_l1','original_censor_adapter']:p=c['baseline' if name=='observed_l1' else 'proposal']
            else:
                p=np.empty(truth.shape,np.float32)
                for o in sorted(set(origin(days).tolist())):
                    v=load_npz(args.output/'predictions'/name/f'origin_{o}.npz');idx=np.flatnonzero(origin(days)==o)
                    p[:,idx]=v['forecast'][:,days[idx]-o-1]
            for cat in ['ALL']+sorted(set(panel.metadata['cat_id'])):
                take=np.ones(panel.n_series,bool) if cat=='ALL' else panel.metadata['cat_id']==cat
                rows.append({'model':name,'block':key,'category':cat,**metric(truth[take],p[take])})
            for h in range(1,8):
                take=days-origin(days)==h
                rows.append({'model':name,'block':key,'category':'ALL','horizon':h,**metric(truth[:,take],p[:,take])})
            mass=np.bincount(inv,weights=truth.sum(axis=1),minlength=len(items))
            for ref in ['baseline','proposal']:
                delta=np.bincount(inv,weights=(np.abs(truth-p)-np.abs(truth-c[ref])).sum(axis=1),minlength=len(items))
                boots=(weights@delta)/(weights@mass)
                uncertainty.append({'model':name,'block':key,'reference':ref,'delta_wape':float(delta.sum()/mass.sum()),
                  'ci95':np.quantile(boots,[.025,.975]).tolist(),'interpretation':'descriptive item bootstrap; no independent or simultaneous guarantee'})
    result={'status':'FOUNDATION_COMPARISON_COMPLETED','scope':protocol['scope'],'protocol_sha256':sha(args.output/'FOUNDATION_PROTOCOL.json'),
      'rows':rows,'paired_comparisons':uncertainty,'series_count':panel.n_series,'finetuning_calls':0,
      'fresh_guardian_opened':False,'external_opened':False,'certificate_issued':False}
    dump(args.output/'FOUNDATION_RESULTS.json',result)
    entries=[args.output/'FOUNDATION_PROTOCOL.json',args.output/'FOUNDATION_RESULTS.json',args.output/'PREDICTION_RECEIPTS.json']+list(args.output.glob('*_WEIGHTS.json'))
    b=io.BytesIO()
    with zipfile.ZipFile(b,'w',zipfile.ZIP_DEFLATED) as z:
        for p in entries:z.writestr(p.name,p.read_bytes())
    write(args.output/'FOUNDATION_RESULTS_FOR_REVIEW.zip',b.getvalue())
    print(json.dumps({'status':result['status'],'result_zip':str(args.output/'FOUNDATION_RESULTS_FOR_REVIEW.zip'),'prediction_checkpoints_preserved':True,'fresh_guardian_opened':False,'external_opened':False}),flush=True)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',type=Path,required=True);ap.add_argument('--cache',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True);ap.add_argument('--context',type=int,default=2048);ap.add_argument('--batch-size',type=int,default=64)
    ap.add_argument('--require-gpu',action='store_true');a=ap.parse_args()
    if a.context<56 or a.context>2048:raise ValueError('Matched context must lie in [56,2048]')
    for n,h in DATA_HASHES.items():
        if sha(a.data/n)!=h:raise RuntimeError('Input hash mismatch: '+n)
    panel=DesignPanel.load(a.data/'design_outcomes_v0_5.npz',a.data/'calendar.csv',a.data/'sell_prices.csv')
    cache={k:load_npz(a.cache/(k+'_aligned.npz')) for k in BLOCKS};meta=load_npz(a.cache/'metadata.npz')
    if not np.array_equal(meta['id'],panel.metadata['id']):raise RuntimeError('Series order mismatch')
    p=freeze(a,panel,cache);print('FOUNDATION_PROTOCOL_READY',flush=True);predictions(a,panel,p);analyze(a,panel,cache,p)
if __name__=='__main__':main()
