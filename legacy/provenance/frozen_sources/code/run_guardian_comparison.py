"""Apply a frozen comparison to the authorized 610-item guardian exactly once.

No fitting, threshold optimization, method selection, or deployment certificate.
An interrupted invocation resumes the identical frozen evaluation.
"""
from pathlib import Path
import argparse,json,hashlib
import numpy as np
import lightgbm as lgb
from run_selective_study import construct,load,dump,save_npz,metric,origin
from run_review_comparison import predict,metrics_by_group,paired_delta

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def bounds(v,mask,meta,reps=10000,alpha=.05/48):
    y=v['truth'].astype(float);err=np.abs(y-v['proposal']);_,code=np.unique(meta['item_id'],return_inverse=True);n=int(code.max()+1)
    components=np.column_stack([np.bincount(code,weights=x,minlength=n) for x in [mask.sum(1),np.full(len(mask),mask.shape[1]),(y*mask).sum(1),(err*mask).sum(1)]])
    rng=np.random.default_rng(20260906);parts=[]
    for start in range(0,reps,200):
        w=rng.multinomial(n,np.full(n,1/n),size=min(200,reps-start));s=w@components
        parts.append(np.column_stack([s[:,0]/s[:,1],np.divide(s[:,3],s[:,2],out=np.full(len(s),np.inf),where=s[:,2]>0)]))
    z=np.concatenate(parts);cov=float(np.quantile(z[:,0],alpha));wr=float(np.quantile(z[:,1],1-alpha))
    return {'item_clusters':n,'bootstrap_reps':reps,'endpoint_alpha':alpha,'coverage_lcb':cov,'wape_ucb':wr if np.isfinite(wr) else None,'scope':'held_out_item_bootstrap_approximation_not_distribution_free'}

def main(a):
    f=json.loads(a.freeze.read_text());assert f['status']=='FROZEN_FOR_SEPARATE_COMPARATIVE_GUARDIAN_AUDIT'
    opened=json.loads((a.input/'GUARDIAN_OPENED.json').read_text());assert opened['freeze_sha256']==sha(a.freeze)
    root=Path(__file__).resolve().parents[1]
    for rel,h in f['immutable_files'].items():assert sha(root/rel)==h,rel
    a.output.mkdir(parents=True,exist_ok=True)
    done=a.output/'GUARDIAN_COMPARISON.json'
    if done.exists():
        old=json.loads(done.read_text());assert old['freeze_sha256']==sha(a.freeze);print('Completed frozen evaluation reused.');return
    cache=a.output/'cache';cache.mkdir(exist_ok=True)
    if not (cache/'shadow_features.npy').exists():construct(a.input,cache)
    meta=load(cache/'metadata.npz');assert len(meta['id'])==6100 and len(np.unique(meta['item_id']))==610
    with np.load(root/'inputs/design_outcomes_v0_5.npz',allow_pickle=False) as z:assert not (set(meta['item_id']) & set(z['item_id']))
    keys=['calibration_a','calibration_b','shadow'];data={k:load(cache/(k+'_aligned.npz')) for k in keys}
    xs={k:np.load(cache/(k+'_features.npy'),mmap_mode='r') for k in keys};records=[];first_masks={}
    for seed in f['seeds']:
        dest=a.output/f'predictions_s{seed}.npz'
        if dest.exists():pred=load(dest)
        else:
            pred={}
            paths={'error':root/f'results/selective_strong/mean_error_s{seed}.txt','direct':root/f'results/selective_strong/contract_excess_s{seed}.txt','demand':root/f'results/review_revision/demand_s{seed}.txt'}
            for name,path in paths.items():
                b=lgb.Booster(model_file=str(path))
                for k in keys:
                    pred[name+'_'+k]=predict(b,xs[k])
                    if name!='direct':pred[name+'110_'+k]=predict(b,xs[k],110)
            save_npz(dest,**pred)
        for policy in [p for p in f['policies'] if p['seed']==seed]:
            name=policy['score'];masks={};r=f['wape_cap']
            for k in keys:
                e=np.maximum(pred['error_'+k],0);mu=np.maximum(pred['demand_'+k],0);p=data[k]['proposal']
                family={'mean_error':e,'relative_error_f':e/np.maximum(p,.25),'relative_error_mu':e/np.maximum(mu,.25),'excess_f':e-r*p,'composite_excess':e-r*mu,'composite_excess_budget220':np.maximum(pred['error110_'+k],0)-r*np.maximum(pred['demand110_'+k],0),'direct_excess':pred['direct_'+k]}
                score=family[name];mask=np.zeros(score.shape,bool)
                for g,t in policy['thresholds'].items():
                    if t is None:continue
                    rows=np.ones(len(meta['id']),bool) if g=='ALL' else meta['cat_id']==g
                    mask[rows]=np.isfinite(score[rows])&(score[rows]<=t)
                masks[k]=mask
            report={**policy,'metrics':{k:metrics_by_group(data[k],masks[k],meta) for k in keys}}
            if seed==f['primary_seed'] and policy['threshold_mode']=='pooled':
                first_masks[name]=masks['shadow']
                if name in ['direct_excess','composite_excess']:
                    report['adjusted_48_endpoint_bounds']={}
                    for k in keys:
                        report['adjusted_48_endpoint_bounds'][k]={}
                        for g in ['ALL']+sorted(set(meta['cat_id'])):
                            rows=np.ones(len(meta['id']),bool) if g=='ALL' else meta['cat_id']==g
                            vv={q:arr[rows] if arr.ndim==2 else arr for q,arr in data[k].items()};mm={q:arr[rows] for q,arr in meta.items()}
                            bd=bounds(vv,masks[k][rows],mm);bd['operating_bounds_pass']=bd['wape_ucb'] is not None and bd['wape_ucb']<=r and bd['coverage_lcb']>=f['coverage_floor']
                            report['adjusted_48_endpoint_bounds'][k][g]=bd
            records.append(report);save_npz(a.output/f"masks_{name}_{policy['threshold_mode']}_s{seed}.npz",**masks)
            print(seed,name,policy['threshold_mode'],report['metrics']['shadow']['ALL'],flush=True)
    contrast=paired_delta(data['shadow'],first_masks['direct_excess'],first_masks['composite_excess'],meta,reps=10000)
    contrast['scope']='prespecified_held_out_item_comparison'
    result={'status':'HELD_OUT_ITEM_COMPARISON_COMPLETED','freeze_sha256':sha(a.freeze),'guardian_item_count':610,'guardian_series_count':6100,'records':records,'primary_direct_minus_composite':contrast,'training_calls':0,'threshold_selection_calls':0,'fresh_guardian_use_count':1,'external_opened':False,'certificate_issued':False,'inference':'Prespecified held-out-item comparison; shared calendar and store shocks remain. Cluster bounds are approximate, not an exact finite-sample certificate.'}
    dump(done,result);print('Frozen guardian comparison completed. External data remain sealed.')

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    for k in ['freeze','input','output']:ap.add_argument('--'+k,type=Path,required=True)
    main(ap.parse_args())
