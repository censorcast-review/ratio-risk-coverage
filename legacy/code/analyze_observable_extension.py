"""Recompute all new reported metrics and summarize observable-label controls."""
from pathlib import Path
import json,io
import numpy as np
from r3_observable import stats
from r2_io import dump,sha
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/observable_extension'
def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def main(branch='observable_extension'):
    global OUT
    assert branch in ['observable_extension','capacity_hit_extension']
    OUT=ROOT/'results'/branch;strict=branch=='observable_extension'
    prefix='OBSERVABLE' if strict else 'CAPACITY_HIT';expected_models=21 if strict else 12
    result=json.loads((OUT/(prefix+'_RESULTS.json')).read_text());records=result['records']
    assert len(records)==expected_models*4
    protocol=json.loads((OUT/'PROTOCOL.json').read_text())
    assert sha(OUT/'PROTOCOL.json')==result['protocol_sha256']
    has_repair=strict and (OUT/'IMPLEMENTATION_REPAIR_RECEIPT.json').exists()
    if has_repair:
        repair=json.loads((OUT/'IMPLEMENTATION_REPAIR_RECEIPT.json').read_text())
        assert sha(OUT/'PROTOCOL.json')==repair['parent_protocol_sha256']
        source_hashes=repair['new_source_hashes']
    else:source_hashes=protocol['source_hashes']
    for name,h in source_hashes.items():assert sha(ROOT/'code'/name)==h
    for p in (OUT/'models').glob('*.json'):
        rr=json.loads(p.read_text());assert sha(p.with_suffix('.txt'))==rr['model_sha256']
    feature_cache=ROOT/'results/observable_extension/cache'
    meta=load(feature_cache/'metadata.npz');blocks={k:load(feature_cache/f'{k}.npz') for k in ['calibration_a','calibration_b','shadow']}
    assert len(meta['id'])==18290 and len(set(meta['item_id']))==1829
    item,inv=np.unique(meta['item_id'],return_inverse=True)
    rng=np.random.default_rng(20260906);weights=rng.multinomial(len(item),np.ones(len(item))/len(item),size=5000).astype(float)
    verified=0;moments=[];boot=[];cache={};selected={x['seed']:x['selected_nb']['method'] for x in result['validation_selection']}
    for rec in records:
        method=rec['method'];seed=rec['seed'];threshold=rec['threshold'];ss=rec['score'];cal=rec['calibration']
        for block in blocks:
            key=(method,seed,block)
            if key not in cache:
                cache.clear();cache[key]=load(OUT/f'{method}_s{seed}_{block}.npz')
            v=cache[key];d=blocks[block];mu=v['mu'];ee=v['expected_error'];y=d['truth'].astype(float);f=d['f'].astype(float)
            assert np.isfinite(mu).all() and np.isfinite(ee).all() and np.all(mu>0) and np.all(ee>=0)
            score=ee-protocol['cap']*mu if ss=='excess' else ee/np.maximum(mu,.25)
            mask=np.zeros(y.shape,bool) if threshold is None else score<=threshold
            for group,expected in rec['metrics'][block].items():
                take=np.ones(len(meta['id']),bool) if group=='ALL' else meta['cat_id']==group
                got=stats(y[take],f[take],mask[take])
                for k,value in got.items():
                    if value is None:assert expected[k] is None
                    else:assert np.isclose(value,expected[k],rtol=1e-12,atol=1e-12),(method,seed,block,k)
                verified+=1
            if ss=='excess' and cal=='model_implied_observable':
                mass=float(mu[mask].sum());actual=float(y[mask].sum())
                moments.append({'method':method,'seed':seed,'block':block,'accepted_n':int(mask.sum()),
                  'model_implied_wape':float(ee[mask].sum()/mass) if mass else None,
                  'realized_wape':rec['metrics'][block]['ALL']['wape'],
                  'implied_total_error':float(ee[mask].sum()),'realized_total_error':float(np.abs(y[mask]-f[mask]).sum()),
                  'implied_demand':mass,'realized_demand':actual})
                if method==selected[seed]:
                    for group in ['ALL','FOODS','HOBBIES','HOUSEHOLD']:
                        take=np.ones(len(meta['id']),bool) if group=='ALL' else meta['cat_id']==group
                        nn=np.bincount(inv,weights=take.astype(float)*y.shape[1],minlength=len(item))
                        aa=np.bincount(inv,weights=mask.sum(axis=1)*take,minlength=len(item))
                        mm=np.bincount(inv,weights=(y*mask).sum(axis=1)*take,minlength=len(item))
                        eea=np.bincount(inv,weights=(np.abs(y-f)*mask).sum(axis=1)*take,minlength=len(item))
                        den=weights@mm;rc=weights@aa/(weights@nn)
                        if not np.all(den>0):
                            boot.append({'seed':seed,'method':method,'block':block,'category':group,'status':'INSUFFICIENT_ACCEPTED_MASS'})
                            continue
                        wape=weights@eea/den
                        assert np.isfinite(wape).all()
                        boot.append({'seed':seed,'method':method,'block':block,'category':group,
                          'row_coverage_ci95':np.quantile(rc,[.025,.975]).tolist(),
                          'wape_ci95':np.quantile(wape,[.025,.975]).tolist(),
                          'interpretation':'descriptive item-bootstrap interval; post-hoc development only'})
    means=[]
    for method in sorted(set(r['method'] for r in records)):
        for ss in ['excess','demand_normalized']:
            for cal in ['model_implied_observable','truth_calibrated_diagnostic']:
                rr=[r for r in records if (r['method'],r['score'],r['calibration'])==(method,ss,cal)]
                assert len(rr)==3
                metrics={}
                for b in blocks:
                    metrics[b]={}
                    for g in ['ALL','FOODS','HOBBIES','HOUSEHOLD']:
                        metrics[b][g]={}
                        for k in ['row_coverage','demand_coverage','wape']:
                            x=[r['metrics'][b][g][k] for r in rr];x=[v for v in x if v is not None]
                            metrics[b][g][k]={'mean':float(np.mean(x)) if x else None,'min':float(min(x)) if x else None,'max':float(max(x)) if x else None,'defined_seeds':len(x)}
                means.append({'method':method,'score':ss,'calibration':cal,'metrics':metrics})
    summary={'status':prefix+'_EXTENSION_VERIFIED','verified_metric_reports':verified,'model_count':len(list((OUT/'models').glob('*.txt'))),
      'scope':'POST_HOC_CONSUMED_DESIGN_ONLY','cap':protocol['cap'],'row_floor':protocol.get('min_row_coverage',protocol.get('row_floor')),
      'selected_nb_by_observed_likelihood':selected,'method_means':means,'conditional_moment_audit':moments,'selected_nb_descriptive_intervals':boot,
      'source_repair_before_completed_fits':has_repair,'fresh_guardian_opened':False,'external_opened':False,'certificate_issued':False}
    assert verified==expected_models*4*12 and summary['model_count']==expected_models
    dump(OUT/(prefix+'_VERIFIED_SUMMARY.json'),summary)
    files=[p for p in OUT.rglob('*') if p.is_file() and 'cache' not in p.relative_to(OUT).parts and p.name!='ARTIFACT_INDEX.json']
    dump(OUT/'ARTIFACT_INDEX.json',{str(p.relative_to(OUT)):sha(p) for p in sorted(files)})
    print(json.dumps({'status':summary['status'],'verified_metric_reports':verified,'model_count':expected_models,'selected_nb':selected},indent=2))
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--branch',default='observable_extension');a=p.parse_args();main(a.branch)
