"""Observable capacity-hit control: S=C means Y>=C, including equality.

Fits and dispersion selection use only S, C and origin features. This control
was designed after seeing the first strict-flag development results; its
statistical status is explicitly post-hoc.
"""
from pathlib import Path
import argparse,json,time,gc
import numpy as np
import lightgbm as lgb
from run_observable_extension import load_npz,save_npz,eval_groups,CAP,SEEDS,BLOCKS
from r3_observable import fit_observable,predict_mean,nll,expected_abs,curve,select
from r2_io import dump,write,sha
ROOT=Path(__file__).resolve().parents[1]
METHODS=[('capacity_hit_poisson',None),('capacity_hit_nb_0p5',.5),('capacity_hit_nb_2',2.),('capacity_hit_nb_10',10.)]
def likelihood_inputs(observed,capacity):
    hit=np.asarray(observed)>=np.asarray(capacity)
    bound=np.where(hit,np.asarray(capacity)-1,np.asarray(observed)).astype(float)
    return bound,hit
def run(data,out):
    out.mkdir(parents=True,exist_ok=True);base=ROOT/'results/observable_extension';cache=base/'cache'
    assert (cache/'READY.json').is_file()
    shard=data/'design_outcomes_v0_5.npz'
    assert sha(shard)=='5d63b6aff7854ed1811f6b5bffd2f397a09d671c6754e0f1345e2313b35e48fd'
    with np.load(shard) as z:capacity=z['capacity']
    protocol={'status':'FROZEN_BEFORE_CAPACITY_HIT_FITS','scope':'POST_HOC_CONSUMED_DESIGN_ONLY',
      'motivation':'Distinguish simulation strict exceedance flag from operationally observable capacity hit',
      'parent_protocol_sha256':sha(base/'PROTOCOL.json'),'data_sha256':sha(shard),
      'source_hashes':{n:sha(Path(__file__).parent/n) for n in ['run_capacity_hit_extension.py','run_observable_extension.py','r3_observable.py']},
      'methods':METHODS,'seeds':SEEDS,'trees':220,'rows_per_seed':600000,
      'censor_rule':'hit = S >= C; uncensored term pmf(S), censored term Pr(Y>=C)=sf(C-1)',
      'dispersion_selection':'minimum censored validation NLL using S and C, not hidden Y',
      'cap':CAP,'row_floor':.35,'fresh_guardian_opened':False,'external_opened':False}
    pp=out/'PROTOCOL.json'
    if pp.exists():assert json.loads(pp.read_text())==json.loads(json.dumps(protocol))
    else:dump(pp,protocol)
    models=out/'models';models.mkdir(exist_ok=True)
    meta=load_npz(cache/'metadata.npz');vv=load_npz(cache/'validation.npz')
    xv=np.load(cache/'validation_x.npy',mmap_mode='r').reshape(-1,34)
    ov,cv=likelihood_inputs(vv['observed'],capacity[:,vv['days']-1])
    dd={k:load_npz(cache/(k+'.npz')) for k in BLOCKS};records=[];selections=[]
    for seed in SEEDS:
        tr=load_npz(cache/f'train_s{seed}.npz');oo,cc=likelihood_inputs(tr['observed'],capacity[tr['series'],tr['days']-1]);val=[]
        for method,shape in METHODS:
            p=models/f'{method}_s{seed}.txt';receipt=p.with_suffix('.json')
            if receipt.exists():
                info=json.loads(receipt.read_text());assert sha(p)==info['model_sha256'];m=lgb.Booster(model_file=str(p))
            else:
                print('Fitting',method,seed,flush=True);start=time.time()
                m,info=fit_observable(tr['x'],oo,cc,method,seed,shape)
                info.update(fit_seconds=time.time()-start,capacity_hit_rule=True,protocol_sha256=sha(pp))
                write(p,m.model_to_string().encode());info['model_sha256']=sha(p);dump(receipt,info)
            mv=predict_mean(m,xv,info);vloss=float(nll(mv,ov.ravel(),cv.ravel(),shape).mean())
            val.append({'method':method,'shape':shape,'observed_validation_nll':vloss})
            scores={k:{} for k in ['excess','demand_normalized']};implied={}
            for k in BLOCKS:
                dest=out/f'{method}_s{seed}_{k}.npz'
                if dest.exists():v=load_npz(dest);mu=v['mu'];ee=v['expected_error']
                else:
                    xx=np.load(cache/(k+'_x.npy'),mmap_mode='r');mu=predict_mean(m,xx.reshape(-1,34),info).reshape(xx.shape[:2]);ee=expected_abs(mu,dd[k]['f'],shape)
                    save_npz(dest,mu=mu,expected_error=ee)
                scores['excess'][k]=ee-CAP*mu;scores['demand_normalized'][k]=ee/np.maximum(mu,.25);implied[k]=(ee,mu)
            for name,ss in scores.items():
                for calibration in ['model_implied_observable','truth_calibrated_diagnostic']:
                    curves=[]
                    for k in ['calibration_a','calibration_b']:
                        e,mass=implied[k] if calibration=='model_implied_observable' else (np.abs(dd[k]['truth']-dd[k]['f']),dd[k]['truth'])
                        curves.append(curve(ss[k],e,mass))
                    t=select(curves,CAP,.35)
                    records.append({'seed':seed,'method':method,'shape':shape,'score':name,'calibration':calibration,'threshold':t,
                      'hidden_labels_used_for_fit':0,'simulation_strict_flag_used':False,
                      'hidden_labels_used_for_threshold':0 if calibration=='model_implied_observable' else sum(dd[k]['truth'].size for k in ['calibration_a','calibration_b']),
                      'metrics':{k:eval_groups(dd[k],ss[k],t,meta) for k in BLOCKS}})
            dump(out/f'records_{method}_s{seed}.json',records[-4:]);print('Completed',method,seed,round(vloss,5),flush=True)
        chosen=min([v for v in val if v['method'].startswith('capacity_hit_nb')],key=lambda v:v['observed_validation_nll'])
        selections.append({'seed':seed,'selected_nb':chosen,'all_validation':val});del tr;gc.collect()
    result={'status':'CAPACITY_HIT_DEVELOPMENT_COMPLETE','protocol_sha256':sha(pp),'records':records,
      'validation_selection':selections,'scope':'POST_HOC_CONSUMED_DESIGN_ONLY',
      'fresh_guardian_opened':False,'external_opened':False,'certificate_issued':False}
    dump(out/'CAPACITY_HIT_RESULTS.json',result);print(result['status'],len(records),flush=True)
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--data',type=Path,required=True);a.add_argument('--output',type=Path,required=True);p=a.parse_args();run(p.data,p.output)
