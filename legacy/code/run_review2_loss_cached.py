"""Recompute loss-control reports from saved predictions, with zero fit calls."""
from pathlib import Path
import argparse,json,numpy as np
from r2_io import dump
from run_selective_study import load,R
from run_review2_objectives import curve,select_threshold,apply,KEYS
from run_review_comparison import SEEDS
def main(a):
    dc=a.design/'cache_design';dd={k:load(dc/(k+'_aligned.npz')) for k in KEYS};dm=load(dc/'metadata.npz')
    gc=a.guardian/'cache';gd={k:load(gc/(k+'_aligned.npz')) for k in KEYS};gm=load(gc/'metadata.npz');records=[]
    original=json.loads((a.loss/'LOSS_RESULTS.json').read_text())
    for seed in SEEDS:
        ep=load(a.design/f'design_predictions_s{seed}.npz');gp=load(a.guardian/f'predictions_s{seed}.npz')
        for loss in ['poisson','tweedie']:
            pred=load(a.loss/f'predictions_{loss}_s{seed}.npz')
            for name in ['composite_excess','relative_error_mu']:
                ss={};gs={}
                for k in KEYS:
                    e=np.maximum(ep['error_'+k],0);m=np.maximum(pred['design_'+k],0)
                    ss[k]=e-R*m if name=='composite_excess' else e/np.maximum(m,.25)
                    e=np.maximum(gp['error_'+k],0);m=np.maximum(pred['guardian_'+k],0)
                    gs[k]=e-R*m if name=='composite_excess' else e/np.maximum(m,.25)
                ca={k:curve(ss[k],dd[k]['truth'],dd[k]['proposal']) for k in KEYS[:2]};t,_=select_threshold(ca['calibration_a'],ca['calibration_b'],'demand')
                rec=dict(seed=seed,demand_loss=loss,score=name,threshold=t,design=apply(dd,dm,ss,t),guardian_posthoc=apply(gd,gm,gs,t),demand_shadow_mse={cohort:float(np.mean((np.maximum(pred[cohort+'_shadow'],0).astype(float)-data['shadow']['truth'])**2)) for cohort,data in [('design',dd),('guardian',gd)]})
                expected=next(x for x in original['records'] if x['seed']==seed and x['demand_loss']==loss and x['score']==name)
                assert rec==expected,(seed,loss,name)
                records.append(rec)
            print('Verified cached loss control',seed,loss,flush=True)
    dump(a.output,dict(status='ALL_12_LOSS_REPORTS_EXACTLY_RECOMPUTED',training_calls=0,external_opened=False,new_guardian_accesses=0,records=records))
if __name__=='__main__':
    ap=argparse.ArgumentParser()
    for k in ['design','guardian','loss','output']:ap.add_argument('--'+k,type=Path,required=True)
    main(ap.parse_args())
