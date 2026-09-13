"""Clarify the side constraint omitted by the pure demand-weighted ranking."""
from pathlib import Path
import numpy as np
from r2_io import dump
from run_selective_study import load,R
from run_review2_objectives import curve,select_threshold,scores,PRIMARY,KEYS
ROOT=Path(__file__).resolve().parents[1];base=ROOT/'results/review2';out=[];selected=[]
for cohort,cache,pred in [('design',base/'cache_design',base/'design_predictions_s20260906.npz'),('guardian_posthoc',ROOT.parent/'evaluation_checkpoints/cache',ROOT.parent/'evaluation_checkpoints/predictions_s20260906.npz')]:
    data={k:load(cache/(k+'_aligned.npz')) for k in KEYS};meta=load(cache/'metadata.npz');pp=load(pred)
    if cohort=='design':
        for name,ss in scores(pp,data).items():
            if name not in PRIMARY:continue
            ca={k:curve(ss[k],data[k]['truth'],data[k]['proposal'],floor=0.) for k in KEYS[:2]}
            for obj in ['row','demand']:
                t,val=select_threshold(ca['calibration_a'],ca['calibration_b'],obj);selected.append(dict(score=name,objective=obj,threshold=t,objective_value=val))
    rows=meta['cat_id']=='HOBBIES';v=data['shadow'];y=v['truth'][rows].astype(float);e=abs(y-v['proposal'][rows]);p=v['proposal'][rows]
    for name in ['excess','ratio']:
        if name=='excess':s=e-R*y
        else:
            s=np.divide(e,y,out=np.full(e.shape,np.finfo(float).max),where=y>0);s[(y==0)&(e==0)]=0
        for floor in [0.,.35]:
            c=curve(s,y,p,floor=floor);ii=np.flatnonzero(c['feasible']);i=int(ii[-1]) if len(ii) else None
            out.append(dict(cohort=cohort,ranking=name,row_floor=floor,row_coverage=float(c['row'][i]) if i is not None else 0,demand_coverage=float(c['demand'][i]) if i is not None else 0,wape=float(c['wape'][i]) if i is not None else None))
dump(base/'objectives/FLOOR_SIDE_CONSTRAINT.json',dict(scope='posthoc_truth_only_diagnostic',hobbies=out,first_seed_floorfree_calibration=selected))
print(out)
