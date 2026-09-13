from pathlib import Path
import numpy as np
from run_selective_study import load,dump,R,FLOOR,frontier
p=Path(__file__).resolve().parents[1]/'results/review_revision';v=load(p/'cache/shadow_aligned.npz');meta=load(p/'cache/metadata.npz')
a=load(p/'mask_direct_excess_pooled_s20260906.npz')['shadow'];b=load(p/'mask_relative_error_f_pooled_s20260906.npz')['shadow']
y=v['truth'].astype(float);f=v['proposal'];err=np.abs(y-f);out={}
for name,mask in [('both',a&b),('direct_only',a&~b),('relative_only',b&~a),('neither',~a&~b)]:
    out[name]={'rows':int(mask.sum()),'row_fraction':float(mask.mean()),'demand_coverage':float(y[mask].sum()/y.sum()),'mean_forecast':float(f[mask].mean()),'mean_demand':float(y[mask].mean()),'zero_demand_fraction':float((y[mask]==0).mean()),'wape':float(err[mask].sum()/y[mask].sum()),'total_contract_excess':float((err[mask]-R*y[mask]).sum())}
oracle=[]
for group in sorted(set(meta['cat_id'])):
    rows=meta['cat_id']==group;yy=y[rows];ff=f[rows];scores=np.abs(yy-ff)-R*yy
    z=frontier(scores,yy,ff,v['baseline'][rows],'absolute');good=z['point_feasible']
    oracle.append({'group':group,'truth_only_oracle_max_coverage_at_cap':float(z['coverage'][good].max()) if good.any() else 0,'deployable':False})
dump(p/'SELECTION_MECHANISM.json',{'acceptance_overlap':out,'oracle':oracle,'cap':R,'scope':'post_hoc_development_diagnostic_not_a_policy'})
print(out);print(oracle)
