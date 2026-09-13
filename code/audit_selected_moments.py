"""Post-fit descriptive accounting of predicted versus realized accepted masses."""
from pathlib import Path
import json
import numpy as np
from policy_audit import scores,mask
R=Path(__file__).resolve().parents[1];src=R/'evidence/matched_censoring';out=R/'reproduction_outputs/matched_moment_accounting.json';out.parent.mkdir(exist_ok=True)
result={'scope':'post-fit descriptive mass accounting; realized outcomes are not conditional-moment oracles','arms':{}}
for arm in ['censored','complete']:
 z=np.load(src/arm/'EVALUATION.npz');f=json.loads((src/arm/'FROZEN.json').read_text());rows=[]
 for p in f['plans']:
  if p['kind']!='relative' or p['value']!=.95:continue
  e,w=z['e'+str(p['e_trees'])],z['w'+str(p['w_trees'])];L,W=abs(z['y']-z['f']),z['y']
  for method in ['demand','weight_descending']:
   a=mask(scores(e,w,p['cap'],method,f['T']),p['policies'][method]['threshold']);E=float(L[a].sum());M=float(W[a].sum());eh=float(e[a].sum());wh=float(w[a].sum());r=p['cap']
   rows.append(dict(e_trees=p['e_trees'],w_trees=p['w_trees'],method=method,rows=int(a.sum()),error=E,exposure=M,predicted_error=eh,predicted_exposure=wh,realized_risk=E/M if M else None,predicted_risk=eh/wh if wh else None,error_mass_bias=(eh/E-1) if E else None,exposure_mass_bias=(wh/M-1) if M else None,error_excess_component=eh-E,exposure_excess_component=-r*(wh-M),excess_discrepancy=(eh-r*wh)-(E-r*M)))
 p=next(v for v in f['plans'] if v['kind']=='relative' and v['value']==.95 and v['e_trees']==v['w_trees']==220)
 e,w=z['e220'],z['w220'];L,W=abs(z['y']-z['f']),z['y'];masks={}
 for o in ['c','d']:
  method=p['selected'][o];masks[o]=mask(scores(e,w,p['cap'],method,f['T']),p['policies'][method]['threshold']) if method is not None else np.zeros(len(W),bool)
 budget={}
 ni=len(z['items']);units=np.repeat(z['series_item_index'],len(z['days']));counts=np.random.default_rng(20260910).multinomial(ni,np.full(ni,1/ni),size=2000)
 for name,a in [('common',masks['c']&masks['d']),('case_only',masks['c']&~masks['d']),('exposure_only',~masks['c']&masks['d']),('union',masks['c']|masks['d']),('case_policy',masks['c']),('exposure_policy',masks['d'])]:
  M=float(W[a].sum());E=float(L[a].sum());budget[name]=dict(rows=int(a.sum()),exposure=M,error=E,risk=E/M if M else None,excess=E-p['cap']*M,mean_forecast=float(z['f'][a].mean()) if a.any() else None)
  bs=counts@np.column_stack([np.bincount(units,weights=a*L),np.bincount(units,weights=a*W)]);budget[name]['risk_ci95']=np.quantile(bs[:,0]/bs[:,1],[.025,.975]).tolist() if M else None
 result['arms'][arm]={'moments':rows,'budget':budget,'cap':p['cap']}
out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
