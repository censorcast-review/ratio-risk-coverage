"""Independent point-result crosscheck of retrospective lambda analysis.

No import of the sensitivity implementation; no writes to lambda/original evidence.
Bootstrap draws are not regenerated in this bounded audit.
"""
from pathlib import Path
import hashlib,json,math
from datetime import datetime,timezone
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
HERE=ROOT/'followup/lambda'
CHECKS=[]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def check(name,ok):
 CHECKS.append({'check':name,'passed':bool(ok)})
 if not ok:raise AssertionError(name)
def near(a,b):return bool(np.isclose(a,b,rtol=0,atol=2e-14))
def measure(a,m):
 n=len(m);N=int(m.sum());W=float(a['exposure'][m].sum());L=float(a['loss'][m].sum())
 return {'n_total':n,'n_accepted':N,'accepted_exposure':W,'accepted_loss':L,'case_coverage':N/n,'exposure_coverage':W/float(a['exposure'].sum()),'risk':L/W if W else None}
def reconstruct(a,s,p):
 return np.array([bool(p['eligible'] and (float(sc)<p['threshold'] or (float(sc)==p['threshold'] and int(tie)<=p['boundary_hash']))) for sc,tie in zip(s,a['tie'])])
def prefix(a,s,cap,floor):
 order=sorted(range(len(s)),key=lambda i:(float(s[i]),int(a['tie'][i])))
 L=W=0.;last=None
 for k,i in enumerate(order,1):
  L+=float(a['loss'][i]);W+=float(a['exposure'][i])
  if k/len(s)>=floor and W>0 and L<=cap*W:last=i
 if last is None:return {'eligible':False,'threshold':None,'boundary_hash':None,'A':None}
 p={'eligible':True,'threshold':float(s[last]),'boundary_hash':int(a['tie'][last])}
 p['A']=measure(a,reconstruct(a,s,p));return p
report=json.loads((HERE/'RESULTS.json').read_text());before={p.name:sha(p) for p in HERE.iterdir() if p.is_file()}
check('retrospective_status',report['analysis_status']=='retrospective; original results already known')
pointrows=[]
for dataset,z in report['datasets'].items():
 base=ROOT/'evidence'/dataset;fixed=json.loads((base/'A_FIXED.json').read_text());primary=json.loads((base/'RESULT.json').read_text())
 arr={r:dict(np.load(base/(r+'_ARRAYS.npz'),allow_pickle=False)) for r in ['A','B','E']}
 for field,orig in [('cap','r'),('design_cap','design_cap'),('floor','floor'),('mean_H_w','mean_H_w')]:check(dataset+'_'+field,near(z[field],fixed[orig]))
 check(dataset+'_legacy_lambda_0.25_is_exposure_weight_0.75',z['sensitivity']['0.75']['mixed_denominator']=='0.25+0.75*w/mean_H(w)')
 for weightkey,sens in z['sensitivity'].items():
  weight=float(weightkey);scores={role:np.array([(float(e)-fixed['r']*float(w))/((1-weight)+weight*float(w)/fixed['mean_H_w']) for e,w in zip(a['Error'],a['exposure'])]) for role,a in arr.items()}
  p=prefix(arr['A'],scores['A'],fixed['design_cap'],fixed['floor'])
  for field in ['eligible','threshold','boundary_hash']:check(dataset+'_'+weightkey+'_policy_'+field,p[field]==sens['Mixed_policy'][field])
  for k,v in p['A'].items():check(dataset+'_'+weightkey+'_A_'+k,near(v,sens['Mixed_policy']['A'][k]))
  masks={role:reconstruct(a,scores[role],p) for role,a in arr.items()}
  for role in ['B','E']:
   m=measure(arr[role],masks[role])
   for k,v in m.items():check(dataset+'_'+weightkey+'_'+role+'_'+k,near(v,sens['Mixed_'+role][k]))
  for role,a in arr.items():
   original=reconstruct(a,a['Mixed'],fixed['policies']['Mixed'])
   check(dataset+'_'+weightkey+'_'+role+'_changed_mask_count',np.count_nonzero(masks[role]!=original)==sens['changed_masks_vs_frozen_Mixed'][role])
   if weight==.5:check(dataset+'_'+role+'_frozen_score_identical',np.array_equal(scores[role],a['Mixed']))
  menu=['Error','Excess','Mixed','Ratio','Descending'];metricsA={}
  for name in menu:
   policy=p if name=='Mixed' else fixed['policies'][name]
   if policy['eligible']:metricsA[name]=measure(arr['A'],masks['A'] if name=='Mixed' else reconstruct(arr['A'],arr['A'][name],policy))
  choices={utility:max(metricsA,key=lambda name:metricsA[name][utility+'_coverage']) for utility in ['case','exposure']}
  comp=sens['selected_comparison_E']
  for utility in choices:check(dataset+'_'+weightkey+'_choice_'+utility,choices[utility]==comp[utility+'_choice'])
  chosenE={u:measure(arr['E'],masks['E'] if name=='Mixed' else reconstruct(arr['E'],arr['E'][name],fixed['policies'][name])) for u,name in choices.items()}
  for code,full in [('c','case'),('d','exposure')]:
   delta=100*(chosenE['exposure'][full+'_coverage']-chosenE['case'][full+'_coverage'])
   check(dataset+'_'+weightkey+'_delta_'+code,near(delta,comp['delta_'+code+'_pp']))
  if weight==.5:
   for field in ['interval_c_pp','interval_d_pp']:check(dataset+'_frozen_'+field,np.allclose(comp[field],primary['primary'][field],rtol=0,atol=2e-14))
   check(dataset+'_frozen_B_upper',near(sens['Mixed_B']['signed_excess_upper'],primary['B']['Mixed']['signed_excess_upper']))
  pointrows.append({'dataset':dataset,'exposure_weight':weight,'denominator':sens['mixed_denominator'],'case_choice':choices['case'],'exposure_choice':choices['exposure'],'delta_c_pp':comp['delta_c_pp'],'delta_d_pp':comp['delta_d_pp'],'mixed_B_risk':sens['Mixed_B']['risk']})
 for name,rec in z['frozen_candidates'].items():
  policy=fixed['policies'][name]
  check(dataset+'_'+name+'_eligibility',rec['A_eligible']==policy['eligible'])
  if policy['eligible']:
   b=measure(arr['B'],reconstruct(arr['B'],arr['B'][name],policy))
   check(dataset+'_'+name+'_B_point_risk',near(b['risk'],rec['B_point_risk']))
  else:check(dataset+'_'+name+'_B_unavailable',rec['B_point_risk'] is None)
 a=arr['A'];order=sorted(range(len(a['exposure'])),key=lambda i:(-float(a['exposure'][i]),int(a['tie'][i])));L=W=0;pool=[]
 for k,i in enumerate(order,1):
  L+=int(a['loss'][i]);W+=int(a['exposure'][i])
  if k/len(order)>=fixed['floor']:pool.append((L/W,k,L,W))
 minrisk,k,L,W=min(pool,key=lambda q:q[0]);rec=z['Descending_A_prefix_diagnostic']
 for field,value in [('n_floor_eligible_prefixes',len(pool)),('minimum_A_risk_among_floor_eligible_prefixes',minrisk),('minimum_accepted_count',k),('minimum_accepted_loss',L),('minimum_accepted_exposure',W),('minimum_at_case_coverage',k/len(order)),('minimum_at_exposure_coverage',W/float(a['exposure'].sum()))]:check(dataset+'_Descending_'+field,near(value,rec[field]))
 for field,cap in [('number_feasible_at_cap',fixed['r']),('number_feasible_at_design_cap',fixed['design_cap'])]:check(dataset+'_Descending_'+field,sum(L<=cap*W for _,k,L,W in pool)==rec[field])
check('lambda_files_unchanged',all(sha(HERE/name)==h for name,h in before.items()))
result={'utc':datetime.now(timezone.utc).isoformat(),'status':'PASS','checks_n':len(CHECKS),'scope':'Independent scalar score formulas, Python sorted-prefix loops, fixed-boundary mask reconstruction, A menu selection, B/E point metrics, E coverage contrasts, Descending minimum-prefix risks. All frozen B point risks checked. Original .5 interval/upper values checked against frozen RESULT. No new bootstrap regeneration; new interval coverage/screen guarantees not asserted.','legacy_coefficient_mapping':'Main convention lambda=.25 is denominator .25+.75*w/T, which is exposure_weight=.75 in the retrospective analysis. The .25 exposure_weight row is the opposite weighting and is also reported transparently.','point_results':pointrows,'checks':CHECKS,'input_sha256':before,'audit_code_sha256':sha(Path(__file__))}
(OUT/'LAMBDA_CROSSCHECK.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'status':'PASS','checks_n':len(CHECKS),'point_results':pointrows},indent=2))
