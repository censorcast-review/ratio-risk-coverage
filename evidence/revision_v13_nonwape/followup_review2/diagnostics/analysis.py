"""Independent, retrospective summaries; does not import experimental code."""
from pathlib import Path
import csv, hashlib, json, re
from datetime import datetime, timezone
import numpy as np

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p): return json.loads(p.read_text())
def save(name, value): (OUT/name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
protected=[p for d in ['evidence','followup'] for p in (ROOT/d).rglob('*') if p.is_file()]
protocol=read(ROOT/'evidence/PROTOCOL.json')
protected += [ROOT/p for p in protocol['code_sha256']]
original={str(p.relative_to(ROOT)):digest(p) for p in protected}
checks=[]
def check(name, condition):
 checks.append({'check':name,'passed':bool(condition)})
 if not condition: raise AssertionError(name)
for name,h in protocol['code_sha256'].items(): check('frozen code hash '+name,digest(ROOT/name)==h)
check('protocol hash',digest(ROOT/'evidence/PROTOCOL.json')==read(ROOT/'evidence/FREEZE_RECEIPT.json')['protocol_sha256'])
results={}
full_csv=[]; policy_csv=[]; decomposition_csv=[]
for dataset in ['bibtex','mediamill']:
 p=ROOT/'evidence'/dataset
 arrays={role:dict(np.load(p/f'{role}_ARRAYS.npz')) for role in 'ABE'}
 flags=dict(np.load(ROOT/'followup/fallback'/f'{dataset}_FALLBACK_FLAGS.npz'))
 af=read(p/'A_FIXED.json'); bs=read(p/'B_SCREEN.json')['results']; old=read(p/'RESULT.json')
 cap=af['r']; selected=af['menu_selected']
 full={}
 for role,z in arrays.items():
  L=float(z['loss'].sum()); W=float(z['exposure'].sum())
  full[role]={'n':len(z['loss']),'loss_mass':L,'exposure_mass':W,'risk':L/W}
  full_csv.append({'dataset':dataset,'role':role,**full[role]})
  check(dataset+role+' fallback w1',np.all(z['exposure'][flags[role]]==1))
  idx=np.flatnonzero(flags[role]); eh=z['Error'][idx]; tie=z['tie'][idx]
  order=np.lexsort((tie,eh))
  for score in ['Excess','Ratio']:
   check(dataset+role+score+' fallback order equals ehat order',np.array_equal(order,np.lexsort((tie,z[score][idx]))))
  check(dataset+role+' fallback Ratio equals ehat',np.array_equal(z['Ratio'][idx],eh))
  check(dataset+role+' fallback Excess equals ehat-r',np.array_equal(z['Excess'][idx],eh-cap))
 ratios={'B_over_A':full['B']['risk']/full['A']['risk'],'E_over_A':full['E']['risk']/full['A']['risk'],'B_minus_A_pp':100*(full['B']['risk']-full['A']['risk']),'E_minus_A_pp':100*(full['E']['risk']-full['A']['risk'])}
 policies={}
 for name,ar in af['policies'].items():
  eligible=ar['eligible']; policies[name]={'A_eligible':eligible}
  for role in 'BE':
   z=arrays[role]; m=z['mask_'+name]; n=len(m); W=float(z['exposure'].sum()); aw=float(z['exposure'][m].sum()); al=float(z['loss'][m].sum())
   met={'accepted_n':int(m.sum()),'accepted_loss':al,'accepted_exposure':aw,'case_coverage_pp':100*float(m.mean()),'exposure_coverage_pp':100*aw/W,'risk':al/aw if aw else None,'mean_signed_excess':float(np.mean(m*(z['loss']-cap*z['exposure'])))} if eligible else None
   policies[name][role]=met
   if eligible:
    check(dataset+name+role+' risk independently reproduced',abs(met['risk']-old[role][name]['risk'])<1e-14)
    for field,stored in [('accepted_n','accepted_n'),('accepted_exposure','accepted_exposure'),('accepted_loss','accepted_loss')]: check(dataset+name+role+field,met[field]==old[role][name][stored])
   row={'dataset':dataset,'role':role,'policy':name,'A_eligible':eligible,**(met or {})}
   if role=='B': row.update(B_signed_excess_upper=bs[name]['signed_excess_upper'] if eligible else None,B_pass=bs[name]['passed'])
   policy_csv.append(row)
  policies[name]['B_signed_excess_upper']=bs[name]['signed_excess_upper'] if eligible else None
  policies[name]['B_pass']=bs[name]['passed']
 gate_contrasts={}
 for utility,gate in [('case','GateCase'),('exposure','GateExposure')]:
  menu=selected[utility]; pair={'gate':gate,'menu':menu,'A_eligible':af['policies'][gate]['eligible']}
  for role in 'BE':
   g=policies[gate][role]; m=policies[menu][role]
   pair[role]={k:g[k]-m[k] for k in ['case_coverage_pp','exposure_coverage_pp','risk']} if g else None
  gate_contrasts[utility]=pair
 z=arrays['E']; g=z['mask_'+selected['exposure']].astype(int)-z['mask_'+selected['case']].astype(int); w=z['exposure']; f=flags['E']
 decomposition={}
 for name,mask in [('all',np.ones(len(g),dtype=bool)),('fallback',f),('nonfallback',~f)]:
  n=int(mask.sum()); W=float(w[mask].sum()); dn=int(g[mask].sum()); dw=float((g[mask]*w[mask]).sum())
  s={'n':n,'exposure_mass':W,'case_count_difference':dn,'exposure_mass_difference':dw,'delta_c_pp':100*dn/n,'delta_d_pp':100*dw/W,'case_weight':n/len(g),'exposure_weight':W/float(w.sum()),'contribution_delta_c_pp':100*dn/len(g),'contribution_delta_d_pp':100*dw/float(w.sum())}
  decomposition[name]=s;decomposition_csv.append({'dataset':dataset,'cohort':name,**s})
 for metric in ['case_count_difference','exposure_mass_difference','contribution_delta_c_pp','contribution_delta_d_pp']:
  check(dataset+' additive '+metric,abs(decomposition['fallback'][metric]+decomposition['nonfallback'][metric]-decomposition['all'][metric])<1e-12)
 results[dataset]={'cap':cap,'selected':selected,'full_coverage':full,'split_risk_comparison':ratios,'policies':policies,'gate_minus_matched_menu':gate_contrasts,'E_fallback_decomposition':decomposition,'joint_success_original':old['joint_success']}

scanfiles=['evidence/PROTOCOL.json','evidence/FREEZE_RECEIPT.json',*protocol['code_sha256'],'audit/STATISTICAL_DESIGN_REVIEW.md']
pattern=re.compile(r'\bpower\b|detectable|sample.?size|effect.?size|\bMDE\b|検出力',re.I)
scan=[]
for filename in scanfiles:
 path=ROOT/filename
 matches=[{'line':i,'text':line} for i,line in enumerate(path.read_text().splitlines(),1) if pattern.search(line)]
 scan.append({'path':filename,'sha256':digest(path),'text_matches':matches,'review':'Inspected protocol/code or design memo; no prospective power calculation or power-based sample-size choice documented.'})
for rel,h in original.items(): check('unchanged '+rel,digest(ROOT/rel)==h)
for filename,rows in [('full_coverage_risks.csv',full_csv),('frozen_policy_B_E_diagnostics.csv',policy_csv),('fallback_accounting.csv',decomposition_csv)]:
 keys=list(dict.fromkeys(k for row in rows for k in row))
 with (OUT/filename).open('w',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader();writer.writerows(rows)
save('RESULTS.json',{'analysis':'Retrospective descriptive; no new fitting/resampling/selection/inference','utc':datetime.now(timezone.utc).isoformat(),'spec_sha256':digest(OUT/'ANALYSIS_SPEC.json'),'script_sha256':digest(Path(__file__)),'datasets':results,'power_documentation':{'scope':'Available frozen protocol/code and archived design-review memo only; absence does not prove absence of an unrecorded calculation.','finding':'No documented prospective power calculation or power-based sample-size design.','inspected':scan},'checks':{'count':len(checks),'all_pass':all(c['passed'] for c in checks),'details':checks},'source_sha256':original})
print(json.dumps({'full_risks':{d:{**v['full_coverage'],**v['split_risk_comparison']} for d,v in results.items()},'fallback':{d:v['E_fallback_decomposition'] for d,v in results.items()},'gate_contrasts':{d:v['gate_minus_matched_menu'] for d,v in results.items()},'checks':len(checks)},indent=2))
