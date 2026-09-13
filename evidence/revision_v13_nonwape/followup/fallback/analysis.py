"""Retrospective frozen-policy subgroup and exposure diagnostics; no refits."""
from __future__ import annotations
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import csv, hashlib, importlib.util, json, pathlib, sys
from datetime import datetime,timezone
import joblib
import numpy as np
from scipy import sparse
from scipy.special import expit
from xml.dom.minidom import parseString
ROOT=pathlib.Path(__file__).resolve().parents[2]
OUT=pathlib.Path(__file__).resolve().parent
SPEC=OUT/'ANALYSIS_SPEC.json'
METHODS=['Error','Excess','Mixed','Ratio','Descending','GateCase','GateExposure']
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(p.read_text())
def write(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
class FixedClassifier: pass
spec=importlib.util.spec_from_file_location('raw_audit_readonly',ROOT/'audit/RAW_MODEL_CHECK.py')
audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)
CHECKS=[]
def check(name,ok):
 CHECKS.append({'check':name,'passed':bool(ok)})
 if not ok:raise AssertionError(name)
def stats(w):
 q=np.quantile(w,[.05,.25,.5,.75,.95],method='linear');mean=float(np.mean(w));sd=float(np.std(w,ddof=0))
 return {'n':len(w),'mean':mean,'sd_population':sd,'cv':sd/mean,'minimum':float(min(w)),'maximum':float(max(w)),'p05':float(q[0]),'p25':float(q[1]),'p50':float(q[2]),'p75':float(q[3]),'p95':float(q[4]),'count_zero':int(sum(w==0)),'share_one':float(np.mean(w==1)),'p95_over_p05':float(q[4]/q[0]) if q[0] else None,'p75_over_p25':float(q[3]/q[1]) if q[1] else None,'IQR_over_median':float((q[3]-q[1])/q[2]) if q[2] else None,'observed_max_over_min':float(max(w)/min(w)) if min(w) else None}
def metrics(a,cohort,loss,w,eligible):
 n=int(cohort.sum());totalw=float(w[cohort].sum());b=a&cohort
 if not eligible:return {'available':False,'cohort_n':n,'cohort_exposure':totalw,'accepted_n':None,'accepted_loss':None,'accepted_exposure':None,'case_coverage':None,'exposure_coverage':None,'risk':None,'reason':'original policy A-ineligible'}
 acceptedn=int(b.sum());acceptedw=float(w[b].sum());acceptedloss=float(loss[b].sum())
 return {'available':True,'cohort_n':n,'cohort_exposure':totalw,'accepted_n':acceptedn,'accepted_loss':acceptedloss,'accepted_exposure':acceptedw,'case_coverage':acceptedn/n if n else None,'exposure_coverage':acceptedw/totalw if totalw else None,'risk':acceptedloss/acceptedw if acceptedw else None}
def contrasts(methodmetrics,first,second):
 a,b=methodmetrics[first],methodmetrics[second]
 if not(a['available'] and b['available']):return {'available':False,'first':first,'second':second}
 return {'available':True,'first':first,'second':second,'delta_c_pp':100*(a['case_coverage']-b['case_coverage']),'delta_d_pp':100*(a['exposure_coverage']-b['exposure_coverage']),'delta_risk':a['risk']-b['risk'] if a['risk'] is not None and b['risk'] is not None else None}
def run():
 spec=read(SPEC);check('analysis_code_hash_declared',sha(pathlib.Path(__file__))==spec['analysis_code_sha256'])
 check('spec_precedes_analysis',datetime.fromisoformat(spec['utc'])<datetime.now(timezone.utc))
 original={}
 protocol=read(ROOT/'evidence/PROTOCOL.json')
 for name,want in protocol['code_sha256'].items():
  p=ROOT/name;check('frozen_code_'+name,sha(p)==want);original[str(p.relative_to(ROOT))]=want
 original['evidence/PROTOCOL.json']=sha(ROOT/'evidence/PROTOCOL.json')
 results={};dispersion=[];metricrows=[];contrastrows=[]
 for dataset in ['bibtex','mediamill']:
  base=ROOT/'evidence'/dataset;fixed=read(base/'A_FIXED.json');oldresult=read(base/'RESULT.json');complete=read(base/'COMPLETE.json')
  for name,want in complete['files'].items():
   p=base/name;check(dataset+'_source_'+name,sha(p)==want);original[str(p.relative_to(ROOT))]=want
  raw=ROOT/'raw'/(dataset+'.rar');check(dataset+'_raw_hash',sha(raw)==read(ROOT/'evidence'/('DOWNLOAD_'+dataset+'.json'))['sha256'])
  files,entries=audit.rar_contents(raw)
  xml=[v for k,v in files.items() if k.lower().endswith('.xml')];check(dataset+'_one_xml',len(xml)==1)
  names=[x.getAttribute('name') for x in parseString(xml[0]).getElementsByTagName('label')]
  testfiles=[(k,v) for k,v in files.items() if k.lower().endswith('.arff') and 'test' in k.lower()];check(dataset+'_one_test',len(testfiles)==1)
  name,data=testfiles[0];check(dataset+'_test_member_hash',hashlib.sha256(data).hexdigest()==read(base/'DATA.json')['metadata']['member_sha256'][name])
  X,Y,_=audit.parse_separately(data,names)
  model=joblib.load(base/'classifier.joblib');indices=dict(np.load(base/'SPLITS.npz',allow_pickle=False))
  check(dataset+'_model_hash',sha(base/'classifier.joblib')==fixed['model_sha256']['classifier.joblib'])
  datasetresult={'r_original':fixed['r'],'design_cap_original':fixed['design_cap'],'floor_original':fixed['floor'],'menu_selected_original':fixed['menu_selected'],'roles':{}}
  fallbackflags={}
  for role in ['A','B','E']:
   rows=indices[role];Z=sparse.csr_matrix(X[rows]).multiply(1/model.scaler.scale_).tocsr()
   prob=np.empty((len(rows),len(model.models)))
   for k,m in enumerate(model.models):
    if isinstance(m,float):prob[:,k]=m
    else:prob[:,k]=expit(np.asarray(Z@m.coef_.T).ravel()+float(m.intercept_[0]))
   pred=prob>=.5;fallback=np.sum(pred,axis=1)==0
   pred[fallback,np.argmax(prob[fallback],axis=1)]=True
   w=np.count_nonzero(pred,axis=1);loss=w-(pred*Y[rows]).sum(1)
   fallbackflags[role]=fallback
   with np.load(base/(role+'_ARRAYS.npz'),allow_pickle=False) as a:
    check(dataset+'_'+role+'_FP_exact',np.array_equal(loss,a['loss']))
    check(dataset+'_'+role+'_w_exact',np.array_equal(w,a['exposure']))
    check(dataset+'_'+role+'_fallback_count',int(fallback.sum())==oldresult['top1_fallback_count'][role])
    cohorts={'all_rows':np.ones(len(rows),bool),'nonfallback_rows':~fallback,'fallback_rows':fallback}
    roledata={'n':len(rows),'fallback_n':int(fallback.sum()),'fallback_share':float(fallback.mean()),'exposure':{},'cohorts':{}}
    for cohortname,cohort in cohorts.items():
     roledata['exposure'][cohortname]=stats(w[cohort]);dispersion.append({'dataset':dataset,'role':role,'cohort':cohortname,**stats(w[cohort])})
     mm={}
     for method in METHODS:
      policy=fixed['policies'][method];eligible=policy['eligible']
      if eligible:
       threshold=policy['threshold'];boundary=policy['boundary_hash']
       mask=(a[method]<threshold)|((a[method]==threshold)&(a['tie']<=boundary))
      else:mask=np.zeros(len(rows),bool)
      if role!='A':check(dataset+'_'+role+'_'+method+'_'+cohortname+'_mask_exact',np.array_equal(mask,a['mask_'+method]))
      mm[method]=metrics(mask,cohort,loss,w,eligible)
      if cohortname=='all_rows' and eligible:
       expected=policy['A'] if role=='A' else oldresult[role][method]
       for key in ['accepted_n','accepted_loss','accepted_exposure','case_coverage','exposure_coverage','risk']:
        check(dataset+'_'+role+'_'+method+'_'+key,np.isclose(mm[method][key],expected[key],rtol=0,atol=2e-15))
      metricrows.append({'dataset':dataset,'role':role,'cohort':cohortname,'method':method,**mm[method]})
     cs={
      'menu_exposure_minus_case':contrasts(mm,fixed['menu_selected']['exposure'],fixed['menu_selected']['case']),
      'gate_case_minus_case_menu':contrasts(mm,'GateCase',fixed['menu_selected']['case']),
      'gate_exposure_minus_exposure_menu':contrasts(mm,'GateExposure',fixed['menu_selected']['exposure'])}
     roledata['cohorts'][cohortname]={'policies':mm,'contrasts':cs}
     for contrast,c in cs.items():contrastrows.append({'dataset':dataset,'role':role,'cohort':cohortname,'contrast':contrast,**c})
    datasetresult['roles'][role]=roledata
  np.savez_compressed(OUT/(dataset+'_FALLBACK_FLAGS.npz'),**fallbackflags)
  results[dataset]=datasetresult
 for rel,want in original.items():check('source_unchanged_'+rel,sha(ROOT/rel)==want)
 report={'utc':datetime.now(timezone.utc).isoformat(),'status':'PASS','checks_n':len(CHECKS),'analysis_spec_sha256':sha(SPEC),'analysis_code_sha256':sha(pathlib.Path(__file__)),'primary_protocol_sha256':sha(ROOT/'evidence/PROTOCOL.json'),'scope':'Retrospective fixed-policy subgroup diagnostics on already evaluated roles. All original predictors, heads, gates, score thresholds, cap, floor, and selected menu rules retained. Nonfallback rows are those with at least one p>=0.5 prediction before top-one fallback. The subgroup changes the evaluation population, not classifier or selection design. No new B-screen approval, confidence interval, replication success claim, tuning, or fitting.','datasets':results,'checks':CHECKS,'original_scientific_sha256':original}
 write(OUT/'RESULT.json',report)
 for name,rows in [('exposure_dispersion.csv',dispersion),('fixed_policy_subgroups.csv',metricrows),('fixed_policy_contrasts.csv',contrastrows)]:
  fields=[]
  for row in rows:
   for key in row:
    if key not in fields:fields.append(key)
  with open(OUT/name,'w',newline='') as f:
   writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
 print(json.dumps({'status':'PASS','checks':len(CHECKS),'datasets':{d:{'E_n':z['roles']['E']['n'],'E_fallback_n':z['roles']['E']['fallback_n'],'E_contrasts':{c:x['contrasts'] for c,x in z['roles']['E']['cohorts'].items()},'E_exposure':z['roles']['E']['exposure']} for d,z in results.items()}},indent=2))
if __name__=='__main__':run()
