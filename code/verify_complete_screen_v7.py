"""Read-only independent verification; imports no experiment implementation."""
from pathlib import Path
import hashlib,json,sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
P=ROOT/'evidence/revision_v7_complete_screen'
OUT=ROOT/'reproduction_outputs/revision_v7_screen_verification'
OUT.mkdir(parents=True,exist_ok=True)
def read(p):return json.loads(p.read_text())
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
checks=[]
def ck(n,v):
 if not v:raise AssertionError(n)
 checks.append(n)
def near(n,a,b,atol=1e-10,rtol=1e-12):ck(n,np.allclose(a,b,rtol=rtol,atol=atol))
def quant(a,qs):
 """Columnwise linear quantiles, explicitly preserving endpoint direction."""
 z=np.sort(np.asarray(a),axis=0);ans=[]
 for q in qs:
  pos=(len(z)-1)*q;lo=int(np.floor(pos));hi=int(np.ceil(pos))
  ans.append(z[lo]+(z[hi]-z[lo])*(pos-lo))
 return np.asarray(ans)
def score(e,w,cap,m,T):
 if m=='error':return e
 if m=='weight_descending':return -w
 z=e-cap*w
 if m=='row':return z
 d=w/T if m=='demand' else .25+.75*w/T
 return np.divide(z,d,out=np.where(z>0,np.inf,np.where(z<0,-np.inf,0.)),where=d>0)
def accept(s,t):
 if t is None:return np.zeros(len(s),bool)
 if t=='all':return np.ones(len(s),bool)
 return s<=t
def stat(L,W,a,u,n):
 return np.array([np.bincount(u,weights=a,minlength=n),
                  np.bincount(u,weights=a*W,minlength=n),
                  np.bincount(u,weights=a*L,minlength=n)]).T
def threshold_check(s,L,W,cap,floor,t,label):
 if L.sum()<=cap*W.sum() and W.sum()>0:
  ck(label+' all',t=='all');return
 ix=np.argsort(s,kind='stable');s=s[ix]
 ends=np.flatnonzero(np.r_[s[1:]!=s[:-1],True]);weights=np.cumsum(W[ix]);loss=np.cumsum(L[ix])
 valid=(ends+1>=floor*len(s))&(weights[ends]>0)&(loss[ends]<=cap*weights[ends]+1e-9)&(s[ends]<np.inf)
 expected=float(s[ends[valid][-1]]) if np.any(valid) else None
 ck(label+' largest whole tie',expected==t)

pr=read(P/'PROTOCOL.json');f=read(P/'FROZEN.json');r=read(P/'RESULTS.json');start=read(P/'START.json');es=read(P/'EVALUATION_STARTED.json');complete=read(P/'COMPLETE.json')
before={str(p.relative_to(P)):sha(p) for p in P.rglob('*') if p.is_file()}
for name,h in complete['files_sha256'].items():ck('final hash '+name,sha(P/name)==h)
for name,h in start['inputs_sha256'].items():ck('input hash '+name,sha(ROOT/name)==h)
for name,h in start['code_sha256'].items():ck('code hash '+name,sha(ROOT/name)==h)
ck('protocol hash',sha(P/'PROTOCOL.json')==start['protocol_sha256']==f['protocol_sha256'])
ck('start hash',sha(P/'START.json')==f['start_sha256'])
ck('freeze hash',sha(P/'FROZEN.json')==es['freeze_sha256'])
ck('calibration stats hash',sha(P/'CALIBRATION_ITEM_STATISTICS.npz')==f['calibration_stats_sha256'])
ck('evaluation hash',sha(ROOT/'evidence/matched_censoring/complete/EVALUATION.npz')==es['evaluation_sha256'])
ck('declared event chronology',pr['utc']<start['utc']<f['utc']<es['utc']<r['completed_utc']<complete['utc'])
ck('no fit',pr['fit_calls']==complete['fit_calls']==0)
ck('no new external opening',pr['new_external_openings']==es['new_external_openings']==complete['new_external_openings']==0)
ck('twenty declared conditions',pr['candidate_family_size']==20 and len(pr['head_pairs'])==4 and len(pr['methods'])==5)
ck('B adjustment',pr['bootstrap']['screen_one_sided_alpha']==.05/20 and pr['bootstrap']['screen_family_size']==20)
ck('later adjustment20',pr['bootstrap']['later_twenty_candidate_upper_quantile']==1-.05/20)
ck('later adjustment8',pr['bootstrap']['later_selected_eight_upper_quantile']==1-.05/8)
near('design margin',pr['design_cap'],.95*pr['report_cap'])
base=read(ROOT/'evidence/matched_censoring/complete/FROZEN.json')
plans={(p['e_trees'],p['w_trees']):p for p in base['plans'] if p['kind']=='relative' and p['value']==.95}
near('old cap',pr['report_cap'],plans[(220,220)]['cap'])
near('score cap fixed',pr['score_cap'],pr['report_cap'])
old_attempt=read(P/'attempt_01_quantile_axis_bug/FROZEN.json')
ck('corrected rerun preserves policies',f['plans']==old_attempt['plans'])

S=np.load(P/'CALIBRATION_ITEM_STATISTICS.npz');E=np.load(P/'EVALUATION_ITEM_STATISTICS.npz')
cal=np.load(ROOT/'evidence/matched_censoring/complete/CALIBRATION.npz');ev=np.load(ROOT/'evidence/matched_censoring/complete/EVALUATION.npz')
items,inv=S['items'],S['series_item_index'];n=len(items)
ck('matching item identities',np.array_equal(items,E['items']) and np.array_equal(items,ev['items']) and np.array_equal(inv,ev['series_item_index']))
ck('calibration block chronology',np.array_equal(cal['blocks'],np.tile(np.r_[np.zeros(35),np.ones(28)],len(inv))))
ck('later chronology',np.array_equal(ev['days'],np.arange(1800,1914)))
A=cal['blocks']==0;B=cal['blocks']==1;ua=np.repeat(inv,35);ub=np.repeat(inv,28);ue=np.repeat(inv,114)
Lc=abs(cal['y']-cal['f']);Wc=cal['y'];Le=abs(ev['y']-ev['f']);We=ev['y'];cap=pr['report_cap'];T=pr['T']
near('A total',S['A_total'],np.array([np.bincount(ua),np.bincount(ua,weights=Wc[A])]).T)
near('B total',S['B_total'],np.array([np.bincount(ub),np.bincount(ub,weights=Wc[B])]).T)
near('evaluation total',E['total'],np.array([np.bincount(ue),np.bincount(ue,weights=We)]).T)
counts=np.random.default_rng(pr['bootstrap']['seed']).multinomial(n,np.ones(n)/n,size=pr['bootstrap']['draws'])
den=counts@E['total'];summary=[];counts_pass=0;nonempty_A=0
for fp,rp in zip(f['plans'],r['plans'],strict=True):
 cell=fp['cell'];ep,wp=fp['e_trees'],fp['w_trees'];near('baseline '+cell,fp['baseline_primary']['cap'],plans[(ep,wp)]['cap']);ck('baseline frozen '+cell,fp['baseline_primary']==plans[(ep,wp)])
 ck('all candidate families '+cell,list(fp['policies'])==pr['methods']==list(rp['candidates']))
 passed={};candidate_boot={}
 for m,mp in fp['policies'].items():
  ident=cell+'__'+m;t=mp['threshold'];sc=score(cal[f'e{ep}'],cal[f'w{wp}'],cap,m,T)
  threshold_check(sc[A],Lc[A],Wc[A],pr['design_cap'],pr['row_floor_A'],t,ident)
  for label,ind,u in [('A',A,ua),('B',B,ub)]:
   actual=stat(Lc[ind],Wc[ind],accept(sc[ind],t),u,n);near(ident+label+' raw reconstruction',S[ident+'__'+label],actual,atol=1e-7)
   sums=S[ident+'__'+label].sum(0);v=mp[label];near(ident+label+' sums',sums,[v['rows'],v['weight'],v['loss']],atol=1e-7)
   near(ident+label+' coverage',sums[:2]/S[label+'_total'].sum(0),[v['c'],v['d']])
   if sums[1]>0:near(ident+label+' ratio risk',sums[2]/sums[1],v['risk'])
  sb=S[ident+'__B'];ex=sb[:,2]-cap*sb[:,1];up=quant(counts@ex/n,[.9975])[0]
  near(ident+' B upper',up,mp['B_adjusted_excess_upper']);near(ident+' B mean',ex.mean(),mp['B_mean_signed_excess_per_item'])
  passed[m]=t is not None and sb[:,1].sum()>0 and up<=0
  ck(ident+' screen decision',passed[m]==mp['B_screen_pass'])
  counts_pass+=int(passed[m]);nonempty_A+=int(t is not None)
  if sb[:,1].sum()>0:
   b=counts@sb;rr=b[:,2]/b[:,1];near(ident+' B risk CI',quant(rr,[.025,.975]),mp['B_risk_ci95']);near(ident+' B risk upper',quant(rr,[.9975])[0],mp['B_risk_upper_family20'])
  se=E[ident];actual=stat(Le,We,accept(score(ev[f'e{ep}'],ev[f'w{wp}'],cap,m,T),t),ue,n);near(ident+' raw evaluation',se,actual,atol=1e-7)
  v=rp['candidates'][m];sums=se.sum(0);near(ident+' evaluation sums',sums,[v['rows'],v['weight'],v['loss']],atol=1e-7);near(ident+' eval coverage',sums[:2]/E['total'].sum(0),[v['c'],v['d']])
  bb=counts@se;candidate_boot[m]=bb;cv=bb[:,:2]/den;ci=quant(cv,[.025,.975]).T
  near(ident+' coverage CI axis',ci,v['coverage_ci95']);ck(ident+' coverage bounds ordered',np.all(ci[:,0]<=ci[:,1]))
  if sums[1]>0:
   near(ident+' eval risk',sums[2]/sums[1],v['risk']);rr=bb[:,2]/bb[:,1]
   near(ident+' eval risk CI',quant(rr,[.025,.975]),v['risk_ci95']);near(ident+' eval upper20',quant(rr,[.9975])[0],v['risk_upper_family20']);near(ident+' eval upper8',quant(rr,[.99375])[0],v['risk_upper_selected8'])
  else:ck(ident+' no false zero risk',v['risk'] is None and v['risk_ci95'] is None)
  old=E[cell+'__baseline__'+m];ov=rp['baseline_candidates'][m];near(ident+' old metrics',old.sum(0),[ov['rows'],ov['weight'],ov['loss']],atol=1e-7)
 for o in ['c','d']:
  candidates=[m for m in pr['methods'] if passed[m]]
  expected=max(candidates,key=lambda m:min(mm[o] for mm in fp['policies'][m]['calibration'])) if candidates else None
  ck(cell+o+' selection',expected==fp['selected'][o]==rp['selected'][o])
  v=rp['selected_policies'][o];ck(cell+o+' selected record',v['method']==expected)
  if expected is not None:
   bm=fp['baseline_primary']['selected'][o];s=E[cell+'__'+expected];old=E[cell+'__baseline__'+bm];delta=(counts@(s-old))[:,:2]/den;point=(s-old).sum(0)[:2]/E['total'].sum(0)
   near(cell+o+' coverage change',point,v['coverage_change_vs_unscreened_primary']);near(cell+o+' coverage cost',-point,v['coverage_cost_unscreened_minus_screened']);near(cell+o+' change CI axis',quant(delta,[.025,.975]).T,v['coverage_change_ci95'])
   ck(cell+o+' full candidate upper below cap',v['metrics']['risk_upper_family20']<cap)
   summary.append({'cell':cell,'utility':o,'method':expected,'case_percent':100*v['metrics']['c'],'exposure_percent':100*v['metrics']['d'],'risk':v['metrics']['risk'],'risk_ci95':v['metrics']['risk_ci95'],'upper20':v['metrics']['risk_upper_family20'],'coverage_cost_pp':(-100*point).tolist()})
 c,d=rp['selected']['c'],rp['selected']['d']
 if c is not None and d is not None:
  delta=(candidate_boot[d]-candidate_boot[c])[:,:2]/den;point=(E[cell+'__'+d]-E[cell+'__'+c]).sum(0)[:2]/E['total'].sum(0);ci=quant(delta,[.025,.975]).T
  near(cell+' contrast point',point,[rp['contrast']['dc'],rp['contrast']['dd']]);near(cell+' contrast CI axis',ci,rp['contrast']['ci95']);ck(cell+' contrast directions',ci[0,1]<0<ci[1,0])
 print('PASS',cell,flush=True)

after={str(p.relative_to(P)):sha(p) for p in P.rglob('*') if p.is_file()};ck('evidence unchanged',before==after)
report={'status':'PASS','checks':len(checks),'names':checks,'nonempty_A':nonempty_A,'B_passers':counts_pass,'B_additional_exclusions':nonempty_A-counts_pass,'selected':summary,
 'conclusions':['All eight selected-condition records have family-20 one-sided bootstrap risk upper bounds below cap 0.6326834234117924.',
                'All four paired contrasts have case-loss CIs strictly negative and exposure-gain CIs strictly positive.',
                'All 16 A-nonempty candidates pass B; B excludes no additional feasible candidate. Attribute observed changes to the combined stricter A-design/B-audit procedure, not uniquely to B screening.',
                'The four cells are shared fitted-head conditions. The eight records include two duplicated descending-exposure policies, so there are six unique selected policies, not eight independent policies.',
                'Bootstrap upper limits are retrospective descriptive approximations conditional on existing fits; no finite-sample distribution-free risk contract or prospective generalization claim follows.']}
(OUT/'REPORT.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='names'},indent=2))
