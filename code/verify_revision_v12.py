"""Independent numerical verification; no imports from selection/run code."""
from pathlib import Path
import json, hashlib, argparse, datetime
from fractions import Fraction as F
import numpy as np
from scipy.optimize import linprog

ROOT=Path(__file__).resolve().parents[1]
METHODS=('error','row','mixed','demand','weight_descending')
CHECKS=0
def check(ok,msg='check'):
    global CHECKS
    assert ok,msg
    CHECKS+=1
def close(x,y,tol=2e-9): check(np.allclose(x,y,atol=tol,rtol=tol),str((x,y)))
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(p):return json.loads(Path(p).read_text())
def arr(p):
    with np.load(p) as a:return {k:a[k] for k in a.files}
def score(e,w,r,T,m):
    if m=='error':return e
    if m=='row':return e-r*w
    if m=='weight_descending':return -w
    den=w/T if m=='demand' else .25+.75*w/T
    q=e-r*w
    return np.divide(q,den,out=np.where(q>0,np.inf,np.where(q<0,-np.inf,0.)),where=den>0)
def accepted(s,t):return np.ones(len(s),bool) if t=='all' else np.zeros(len(s),bool) if t is None else s<=t
def metric(L,W,a):
    mass=float(W[a].sum());loss=float(L[a].sum())
    return dict(rows=int(a.sum()),c=float(a.mean()),d=mass/W.sum(),loss=loss,weight=mass,risk=loss/mass if mass else None)
def metric_check(got,want):
    for k,v in got.items():
        if v is None:check(want[k] is None)
        else:close(v,want[k])
def last_threshold(s,L,W,blocks,r,floor):
    keys=np.unique(blocks)
    if all(W[blocks==b].sum()>0 and L[blocks==b].sum()<=r*W[blocks==b].sum() for b in keys):return 'all'
    ts=np.unique(s[s<np.inf]);keep=np.ones(len(ts),bool)
    for b in keys:
        ids=np.flatnonzero(blocks==b);order=ids[np.argsort(s[ids],kind='stable')]
        k=np.searchsorted(s[order],ts,side='right')
        mass=np.r_[0,np.cumsum(W[order])][k];loss=np.r_[0,np.cumsum(L[order])][k]
        keep&=(k>=floor*len(ids))&(mass>0)&(loss<=r*mass+1e-9)
    return float(ts[np.flatnonzero(keep)[-1]]) if keep.any() else None
def stats(L,W,a,units,n):
    # Sort/reduceat is separate from the production bincount aggregation.
    order=np.argsort(units,kind='stable');u=units[order]
    starts=np.r_[0,np.flatnonzero(u[1:]!=u[:-1])+1];out=np.zeros((n,3))
    for j,v in enumerate([a,a*W,a*L]):out[u[starts],j]=np.add.reduceat(v[order],starts)
    return out
def select(menu,obj):
    eligible=[m for m in METHODS if menu[m]['threshold'] is not None]
    return max(eligible,key=lambda m:min(x[obj] for x in menu[m]['calibration'])) if eligible else None

def seeds():
    master=read(ROOT/'evidence/revision_v12_seeds/PROTOCOL.json');records=[]
    for rel,h in master['code_sha256'].items():check(sha(ROOT/rel)==h,rel)
    for seed in master['additional_seeds']:
        o=ROOT/f'evidence/revision_v12_seeds/seed_{seed}';p=read(o/'PROTOCOL.json')
        check(p['seed']==seed and p['bootstrap_seed']==20260910)
        rows=arr(o/'TRAINING_ROWS.npz');rng=np.random.default_rng(seed)
        for name,lo,hi in [('point',1314,1554),('head',1555,1610)]:
            nd=hi-lo+1;ix=np.sort(rng.choice(18290*nd,600000,replace=False))
            check(np.array_equal(rows[name+'_series'],ix//nd));check(np.array_equal(rows[name+'_days'],lo+ix%nd))
        complete=read(o/'COMPLETE.json');check(complete['fit_calls']==10)
        for rel,h in complete['files'].items():check(sha(o/rel)==h,rel)
        for arm in ['censored','complete']:
            a=o/arm;calpath=a/'CALIBRATION.npz'
            if seed==20260914 and arm=='censored':
                recovery=ROOT/'evidence/revision_v12_seeds/array_recovery';receipt=read(recovery/'COMPLETE.json');rp=read(recovery/'PROTOCOL.json')
                check(receipt['fit_calls']==0)
                check(sha(calpath)==receipt['original_sha256'])
                check(sha(recovery/'PROTOCOL.json')==receipt['protocol_sha256'])
                for rel,h in rp['source_sha256'].items():check(sha(ROOT/rel)==h,rel)
                for rel,h in rp['frozen_sha256'].items():check(sha(ROOT/rel)==h,rel)
                check(receipt['exact_retained_array_matches']==['e220','e660','f','w220'])
                cal={}
                for key,rec in receipt['arrays'].items():
                    q=recovery/'arrays'/f'{key}.npy';check(sha(q)==rec['sha256']);check(q.stat().st_size==rec['bytes']);cal[key]=np.load(q)
                check(set(cal)=={'f','e220','e660','w220','w660','y','blocks'})
            else:cal=arr(calpath)
            ev=arr(a/'EVALUATION.npz');fr=read(a/'FROZEN.json');res=read(a/'RESULTS.json')
            for name,h in fr['models'].items():check(sha(a/name)==h)
            check(np.array_equal(ev['days'],np.arange(1800,1914)))
            L=np.abs(cal['y']-cal['f']);W=cal['y'];blocks=cal['blocks']
            full=max(L[blocks==b].sum()/W[blocks==b].sum() for b in [0,1]);close(full,fr['full_calibration_reference'])
            Le=np.abs(ev['y']-ev['f']);We=ev['y'];units=np.repeat(ev['series_item_index'],114);n=len(ev['items'])
            total=np.column_stack([np.bincount(units),np.bincount(units,weights=We)])
            counts=np.random.default_rng(20260910).multinomial(n,np.full(n,1/n),size=2000)
            primary=[p for p in res['menus'] if p['kind']=='relative' and p['value']==.95];check(len(primary)==4)
            for p in primary:
                close(p['cap'],.95*full);r=p['cap'];e=cal['e'+str(p['e_trees'])];w=cal['w'+str(p['w_trees'])];st={}
                for m in METHODS:
                    s=score(e,w,r,fr['T'],m);t=last_threshold(s,L,W,blocks,r,.35);check(t==p['policies'][m]['threshold'],m)
                    ac=accepted(s,t)
                    for j,b in enumerate([0,1]):metric_check(metric(L[blocks==b],W[blocks==b],ac[blocks==b]),p['policies'][m]['calibration'][j])
                    se=score(ev['e'+str(p['e_trees'])],ev['w'+str(p['w_trees'])],r,fr['T'],m);ae=accepted(se,t)
                    metric_check(metric(Le,We,ae),p['test'][m]);st[m]=stats(Le,We,ae,units,n)
                c,d=select(p['policies'],'c'),select(p['policies'],'d');check(p['selected']==dict(c=c,d=d))
                if c is not None and d is not None:
                    delta=(counts@(st[d]-st[c]))[:,:2]/(counts@total);ci=np.quantile(delta,[.025,.975],axis=0).T
                    close(ci,p['contrast']['ci95']);close(p['test'][d]['c']-p['test'][c]['c'],p['contrast']['dc']);close(p['test'][d]['d']-p['test'][c]['d'],p['contrast']['dd'])
                records.append(dict(seed=seed,arm=arm,heads=[p['e_trees'],p['w_trees']],verified=True))
            print('VERIFIED seed',seed,arm,flush=True)
    return records

def external():
    o=ROOT/'evidence/revision_v12_external';proto=read(o/'PROTOCOL.json');f=read(o/'FROZEN_A.json');result=read(o/'RESULTS.json')
    check(sha(o/'PROTOCOL.json')==(o/'PROTOCOL.sha256').read_text().split()[0])
    for rel,h in proto['code_sha256'].items():check(sha(ROOT/rel)==h)
    for k,h in f['model_sha256'].items():check(sha(o/f'MODEL_{k}.txt')==h)
    dates=[read(o/n)['utc'] for n in ['PROTOCOL.json','INPUT_RECEIPT.json','START.json','FROZEN_A.json','B_OPEN.json','E_OPEN.json','IMPLEMENTATION_AMENDMENT.json','RESUME_START.json','COMPLETE.json']]
    check(dates==sorted(dates),'chronology')
    check(read(o/'COMPLETE.json')['fit_calls']==3)
    A=arr(o/'A_ARRAYS.npz');L=np.abs(A['y']-A['f']);W=A['y'];r=.95*L.sum()/W.sum();close(r,f['report_cap']);close(.95*r,f['design_cap'])
    for m in METHODS:
        s=score(A['e'],A['w'],r,f['T'],m);t=last_threshold(s,L,W,np.zeros(len(W)),.95*r,.40)
        check(t==f['plans'][m]['threshold']);metric_check(metric(L,W,accepted(s,t)),f['plans'][m]['calibration'][0])
    c,d=select(f['plans'],'c'),select(f['plans'],'d');check(f['selected']==dict(c=c,d=d))
    outputs={}
    for name,lo,hi,seed in [('B',313,364,2026091201),('E',365,400,2026091202)]:
        a=arr(o/f'{name}_ARRAYS.npz');saved=arr(o/f'{name}_WEEKLY_STATISTICS.npz');n=hi-lo+1
        check(a['week'].min()>=lo and a['week'].max()<=hi)
        rng=np.random.default_rng(seed);starts=rng.integers(0,n,(20000,(n+3)//4));counts=np.zeros((20000,n),int)
        for i,st in enumerate(starts):counts[i]=np.bincount(np.concatenate([(np.arange(4)+v)%n for v in st])[:n],minlength=n)
        check(np.array_equal(counts,saved['counts']));check(np.all(counts.sum(axis=1)==n))
        L=np.abs(a['y']-a['f']);W=a['y'];u=a['week']-lo
        total=stats(L,W,np.ones(len(W),bool),u,n)[:,:2];close(total,saved['total'])
        st={}
        for m in METHODS:
            accept=accepted(score(a['e'],a['w'],r,f['T'],m),f['plans'][m]['threshold']);metric_check(metric(L,W,accept),result[name][m])
            st[m]=stats(L,W,accept,u,n);close(st[m],saved[m]);dr=counts@st[m]
            upper=float(np.quantile(dr[:,2]-r*dr[:,1],.99));close(upper,result[name][m]['signed_excess_U5'])
            nz=(dr[:,1]>0).all();passed=bool(f['plans'][m]['threshold'] is not None and nz and upper<=0);check(passed==result[name][m]['passed'])
            if nz:close(np.quantile(dr[:,2]/dr[:,1],.99),result[name][m]['risk_U5'])
        outputs[name]=(counts,total,st)
    success=False
    if c is not None and d is not None:
        counts,total,st=outputs['E'];v=(counts@(st[d]-st[c]))[:,:2]/(counts@total);ci=np.quantile(v,[.025,.975],axis=0).T
        close(ci,result['contrast']['ci95']);both=result['B'][c]['passed'] and result['B'][d]['passed']
        success=bool(both and ci[0,1]<0 and ci[1,0]>0)
    check(success==result['primary_success']);print('VERIFIED external',flush=True)
    return dict(success=success,selected=result['selected'],empty_weeks_preserved=[370,371,400])

def theory():
    rng=np.random.default_rng(20260912);runs=0
    for _ in range(250):
        n=8;P=np.full(n,1/n);w=rng.uniform(.5,3,n);r=.8;g=rng.uniform(.1,.8,n);g[:3]=-rng.uniform(.1,.35,3)
        ell=r*w+g
        if np.any(ell<0):continue
        a0=np.r_[np.ones(3),np.zeros(5)];s=-np.dot(P*g,a0)
        wh=w+rng.uniform(-.002,.002,n);lh=ell+rng.uniform(-.002,.002,n);gh=lh-r*wh
        eM=np.dot(P,np.abs(wh-w));eG=np.dot(P,np.abs(lh-ell))+r*eM
        for tau in [0,eG]:
            true=linprog(-P*w,A_ub=np.array([P*g,-P]),b_ub=[0,-.2],bounds=[(0,1)]*n,method='highs')
            est=linprog(-P*wh,A_ub=np.array([P*gh,-P]),b_ub=[-tau,-.2],bounds=[(0,1)]*n,method='highs')
            check(true.success and est.success);check(eG+tau<=s)
            loss=np.dot(P*w,true.x-est.x);bound=(eG+tau)/s*np.dot(P*w,true.x-a0)+2*eM
            check(loss<=bound+1e-8);G=np.dot(P*g,est.x);M=np.dot(P*w,est.x)
            check(G<=eG-tau+1e-8);check(max(G/M,0)<=max(eG-tau,0)/M+1e-8)
            runs+=1
    for den in [10,100,1000,10000,1000000]:
        eps=F(1,den);r=F(1,2);a=[eps,F(1)];G=(eps*a[0]-eps**2*a[1])/2
        check(G==0);check((a[0]+a[1])/2==(1+eps)/2)
        w=[2*eps,2*(1-eps)];ell=[2*(r+1)*eps,(r+2)*w[1]]
        check((w[0]+w[1])/2==1);check(ell[0]/w[0]==r+1);check((ell[0]-2*r*eps)/2==eps)
    return dict(LP_comparisons=runs,rational_counterexamples=10,interpretation='Numerical falsification checks complement the written proof; not a proof of population assumptions')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--part',choices=['seeds','external','theory'],required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    result=globals()[a.part]()
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(dict(status='PASS',part=a.part,checks=CHECKS,result=result,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),verifier_sha256=sha(__file__)),indent=2)+'\n')
    print('PASS',a.part,CHECKS,flush=True)
