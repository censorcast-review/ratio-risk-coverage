"""Independent checks of the policy audit and compatibility with frozen evidence."""
from pathlib import Path
import argparse, hashlib, json
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
def load(p): return json.loads(Path(p).read_text())

def main(out):
    P=ROOT/'evidence/policy_choice'; f=load(P/'FROZEN.json'); r=load(P/'RESULTS.json')
    checks=[]
    def ck(name,condition):
        if not condition: raise AssertionError(name)
        checks.append(name)
    def near(name,a,b,tol=1e-10): ck(name,np.allclose(a,b,atol=tol,rtol=0))
    for name,h in load(P/'COMPLETE.json')['files'].items():
        ck('hash:'+name,hashlib.sha256((P/name).read_bytes()).hexdigest()==h)
    for name,h in f['inputs'].items():
        ck('input:'+name,hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==h)
    ck('freeze precedes evaluation',f['utc']<load(P/'EVALUATION_STARTED.json')['utc'])
    ck('no new opening',load(P/'COMPLETE.json')['external_openings']==0)
    stat=np.load(P/'TEST_STATISTICS.npz'); screen=np.load(P/'SCREEN_STATISTICS.npz')
    methods=('error','row','mixed','demand','weight_descending')
    cap=.85*.7530939208
    old=load(ROOT/'evidence/accuracy_selection/FROZEN_SELECTORS.json')
    bike=load(ROOT/'evidence/nonretail/bike/FROZEN.json')
    for section in ('menus','screens'):
        for rec in r[section]:
            model=rec['model']; prefix=section+'__'+model+'__'
            primary=(model=='bike' and rec['factor']==.85) or (model!='bike' and rec['cap']==cap)
            candidates=[m for m in methods if rec['policies'][m]['threshold'] is not None and
                        (section!='screens' or rec['policies'][m]['passes_screen'])]
            for obj in ('c','d'):
                expected=max(candidates,key=lambda m:min(v[obj] for v in rec['policies'][m]['calibration'])) if candidates else None
                ck(f'{section}:{model}:{rec["cap"]}:choice:{obj}',rec['selected'][obj]==expected)
            if not primary: continue
            total=stat[('Bike' if model=='bike' else 'M5')+'__total']; n=len(total)
            counts=np.random.default_rng(20260911).multinomial(n,np.full(n,1/n),size=4000)
            for m in methods:
                s=stat[prefix+m]; sums=s.sum(0); report=rec['test'][m]
                near(model+m+section+' sums',sums,[report['rows'],report['weight'],report['loss']],1e-6)
                near(model+m+section+' coverage',sums[:2]/total.sum(0),[report['c'],report['d']])
                if sums[1]>0:
                    b=counts@s;ci=np.quantile(b[:,2]/b[:,1],[.025,.975])
                    near(model+m+section+' risk',ci,report['risk_ci95'])
                else: ck(model+m+section+' no fake zero risk',report['risk'] is None)
                if section=='screens':
                    sB=screen[model+'__'+m]; ex=sB[:,2]-cap*sB[:,1]
                    up=np.quantile(counts@ex/n,1-.05/15)
                    near(model+m+' screen bound',up,rec['policies'][m]['excess_upper_B'])
                    ck(model+m+' screen',rec['policies'][m]['passes_screen']==
                       (rec['policies'][m]['threshold'] is not None and sB[:,1].sum()>0 and up<=0))
                elif model!='bike' and m in ('row','mixed','demand'):
                    lam={'row':'1.0','mixed':'0.25','demand':'0.0'}[m]
                    near(model+m+' original threshold',rec['policies'][m]['threshold'],old['policies'][model][lam],1e-9)
                elif model=='bike':
                    prev=next(p for p in bike['policies'] if p['factor']==.85 and p['method']==m)
                    if prev['threshold'] is None: ck('Bike '+m+' infeasible',rec['policies'][m]['threshold'] is None)
                    else:
                        # Pure-ratio scores differ by a fixed scale T only.
                        expected=prev['threshold']*(rec['T'] if m=='demand' else 1)
                        near('Bike '+m+' threshold',expected,rec['policies'][m]['threshold'],1e-9)
            a,b=[rec['selected'][o] for o in ('c','d')]
            if a is not None and b is not None:
                diff=stat[prefix+b]-stat[prefix+a]; boot=(counts@diff)[:,:2]/(counts@total)
                near(model+section+' contrast95',np.quantile(boot,[.025,.975],axis=0).T,rec['contrast']['ci95'])
                near(model+section+' contrast99.375',np.quantile(boot,[.003125,.996875],axis=0).T,rec['contrast']['ci99_375'])
                near(model+section+' budget union',sum(rec['budget'][k]['excess'] for k in ('common','case_only','exposure_only')),rec['budget']['union']['excess'],1e-6)
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps({'status':'PASS','checks':len(checks),'names':checks},indent=2)+'\n')
    print('PASS',len(checks),'policy-choice checks')

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--output',type=Path,default=Path('reproduction_outputs/policy_choice_verification.json'));main(a.parse_args().output)
