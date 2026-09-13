"""A controlled counterexample and repeated-sample bound experiment.

The construction is a mechanism demonstration, not an estimate of retail effect size.
All model scores here are analytic conditional expectations, not fitted models.
"""
import argparse,json,time
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score

def sample(rng,n):
    group=rng.random(n)<.2
    mu=np.where(group,10.,1.)
    er=np.where(group,2.,rng.uniform(.8,1.,n))
    pred=mu-er
    y=mu+rng.uniform(-.2,.2,n)
    # Error is positive throughout, so E|Y-p| = er exactly.
    score={'absolute_error':er,'relative_error':er/mu,'contract_excess':er-.3*mu}
    return y,pred,score,group

def max_frontier(y,p,s,cap=.3):
    idx=np.argsort(s,kind='stable');ss=s[idx];end=np.r_[np.flatnonzero(ss[1:]!=ss[:-1]),len(s)-1]
    mass=np.cumsum(y[idx])[end];err=np.cumsum(np.abs(y-p)[idx])[end]
    good=err<=cap*mass
    if not good.any():return 0.
    return float((end[good].max()+1)/len(y))

def main(out,reps):
    out.mkdir(parents=True,exist_ok=True);start=time.time();rng=np.random.default_rng(20260906)
    y,p,s,g=sample(rng,200000)
    counter={k:dict(auc_high_error_type=float(roc_auc_score(g,v)),
                   maximum_empirical_coverage=max_frontier(y,p,v)) for k,v in s.items()}
    rows=[]
    # Prespecified thresholds, bounded excess width 2.5 from this generator.
    thresholds=np.linspace(-1.3,.9,101);alpha=.05;K=len(thresholds)
    for n in (1000,5000):
        for repetition in range(reps):
            rng=np.random.default_rng(1000000+n*1000+repetition)
            yc,pc,sc,gc=sample(rng,n);yt,pt,st,gt=sample(rng,20000)
            gval=np.abs(yc-pc)-.3*yc
            a=sc['contract_excess'][:,None]<=thresholds
            empirical=(a*gval[:,None]).mean(axis=0)
            coverage=a.mean(axis=0)
            margin=2.5*np.sqrt(np.log(2*K/alpha)/(2*n))
            covmargin=np.sqrt(np.log(2*K/alpha)/(2*n))
            good=(empirical+margin<=0)&(coverage-covmargin>=.30)
            if good.any():
                threshold=thresholds[np.flatnonzero(good)[-1]];accepted=st['contract_excess']<=threshold
                wape=float(np.abs(yt[accepted]-pt[accepted]).sum()/yt[accepted].sum())
                cov=float(accepted.mean())
                rows.append(dict(n=n,repetition=repetition,issued=True,coverage=cov,wape=wape,
                                 population_violation_proxy=bool(wape>.3 or cov<.3)))
            else:rows.append(dict(n=n,repetition=repetition,issued=False,coverage=0.,wape=None,population_violation_proxy=False))
    summary=[]
    for n in (1000,5000):
        group=[r for r in rows if r['n']==n];issued=[r for r in group if r['issued']]
        summary.append(dict(calibration_n=n,repetitions=reps,issued_count=len(issued),
                            issuance_rate=len(issued)/reps,mean_test_coverage_if_issued=float(np.mean([r['coverage'] for r in issued])) if issued else None,
                            mean_test_wape_if_issued=float(np.mean([r['wape'] for r in issued])) if issued else None,
                            test_proxy_violations=sum(r['population_violation_proxy'] for r in group)))
    result=dict(status='SYNTHETIC_EXPERIMENT_COMPLETE',counterexample=counter,monte_carlo=summary,
          construction={'group_B_probability':.2,'group_A_mean':1.,'group_B_mean':10.,'noise_uniform':[-.2,.2],
                        'error_A_uniform':[.8,1.],'error_B':2.,'contract_wape':.3,'minimum_coverage':.3},
          uncertainty='Finite-sample Hoeffding union bound uses known synthetic support; 20,000-row test is only a population proxy.',
          seconds=time.time()-start)
    (out/'SYNTHETIC_RESULTS.json').write_text(json.dumps(result,indent=2))
    (out/'replicates.json').write_text(json.dumps(rows,indent=2));print(json.dumps(result,indent=2))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);ap.add_argument('--reps',type=int,default=200)
    a=ap.parse_args();main(a.output,a.reps)
