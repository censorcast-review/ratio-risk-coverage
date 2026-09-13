"""Validation-only joint nonnegative LAD calibration with a convex certificate.

This follow-up was specified after the first seed's original finite strength
grid chose zero. All new candidate selection remains on point validation.
No new UCI test policy is evaluated. M5 evaluations remain retrospective.
"""
from pathlib import Path
import argparse,json,sys
from datetime import datetime,timezone
import numpy as np
from scipy.optimize import linprog
from audit_observed_data import sha,write,metrics,optimal_scale,rmsse_scale
from run_m5_observable import SEEDS,POWERS

def solve(y,b,u,tol=1e-7,max_iter=250):
    y=np.asarray(y,float).ravel();b=np.asarray(b,float).ravel();u=np.asarray(u,float).ravel()
    T=y.sum();bs=b.sum();us=u.sum();s=optimal_scale(y,b)
    start=np.array([s*bs/T,0.]);R0=np.abs(y-s*b).sum()/T
    if us==0 or bs==0:return dict(beta_base=s,beta_uplift=0.,wape=float(R0),lower_bound=float(R0),gap=0.,iterations=0)
    # Coordinates are forecast-mass fractions. For any improvement over R0,
    # a+b <= 1+R0 follows from sum|Y-f| >= sum f - sum Y.
    B=b*T/bs;U=u*T/us;limit=1+R0;point=start;best=float('inf');bestp=None;cuts=[];rhs=[];lb=0.
    for it in range(max_iter):
        residual=point[0]*B+point[1]*U-y
        value=float(np.abs(residual).sum()/T);sgn=np.sign(residual)
        grad=np.array([np.dot(sgn,B),np.dot(sgn,U)])/T
        if value<best:best=value;bestp=point.copy()
        cuts.append([float(grad[0]),float(grad[1]),-1.]);rhs.append(float(grad@point-value))
        opt=linprog([0.,0.,1.],A_ub=np.array([[1.,1.,0.]]+cuts),b_ub=np.array([limit]+rhs),
                    bounds=[(0.,limit),(0.,limit),(0.,None)],method='highs',
                    options={'dual_feasibility_tolerance':1e-9,'primal_feasibility_tolerance':1e-9})
        assert opt.success,opt.message
        lb=float(opt.fun)
        if best-lb<=tol:break
        point=opt.x[:2]
    assert best-lb<=tol+1e-8,(it,best,lb)
    return dict(beta_base=float(bestp[0]*T/bs),beta_uplift=float(bestp[1]*T/us),
        wape=best,lower_bound=lb,gap=float(best-lb),iterations=it+1,
        implied_alpha=float(bestp[1]*bs/(bestp[0]*us)) if bestp[0]>0 else None)

def calibrate(y,p):
    base=p['l1'].astype(float);h=p['hit'].astype(float);records=[];choices={}
    for kind,mean,powers in [('observed_mean_gate','poisson',POWERS),('censored_mean_gate','em',POWERS),('censored_mean_ungated','em',[0.])]:
        own=[]
        for power in powers:
            uplift=h**power*np.maximum(p[mean].astype(float)-base,0)
            fit=solve(y,base,uplift)
            row=dict(kind=kind,mean=mean,power=power,**fit);records.append(row);own.append(row)
        choices[kind]=min(own,key=lambda r:r['wape'])
    return choices,records

def pred(p,c):
    b=p['l1'].astype(float);u=p['hit'].astype(float)**c['power']*np.maximum(p[c['mean']].astype(float)-b,0)
    return c['beta_base']*b+c['beta_uplift']*u

def run(root,data,out,seed):
    out.mkdir(parents=True,exist_ok=True)
    proto=out/'PROTOCOL.json'
    if not proto.exists():write(proto,dict(created_utc=datetime.now(timezone.utc).isoformat(),
        scope='POST_HOC_VALIDATION_CALIBRATION_AND_CONSUMED_M5_EVALUATION',
        motivation='First M5 seed selects alpha=0 on the original finite grid; UCI scale optimum exceeds old ceiling',
        source_sha256=sha(__file__),parent_protocol_sha256=sha(root/'evidence/revision/m5_observable/PROTOCOL.json'),
        seeds=SEEDS,mean_controls=['observed Poisson','capacity-hit Poisson EM'],powers=POWERS,
        calibration='joint nonnegative LAD; supporting-plane LP with empirical objective gap <= 1e-7',
        coefficient_domain='all improving nonnegative coefficients contained by total forecast <= target+baseline error',
        test_status='M5 already consumed; UCI no new test predictions or policy evaluations',new_fits=0))
    assert json.loads(proto.read_text())['source_sha256']==sha(__file__)
    source=root/'evidence/revision/m5_observable'/f'seed_{seed}';dest=out/f'seed_{seed}';dest.mkdir(exist_ok=True)
    z=np.load(data/'data/design_outcomes_v0_5.npz');yall=z['truth'];cats=z['cat_id'];q=rmsse_scale(yall[:,:1313]);p=np.load(source/'validation_heads.npz');days=p['days'];y=yall[:,days-1]
    chosen,records=calibrate(y,p)
    write(dest/'POLICY_FREEZE.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),seed=seed,choices=chosen,
        all_validation=records,source_heads_sha256=sha(source/'validation_heads.npz'),
        protocol_sha256=sha(proto),eval_blocks_used_for_selection=0))
    print('JOINT CHOICES',seed,json.dumps(chosen),flush=True)
    # Other seeds retain sufficient statistics, while first seed heads are saved.
    # Recompute their block heads from frozen models if needed, via the parent
    # runner's observable feature path. Evaluation is in a separate finalize step.

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--root',type=Path,required=True);a.add_argument('--data',type=Path,required=True);a.add_argument('--output',type=Path,required=True);a.add_argument('--seed',type=int,required=True);v=a.parse_args();run(v.root,v.data,v.output,v.seed)
