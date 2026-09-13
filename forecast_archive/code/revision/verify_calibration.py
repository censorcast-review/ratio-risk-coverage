"""Independent epigraph LP checks for the two-coefficient calibration solver."""
import json
from pathlib import Path
import numpy as np
from scipy.optimize import linprog
from refine_calibration import solve
rng=np.random.default_rng(731);checks=[]
for i in range(36):
    n=20+i%11;y=rng.poisson(3,size=n).astype(float)
    b=rng.exponential(2,size=n);u=rng.exponential(3,size=n)
    b[rng.random(n)<.25]=0;u[rng.random(n)<.2]=0
    f=solve(y,b,u,tol=1e-7)
    # Independent, full LAD LP: beta0, beta1 and one error slack per row.
    A=np.concatenate([np.column_stack([b,u]),-np.eye(n)],axis=1)
    B=np.concatenate([np.column_stack([-b,-u]),-np.eye(n)],axis=1)
    lp=linprog(np.r_[0.,0.,np.ones(n)/y.sum()],A_ub=np.r_[A,B],b_ub=np.r_[y,-y],bounds=[(0,None)]*(n+2),method='highs')
    assert lp.success
    assert abs(f['wape']-lp.fun)<1.1e-7,(f,lp.fun)
    assert f['lower_bound']<=lp.fun+1e-8
    checks.append(abs(f['wape']-lp.fun))
report=dict(status='PASS',independent_full_epigraph_lp_cases=len(checks),max_objective_difference=max(checks))
out=Path(__file__).resolve().parents[2]/'verification/CALIBRATION_VERIFICATION.json';out.write_text(json.dumps(report,indent=2));print(report)
