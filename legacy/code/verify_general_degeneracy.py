from fractions import Fraction as F
from itertools import combinations
from pathlib import Path
import argparse
import json
from r2_io import dump
from reproduction_io import ROOT,safe_output
import numpy as np
from scipy.optimize import linprog

records=[]
for r in [F(1,10),F(1,2),F(9,10)]:
 for c0 in [F(0),F(7,20),F(9,10)]:
  k=(1+c0)/2
  p=(1-k)/4
  q=1-k-p
  for eps in [p/(q*10), p/(q*100000)]:
   prob=[k,q,p]
   y=[q*(1-r)*eps/(r*k),eps,F(1)]
   f=[y[0],F(0),(1-r)*(1-q*eps/p)]
   e=[abs(a-b) for a,b in zip(y,f)]
   m=[a-r*b for a,b in zip(e,y)]
   T=sum(a*b for a,b in zip(prob,y))
   row=[F(1),F(1),F(0)]
   dem=[F(1),F(0),F(1)]
   def calc(a):
    c=sum(b*d for b,d in zip(prob,a)); M=sum(b*d*t for b,d,t in zip(prob,a,y)); E=sum(b*d*t for b,d,t in zip(prob,a,e))
    return c,M/T,E/M
   cr,dr,Rr=calc(row);cd,dd,Rd=calc(dem)
   assert Rr==Rd==r and cr>c0 and cd>c0
   assert cr-cd==q-p
   assert dr==q*eps/(r*p+q*eps)
   assert dd-dr==r*(p-q*eps)/(r*p+q*eps)
   delta=q*(1-r)*eps/p
   assert dd-dr==q*eps/T*((1-r)/delta-1)
   # Enumerate every vertex in the 3-variable feasibility polytope using exact arithmetic.
   # Every optimum includes context K, so its acceptance is fixed to 1.
   A=[(F(1),F(0),F(0)),(F(1),F(0),F(1)),(F(0),F(1),F(0)),(F(0),F(1),F(1)),(q*m[1],p*m[2],-k*m[0]),(q,p,c0-k)]
   vertices=set()
   for aa,bb in combinations(A,2):
    u,v,w=aa;x,z,t=bb;den=u*z-v*x
    if not den:continue
    av=(w*z-v*t)/den; bv=(u*t-w*x)/den
    if 0<=av<=1 and 0<=bv<=1 and k+q*av+p*bv>=c0 and k*m[0]+q*av*m[1]+p*bv*m[2]<=0:
     vertices.add((av,bv))
   vals=[((F(1),a,b),calc([F(1),a,b])) for a,b in vertices]
   rowmax=max(v[1][0] for v in vals);demmax=max(v[1][1] for v in vals)
   assert [a for a,v in vals if v[0]==rowmax]==[tuple(row)]
   assert [a for a,v in vals if v[1]==demmax]==[tuple(dem)]
   B=q*(1-r)*eps
   aineq=np.array([[float(a*b/B) for a,b in zip(prob,m)],[-float(a) for a in prob]])
   bineq=np.array([0.,-float(c0)])
   for obj,exp in [(prob,row),([a*b/T for a,b in zip(prob,y)],dem)]:
    lp=linprog(-np.array(list(map(float,obj))),A_ub=aineq,b_ub=bineq,bounds=[(0,1)]*3,method='highs')
    assert lp.success
    assert np.allclose(lp.x,list(map(float,exp)),atol=1e-6)
   records.append(dict(r=str(r),c0=str(c0),epsilon=str(eps),masses=list(map(str,prob)),row_coverage=str(cr),demand_policy_row_coverage=str(cd),row_policy_demand_coverage=str(dr),demand_policy_demand_coverage=str(dd),demand_gap=str(dd-dr),exact_vertices=len(vertices)))
# A sequence demonstrates simultaneous approach to both sharp bounds.
for c0 in [F(0),F(7,20),F(9,10)]:
 r=F(1,2)
 for n in [100,1000,10000]:
  eta=F(1,n);k=c0+eta*(1-c0);p=eta*(1-c0)/4;q=1-k-p;eps=p*eta/q
  gapc=q-p;gapd=r*(p-q*eps)/(r*p+q*eps)
  assert gapc>1-c0-2*eta and gapd>1-4*eta
result={'status':'PASS','scope':'new theorem only; no retail fits or data opened','exact_and_LP_cases':len(records),'sharp_bound_sequence_cases':9,'records':records}
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output',type=Path,default=ROOT/'reproduction_outputs/general_degeneracy/GENERAL_THEOREM_CHECK.json')
args=parser.parse_args()
dump(safe_output(args.output),result)
print(json.dumps({k:v for k,v in result.items() if k!='records'}))
