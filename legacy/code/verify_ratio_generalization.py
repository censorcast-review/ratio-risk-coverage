"""Exact and LP verification of the generalized ratio-risk reversal.

Only synthetic laws are constructed. No retail data or fitted object is read.
"""
from fractions import Fraction as F
from itertools import combinations, product
from pathlib import Path
import argparse
import json
from reproduction_io import ROOT,safe_output
import numpy as np
from scipy.optimize import linprog
from r2_io import dump

ROOT=Path(__file__).resolve().parents[1]

def dot(a,b): return sum(x*y for x,y in zip(a,b))
def solve3(mat,rhs):
    rows=[list(a)+[b] for a,b in zip(mat,rhs)]
    for i in range(3):
        pivot=next((j for j in range(i,3) if rows[j][i]),None)
        if pivot is None:return None
        rows[i],rows[pivot]=rows[pivot],rows[i]
        scale=rows[i][i];rows[i]=[v/scale for v in rows[i]]
        for j in range(3):
            if i!=j:
                scale=rows[j][i];rows[j]=[x-scale*y for x,y in zip(rows[j],rows[i])]
    return tuple(row[-1] for row in rows)

def enumerate_vertices(mass,excess,c0):
    constraints=[]
    for i in range(3):
        e=tuple(F(int(i==j)) for j in range(3))
        constraints.extend([(e,F(1)),(tuple(-v for v in e),F(0))])
    constraints.extend([(tuple(p*m for p,m in zip(mass,excess)),F(0)),(tuple(-p for p in mass),-c0)])
    vertices=set()
    for chosen in combinations(constraints,3):
        point=solve3([a for a,b in chosen],[b for a,b in chosen])
        if point is not None and all(dot(a,point)<=b for a,b in constraints):vertices.add(point)
    return vertices

records=[]
for (r,rho),c0,n in product([(F(1,10),F(1)),(F(1,2),F(1)),(F(9,10),F(1)),(F(3,2),F(2))],[F(0),F(7,20),F(9,10)],[10,100000]):
    k=(1+c0)/2;p=(1-k)/4;q=1-k-p;eps=p/(q*n)
    mass=[k,q,p]
    w=[q*(rho-r)*eps/(r*k),eps,F(1)]
    loss=[F(0),rho*eps,r+q*(rho-r)*eps/p]
    m=[ell-r*wi for ell,wi in zip(loss,w)]
    T=dot(mass,w);B=q*(rho-r)*eps
    assert T==p+q*rho*eps/r
    assert all(wi>0 and 0<=ell<=rho*wi for ell,wi in zip(loss,w))
    aR=(F(1),F(1),F(0));aW=(F(1),F(0),F(1))
    dmass=[p*wi/T for p,wi in zip(mass,w)]
    for a in [aR,aW]:
        assert dot(a,[p*ell for p,ell in zip(mass,loss)])/dot(a,[p*wi for p,wi in zip(mass,w)])==r
        assert dot(a,mass)>c0
    gc=dot(mass,aR)-dot(mass,aW);gd=dot(dmass,aW)-dot(dmass,aR)
    assert gc==q-p
    assert dot(dmass,aR)==q*rho*eps/(r*p+q*rho*eps)
    assert gd==r*(p-q*eps)/(r*p+q*rho*eps)
    delta=loss[2]/w[2]-r
    assert gd==(q*eps/T)*((rho-r)/delta-1)
    assert gd==(q*eps/T)*(rho-loss[2]/w[2])/(loss[2]/w[2]-r)
    vertices=enumerate_vertices(mass,m,c0)
    for obj,expected in [(mass,aR),(dmass,aW)]:
        best=max(dot(a,obj) for a in vertices)
        assert [a for a in vertices if dot(a,obj)==best]==[expected]
        Aub=np.array([[float(p*mi/B) for p,mi in zip(mass,m)],[-float(p) for p in mass]])
        lp=linprog(-np.array(list(map(float,obj))),A_ub=Aub,b_ub=[0,-float(c0)],bounds=[(0,1)]*3,method='highs')
        assert lp.success and np.allclose(lp.x,list(map(float,expected)),atol=1e-6)
    if rho==1:
        scale=1/max(F(1),*w)
        for ell,wi in zip(loss,w):
            ell*=scale;wi*=scale
            joint=[ell,wi-ell,1-wi]
            assert min(joint)>=0 and sum(joint)==1
            assert dot(joint,[F(1),F(0),F(0)])==ell
            assert dot(joint,[F(1),F(1),F(0)])==wi
        y=w;f=[w[0],F(0),(1-r)*(1-q*eps/p)]
        assert all(fi>=0 for fi in f)
        assert [abs(yi-fi) for yi,fi in zip(y,f)]==loss
    records.append({'r':str(r),'rho':str(rho),'c0':str(c0),'epsilon':str(eps),'vertices':len(vertices),'row_gap':str(gc),'exposure_gap':str(gd),'nested_binary_realization':rho==1})

sharp=[]
for (r,rho),c0,n in product([(F(1,10),F(1)),(F(1,2),F(1)),(F(9,10),F(1)),(F(3,2),F(2))],[F(0),F(7,20),F(9,10)],[100,1000,10000]):
    eta=F(1,n);k=c0+(1-c0)*eta/8;p=(1-c0)*eta/8;q=1-k-p
    eps=p*r*eta/(q*4*(r+rho))
    gc=q-p;gd=r*(p-q*eps)/(r*p+q*rho*eps)
    assert gc>1-c0-eta and gd>1-eta and gc<=1-c0 and gd<=1
    sharp.append({'r':str(r),'rho':str(rho),'c0':str(c0),'eta':str(eta),'row_gap':str(gc),'exposure_gap':str(gd)})

# Necessary strict orderings for this exhausted-budget, three-context family.
# All three regimes are covered: reversal, no row advantage, no exposure advantage.
ordering_cases=0
for q,p,eps in product([F(1,5),F(2,5)],[F(1,5),F(3,10)],[F(1,10),F(1),F(2)]):
    k=1-q-p
    if k<=0:continue
    # Reduced budget a_U+a_V <= 1; optimizing any linear objective needs only these three vertices.
    verts=[(F(0),F(0)),(F(1),F(0)),(F(0),F(1))]
    rows=[q*a+p*b for a,b in verts];exposure=[q*eps*a+p*b for a,b in verts]
    row_unique=rows[1]>max(rows[0],rows[2]);exp_unique=exposure[2]>max(exposure[0],exposure[1])
    assert row_unique==(q>p) and exp_unique==(q*eps<p)
    ordering_cases+=1

# Counterexample to universal applicability: constant W makes both utility functionals identical.
constant_cases=0
mass=[F(1,2),F(1,3),F(1,6)]
for exposure in [F(1,10),F(1),F(7)]:
    T=sum(p*exposure for p in mass)
    for a in product([F(0),F(1,2),F(1)],repeat=3):
        assert sum(p*x*exposure for p,x in zip(mass,a))/T==dot(mass,a)
        constant_cases+=1

perturb_count=mixed_count=0
for w,rate,dw_ratio,de_sign in product([F(1,10**6),F(1,1000),F(1,2),F(1),F(10)],[F(0),F(1,10),F(1),F(2)],[F(-1,2),F(0),F(1,2)],[-1,0,1]):
    ell=rate*w;dw=dw_ratio*w;de=de_sign*(ell+w)/10
    if ell+de<0:continue
    beta=F(1,2);hatw=w+dw;r=F(3,5);m=ell-r*w
    exact=abs((ell+de)/hatw-ell/w)
    identity=abs(w*de-ell*dw)/(w*abs(hatw))
    bound=(abs(de)+(ell/w)*abs(dw))/((1-beta)*w)
    assert exact==identity and exact<=bound
    assert abs((ell+de-r*hatw)-m)<=abs(de)+r*abs(dw)
    perturb_count+=1
    for lam,T in product([F(1,100),F(1,4),F(1,2),F(1)],[F(1,2),F(3)]):
        b=(1-lam)/T;den=lam+b*w;hatden=lam+b*hatw
        got=abs((ell+de-r*hatw)/hatden-m/den)
        bnd=(abs(de)+r*abs(dw))/lam+abs(m)*b*abs(dw)/(lam*lam)
        assert got<=bnd and den>=lam and hatden>=lam
        # Algebraic derivative quotient-rule numerator, verified exactly.
        assert -r*den-b*m==-(r*lam+b*ell)
        mixed_count+=1

result={'status':'PASS','scope':'Synthetic nonnegative ratio-risk laws only; no retail labels, fits, caches, or external partition read.','family_exact_vertex_and_LP_cases':len(records),'sharp_sequence_cases':len(sharp),'nested_binary_realization_cases':sum(x['nested_binary_realization'] for x in records),'strict_ordering_cases':ordering_cases,'constant_denominator_counterexamples':constant_cases,'finite_ratio_perturbation_cases':perturb_count,'mixed_score_perturbation_cases':mixed_count,'universal_claim_caveat':'Sharpness is existential across realizable joint laws. Ratio form alone does not imply reversal for every fixed loss/exposure metric class.','records':records,'sharp_sequences':sharp}
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output',type=Path,default=ROOT/'reproduction_outputs/ratio_generalization/THEORY_CHECKS.json')
args=parser.parse_args()
dump(safe_output(args.output),result)
print(json.dumps({k:v for k,v in result.items() if k not in ['records','sharp_sequences']},indent=2))
