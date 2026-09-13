"""Exact constructions and independent LP checks; analytic proofs are in the PDF."""
from fractions import Fraction as F
from pathlib import Path
import json
import numpy as np
from scipy.optimize import linprog

ROOT=Path(__file__).resolve().parents[1];count=0
def check(v):
    global count
    assert v;count+=1
def dot(a,b):return sum(x*y for x,y in zip(a,b))
def lp(prob,w,ell,r):
    p=np.array(prob,dtype=float);w=np.array(w,dtype=float);e=np.array(ell,dtype=float)
    a=linprog(-p,A_ub=[p*(e-float(r)*w)],b_ub=[0.],bounds=[(0,1)]*len(p),method='highs')
    b=linprog(-p*w,A_ub=[p*(e-float(r)*w)],b_ub=[0.],bounds=[(0,1)]*len(p),method='highs')
    return a,b

# Sharp construction: exact excess, binary realizability and WAPE realization.
for r in [F(1,10),F(1,2),F(4,5)]:
    k,q,p=F(2,5),F(1,2),F(1,10)
    for eps in [F(1,100),F(1,1000),F(1,10000)]:
        w=[q*(1-r)*eps/(r*k),eps,F(1)];e=[F(0),eps,r+q*(1-r)*eps/p];mass=[k,q,p]
        g=[pi*(ei-r*wi) for pi,ei,wi in zip(mass,e,w)]
        check(g==[-q*(1-r)*eps,q*(1-r)*eps,q*(1-r)*eps])
        check(all(0<=ei<=wi for ei,wi in zip(e,w)))
        f=[w[0],F(0),(1-r)*(1-q*eps/p)]
        check([abs(yi-fi) for yi,fi in zip(w,f)]==e)
        a,b=lp(mass,w,e,r);check(a.success and b.success)
        check(np.allclose(a.x,[1,1,0],atol=1e-7));check(np.allclose(b.x,[1,0,1],atol=1e-7))
        T=dot(mass,w);gap=(p-q*eps)/T
        check(gap==r*(p-q*eps)/(r*p+q*eps))

# Fixed two-label example.
p=[F(2,5),F(2,5),F(1,5)];w=[F(1),F(1),F(2)];e=[F(0),F(3,10),F(1,2)];r=F(1,10)
ar=[F(1),F(1,2),F(0)];aw=[F(1),F(0),F(2,3)];T=dot(p,w)
check(dot(p,ar)==F(3,5));check(dot(p,aw)==F(8,15))
check(dot([pi*wi for pi,wi in zip(p,w)],ar)/T==F(1,2))
check(dot([pi*wi for pi,wi in zip(p,w)],aw)/T==F(5,9))
a,b=lp(p,w,e,r);check(np.allclose(a.x,np.array(ar,float)));check(np.allclose(b.x,np.array(aw,float)))
x,y=F(1,15),F(1,18);D=F(1,3);check(y<=(D-x)/(1-D*x))

# Random finite laws verify the joint envelope against independent linear programs.
rng=np.random.default_rng(731);envelope_cases=0;two_cases=0
for n in [2,3,5]:
    for _ in range(40):
        p=rng.dirichlet(np.ones(n));w=rng.uniform(1,4,n);e=w*rng.uniform(0,1,n);e[0]=0
        a,b=lp(p,w,e,.3);check(a.success and b.success)
        x=p@(a.x-b.x);y=(p*w)@(b.x-a.x)/(p@w);D=(w.max()-w.min())/(w.max()+w.min())
        check(x>=-1e-9 and y>=-1e-9 and y<=(D-x)/(1-D*x)+1e-9);envelope_cases+=1
        if n==2:check(np.allclose(a.x,b.x,atol=1e-8));two_cases+=1

# Absolute-loss uplift identity on atoms, using exact piecewise integration.
mass=[F(1,2),F(1,3),F(1,6)];ys=[F(0),F(1),F(5)]
for b,u in [(F(0),F(2)),(F(1,2),F(3)),(F(2),F(1,3))]:
    direct=sum(pi*(abs(y-b-u)-abs(y-b)) for pi,y in zip(mass,ys))
    knots=sorted({b,b+u}|{y for y in ys if b<y<b+u});integ=F(0)
    for l,h in zip(knots[:-1],knots[1:]):
        mid=(l+h)/2;cdf=sum(pi for pi,y in zip(mass,ys) if y<=mid);integ+=(h-l)*(2*cdf-1)
    check(direct==integ)
out=ROOT/'verification';out.mkdir(exist_ok=True)
report={'status':'PASS','assertions':count,'bounded_exposure_lp_laws':envelope_cases,'two_context_lp_laws':two_cases,'scope':'finite construction checks accompany, and do not replace, the analytic proofs'}
(out/'THEORY_VERIFICATION.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
