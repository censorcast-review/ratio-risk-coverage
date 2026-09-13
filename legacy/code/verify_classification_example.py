"""Verify fixed-classifier precision examples; synthetic laws only.

Default output is outside archived results so verification never invalidates
package hashes. An explicit --output is required to create an archival report.
"""
import argparse
from fractions import Fraction as F
from itertools import combinations, product
from pathlib import Path
import json
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
                scale=rows[j][i]
                rows[j]=[x-scale*y for x,y in zip(rows[j],rows[i])]
    return tuple(row[-1] for row in rows)
def vertices(mass,excess,c0):
    constraints=[]
    for i in range(3):
        e=tuple(F(int(i==j)) for j in range(3))
        constraints.extend([(e,F(1)),(tuple(-v for v in e),F(0))])
    constraints.extend([(tuple(p*m for p,m in zip(mass,excess)),F(0)),(tuple(-p for p in mass),-c0)])
    found=set()
    for chosen in combinations(constraints,3):
        point=solve3([a for a,b in chosen],[b for a,b in chosen])
        if point is not None and all(dot(a,point)<=b for a,b in constraints):found.add(point)
    return found

def run():
    mass=[F(2,5),F(2,5),F(1,5)]
    # Fixed two-label classifier; the gate observes the entire context X.
    prediction=[(1,0),(1,0),(1,1)]
    truth=[[(F(1),(1,0))],[(F(3,10),(0,0)),(F(7,10),(1,0))],[(F(1,2),(1,0)),(F(1,2),(1,1))]]
    ell=[];w=[]
    atom_checks=0
    for h,law in zip(prediction,truth):
        assert sum(p for p,y in law)==1
        wi=F(sum(h));ei=F(0)
        for p,y in law:
            li=F(sum(hi*(1-yi) for hi,yi in zip(h,y)))
            assert 0<=li<=wi<=2
            assert 0<=li/2<=wi/2<=1
            ei+=p*li;atom_checks+=1
        ell.append(ei);w.append(wi)
    assert ell==[F(0),F(3,10),F(1,2)] and w==[F(1),F(1),F(2)]
    r=F(1,10);c0=F(7,20);T=dot(mass,w)
    m=[ei-r*wi for ei,wi in zip(ell,w)]
    assert T==F(6,5) and m==[F(-1,10),F(1,5),F(3,10)]
    vs=vertices(mass,m,c0)
    row=(F(1),F(1,2),F(0));expo=(F(1),F(0),F(2,3))
    exposure=[p*wi/T for p,wi in zip(mass,w)]
    records={}
    for name,objective,expected in [('rows',mass,row),('positive_exposure',exposure,expo)]:
        best=max(dot(objective,a) for a in vs)
        maximizers=[a for a in vs if dot(objective,a)==best]
        assert maximizers==[expected],(name,maximizers)
        solution=linprog(-np.array(list(map(float,objective))),
            A_ub=np.array([[float(p*mi) for p,mi in zip(mass,m)],[-float(p) for p in mass]]),
            b_ub=[0,-float(c0)],bounds=[(0,1)]*3,method='highs')
        assert solution.success and np.allclose(solution.x,list(map(float,expected)),atol=1e-10)
        E=dot(expected,[p*ei for p,ei in zip(mass,ell)])
        W=dot(expected,[p*wi for p,wi in zip(mass,w)])
        assert E/W==r
        # Acceptance is Bernoulli(a(X)) with an independent uniform coin;
        # summing the joint-label law explicitly recovers the same masses.
        direct_L=direct_W=F(0)
        for px,ax,h,law in zip(mass,expected,prediction,truth):
            for py,y in law:
                direct_L+=px*ax*py*sum(hi*(1-yi) for hi,yi in zip(h,y))
                direct_W+=px*ax*py*sum(h)
        assert direct_L==E and direct_W==W
        records[name]={'acceptance':[str(a) for a in expected],'case_coverage':str(dot(mass,expected)),
          'positive_exposure_coverage':str(dot(exposure,expected)),'false_positive_mass':str(E),
          'positive_prediction_mass':str(W),'risk':str(E/W),'micro_precision':str(1-E/W)}
    assert records['rows']['case_coverage']=='3/5'
    assert records['rows']['positive_exposure_coverage']=='1/2'
    assert records['positive_exposure']['case_coverage']=='8/15'
    assert records['positive_exposure']['positive_exposure_coverage']=='5/9'
    assert dot(mass,row)-dot(mass,expo)==F(1,15)
    assert dot(exposure,expo)-dot(exposure,row)==F(1,18)

    # Every budget-saturating convex interpolation is a genuinely randomized
    # admissible whole-example gate with risk 1/10; endpoints are unique optima.
    frontier=[]
    for j in range(21):
        t=F(j,20);a=tuple(t*x+(1-t)*y for x,y in zip(row,expo))
        assert dot(a,[p*ei for p,ei in zip(mass,ell)])==r*dot(a,[p*wi for p,wi in zip(mass,w)])
        assert dot(mass,a)==F(8,15)+t/F(15)
        assert dot(exposure,a)==F(5,9)-t/F(18)
        frontier.append({'t':str(t),'c':str(dot(mass,a)),'d':str(dot(exposure,a))})

    # Ordinary binary precision with an unrestricted gate observing h(X):
    # accepting every predicted-negative case costs neither risk nor exposure,
    # hence max c = P(h=0) + P(h=1) max d. Check exact polytope optima.
    binary_cases=0;no_positive_feasible=0;completion_cases=0
    bm=[F(1,2),F(1,3),F(1,6)]
    for h,eta,cap,floor in product(
       list(product([0,1],repeat=3))[1:],
       [(F(0),F(1,3),F(2,3)),(F(1,4),F(1,2),F(3,4)),(F(0),F(0),F(1)),(F(1),F(0),F(1))],
       [F(1,4),F(1,2),F(3,4)], [F(0),F(7,20),F(3,4)]):
        bw=list(map(F,h));bt=dot(bm,bw);be=[hi*et for hi,et in zip(bw,eta)]
        bv=vertices(bm,[ei-cap*wi for ei,wi in zip(be,bw)],floor)
        obj=[px*wi/bt for px,wi in zip(bm,bw)]
        if not bv or max(dot(obj,a) for a in bv)==0:
            no_positive_feasible+=1;continue
        best_d=max(dot(obj,a) for a in bv);best_c=max(dot(bm,a) for a in bv)
        assert best_c==1-bt+bt*best_d
        assert all(dot(obj,a)==best_d for a in bv if dot(bm,a)==best_c)
        for a in bv:
            ap=tuple(F(1) if not hi else ai for ai,hi in zip(a,h))
            assert dot(obj,ap)==dot(obj,a)
            assert dot(bm,ap)==1-bt+bt*dot(obj,a)
            assert dot(ap,[px*ei for px,ei in zip(bm,be)])==dot(a,[px*ei for px,ei in zip(bm,be)])
            completion_cases+=1
        binary_cases+=1
    return {'status':'PASS','scope':'Synthetic fixed deterministic classifiers and independent randomized gates only; no fitted model or evaluation observations read.',
      'multilabel':{'acceptance_unit':'whole example; two labels share the same acceptance draw',
        'classifier_information':'gate sees all X used by the fixed classifier',
        'mass':[str(x) for x in mass],'mean_false_positives':[str(x) for x in ell],
        'positive_predictions':[str(x) for x in w],'risk_cap':str(r),'row_floor':str(c0),
        'total_positive_exposure':str(T),'exact_feasible_vertices':len(vs),'independent_LP_checks':2,
        'joint_label_atoms':atom_checks,'policies':records,'row_gap':'1/15','exposure_gap':'1/18',
        'frontier':frontier,'bounded_normalization':'Divide L and W by 2; risks, coverages, and optimizers are unchanged.'},
      'binary_no_reversal':{'exact_feasible_cases':binary_cases,'no_positive_feasible_cases':no_positive_feasible,
        'exact_policy_completions':completion_cases,'identity':'max c = 1 - T + T max d',
        'scope':'Unrestricted binary-classification gate observing fixed classifier output; nonempty accepted-positive denominator. An exposure optimizer can reject negative predictions gratuitously, but its negative-completed version also maximizes rows.'}}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,
      default=ROOT/'reproduction_outputs/classification_validation/CLASSIFICATION_CHECKS.json')
    args=p.parse_args();args.output=__import__('reproduction_io').safe_output(args.output);result=run();dump(args.output,result)
    print(json.dumps({'status':result['status'],'multilabel_vertices':result['multilabel']['exact_feasible_vertices'],
       'independent_LP_checks':2,'binary_no_reversal':result['binary_no_reversal'],'output':str(args.output)},indent=2))
