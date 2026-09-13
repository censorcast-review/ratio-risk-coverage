"""Verify bounded-exposure reversal geometry using synthetic rational laws.

No training data, frozen evaluation cache, or original report is read. Default
outputs remain separate from immutable publication evidence.
"""
import argparse
from fractions import Fraction as F
from itertools import product
from pathlib import Path
import json
import numpy as np
from scipy.optimize import linprog
from r2_io import dump
from reproduction_io import safe_output
from verify_classification_example import dot, vertices

ROOT=Path(__file__).resolve().parents[1]

def envelope(x,u,v,c0=F(0)):
    """Sharp closure bound for ordered coverage gaps, including a row floor."""
    amax=min((1-x)/2,1-c0-x)
    return ((v-u)*amax-u*x)/(u+(v-u)*amax)

def check_pair(mass,w,row,exposure,c0=F(0)):
    T=dot(mass,w);u=min(w);v=max(w)
    cr=dot(mass,row);cw=dot(mass,exposure)
    x=cr-cw;y=dot([p*wi/T for p,wi in zip(mass,w)],exposure)-dot([p*wi/T for p,wi in zip(mass,w)],row)
    if x<0 or y<0 or cw<c0:return False
    D=(v-u)/(v+u)
    assert x<=D and y<=D
    assert x+y<=D*(1+x*y)
    assert x<=min(D,(1-u/v)*(1-c0))
    assert y<=envelope(x,u,v,c0)
    a=dot(mass,[max(aw-ar,F(0)) for ar,aw in zip(row,exposure)])
    b=dot(mass,[max(ar-aw,F(0)) for ar,aw in zip(row,exposure)])
    assert x==b-a and a<=min((1-x)/2,1-c0-x)
    assert y<=((v-u)*a-u*x)/(u+(v-u)*a)
    return True

def lp(mass,w,ell,cap,c0,objective):
    excess=[ei-cap*wi for ei,wi in zip(ell,w)]
    out=linprog(-np.array(list(map(float,objective))),
        A_ub=np.array([[float(p*mi) for p,mi in zip(mass,excess)],[-float(p) for p in mass]]),
        b_ub=[0,-float(c0)],bounds=[(0,1)]*len(mass),method='highs')
    assert out.success
    return out.x

def sharp_law(m,cap,c0,x,a):
    """A fixed m-label classifier with constant output at each context."""
    u=F(1);v=F(m);k=1-2*a-x;p=a+x;q=a
    assert k>0 and p>q>0 and m*q>p
    assert k+q>=c0
    B=min(cap*k,(1-cap)*p,(1-cap)*v*q)/2
    mass=[k,p,q];w=[u,u,v]
    ell=[cap-B/k,cap+B/p,cap*v+B/q]
    assert all(0<=ei<=wi for ei,wi in zip(ell,w))
    mr=[ei-cap*wi for ei,wi in zip(ell,w)]
    assert [px*mi for px,mi in zip(mass,mr)]==[-B,B,B]
    expected={'rows':(F(1),F(1),F(0)),'exposure':(F(1),F(0),F(1))}
    T=dot(mass,w);expo_obj=[px*wi/T for px,wi in zip(mass,w)]
    vs=vertices(mass,mr,c0)
    for name,obj in [('rows',mass),('exposure',expo_obj)]:
        best=max(dot(obj,aa) for aa in vs)
        best_vertices=[aa for aa in vs if dot(obj,aa)==best]
        assert best_vertices==[expected[name]]
        sol=lp(mass,w,ell,cap,c0,obj)
        assert np.allclose(sol,list(map(float,expected[name])),atol=1e-9)
        aa=expected[name]
        assert dot(aa,[px*ei for px,ei in zip(mass,ell)])/dot(aa,[px*wi for px,wi in zip(mass,w)])==cap
    assert check_pair(mass,w,expected['rows'],expected['exposure'],c0)
    y=(m*q-p)/T
    assert y==((v-u)*a-u*x)/(u+(v-u)*a)
    # At K and U, predict one positive label; at V, predict all m labels.
    # Set all predicted-positive labels false jointly with probability ell/w.
    # This explicitly realizes the conditional false-positive count, rather
    # than merely postulating moments unattached to a fixed classifier.
    joint_atoms=[]
    for name,wi,ei in zip(['K','U','V'],w,ell):
        h=(1,)*int(wi)+(0,)*(m-int(wi))
        law=[(ei/wi,(0,)*m),(1-ei/wi,h)]
        assert sum(py for py,_ in law)==1
        realized=sum(py*sum(hi*(1-yi) for hi,yi in zip(h,yy)) for py,yy in law)
        assert realized==ei and sum(h)==wi
        joint_atoms.append({'context':name,'prediction':list(h),'all_predicted_labels_false_probability':str(ei/wi)})
    return {'m':m,'risk_cap':str(cap),'row_floor':str(c0),'probabilities':list(map(str,mass)),
        'conditional_false_positives':list(map(str,ell)),'positive_prediction_counts':list(map(str,w)),
        'slack_budget':str(B),'row_gap':str(x),'exposure_gap':str(y),
        'floor_aware_envelope':str(envelope(x,u,v,c0)),'exact_feasible_vertices':len(vs),
        'independent_LP_checks':2,'fixed_classifier_realization':joint_atoms}

def run():
    # Exact geometric checks include genuinely randomized gates and a
    # nonconstant exposure value inside the enclosing endpoint interval.
    pairs=0
    masses=[[F(1,3)]*3,[F(1,2),F(1,3),F(1,6)]]
    weights=[[F(1)]*3,[F(1),F(1),F(2)],[F(1),F(2),F(3)],
             [F(2),F(3),F(5)],[F(1),F(3,2),F(2)]]
    gates=list(product([F(0),F(1,2),F(1)],repeat=3))
    for mass,w,ar,aw,c0 in product(masses,weights,gates,gates,[F(0),F(7,20),F(3,4)]):
        pairs+=int(check_pair(mass,w,ar,aw,c0))

    # Both equalities of the unrestricted envelope hold exactly for disjoint
    # endpoint-weight gates. Their risk-optimal approximations appear below.
    exact_envelope=[]
    for m,part in product([2,3,5],[F(0),F(1,4),F(1,2),F(3,4),F(1)]):
        D=F(m-1,m+1);x=D*part
        p=(1+x)/2;q=(1-x)/2
        mass=[p,q,F(0)];w=[F(1),F(m),F(1)]
        ar=(F(1),F(0),F(0));aw=(F(0),F(1),F(0))
        y=(m*q-p)/(p+m*q)
        assert y==(D-x)/(1-D*x)
        assert check_pair(mass,w,ar,aw)
        exact_envelope.append({'m':m,'row_gap':str(x),'exposure_gap':str(y)})

    # Three-context fixed-classifier laws approach every interior frontier
    # point from strictly ordered unique optima. A binding row floor can
    # leave positive common mass and permit exact attainment.
    records=[];limits=[];attainments=[]
    for m,c0,cap in product([2,3,5],[F(0),F(7,20),F(3,4)],[F(1,10),F(2,3)]):
        D=F(m-1,m+1);xmax=min(D,F(m-1,m)*(1-c0));x=xmax/2
        am=min((1-x)/2,1-c0-x);amin=x/(m-1)
        upper=envelope(x,F(1),F(m),c0)
        previous=None
        for j in [1,2,3]:
            a=am-(am-amin)/10**j
            rec=sharp_law(m,cap,c0,x,a);records.append(rec)
            y=F(rec['exposure_gap'])
            assert y<upper and (previous is None or y>previous)
            previous=y
        limits.append({'m':m,'risk_cap':str(cap),'row_floor':str(c0),
          'fixed_row_gap':str(x),'limiting_exposure_gap':str(upper),
          'third_approximation_gap_to_bound':str(upper-previous)})
        if 1-2*am-x>0:
            rec=sharp_law(m,cap,c0,x,am)
            assert rec['exposure_gap']==rec['floor_aware_envelope']
            attainments.append(rec)

    # Nonconstant w is not sufficient: zero loss makes accept-all uniquely
    # optimal for each objective, at every cap and any feasible row floor.
    mass=[F(1,3)]*3;w=[F(1),F(1),F(2)];ell=[F(0)]*3
    vv=vertices(mass,ell,F(7,20));T=dot(mass,w)
    for obj in [mass,[p*wi/T for p,wi in zip(mass,w)]]:
        assert [aa for aa in vv if dot(obj,aa)==max(dot(obj,x) for x in vv)]==[(F(1),)*3]
        assert np.allclose(lp(mass,w,ell,F(1,10),F(7,20),obj),1)

    # Even a universal characterization for a fixed feature distribution
    # needs a sufficiently rich support: with two positive-weight atoms,
    # both utilities accept every nonpositive-excess atom and then the only
    # remaining positive-excess atom to the same extent. Check exact optima.
    two_atom_cases=0;no_positive_feasible=0
    for p,eta1,eta2,cap,c0 in product([F(1,4),F(1,2),F(3,4)],
        [F(0),F(1,4),F(1,2),F(1)], [F(0),F(1,4),F(1,2),F(1)],
        [F(1,10),F(1,3),F(3,4)],[F(0),F(7,20),F(3,4)]):
        mass=[p,1-p,F(0)];w=[F(1),F(2),F(1)]
        ell=[eta1,2*eta2,F(0)];T=dot(mass,w)
        mm=[ei-cap*wi for ei,wi in zip(ell,w)]
        vv=vertices(mass,mm,c0)
        ex=[px*wi/T for px,wi in zip(mass,w)]
        if not vv or max(dot(ex,aa) for aa in vv)==0:
            no_positive_feasible+=1;continue
        best_c=max(dot(mass,aa) for aa in vv);best_d=max(dot(ex,aa) for aa in vv)
        assert all(dot(ex,aa)==best_d for aa in vv if dot(mass,aa)==best_c)
        assert all(dot(mass,aa)==best_c for aa in vv if dot(ex,aa)==best_d)
        two_atom_cases+=1

    return {'status':'PASS','scope':'Synthetic rational laws only; no learned policy or evaluation data read.',
       'statement':{'positive_exposure':'0 < u <= w(X) <= v < infinity',
         'gaps':'x=c(a_R)-c(a_W)>=0; y=d_W(a_W)-d_W(a_R)>=0',
         'D':'(v-u)/(v+u)',
         'envelope':'y <= (D-x)/(1-D*x)',
         'individual_bounds':'x <= D; y <= D',
         'floor_refinement':'a_star=min((1-x)/2,1-c0-x); y<=((v-u)*a_star-u*x)/(u+(v-u)*a_star)',
         'floor_row_bound':'x <= min(D,(1-u/v)*(1-c0))',
         'sharpness':'The floor-aware frontier is the closure of gaps achieved by unique row and exposure optima of fixed m-label classifiers with whole-example abstention, any fixed 0<r<1, and 0<=c0<1; m>=2 integer. Interior points with positive common mass attain it exactly.'},
       'exact_ordered_pair_checks':pairs,'exact_unrestricted_envelope_checks':exact_envelope,
       'unique_optimum_approximations':records,'limiting_frontier_points':limits,'exact_floor_frontier_attainments':attainments,
       'independent_LP_checks':2*(len(records)+len(attainments))+2,
       'nonconstant_zero_loss_counterexample':{'probabilities':['1/3']*3,'w':['1','1','2'],'ell':['0']*3,'unique_common_optimum':['1']*3},
       'two_atom_no_reversal':{'exact_feasible_cases':two_atom_cases,'no_positive_feasible_cases':no_positive_feasible,
          'qualification':'A fixed two-atom positive-weight feature support is an additional counterexample to a proposed universal iff based only on nonconstant w.'}}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,
       default=ROOT/'reproduction_outputs/bounded_exposure_validation/BOUNDED_EXPOSURE_CHECKS.json')
    args=p.parse_args();args.output=safe_output(args.output)
    result=run();dump(args.output,result)
    print(json.dumps({'status':result['status'],'exact_ordered_pair_checks':result['exact_ordered_pair_checks'],
       'unique_optimum_approximations':len(result['unique_optimum_approximations']),
       'independent_LP_checks':result['independent_LP_checks'],'two_atom_no_reversal':result['two_atom_no_reversal'],
       'output':str(args.output)},indent=2))
