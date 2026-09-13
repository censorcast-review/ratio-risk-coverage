"""Small exhaustive checks of objectives, feasibility and atom handling."""
import unittest,itertools
import numpy as np
from run_review2_objectives import curve,select_threshold

class Objectives(unittest.TestCase):
    def test_nested_chain_matches_exhaustive_threshold_search(self):
        rng=np.random.default_rng(432)
        for _ in range(100):
            s=[rng.integers(0,5,12),rng.integers(0,5,15)]
            y=[rng.integers(0,6,12),rng.integers(0,6,15)]
            p=[rng.uniform(0,5,12),rng.uniform(0,5,15)]
            cs=[curve(x,a,b,cap=.8,floor=.2) for x,a,b in zip(s,y,p)]
            candidates=[]
            for t in sorted(set(s[0])|set(s[1])):
                vals=[];valid=True
                for ss,yy,pp in zip(s,y,p):
                    mask=ss<=t;mass=yy[mask].sum()
                    if mask.mean()<.2 or mass==0 or abs(yy[mask]-pp[mask]).sum()/mass>.8:valid=False
                    vals.append((mask.mean(),mass/yy.sum()))
                if valid:candidates.append((t,min(q[0] for q in vals),min(q[1] for q in vals)))
            answers=[]
            for index,obj in [(1,'row'),(2,'demand')]:
                observed,_=select_threshold(*cs,obj)
                expected=max(candidates,key=lambda v:(v[index],v[0]))[0] if candidates else None
                self.assertEqual(observed,expected);answers.append(observed)
            self.assertEqual(*answers)

    def test_different_families_have_different_optima(self):
        # Exhaust every deterministic acceptance set: row and mass objectives
        # disagree even though the forecast and risk cap are identical.
        y=np.array([6.,.1,.1,.1,.1,8.]);p=np.array([6.,.7,.71,.72,.73,2.]);cap=.5
        sets=[]
        for bits in itertools.product([False,True],repeat=6):
            a=np.array(bits);mass=y[a].sum()
            if mass and abs(y[a]-p[a]).sum()<=cap*mass:sets.append((a.sum(),mass,bits))
        row=max(sets,key=lambda v:(v[0],v[1]));mass=max(sets,key=lambda v:(v[1],v[0]))
        self.assertEqual(row[0],5);self.assertEqual(mass[0],3)
        self.assertAlmostEqual(row[1],6.4);self.assertAlmostEqual(mass[1],14.1)
        self.assertNotEqual(row[2],mass[2])

    def test_zero_demand_is_not_a_valid_ratio_policy(self):
        c=curve(np.arange(5),np.zeros(5),np.zeros(5),cap=.8,floor=.2)
        self.assertFalse(c['feasible'].any())
        self.assertEqual(select_threshold(c,c,'demand'),(None,0.))

    def test_score_atoms_are_indivisible(self):
        c=curve(np.array([1.,1.,2.]),np.ones(3),np.array([1.,0.,1.]),cap=.4,floor=0)
        self.assertEqual(c['threshold'].tolist(),[1.,2.])
        self.assertAlmostEqual(c['row'][0],2/3)

if __name__=='__main__':unittest.main()
