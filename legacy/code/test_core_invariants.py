"""Regression tests for material scientific failure modes, with no retail holdout reads."""
import unittest
import tempfile
from pathlib import Path
import numpy as np
from run_selective_study import frontier, choose, metric, bootstrap, origin, construct

class ScientificInvariants(unittest.TestCase):
    def test_category_metrics_do_not_mutate_acceptance(self):
        y=np.array([[1.,2.],[3.,4.]]);a=np.ones_like(y,dtype=bool);saved=a.copy()
        _=metric(y,y,y+1,a)
        np.testing.assert_array_equal(a,saved)
        meta={'item_id':np.array(['a','b'])}
        _=bootstrap({'truth':y,'proposal':y,'baseline':y+1,'target_days':np.array([8,9])},a,meta,reps=10)
        np.testing.assert_array_equal(a,saved)

    def test_threshold_frontier_never_splits_ties(self):
        s=np.array([0.,0.,1.,1.]);y=np.ones(4);p=np.array([1.,1.,0.,0.])
        f=frontier(s,y,p,np.zeros(4),'absolute')
        np.testing.assert_array_equal(f['n'],[2,4])

    def test_fixed_absolute_and_paired_contracts_are_distinct(self):
        y=np.array([100.,100.]);p=np.array([90.,90.]);b=np.array([99.,99.]);s=np.array([0.,1.])
        self.assertTrue(frontier(s,y,p,b,'absolute')['point_feasible'].any())
        self.assertFalse(frontier(s,y,p,b,'paired')['point_feasible'].any())

    def test_history_at_origin_invariant_to_future_outcomes(self):
        o=int(origin(np.array([19]))[0]);self.assertEqual(o,14)
        history=np.arange(1.,31.);altered=history.copy();altered[o:]+=10000
        self.assertEqual(history[:o].sum(),altered[:o].sum())
        self.assertNotEqual(history[:18].sum(),altered[:18].sum())

    def test_censor_excess_upper_envelope(self):
        for f in [0.,2.,5.,10.]:
            for c,u in [(1.,3.),(3.,12.)]:
                r=.3;z=np.linspace(c,u,2001);g=np.abs(z-f)-r*z
                upper=max(abs(c-f)-r*c,abs(u-f)-r*u)
                self.assertLessEqual(float(g.max()),upper+1e-12)

    def test_actual_feature_builder_is_invariant_after_forecast_origin(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);obs=np.tile(np.arange(1913)%5+1.,(2,1));cen=np.zeros_like(obs)
            meta={k:np.array(['a','b']) for k in ['id','item_id','cat_id','dept_id','store_id','state_id']}
            for version in [0,1]:
                src=root/str(version);src.mkdir();out=src/'cache';out.mkdir()
                current=obs.copy()
                if version:current[:,1799:]+=1000 # First changed outcome is day 1800.
                np.savez(src/'design_outcomes_v0_5.npz',**meta,observed=current,censored=cen)
                for key,start in [('selection_base',1314),('risk_train',1434),('calibration_a',1554),('calibration_b',1674),('shadow',1794)]:
                    days=np.arange(start,start+120);shape=(2,120)
                    v={k:np.full(shape,2.) for k in ['baseline','proposal','raw_poisson_histgb','censored_poisson_em','prior_observed_wape','instantaneous_absolute_error']}
                    v.update(truth=obs[:,days-1],observed=current[:,days-1],censored=cen[:,days-1],censor_probability=np.zeros(shape),target_days=days)
                    np.savez(src/(key+'_v0_5.npz'),**v)
                construct(src,out)
            x=np.load(root/'0/cache/shadow_features.npy');changed=np.load(root/'1/cache/shadow_features.npy')
            # d_1800...1806 have origin d_1799. The next week can use the change.
            np.testing.assert_array_equal(x[:,:7],changed[:,:7])
            self.assertGreater(float(np.abs(x[:,7:]-changed[:,7:]).sum()),0)

if __name__=='__main__':unittest.main()
