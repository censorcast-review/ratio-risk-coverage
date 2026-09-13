import unittest
from types import SimpleNamespace
import numpy as np
import torch
from run_foundation_comparison import make_inputs,origin,median_array,metric

class FoundationTests(unittest.TestCase):
    def panel(self):
        x=np.arange(120,dtype=np.float32).reshape(3,40)
        return SimpleNamespace(observed=x.copy(),capacity=x+10,censored=np.zeros_like(x),
          metadata={'state_id':np.array(['CA','CA','TX'])},
          calendar_arrays={k:np.arange(40,dtype=np.float32) for k in ['wday','month','event_any','event_type','snap_CA','snap_TX']},
          price_matrix=np.ones_like(x),day_to_week_index=np.arange(40))
    def test_future_observations_never_enter_inputs(self):
        p=self.panel();a=make_inputs(p,[0,1],28,7,20,True)
        p.observed[:,28:]+=100000;p.capacity[:,28:]+=100000;p.censored[:,:]=1
        b=make_inputs(p,[0,1],28,7,20,True)
        for aa,bb in zip(a,b):
            np.testing.assert_array_equal(aa['target'],bb['target'])
            for group in ['past_covariates','future_covariates']:
                for k in aa[group]:np.testing.assert_array_equal(aa[group][k],bb[group][k])
        np.testing.assert_array_equal(a[0]['target'],np.arange(8,28))
    def test_horizons_and_origin_week(self):
        d=np.arange(1554,1914);h=d-origin(d)
        self.assertTrue(np.all((h>=1)&(h<=7)));self.assertTrue(np.all(origin(d)%7==0))
    def test_two_model_output_conventions(self):
        q=torch.arange(2*7*3,dtype=torch.float32).reshape(2,7,3)
        np.testing.assert_array_equal(median_array(q,2,7),median_array([q[0:1],q[1:2]],2,7))
    def test_zero_mass_and_nonfinite_forecasts(self):
        self.assertIsNone(metric(np.zeros(10),np.ones(10))['wape'])
        with self.assertRaises(RuntimeError):median_array(torch.full((2,7,3),float('nan')),2,7)
if __name__=='__main__':unittest.main()
