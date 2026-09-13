import unittest
import numpy as np
import pandas as pd
from run_study import features,censor

class LeakageTests(unittest.TestCase):
    def test_future_observations_do_not_change_forecast_features(self):
        rng=np.random.default_rng(7);obs=rng.poisson(3,(4,140)).astype(float)
        cap=np.full_like(obs,5.);dates=pd.date_range('2009-12-01',periods=140)
        days=np.arange(70,77);before=features(obs,cap,dates,days)
        obs[:,70:]=90000;cap[:,70:]=100000
        np.testing.assert_array_equal(before,features(obs,cap,dates,days))
        np.testing.assert_array_equal(before[:7,1],np.arange(1,8))

    def test_censoring_capacity_is_causal(self):
        y=np.ones((3,160))*4;obs,cap=censor(y)
        changed=y.copy();changed[:,100:]=500
        obs2,cap2=censor(changed)
        np.testing.assert_array_equal(cap[:,:101],cap2[:,:101])
        np.testing.assert_array_equal(obs[:,:100],obs2[:,:100])
        self.assertTrue(np.all(obs[:,56:]<=cap[:,56:]))

if __name__=='__main__':unittest.main()
