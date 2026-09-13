import unittest
import numpy as np
from scipy.stats import poisson
from run_capacity_hit_extension import replace_history,likelihood_inputs
from r3_observable import nll

class CapacityHitTests(unittest.TestCase):
    def test_history_columns_use_only_pre_origin_capacity_hits(self):
        x=np.zeros((2,34),np.float32);h=np.zeros((2,60),np.int32);h[0,21:28]=1
        cs=np.pad(np.cumsum(h,axis=1),((0,0),(1,0)))
        rows=np.array([0,1]);days=np.array([29,30]);a=replace_history(x,rows,days,cs)
        np.testing.assert_allclose(a[:,28],[1,0]);np.testing.assert_allclose(a[:,29],[.25,0])
        h[:,28:]=1;future=np.pad(np.cumsum(h,axis=1),((0,0),(1,0)))
        np.testing.assert_array_equal(a,replace_history(x,rows,days,future))
    def test_capacity_equality_uses_inclusive_tail(self):
        bound,hit=likelihood_inputs(np.array([1,2,3]),np.array([2,2,3]))
        np.testing.assert_array_equal(hit,[False,True,True])
        expected=-np.log([poisson.pmf(1,2),poisson.sf(1,2),poisson.sf(2,2)])
        np.testing.assert_allclose(nll(np.full(3,2.),bound,hit),expected)
if __name__=='__main__':unittest.main()
