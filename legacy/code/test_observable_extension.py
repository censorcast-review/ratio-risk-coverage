"""Numerical identities and semantic boundaries, not implementation mirrors."""
import unittest,numpy as np
from scipy.stats import poisson,nbinom
from r3_observable import nll,derivatives,expected_abs,tail_terms,fit_observable,predict_mean
class ObservableTests(unittest.TestCase):
    def test_likelihood_derivatives_by_finite_differences(self):
        o=np.array([0,1,3,10,2,4,12],float);c=np.array([0,0,0,0,1,1,1],bool);z=np.log([.3,2,5,8,1,6,8]);h=1e-4
        for shape in [None,.5,2.,10.]:
            g,H=derivatives(z,o,c,shape);f0=nll(np.exp(z),o,c,shape);fp=nll(np.exp(z+h),o,c,shape);fm=nll(np.exp(z-h),o,c,shape)
            np.testing.assert_allclose(g,(fp-fm)/(2*h),rtol=2e-6,atol=2e-6)
            np.testing.assert_allclose(H,(fp-2*f0+fm)/h**2,rtol=3e-4,atol=3e-5)
    def test_absolute_error_moment_by_enumeration(self):
        for shape in [None,.5,2.,10.]:
            mu=np.array([.01,.3,2.,10.,50.]);f=np.array([0.,1.,2.5,12.,45.2]);y=np.arange(10000)[:,None]
            pm=poisson.pmf(y,mu) if shape is None else nbinom.pmf(y,shape,shape/(shape+mu))
            exact=(np.abs(y-f)*pm).sum(0)
            np.testing.assert_allclose(expected_abs(mu,f,shape),exact,rtol=1e-10,atol=1e-10)
    def test_extreme_censored_tail_is_finite(self):
        for shape in [None,.5,2.,10.]:
            a,l=tail_terms(np.array([1e-6,.01]),np.array([1000.,500.]),shape)
            self.assertTrue(np.isfinite(a).all() and np.isfinite(l).all());self.assertTrue((a>0).all())
    def test_uncensored_and_censored_at_capacity_differ(self):
        self.assertNotEqual(nll(np.array([3.]),np.array([2.]),np.array([False]))[0],nll(np.array([3.]),np.array([2.]),np.array([True]))[0])
    def test_hidden_outcome_absent_from_fitter_interface(self):
        import inspect
        self.assertNotIn('truth',inspect.signature(fit_observable).parameters)
if __name__=='__main__':unittest.main()
