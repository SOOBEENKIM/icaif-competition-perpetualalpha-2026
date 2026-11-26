import unittest
import numpy as np
from auto_pair_selection.selection import IDS
from .scaled_coherence import project


class ScaledCoherenceTests(unittest.TestCase):
    def test_complete_cohort_weighted_error_identity(self):
        rng=np.random.default_rng(31)
        for _ in range(20):
            beta=np.exp(rng.normal(0,.3,45));mu=rng.normal(0,.01,45);unit=np.exp(rng.normal(-5,1,45));r=rng.normal(0,.01,10)
            y=np.array([(r[a]-bt*r[b])/(1+bt) for (a,b),bt in zip(IDS,beta)])
            p,a=project(mu,beta,IDS,np.ones(45,bool),unit)
            self.assertAlmostEqual(float(np.sum(((mu-y)/unit)**2)-np.sum(((p-y)/unit)**2)),a['removed_energy'],places=8)
            self.assertTrue(0<=a['stein_scale']<=1)

    def test_uniform_risk_units_recover_unweighted_projection(self):
        from .coherence import project as ordinary
        rng=np.random.default_rng(32);mu=rng.normal(0,.01,45);beta=np.ones(45);ok=np.ones(45,bool)
        p,_=project(mu,beta,IDS,ok,np.ones(45)*.01);q,_=ordinary(mu,beta,IDS,ok)
        np.testing.assert_allclose(p,q,atol=1e-14)

    def test_no_incoherence_does_not_create_shrinkage(self):
        r=np.linspace(-.01,.01,10);beta=np.ones(45);mu=np.array([(r[a]-r[b])/2 for a,b in IDS])
        p,a=project(mu,beta,IDS,np.ones(45,bool),np.linspace(.001,.01,45))
        np.testing.assert_allclose(p,mu,atol=1e-14);self.assertAlmostEqual(a['stein_scale'],1.)


if __name__=='__main__':unittest.main()
