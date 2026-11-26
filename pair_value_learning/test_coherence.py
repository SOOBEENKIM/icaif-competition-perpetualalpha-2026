import unittest
import numpy as np
from .coherence import project
from auto_pair_selection.selection import IDS


class CoherenceTests(unittest.TestCase):
    def test_impossible_triangle_removed(self):
        # A>B, B>C, C>A by identicalpositiveamount cannot all hold.
        p=[(0,1),(1,2),(2,0)];out,audit=project(np.ones(3)*.01,np.ones(3),p,np.ones(3,bool))
        np.testing.assert_allclose(out,0,atol=1e-15);self.assertEqual(audit['rank'],2)

    def test_coherent_forecasts_unchanged(self):
        r=np.linspace(-.02,.02,10);b=np.ones(45);mu=np.array([(r[a]-r[c])/2 for a,c in IDS])
        out,_=project(mu,b,IDS,np.ones(45,bool));np.testing.assert_allclose(out,mu,atol=1e-15)

    def test_squared_error_identity_for_source_frozen_hedges(self):
        rng=np.random.default_rng(5)
        for _ in range(20):
            b=np.exp(rng.normal(0,.3,45));mu=rng.normal(0,.01,45);r=rng.normal(0,.01,10)
            y=np.array([(r[a]-bt*r[c])/(1+bt) for (a,c),bt in zip(IDS,b)])
            out,audit=project(mu,b,IDS,np.ones(45,bool))
            self.assertAlmostEqual(float(np.sum((mu-y)**2)-np.sum((out-y)**2)),audit['removed_energy'])

    def test_invalid_pair_cannot_change_valid_projection(self):
        mu=np.ones(45)*.01;b=np.ones(45);valid=np.ones(45,bool);valid[2]=False
        a,_=project(mu,b,IDS,valid);mu[2]=1e10;z,_=project(mu,b,IDS,valid)
        np.testing.assert_array_equal(a[valid],z[valid])


if __name__=='__main__':unittest.main()
