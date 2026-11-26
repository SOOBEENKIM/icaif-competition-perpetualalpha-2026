import unittest
import numpy as np
from .top_of_book_bound import unit_bound

class BoundTests(unittest.TestCase):
    def test_dollar_accounting_at_known_two_leg_prices(self):
        value=unit_bound(np.array([.6,-.4]),np.array([100.,200.]),np.array([100.1,199.8]),
            np.array([99.9,200.2]),np.array([101.,198.]),np.array([100.,200.]),np.zeros(2),24,9,10)
        # Per 1 gross: price cash .009, entry fees .00090018,
        # reverse-side spread .001, exit fee+reserve .0019.
        self.assertAlmostEqual(value,51.9982,places=9)
    def test_worse_depth_cannot_exceed_touch_bound(self):
        w=np.array([.6,-.4]);p=np.array([100.,200.]);entry=np.array([100.1,199.8]);exit_=np.array([99.9,200.2]);terminal=np.array([101.,198.])
        best=unit_bound(w,p,entry,exit_,terminal,p,np.array([.0001,-.0001]),24,9,10)
        for adverse in [.0001,.001,.01]:
            worse=unit_bound(w,p,entry*(1+np.sign(w)*adverse),exit_*(1-np.sign(w)*adverse),terminal,p,np.array([.0001,-.0001]),24,9,10)
            self.assertLess(worse,best)
    def test_ratio_tolerance_endpoints_bound_all_interior_ratios(self):
        rng=np.random.default_rng(1017)
        for _ in range(20):
            beta=rng.uniform(.1,10);side=rng.choice([-1,1]);marks=np.array([100.,200.]);signs=side*np.array([1.,-1.])
            entry=marks*(1+signs*.001);exit_=marks*(1-signs*.001);terminal=marks*(1+rng.normal(0,.02,2));fund=rng.normal(0,.0001,2)
            def value(ratio):
                w=signs*np.array([1.,beta*ratio]);w/=abs(w).sum()
                return unit_bound(w,marks,entry,exit_,terminal,marks,fund,23.5,9,10)
            ceiling=max(value(.95),value(1.05))
            self.assertTrue(all(value(r)<=ceiling+1e-9 for r in np.linspace(.95,1.05,51)))

if __name__=='__main__':unittest.main()
