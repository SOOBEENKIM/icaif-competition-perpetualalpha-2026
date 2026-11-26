import unittest
import numpy as np
from .adaptive_hedge import fit_pair


class AdaptiveHedgeTests(unittest.TestCase):
    def test_prefix_and_pair_order_symmetry(self):
        rng=np.random.default_rng(991);n=1700;trend=np.cumsum(rng.normal(0,.002,n));spread=np.zeros(n)
        for i in range(1,n):spread[i]=.98*spread[i-1]+rng.normal(0,.001)
        logs=np.c_[trend+spread,trend];active=np.ones_like(logs,bool)
        a=fit_pair(logs,active,1550,[0,1]);b=fit_pair(logs,active,1550,[1,0]);self.assertIsNotNone(a);self.assertIsNotNone(b)
        self.assertAlmostEqual(a['beta']*b['beta'],1,places=12)
        self.assertAlmostEqual(a['rho'],b['rho'],places=12)
        logs[1551:]=np.nan;active[1551:]=False
        self.assertEqual(a,fit_pair(logs,active,1550,[0,1]));self.assertLessEqual(a['last_price_index'],1550)


if __name__=='__main__':unittest.main()
