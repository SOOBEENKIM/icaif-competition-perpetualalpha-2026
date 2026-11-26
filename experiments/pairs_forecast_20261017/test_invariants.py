import copy
import unittest
from types import SimpleNamespace
import numpy as np
from .model import design, labels
from .policy import balanced_inventory


class Invariants(unittest.TestCase):
    def test_features_cannot_read_future(self):
        rng = np.random.default_rng(17); n = 1100
        close = 100 * np.exp(np.cumsum(rng.normal(0, .002, (n, 3)), axis=0))
        m = SimpleNamespace(close=close, times=np.arange(n)*3600, volume=np.ones((n,3)), trades=np.ones((n,3)), funding=np.zeros((n,3)), symbols=('BTC','ETH','HYPE'))
        cfg = dict(pairs=[['ETH','BTC'],['HYPE','BTC']])
        a = design(m, cfg); changed = copy.deepcopy(m); changed.close[950:] *= 10
        b = design(changed, cfg)
        np.testing.assert_allclose(a['x'][:950], b['x'][:950], equal_nan=True)
        np.testing.assert_allclose(a['beta'][:950], b['beta'][:950], equal_nan=True)

    def test_forward_label_uses_next_entry_and_fixed_hedge(self):
        m = SimpleNamespace(times=np.arange(6)*3600, close=np.array([[1.,1.],[100.,100.],[110.,100.],[110.,100.],[110.,100.],[110.,100.]]), funding=np.zeros((6,2)))
        d = dict(beta=np.ones((6,1)), vol=np.ones((6,1)), fresh=np.ones((6,1),bool), pairs=[[0,1]])
        y, maturity = labels(m,d,1)
        self.assertAlmostEqual(y[0,0], .05)
        self.assertEqual(maturity[0],7200)
        self.assertAlmostEqual(y[1,0],0.)

    def test_balanced_partial_is_not_an_unhedged_failure(self):
        ok, q = balanced_inventory([8.,-16.],[10.,-20.],.05)
        self.assertTrue(ok);np.testing.assert_array_equal(q,[8.,-16.])

    def test_imbalance_repair_only_reduces(self):
        ok,q=balanced_inventory([8.,-10.],[10.,-20.],.05)
        self.assertFalse(ok);np.testing.assert_array_equal(q,[5.,-10.])
        self.assertTrue(np.all(abs(q)<=abs([8.,-10.])))
        ok,q=balanced_inventory([8.,0.],[10.,-20.],.05)
        self.assertFalse(ok);np.testing.assert_array_equal(q,[0.,0.])


if __name__ == '__main__':unittest.main()
