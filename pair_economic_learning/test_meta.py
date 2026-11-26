import unittest
import numpy as np
from .meta import outcome,fit
from .model import Spec
from .test_model import EconomicLearningTests


class CostLabelTests(unittest.TestCase):
    def test_flat_prices_charge_both_legs_roundtrip(self):
        q=np.array([.005,-.005]);p=np.array([100.,100.])
        net,both=outcome(q,p,p,p,.01,0.,0.)
        self.assertTrue(both);self.assertAlmostEqual(net,-.0038)

    def test_leg_limits_checked_independently(self):
        q=np.array([.005,-.005]);p=np.array([100.,100.])
        _,both=outcome(q,p,np.array([110.,100.]),p,.01,0.,0.)
        self.assertFalse(both)

    def test_price_and_actual_cash_are_additive(self):
        q=np.array([.005,-.005]);p=np.array([100.,100.])
        x,_=outcome(q,p,p,p,.01,.001,0.)
        y,_=outcome(q,p,p,p,.01,.001,.002)
        self.assertAlmostEqual(y-x,.002)

    def test_unmatured_labels_cannot_change_gate(self):
        f=EconomicLearningTests().fixture();net=f['y'].copy();both=np.ones_like(f['valid']);source=600*3600
        aa,audit=fit(f,net,both,source,Spec(min_origins=20))
        net[f['maturity']>=source]=1e6
        bb,audit2=fit(f,net,both,source,Spec(min_origins=20))
        self.assertEqual(audit,audit2)
        for k in aa:self.assertEqual(bytes(aa[k].save_raw()),bytes(bb[k].save_raw()))


if __name__=='__main__':unittest.main()
