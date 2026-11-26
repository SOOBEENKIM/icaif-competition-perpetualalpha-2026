import unittest
import numpy as np
from experiments.shared_execution_pairs_20261017.account_v2 import SharedAccount
from .settlements import align,records,charge

class SettlementTiming(unittest.TestCase):
    def test_late_event_is_not_available_at_hour(self):
        times=np.arange(0,7201,300)
        rows=[dict(time=3600019,coin='par:BTC',fundingRate='.001')]
        rate,stamp,source=align(rows,times,0,'par:BTC')
        self.assertEqual(stamp[np.searchsorted(times,3600)],-1)
        i=np.searchsorted(times,3900)
        self.assertEqual(stamp[i],3600019);self.assertEqual(source[i],3600)
        self.assertEqual(rate[i],.001)
        self.assertEqual(stamp[-1],-1)
    def test_after_cutoff_event_rejected(self):
        with self.assertRaises(ValueError):
            records([dict(time=7200010,coin='par:BTC',fundingRate='.001')],'par:BTC',7200000)
    def test_missing_interior_event_fails(self):
        with self.assertRaises(ValueError):align([],np.arange(0,7201,300),0,'par:BTC')
    def test_old_cutoff_payment_not_charged_twice(self):
        rows=[dict(time=19,coin='par:BTC',fundingRate='.001'),dict(time=3600019,coin='par:BTC',fundingRate='.002')]
        _,stamps,_=align(rows,np.arange(0,7201,300),0,'par:BTC')
        self.assertEqual((stamps>=0).sum(),1)
    def test_inventory_before_next_fill_pays_actual_settlement(self):
        a=SharedAccount(1e6,('BTC',))
        for ledger in [a.physical,a.owners['base']]:ledger.fill(0,10,100,0,3600)
        cash=charge(a,3900,np.array([100.]),np.array([.001]),np.array([True]),3600019,1.)
        self.assertEqual(cash,-1.);self.assertEqual(a.physical.funding,-1.)
        for ledger in [a.physical,a.owners['base']]:ledger.fill(0,-10,100,0,3900)
        self.assertEqual(a.physical.funding,-1.)
        self.assertEqual(a.physical.funding_events[-1]['settlement_time_ms'],3600019)
        self.assertEqual(a.physical.funding_events[-1]['recognized_bar_time'],3900)
        self.assertAlmostEqual(a.reconcile(np.array([100.])),999999.)
    def test_haircut_is_on_net_physical_receipt(self):
        a=SharedAccount(1e6,('BTC',))
        a.physical.fill(0,-5,100,0,3600)
        a.owners['base'].fill(0,-10,100,0,3600)
        a.owners['pair'].fill(0,5,100,0,3600)
        self.assertEqual(charge(a,3900,np.array([100.]),np.array([.001]),np.array([True]),3600019,.5),.25)
        self.assertEqual(a.owners['base'].funding,.5)
        self.assertEqual(a.owners['pair'].funding,-.25)
        self.assertAlmostEqual(a.reconcile(np.array([100.])),1000000.25)

if __name__=='__main__':unittest.main()
