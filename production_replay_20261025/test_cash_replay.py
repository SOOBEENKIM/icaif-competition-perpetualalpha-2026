import unittest
from . import test_books as fixtures
from .cash_replay import CashVenue


class CashReplayTests(unittest.TestCase):
    def case(self,qty=1.,rate=.001,unit=-.1):
        v,p,t=fixtures.RecordedBookTests().case()
        hour=int(t//3600)+1;at=hour*3600
        funding=[dict(time=at*1000,coin='pas:BTC',fundingRate=rate)]
        units=[] if unit is None else [dict(coin='pas:BTC',hour=hour,rate=rate,
            cash_per_unit=unit,maximum_unit_error=1e-8)]
        c=CashVenue(v.streams,v.profile,v.sc,t,at+10,funding,cash_units=units)
        for ledger in [p.account.physical,p.account.owners['base']]:ledger.fill(0,qty,100.,0.,t)
        c.account=p.account
        return c,p.account,at

    def test_cash_applies_only_once_and_after_settlement(self):
        c,a,at=self.case()
        c.settle(at*1000-1,a);self.assertEqual(a.physical.funding,0.)
        c.settle(at*1000,a);c.settle((at+1)*1000,a)
        self.assertAlmostEqual(a.physical.funding,-.1)
        self.assertAlmostEqual(a.owners['base'].funding,-.1)

    def test_receipts_haircut_does_not_discount_payments(self):
        c,a,at=self.case(qty=-2.);c.settle(at*1000,a)
        self.assertAlmostEqual(a.physical.funding,.1)
        c,a,at=self.case(qty=2.);c.settle(at*1000,a)
        self.assertAlmostEqual(a.physical.funding,-.2)

    def test_missing_nonzero_settlement_invalidates_run(self):
        c,a,at=self.case(unit=None);c.settle(at*1000,a)
        self.assertEqual(c.failures[0]['kind'],'missing_verified_settlement')
        self.assertEqual(c.cash_records,[])

    def test_zero_rate_needs_no_inferred_oracle(self):
        c,a,at=self.case(rate=0.,unit=None);c.settle(at*1000,a)
        self.assertFalse(c.failures);self.assertEqual(a.physical.funding,0.)

    def test_same_time_fill_is_not_treated_as_pre_settlement_inventory(self):
        c,a,at=self.case()
        for ledger in [a.physical,a.owners['base']]:ledger.fill(0,5.,100.,0.,at)
        c.settle(at*1000,a);self.assertAlmostEqual(a.physical.funding,-.1)


if __name__=='__main__':unittest.main()
