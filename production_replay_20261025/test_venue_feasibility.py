import unittest
from . import test_books as fixtures
from .venue_feasibility import assess


class VenueFeasibilityTests(unittest.TestCase):
    def case(self):
        v,p,t=fixtures.RecordedBookTests().case()
        v.profile['venue_caps']=dict(per_market=1000.,total=1000000.)
        for symbol,stream in v.streams.items():
            for _,_,c in stream['ctx']:c['openInterest']='9.5' if symbol=='BTC' else '0'
        result=dict(snapshots=[dict(qty=[1.]+[0.]*9)],orders=[dict(time=t,arrival=t+1,symbol='BTC',side=-1,size=2.,filled=2.)])
        return result,v

    def test_paper_reversal_cannot_bypass_actual_capacity_check(self):
        result,v=self.case();r=assess(result,v.streams,v.profile,v.sc,[0.]*10)
        self.assertFalse(r['complete']);self.assertIn('reversal',r['failures'][0]['reason'])

    def test_missing_oi_is_not_assumed_zero(self):
        result,v=self.case()
        for _,_,c in v.streams['ETH']['ctx']:del c['openInterest']
        r=assess(result,v.streams,v.profile,v.sc,[0.]*10)
        self.assertFalse(r['complete']);self.assertIn('unavailable',r['failures'][0]['reason'])

    def test_observed_spare_capacity_is_accepted(self):
        result,v=self.case();v.profile['venue_caps']['per_market']=2000.
        r=assess(result,v.streams,v.profile,v.sc,[0.]*10)
        self.assertTrue(r['complete']);self.assertEqual(r['checks'],1)


if __name__=='__main__':unittest.main()
