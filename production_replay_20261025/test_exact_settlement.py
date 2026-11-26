import copy,unittest
from . import test_books as fixtures
from .exact_settlement import timed_units,synchronize_initial,run_exact
from candidate_handoff.portfolio import CandidatePortfolio


class HoldPortfolio(CandidatePortfolio):
    def step_candidate(self,*args):pass


class ExactSettlementTests(unittest.TestCase):
    def unit(self,at,rate=.001):
        return dict(coin='pas:BTC',hour=int(at//3600),cash_per_unit=-.1,rate=rate,
                    maximum_unit_error=1e-8,settlement_time_ms=at*1000)

    def test_missing_prior_cash_is_added_once_without_changing_seed(self):
        v,p,t=fixtures.RecordedBookTests().case()
        for ledger in [p.account.physical,p.account.owners['base']]:ledger.fill(0,2.,100.,0.,t-100)
        u=self.unit(t+.012)
        fixed,events=synchronize_initial(p,[u],t+300)
        self.assertAlmostEqual(fixed.account.physical.funding,-.2)
        self.assertEqual(p.account.physical.funding,0.)
        again,repeated=synchronize_initial(fixed,[u],t+300)
        self.assertEqual(repeated,[]);self.assertAlmostEqual(again.account.physical.funding,-.2)
        self.assertEqual(len(events),1)

    def test_settlement_after_exact_start_cannot_enter_initial_cash(self):
        v,p,t=fixtures.RecordedBookTests().case()
        for ledger in [p.account.physical,p.account.owners['base']]:ledger.fill(0,1.,100.,0.,t-1)
        fixed,events=synchronize_initial(p,[self.unit(t+.012)],t)
        self.assertFalse(events);self.assertEqual(fixed.account.physical.funding,0.)

    def test_missing_initial_payment_is_not_silently_omitted(self):
        v,p,t=fixtures.RecordedBookTests().case()
        for ledger in [p.account.physical,p.account.owners['base']]:ledger.fill(0,1.,100.,0.,t-1)
        expected=[dict(time=t*1000,coin='pas:BTC',fundingRate=.001)]
        fixed,_=synchronize_initial(p,[],t+300,expected)
        self.assertEqual(fixed.research_initial_cash_missing[0]['kind'],'missing_initial_funding_cash')
        expected[0]['fundingRate']=0.
        fixed,_=synchronize_initial(p,[],t+300,expected)
        self.assertEqual(fixed.research_initial_cash_missing,[])

    def test_team_timestamp_is_used_instead_of_rounded_hour(self):
        u=self.unit(3600.)
        rows=[('team',dict(time=3600012,delta=dict(coin='pas:BTC',szi='1'))),
              ('other',dict(time=3600025,delta=dict(coin='pas:BTC',szi='2')))]
        result=timed_units([u],rows,'TEAM')[0]
        self.assertEqual(result['settlement_time_ms'],3600012)
        self.assertTrue(result['team_timestamp_observed'])

    def test_missing_team_market_retains_timestamp_uncertainty(self):
        u=self.unit(3600.)
        rows=[('a',dict(time=3600012,delta=dict(coin='pas:BTC',szi='1'))),
              ('b',dict(time=3600025,delta=dict(coin='pas:BTC',szi='2')))]
        r=timed_units([u],rows,'team')[0]
        self.assertFalse(r['team_timestamp_observed']);self.assertEqual(r['timing_bounds_ms'],[3600012,3600025])

    def test_post_start_settlement_enters_pnl_at_actual_time(self):
        v,p,t=fixtures.RecordedBookTests().case();p.__class__=HoldPortfolio
        for ledger in [p.account.physical,p.account.owners['base']]:ledger.fill(0,1.,100.,0.,t-1)
        u=self.unit(t+1.012)
        records=[(v.profile['team_wallet'],dict(time=int((t+1.012)*1000),delta=dict(coin='pas:BTC',szi='1')))]
        result=run_exact(v.streams,[],v.profile,v.sc,t,t+10,p,'incumbent',
            [dict(time=t*1000,coin='pas:BTC',fundingRate=.001)],cash_units=[u],records=records)
        self.assertEqual(result['snapshots'][0]['funding'],0.)
        self.assertAlmostEqual(result['net_pnl'],-.1)
        self.assertEqual(result['initial_cash_synchronization'],[])


if __name__=='__main__':unittest.main()
