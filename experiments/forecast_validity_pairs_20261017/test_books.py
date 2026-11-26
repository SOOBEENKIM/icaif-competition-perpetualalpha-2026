import copy
import unittest
import numpy as np
from experiments.shared_execution_pairs_20261017 import test_book_session as fixture
from experiments.shared_execution_pairs_20261017.account_v2 import SharedAccount
from experiments.basis_carry_20261017.probe import asof
from .book_plan import plan_entry
from .book_session import FixedForecastEntrySession


class FixedForecastBookTests(unittest.TestCase):
    def setUp(self):
        f=fixture.BookSessionTests();f.setUp()
        self.cfg,self.p,self.sc,self.streams,self.marks,self.sigma=f.cfg,f.p,f.sc,f.streams,f.marks,f.sigma
        self.intent=dict(pair=['ETH','BTC'],ids=[1,0],direction=1,source_beta=1.,source_prices=[100.,100.],
                         terminal_prices=[102.,98.],forecast_source=0,horizon=24,account_marks=self.marks.tolist())
        self.cfg['book_intent_lifetime_hours']=6;self.cfg['funding_forecast_hours']=24
        self.account=SharedAccount(1e6,self.p['symbols'])
        self.session=FixedForecastEntrySession(self.account,self.intent,self.cfg,self.p,self.sc)

    def quotes(self):
        return {s:asof(self.streams[s],0,self.cfg) for s in ['BTC','ETH']}

    def test_frozen_source_hedge_repriced_from_current_books(self):
        q=self.quotes();self.intent.update(source_beta=.5,source_prices=[100.,200.],terminal_prices=[120.,180.])
        for s,price in [('ETH',110.),('BTC',190.)]:
            q[s]['book']['levels']=[[dict(px=str(price-.01),sz='100')],[dict(px=str(price+.01),sz='100')]]
        r=plan_entry(self.intent,q,0,self.cfg,self.p,self.sc,np.zeros(10),1e6,self.sigma)
        self.assertTrue(r['accepted']);self.assertAlmostEqual(abs(r['target'][1]/r['target'][0]),.25,places=3)

    def test_remaining_horizon_not_reset_by_later_entry(self):
        q=self.quotes()
        for v in q.values():
            v['received']={k:7200000 for k in v['received']};v['book']['time']=7200000
        r=plan_entry(self.intent,q,7200,self.cfg,self.p,self.sc,np.zeros(10),1e6,self.sigma)
        self.assertTrue(r['accepted']);self.assertEqual(r['remaining_hours'],22)
        cfg=copy.deepcopy(self.cfg);cfg['book_intent_lifetime_hours']=1
        self.assertEqual(plan_entry(self.intent,q,7200,cfg,self.p,self.sc,np.zeros(10),1e6,self.sigma)['reason'],'expired_signal')

    def test_oracle_is_not_an_executable_entry_or_exit(self):
        q=self.quotes();q['ETH']['oracle']=1.;q['BTC']['oracle']=100000.
        for v in q.values():v['ctx']['funding']='0'
        r=plan_entry(self.intent,q,0,self.cfg,self.p,self.sc,np.zeros(10),1e6,self.sigma)
        self.assertTrue(r['accepted']);self.assertEqual(r['entry_preview'],[100.01,99.99])
        self.assertEqual(r['exit_preview'],[99.99,100.01])

    def test_no_price_edge_is_rejected_after_costs(self):
        self.intent['terminal_prices']=[100.,100.]
        r=plan_entry(self.intent,self.quotes(),0,self.cfg,self.p,self.sc,np.zeros(10),1e6,self.sigma)
        self.assertEqual(r['reason'],'edge_below_full_cost')

    def test_both_entries_pay_fees_but_open_inventory_is_not_a_completed_trade(self):
        self.session.enter(0,self.streams,self.marks,self.sigma)
        self.assertEqual(self.session.state,'open');self.assertEqual(len(self.account.physical.fills),2)
        self.assertGreater(self.account.physical.fees,0.)
        self.assertIsNone(self.session.advance(4,self.streams,self.marks,self.sigma))
        self.assertTrue(np.any(self.account.physical.qty))

    def test_missing_leg_gets_one_paid_completion(self):
        self.streams['BTC']['book'][1][2]['levels'][0][0]['px']='97'
        self.session.enter(0,self.streams,self.marks,self.sigma)
        self.assertEqual(self.session.state,'completion')
        held=self.account.owners['pair'].qty[1]
        self.session.advance(2,self.streams,self.marks,self.sigma)
        self.assertEqual(self.session.state,'open');self.assertEqual(self.session.completions,1)
        self.assertEqual(self.account.owners['pair'].qty[1],held)

    def test_failed_completion_retains_actual_fill_then_pays_close(self):
        for i in [1,3]:self.streams['BTC']['book'][i][2]['levels'][0][0]['px']='90'
        self.session.enter(0,self.streams,self.marks,self.sigma)
        self.session.advance(2,self.streams,self.marks,self.sigma)
        self.assertEqual(self.session.state,'closing')
        self.session.advance(4,self.streams,self.marks,self.sigma)
        self.assertEqual(self.session.state,'closed')
        self.assertGreater(self.account.physical.fees,0.)

    def test_unknown_arrival_does_not_allow_repeated_orders(self):
        self.streams['BTC']['book']=self.streams['BTC']['book'][:1]
        self.streams['BTC']['book_times']=self.streams['BTC']['book_times'][:1]
        self.session.enter(0,self.streams,self.marks,self.sigma)
        self.assertEqual(self.session.state,'reconcile_required')
        count=len(self.session.events);self.session.advance(4,self.streams,self.marks,self.sigma)
        self.assertEqual(count,len(self.session.events))


if __name__=='__main__':unittest.main()
