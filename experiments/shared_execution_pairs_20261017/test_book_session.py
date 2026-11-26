import copy,unittest
import numpy as np
from experiments.persistent_intent_pairs_20261017 import test_book_plan as fixture
from .account import SharedAccount
from .book_session import BookPairSession


class BookSessionTests(unittest.TestCase):
    def setUp(self):
        fixture.BookPlannerTests.setUp(self)
        self.intent['forecast_source']=0
        self.cfg.update(completion_ack_delay_ms=1000,completion_window_seconds=10,future_exit_observation_grace_ms=5000)
        self.sc['latency_ms']=1000
        self.account=SharedAccount(1e6,self.p['symbols']);self.marks=np.ones(10)*100;self.sigma=np.eye(10)*1e-7
        self.streams={}
        for symbol in ('BTC','ETH'):
            times=list(range(0,21000,1000))
            self.streams[symbol]=dict(book=[(t,0,dict(time=t,levels=[[dict(px='99.99',sz='100')],[dict(px='100.01',sz='100')]])) for t in times],
                book_times=times,ctx=[(t,0,dict(oraclePx='100',funding='0')) for t in times],ctx_times=times,resets=[])
        self.session=BookPairSession(self.account,self.intent,self.cfg,self.p,self.sc)

    def change(self,symbol,index,bid,ask=None,size=None):
        book=self.streams[symbol]['book'][index][2]
        book['levels'][0][0]['px']=str(bid);book['levels'][1][0]['px']=str(ask if ask is not None else bid+.02)
        if size is not None:
            for level in book['levels']:level[0]['sz']=str(size)

    def test_real_book_plan_enters_and_exits_with_one_ledger(self):
        self.session.enter(0,self.streams,self.marks,self.sigma);self.assertEqual(self.session.state,'open')
        for index in (4,5):self.change('ETH',index,106.)
        self.session.advance(4,self.streams,self.marks,self.sigma)
        self.assertEqual(self.session.state,'closed');self.assertEqual(len(self.account.physical.fills),4)
        self.assertGreater(self.account.physical.fees,0);self.account.reconcile(self.marks)

    def test_baseline_symbol_ownership_does_not_block_pair_or_get_closed(self):
        self.account.physical.fill(0,10,100,0,0);self.account.owner_fill('base',0,10,100,0,0,'fixture')
        self.session.enter(0,self.streams,self.marks,self.sigma);self.assertEqual(self.session.state,'open')
        for index in (4,5):self.change('ETH',index,106.)
        self.session.advance(4,self.streams,self.marks,self.sigma)
        self.assertEqual(self.session.state,'closed');self.assertEqual(self.account.owners['base'].qty[0],10)
        self.assertEqual(self.account.physical.qty[0],10)

    def test_one_unfilled_leg_gets_one_cash_bounded_retry_after_ack(self):
        self.change('BTC',1,97.)
        self.session.enter(0,self.streams,self.marks,self.sigma);self.assertEqual(self.session.state,'completion')
        held=self.account.owners['pair'].qty[1];self.assertIsNone(self.session.advance(1,self.streams,self.marks,self.sigma))
        self.session.advance(2,self.streams,self.marks,self.sigma)
        self.assertEqual(self.session.state,'open');self.assertEqual(self.session.completions,1)
        self.assertEqual(self.account.owners['pair'].qty[1],held)
        budget=next(e for e in self.session.events if e['kind']=='completion_budget')
        self.assertGreaterEqual(budget['minimum_expected_cash_at_limits'],budget['buffer_cash']-1e-8)

    def test_failed_completion_closes_held_leg_and_pays_cost(self):
        self.change('BTC',1,97.);self.change('BTC',3,90.)
        self.session.enter(0,self.streams,self.marks,self.sigma);self.session.advance(2,self.streams,self.marks,self.sigma)
        self.assertEqual(self.session.state,'closing')
        self.session.advance(4,self.streams,self.marks,self.sigma)
        self.assertEqual(self.session.state,'closed');self.assertGreater(self.account.physical.fees,0)

    def test_partial_exit_is_not_marked_closed_and_retry_is_paid(self):
        self.session.enter(0,self.streams,self.marks,self.sigma)
        self.change('ETH',4,106.);self.change('ETH',5,106.,size=1.)
        self.session.advance(4,self.streams,self.marks,self.sigma);self.assertEqual(self.session.state,'closing')
        self.assertGreater(self.account.owners['pair'].qty[1],0)
        self.session.advance(6,self.streams,self.marks,self.sigma);self.assertEqual(self.session.state,'closed')
        self.assertEqual(sum(f['asset']==1 and f['reduce_only'] for f in self.account.physical.fills),2)

    def test_missing_arrival_stays_unknown_and_does_not_send_retry(self):
        st=self.streams['BTC'];st['book']=st['book'][:1];st['book_times']=st['book_times'][:1]
        self.session.enter(0,self.streams,self.marks,self.sigma);self.assertEqual(self.session.state,'reconcile_required')
        count=len(self.session.events);self.assertIsNone(self.session.advance(4,self.streams,self.marks,self.sigma))
        self.assertEqual(len(self.session.events),count);self.assertEqual(self.account.unknown,{'BTC'})

    def test_completion_expiry_unwinds_instead_of_late_hedge(self):
        self.change('BTC',1,97.)
        self.session.enter(0,self.streams,self.marks,self.sigma)
        self.session.advance(12,self.streams,self.marks,self.sigma)
        self.assertEqual(self.session.state,'closed');self.assertEqual(self.session.completions,0)


if __name__=='__main__':unittest.main()
