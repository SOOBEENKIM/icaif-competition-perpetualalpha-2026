"""Quote features through real target, risk, account, and receipt handling."""
import copy
import unittest
from unittest.mock import patch
import numpy as np
from candidate_handoff import test_handoff as fixtures
from candidate_handoff.portfolio import adopt
from competition_live.profile import START
from competition_live.runtime import fingerprint
from .quote_history import Activated, QuoteHistoryPortfolio, quote_target


class QuoteIntegrationTests(unittest.TestCase):
    def case(self, mode='quote_inputs'):
        helper=fixtures.HandoffTests();helper.setUp()
        _,original,portfolio,clock,orders,current=helper.fixture()
        p=adopt(portfolio)
        p.account.executor=original.executor
        p.account.collect_fills=original.collect_fills
        p.account.checkpoint=lambda *args:None
        p.account.refresh_market=current
        p.__class__=Activated;p.activation=START;p.quote_mode=mode
        packet=helper.packet(clock[0])
        f=packet['body']['frame'];f['anchor_weights']=[.04,-.04]+[0.]*8
        packet['sha256']=fingerprint(packet['body'])
        streams={}
        for j,symbol in enumerate(p.p['symbols']):
            stream=copy.deepcopy(current(symbol))
            for point,price in [(START-10800,81000. if j==0 else 79000.),(START,80000.)]:
                at=(point-1)*1000
                stream['book'].append((at,1,dict(coin='pas:'+symbol,time=at,
                    levels=[[dict(px=str(price-1),sz='100')],[dict(px=str(price+1),sz='100')]])))
                stream['ctx'].append((at,1,dict(markPx=str(price),oraclePx=str(price),funding='0')))
            for field in ['book','ctx']:
                stream[field].sort(key=lambda x:x[0]);stream[field+'_times']=[x[0] for x in stream[field]]
            streams[symbol]=stream
        return p,clock,orders,streams,packet,helper.p

    def step(self,case):
        p,clock,orders,streams,packet,profile=case
        with patch('competition_live.live_account.time.time',side_effect=lambda:clock[0]):
            p.step_candidate(clock[0],streams,packet,profile)

    def test_signed_quote_features_reach_actual_account_path(self):
        case=self.case();p,clock,orders,streams,packet,profile=case
        expected,trace=quote_target(packet['body']['frame'],np.full(10,80000.),np.zeros(10),50000.,p.p,streams,p.cfg)
        self.assertGreater(expected['target_qty'][0],0.)
        self.assertLess(expected['target_qty'][1],0.)
        self.step(case)
        self.assertIsInstance(p,QuoteHistoryPortfolio)
        np.testing.assert_allclose(p.account.physical.qty,expected['target_qty'])
        self.assertEqual({o.coin for o in orders},{'pas:BTC','pas:ETH'})
        self.assertTrue(any(o.is_buy for o in orders));self.assertTrue(any(not o.is_buy for o in orders))
        self.assertFalse(p.account.pending_orders);self.assertFalse(p.account.unknown)
        p.account.reconcile(np.full(10,80000.))
        self.assertLessEqual(p.account.maximum_error,1e-8)
        self.assertTrue(all(not row['valid'] for row in trace['quote_history'][:10]))

    def test_activation_keeps_owned_and_reserved_inventory(self):
        case=self.case();p,clock,orders,streams,packet,profile=case
        p.account.physical_fill(2,.01,80000.,0.,START-10)
        p.account.owner_fill('base',2,.01,80000.,0.,START-10,'seed')
        p.account.reserved.qty[2]=-.02
        before_reserved=p.account.reserved.qty.copy()
        before_cash=p.account.physical.cash
        # An ordinary zero target may reduce our HYPE. The teammate's virtual
        # position is never overwritten by migration or by our strategy target.
        self.step(case)
        np.testing.assert_array_equal(p.account.reserved.qty,before_reserved)
        self.assertGreaterEqual(len(p.account.physical.fills),1)
        self.assertEqual(p.account.physical.fills[0]['signed_qty'],.01)
        self.assertIsNotNone(before_cash)

    def test_missing_current_or_lag_quote_vetoes_additions(self):
        for missing in ['current','lag']:
            with self.subTest(missing=missing):
                case=self.case();p,clock,orders,streams,packet,profile=case
                for s in streams.values():
                    point=START if missing=='current' else START-10800
                    for field in ['book','ctx']:
                        s[field]=[row for row in s[field] if row[0]!=(point-1)*1000]
                        s[field+'_times']=[row[0] for row in s[field]]
                self.step(case)
                self.assertFalse(orders);np.testing.assert_array_equal(p.account.physical.qty,0.)

    def test_history_only_keeps_original_deadline(self):
        case=self.case('history_only');self.step(case)
        p=case[0]
        events=[e for e in p.events if e['kind']=='quote_history_decision']
        self.assertEqual(len(events),1)
        self.assertEqual(events[0]['quote_mode'],'history_only')
        if p.base_task is not None:self.assertEqual(p.base_task['entry_expires'],START+900)


if __name__=='__main__':unittest.main()
