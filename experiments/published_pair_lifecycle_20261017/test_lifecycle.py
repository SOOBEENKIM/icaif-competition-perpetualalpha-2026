import copy
import unittest
import numpy as np
from experiments.forecast_validity_pairs_20261017.test_books import FixedForecastBookTests
from .account import BookAccount
from .lifecycle import PairLifecycle


class LifecycleTests(unittest.TestCase):
    def setUp(self):
        f = FixedForecastBookTests(); f.setUp()
        self.cfg, self.p, self.sc = f.cfg, f.p, f.sc
        self.streams, self.marks, self.sigma = f.streams, f.marks, f.sigma
        self.cfg.update(book_intent_lifetime_hours=1, exit_lead_seconds=6,
            dense_policy={'stop_sigma': 3.},
            renewal_policy={'maximum_holding_hours': 168, 'decision_cadence_hours': 6, 'incremental_buffer_bps': 5},
            rehedge_policy={'gross_tolerance': .05})
        self.intent = dict(f.intent, available_time=0, source_vol=.01)
        self.account = BookAccount(1e6, self.p['symbols'])
        self.funding_clock = 0
        self.session = PairLifecycle(self.account, self.intent, self.cfg, self.p, self.sc,
                                      on_time=self.funding)

    def funding(self, at, account):
        self.assertGreaterEqual(at, self.funding_clock)
        for bucket in range((self.funding_clock//3600000+1)*3600000, int(at)//3600000*3600000+1, 3600000):
            account.charge_funding(bucket/1000, self.marks, np.zeros(10), np.ones(10, bool))
        self.funding_clock = at

    def window(self, seconds, *, eth=100., btc=100., eth_size=100., btc_size=100.):
        for symbol, mid, size in [('ETH', eth, eth_size), ('BTC', btc, btc_size)]:
            stream = self.streams[symbol]
            for sec in seconds:
                at = int(sec*1000)
                book = dict(time=at, levels=[[dict(px=str(mid-.01), sz=str(size))], [dict(px=str(mid+.01), sz=str(size))]])
                if at in stream['book_times']:
                    stream['book'][stream['book_times'].index(at)] = (at, 0, book)
                else:
                    stream['book'].append((at, 0, book))
                    stream['ctx'].append((at, 0, dict(oraclePx=str(mid), funding='0')))
            for field in ['book', 'ctx']:
                stream[field].sort(key=lambda r:r[0]); stream[field+'_times'] = [r[0] for r in stream[field]]

    def open(self):
        self.session.enter(0, self.streams, self.marks, self.sigma)
        self.assertEqual(self.session.state, 'open')

    def forecast(self, source=18*3600, beta=1., terminal=(102., 98.)):
        return dict(self.intent, forecast_source=source, available_time=source+.5,
                    source_beta=beta, terminal_prices=list(terminal))

    def test_complete_entry_to_paid_maturity_exit_with_ownership(self):
        self.account.physical.fill(1, 2., 100., 0., 0)
        self.account.owner_fill('base', 1, 2., 100., 0., 0, 'fixture')
        self.open()
        now=24*3600-6
        self.window(range(now, now+4), eth=102., btc=98.)
        self.session.advance(now, self.streams, self.marks, self.sigma)
        self.assertEqual(self.session.state, 'closed')
        self.assertEqual(self.session.closed_reason, 'forecast_maturity')
        self.assertTrue(self.session.was_balanced)
        self.assertFalse(np.any(self.account.owners['pair'].qty))
        self.assertEqual(self.account.owners['base'].qty[1], 2.)
        self.assertGreater(self.account.physical.fees, 0.)
        self.account.reconcile(self.marks)

    def test_no_resize_renewal_is_no_fill_and_preserves_stop(self):
        self.open(); count=len(self.account.physical.fills); bound=self.session.stop_bound
        now=18*3600+1; self.window(range(now-1, now+5))
        self.session.advance(now, self.streams, self.marks, self.sigma, forecast=self.forecast())
        self.assertEqual(self.session.maturity, 42*3600)
        self.assertEqual(len(self.account.physical.fills), count)
        self.assertEqual(self.session.stop_bound, bound)
        old_exit=24*3600-6; self.window(range(old_exit, old_exit+3))
        self.session.advance(old_exit, self.streams, self.marks, self.sigma)
        self.assertEqual(self.session.state, 'open')

    def test_changed_hedge_requires_paid_fills_before_maturity_moves(self):
        self.open(); count=len(self.account.physical.fills); anchor=self.session.stop_origin
        now=18*3600+1; self.window(range(now-1, now+6))
        self.session.advance(now, self.streams, self.marks, self.sigma, forecast=self.forecast(beta=2.))
        self.assertEqual(self.session.state, 'open')
        self.assertEqual(self.session.maturity, 42*3600)
        self.assertGreater(len(self.account.physical.fills), count)
        self.assertEqual(self.session.stop_origin, anchor)
        q=self.account.owners['pair'].qty[self.session.ids]
        self.assertLess(abs(abs(q[1]/q[0])/2.-1), .05)
        self.account.reconcile(self.marks)

    def test_partial_hedge_adjustment_stays_paid_and_unwinds(self):
        self.open(); now=18*3600+1
        self.window(range(now-1, now+8))
        # Decision sees both books. One independent arrival has no extra short liquidity.
        self.window([now+1], btc_size=0.)
        old_maturity=self.session.maturity
        self.session.advance(now, self.streams, self.marks, self.sigma, forecast=self.forecast(beta=2.))
        self.assertEqual(self.session.state, 'closing')
        self.assertEqual(self.session.maturity, old_maturity)
        self.assertEqual(self.session.closed_reason, 'partial_adjustment_hedge_mismatch')
        self.session.advance(self.session.ready_at, self.streams, self.marks, self.sigma)
        self.assertEqual(self.session.state, 'closed')
        self.assertGreater(self.account.physical.fees, 0.)

    def test_zero_adjustment_fill_retains_old_maturity(self):
        self.open(); now=18*3600+1
        self.window(range(now-1, now+8)); self.window([now+1], eth_size=0., btc_size=0.)
        count=len(self.account.physical.fills)
        self.session.advance(now, self.streams, self.marks, self.sigma, forecast=self.forecast(beta=2.))
        self.assertEqual(self.session.state, 'open'); self.assertEqual(self.session.maturity, 24*3600)
        self.assertEqual(len(self.account.physical.fills), count)
        self.session.advance(self.session.ready_at, self.streams, self.marks, self.sigma, forecast=self.forecast(beta=2.))
        self.assertEqual(len(self.account.physical.fills), count)

    def test_original_price_stop_cannot_be_cancelled_by_new_forecast(self):
        self.open(); now=18*3600+1
        self.window(range(now-1, now+5), eth=60., btc=140.)
        self.session.advance(now, self.streams, self.marks, self.sigma, forecast=self.forecast(terminal=(150., 50.)))
        self.assertEqual(self.session.state, 'closed'); self.assertEqual(self.session.closed_reason, 'pair_return_stop')
        self.assertEqual(self.session.maturity, 24*3600)

    def test_late_forecast_and_hard_deadline_cannot_extend_holding(self):
        self.open(); now=18*3600+1; self.window(range(now-1, now+5))
        future=self.forecast(); future['available_time']=now+1
        self.session.advance(now, self.streams, self.marks, self.sigma, forecast=future)
        self.assertEqual(self.session.maturity, 24*3600)
        self.session.cfg['renewal_policy']['maximum_holding_hours']=30
        self.session.advance(now+1, self.streams, self.marks, self.sigma, forecast=self.forecast())
        self.assertEqual(self.session.maturity, 24*3600)

    def test_partial_maturity_exit_remains_open_until_all_receipts_flat(self):
        self.open(); now=24*3600-6
        self.window(range(now, now+7)); self.window([now+1], eth_size=1.)
        self.session.advance(now, self.streams, self.marks, self.sigma)
        self.assertEqual(self.session.state, 'closing')
        self.assertTrue(np.any(self.account.owners['pair'].qty))
        self.session.advance(self.session.ready_at, self.streams, self.marks, self.sigma)
        self.assertEqual(self.session.state, 'closed')

    def test_unknown_exit_does_not_enable_duplicate_retry(self):
        self.open(); now=24*3600-6; self.window([now])
        self.session.advance(now, self.streams, self.marks, self.sigma)
        self.assertEqual(self.session.state, 'reconcile_required')
        count=len(self.account.physical.fills)
        self.window(range(now+1, now+10))
        self.session.advance(now+8, self.streams, self.marks, self.sigma)
        self.assertEqual(len(self.account.physical.fills), count)

    def test_future_publication_cannot_be_used_for_entry(self):
        self.session.intent['available_time']=1
        with self.assertRaisesRegex(ValueError, 'not available'):
            self.session.enter(0, self.streams, self.marks, self.sigma)

    def test_known_zero_fill_allows_a_distinct_bounded_later_attempt(self):
        self.window([1],eth_size=0.,btc_size=0.)
        self.session.enter(0,self.streams,self.marks,self.sigma)
        self.assertEqual(self.session.state,'unfilled')
        later=PairLifecycle(self.account,self.intent,self.cfg,self.p,self.sc,on_time=self.funding)
        later.enter(3,self.streams,self.marks,self.sigma)
        self.assertEqual(later.state,'open')
        self.assertEqual(len(self.account.seen),2)


class ExecutionTests(LifecycleTests):
    # Reuse the fixture, not its tests, by overriding discovery below.
    def packet(self, now, target, limit):
        pair=np.zeros(10); pair[1]=target
        raw=dict(packet_id=str(now),decision_time=now,source_available=now,source_close=now,
                 source_prices=self.marks.tolist(),risk_forced=False,
                 owner_targets={'base':[0.]*10,'pair':pair.tolist()},owner_limits={'pair':{1:limit}})
        return self.account.seal(raw, self.p, {**self.sc,'impact_bps':0.})

    def test_integer_price_limit_is_enforced_at_independent_arrival(self):
        packet=self.packet(0, 1., 100.005)
        self.assertEqual(packet['sealed_owner_limits']['pair',1],100.005)
        result=self.account.execute_books(packet,self.streams,self.marks,self.sigma,self.p,self.sc,self.cfg)
        self.assertEqual(result['orders'][0]['filled_qty'],0.)
        self.assertFalse(self.account.physical.fills)

    def test_colliding_limit_identity_is_rejected(self):
        packet=self.packet(0,1.,100.005)
        packet['owner_limits']['pair']={1:100.005,'1':110.}
        with self.assertRaisesRegex(ValueError,'Duplicate'):
            self.account.seal(packet,self.p,{**self.sc,'impact_bps':0.})

    def test_received_but_pre_submission_server_book_cannot_fill(self):
        for _,_,book in self.streams['ETH']['book']:book['time']=-1
        result=self.account.execute_books(self.packet(0,1.,101.),self.streams,self.marks,self.sigma,self.p,self.sc,self.cfg)
        self.assertEqual(result['unknown_symbols'],['ETH']); self.assertFalse(self.account.physical.fills)

    def test_same_server_snapshot_liquidity_is_not_consumed_twice(self):
        self.window([1,2],eth_size=1.)
        self.streams['ETH']['book'][2][2]['time']=1000
        for now in [0,1]:
            self.account.execute_books(self.packet(now,2.,101.),self.streams,self.marks,self.sigma,self.p,self.sc,self.cfg)
        self.assertEqual(self.account.owners['pair'].qty[1],1.)


# Avoid counting inherited lifecycle tests twice.
def load_tests(loader, tests, pattern):
    suite=unittest.TestSuite(loader.loadTestsFromTestCase(LifecycleTests))
    suite.addTests(ExecutionTests(name) for name in ExecutionTests.__dict__ if name.startswith('test_'))
    return suite


if __name__=='__main__':unittest.main()
