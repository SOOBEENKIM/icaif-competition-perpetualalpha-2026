import copy
import unittest
from .liquidation_audit import terminal_value, consume, value_at


class LiquidationAuditTest(unittest.TestCase):
    def setUp(self):
        self.profile = dict(trading=dict(symbols=['BTC'], execution=dict(decimals=[3])))
        self.sc = dict(depth_fraction=1., fee_bps=10.)
        book = dict(time=100000, levels=[[dict(px='99', sz='2')], [dict(px='101', sz='2')]])
        self.stream = dict(BTC=dict(book=[(100000, 1, book)], book_times=[100000],
            ctx=[(100000, 1, dict(markPx='100', oraclePx='100'))], ctx_times=[100000], resets=[]))
        self.snap = dict(time=100., qty=[1.], marks=[100.], equity=1000.)

    def test_long_and_short_pay_spread_and_fee(self):
        long = terminal_value(self.stream, self.profile, self.sc, self.snap)
        short = terminal_value(self.stream, self.profile, self.sc, dict(self.snap, qty=[-1.]))
        self.assertAlmostEqual(long['complete_liquidation_equity'], 998.901)
        self.assertAlmostEqual(short['complete_liquidation_equity'], 998.899)

    def test_missing_depth_is_not_free_liquidation(self):
        r = terminal_value(self.stream, self.profile, self.sc, dict(self.snap, qty=[3.]))
        self.assertFalse(r['complete'])
        self.assertIsNone(r['complete_liquidation_equity'])
        self.assertEqual(r['legs'][0]['residual'], 1.)

    def test_recorded_fills_are_not_available_again(self):
        orders = [dict(arrival=100., symbol='BTC', side=-1, filled=1.5, limit=99.)]
        r = terminal_value(self.stream, self.profile, self.sc, self.snap, orders)
        self.assertEqual(r['legs'][0]['filled'], .5)
        self.assertFalse(r['complete'])

    def test_future_books_cannot_fix_missing_depth(self):
        altered = copy.deepcopy(self.stream)
        b = dict(time=101000, levels=[[dict(px='200', sz='100')], [dict(px='201', sz='100')]])
        altered['BTC']['book'].append((101000, 1, b)); altered['BTC']['book_times'].append(101000)
        self.assertEqual(terminal_value(self.stream, self.profile, self.sc, self.snap),
                         terminal_value(altered, self.profile, self.sc, self.snap))

    def test_stale_or_reset_book_invalid(self):
        r = terminal_value(self.stream, self.profile, self.sc, dict(self.snap, time=120.))
        self.assertFalse(r['complete'])
        self.stream['BTC']['resets'] = [100001]
        r = terminal_value(self.stream, self.profile, self.sc, dict(self.snap, time=100.01))
        self.assertFalse(r['complete'])

    def test_impossible_recorded_consumption_raises(self):
        with self.assertRaises(ValueError):
            consume({99.: (2., 2.)}, 3., -1, 99.)

    def test_common_clock_uses_arrival_not_rounded_fill_second(self):
        result = dict(snapshots=[dict(self.snap,fees=0.,funding=0.)],
             orders=[dict(arrival=100.25,symbol='BTC',side=1,filled=1.,price=101.)],
             fills=[dict(asset=0,signed_qty=1.,price=101.,fee=.101,time=100)],funding=[])
        before=value_at(self.stream,self.profile,self.sc,result,100.1)
        after=value_at(self.stream,self.profile,self.sc,result,100.3)
        self.assertEqual(before['qty'],[1.]);self.assertEqual(after['qty'],[2.])
        self.assertEqual(before['equity'],1000.)
        self.assertAlmostEqual(after['equity'],998.899)

    def test_endpoint_funding_cash_counted_once(self):
        result=dict(snapshots=[dict(self.snap,fees=0.,funding=0.)],orders=[],fills=[],
                    funding=[dict(time=100250,delta=dict(usdc='.2'))])
        r=value_at(self.stream,self.profile,self.sc,result,100.3)
        self.assertAlmostEqual(r['equity'],1000.2)
        self.assertAlmostEqual(r['funding'],.2)


if __name__ == '__main__':
    unittest.main()
