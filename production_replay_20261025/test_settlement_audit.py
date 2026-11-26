import unittest
from .settlement_audit import cash_units, path_cash_sensitivity


def row(q, cash, at=3600000, rate=.0001):
    return dict(time=at, delta=dict(coin='pas:BTC', szi=str(q), usdc=str(cash), fundingRate=str(rate)))


class SettlementAuditTests(unittest.TestCase):
    def test_both_sides_and_rounding_intersect(self):
        units, bad = cash_units([('a',row(10,-.100001)), ('b',row(-30,.3))])
        self.assertFalse(bad)
        self.assertLessEqual(units[0]['cash_per_unit_lower'], -.01)
        self.assertGreaterEqual(units[0]['cash_per_unit_upper'], -.01)

    def test_inconsistent_cash_cannot_be_averaged_away(self):
        units, bad = cash_units([('a',row(10,-.1)), ('b',row(10,-.2))])
        self.assertEqual(units, [])
        self.assertEqual(len(bad), 1)

    def test_duplicate_wallet_event_not_counted_twice(self):
        units, _ = cash_units([('a',row(10,-.1)), ('a',row(10,-.1))])
        self.assertEqual(units[0]['observations'], 1)

    def test_strict_pre_settlement_position_and_receipt_haircut(self):
        units, _ = cash_units([('a',row(10,-.1))])
        old = [dict(asset=0, time=0., signed_qty=-10.)]
        result = dict(fills=[dict(asset=0, time=3600., signed_qty=100.)])
        r = path_cash_sensitivity(result, old, units, ['BTC'], 0, 3600, .5,
                                  [dict(time=3600000,coin='pas:BTC',fundingRate='.0001')])
        self.assertAlmostEqual(r['funding'], .05)
        self.assertFalse(r['policy_feedback_replayed'])

    def test_missing_settlement_is_not_zero_cash(self):
        r = path_cash_sensitivity(dict(fills=[]), [dict(asset=0,time=0.,signed_qty=10.)],
             [], ['BTC'], 0, 3600, 1., [dict(time=3600000,coin='pas:BTC',fundingRate='.0001')])
        self.assertFalse(r['complete'])
        self.assertIsNone(r['funding'])


if __name__ == '__main__':
    unittest.main()
