import copy
import unittest

import numpy as np

from bot.execution import OutcomeUnknown
from . import test_books as fixtures
from .timestamp_ledger import install_in_copy,ambiguous_funding_records,MillisecondLedger


class TimestampLedgerTests(unittest.TestCase):
    def example(self,corrected):
        _,p,t=fixtures.RecordedBookTests().case()
        if corrected:install_in_copy(p)
        for l in [p.account.physical,p.account.owners['base']]:
            l.fill(0,1.,100.,0.,t-1)
            l.fill(0,1.,100.,0.,t+.8)
        return p,t

    def test_original_same_second_bug_is_reproduced(self):
        p,t=self.example(False)
        with self.assertRaisesRegex(OutcomeUnknown,'settled fill history'):
            p.account.apply_actual_funding([dict(time=round((t+.1)*1000),
                delta=dict(coin='pas:BTC',szi='1',usdc='-.1'))])

    def test_after_payment_fill_does_not_receive_preceding_funding(self):
        p,t=self.example(True)
        row=dict(time=round((t+.1)*1000),delta=dict(coin='pas:BTC',szi='1',usdc='-.1'))
        p.account.apply_actual_funding([row]);p.account.apply_actual_funding([row])
        self.assertAlmostEqual(p.account.physical.funding,-.1)
        self.assertAlmostEqual(p.account.owners['base'].funding,-.1)
        self.assertEqual(p.account.physical.fills[-1]['time_ms'],round((t+.8)*1000))
        self.assertAlmostEqual(p.account.physical.reconcile(np.full(10,100.)),49999.9)

    def test_same_millisecond_order_uses_existing_strict_prior_rule(self):
        p,t=self.example(True)
        p.account.apply_actual_funding([dict(time=round((t+.8)*1000),
            delta=dict(coin='pas:BTC',szi='1',usdc='-.1'))])
        self.assertAlmostEqual(p.account.physical.funding,-.1)

    def test_adaptation_preserves_cash_positions_and_old_unknown_precision(self):
        p,t=self.example(False);old=copy.deepcopy(p.account.physical)
        install_in_copy(p)
        np.testing.assert_array_equal(p.account.physical.qty,old.qty)
        self.assertEqual(p.account.physical.shadow_cash,old.shadow_cash)
        self.assertEqual(p.account.physical.fills,old.fills)
        p.account.physical.funding_events=[dict(coin='pas:BTC',time=t+.1)]
        self.assertEqual(len(ambiguous_funding_records(p.account.physical)),1)

    def test_invalid_time_cannot_partially_mutate_ledger(self):
        l=MillisecondLedger(50000.,['BTC'])
        for bad in [float('nan'),float('inf'),-1.,1.0001]:
            with self.assertRaises(ValueError):l.fill(0,1,100.,0.,bad)
        self.assertEqual(l.fills,[]);self.assertEqual(l.shadow_cash,50000.)


if __name__=='__main__':unittest.main()
