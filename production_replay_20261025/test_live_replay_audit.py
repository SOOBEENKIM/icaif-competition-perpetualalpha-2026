import copy,unittest
import numpy as np
from replay_lab.common_comparison.engine import Ledger
from .live_replay_audit import ledger_value


class ActualLedgerAuditTests(unittest.TestCase):
    def test_reconstruction_matches_real_ledger_and_excludes_future_fills(self):
        ledger=Ledger(50000.,['BTC','ETH'])
        ledger.fill(0,2.,100.,9.,1)
        ledger.fill(0,-.5,110.,9.,2)
        ledger.fill(1,-1.,200.,9.,2)
        ledger.cash-=.5;ledger.shadow_cash-=.5;ledger.funding-=.5
        ledger.funding_events.append(dict(time=2.012,signed_amount=-.5))
        marks=np.array([105.,190.]);expected=ledger.reconcile(marks)
        ledger.fill(0,5.,2000.,9.,10)
        got=ledger_value(ledger,marks,3.)
        self.assertAlmostEqual(got['equity'],expected)
        self.assertEqual(got['qty'],[1.5,-1.])
        self.assertEqual(got['funding'],-.5)

    def test_exact_boundary_rounding_is_reported(self):
        ledger=Ledger(50000.,['BTC']);ledger.fill(0,1.,100.,9.,3)
        self.assertEqual(ledger_value(ledger,np.array([100.]),3)['boundary_fill_timestamp_ambiguities'],1)


if __name__=='__main__':unittest.main()
