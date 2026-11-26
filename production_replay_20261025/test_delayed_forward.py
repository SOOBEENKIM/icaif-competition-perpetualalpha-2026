import unittest
from types import SimpleNamespace
from unittest.mock import Mock,patch

from bot.execution import OrderRejected
from candidate_handoff.risk_priority import DeadlineGateway
from .delayed_forward_review import delayed_submit


class DelayedSubmissionTests(unittest.TestCase):
    def venue(self,fee=9.):
        v=SimpleNamespace(now=100.,end=1000.,sc={'fee_bps':fee},account=object(),settle=Mock(),gateway=Mock())
        v.gateway.send.return_value='receipt'
        return v

    def test_funding_clock_advances_before_transmission(self):
        v=self.venue();order=object()
        def send(o):
            self.assertEqual(v.now,190.)
            v.settle.assert_called_once_with(190000,v.account)
            self.assertIs(o,order)
            return 'receipt'
        v.gateway.send.side_effect=send
        self.assertEqual(delayed_submit(v,order),'receipt')

    def test_deadline_expires_during_reconciliation_without_transmission(self):
        v=self.venue();sender=v.gateway
        v.gateway=DeadlineGateway(sender,180.)
        with patch('candidate_handoff.risk_priority.time.time',side_effect=lambda:v.now):
            with self.assertRaises(OrderRejected):delayed_submit(v,object())
        sender.send.assert_not_called()

    def test_fixed_end_is_not_crossed(self):
        v=self.venue();v.end=150.
        with self.assertRaises(OrderRejected):delayed_submit(v,object())
        self.assertEqual(v.now,150.)
        v.gateway.send.assert_not_called()

    def test_stress_delays_each_serial_order(self):
        v=self.venue(12.)
        delayed_submit(v,object());delayed_submit(v,object())
        self.assertEqual(v.now,340.)


if __name__=='__main__':unittest.main()
