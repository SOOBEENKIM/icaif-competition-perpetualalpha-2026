import unittest
from .replication_temporal_audit import audit_path,audit_actual,START


class ReplicationTemporalTests(unittest.TestCase):
    def test_every_same_second_cash_order_is_flagged(self):
        r=dict(orders=[dict(symbol='BTC',arrival=START+.8,filled=1.)],
            funding=[dict(time=START*1000+10,delta=dict(coin='pas:BTC'))])
        self.assertEqual(len(audit_path(r)),1)
        r['orders'][0]['arrival']=START+1.8
        self.assertEqual(audit_path(r),[])

    def test_unfilled_order_cannot_change_settlement_inventory(self):
        r=dict(orders=[dict(symbol='BTC',arrival=START+.8,filled=0.)],
            funding=[dict(time=START*1000+10,delta=dict(coin='pas:BTC'))])
        self.assertEqual(audit_path(r),[])

    def test_actual_old_precision_is_never_guessed(self):
        r=dict(fills=[dict(symbol='BTC',time=START)],
            funding_events=[dict(coin='pas:BTC',time=START+.012)])
        self.assertEqual(len(audit_actual(r)),1)
        r['fills'][0].update(time=START+.8,time_ms=START*1000+800)
        self.assertEqual(audit_actual(r),[])


if __name__=='__main__':unittest.main()
