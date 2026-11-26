import unittest
import numpy as np
from .policy import AdjustmentController
from .metrics import official_metrics


class PolicyTests(unittest.TestCase):
    def setUp(self):
        self.p={'risk':dict(gross_fraction=.6,net_fraction=.4,asset_fraction=.15,asset_notional=150000,
            daily_volatility=.005,sizing_headroom=.9,normal_turnover_fraction=.15),
            'execution':dict(decimals=[4,4])}
        self.prices=np.array([100.,100.]);self.cov=np.eye(2)*.000001

    def call(self,c,t=43200,qty=(0.,0.),raw=(.04,0.),retention=(1.,1.),halted=False,valid=True):
        return c.propose(t,self.prices,np.array(qty),1e6,np.array(raw),self.cov,
            np.ones(2)*1e5,np.ones(2),self.p,buffer=.2,retention=np.array(retention),halted=halted,valid_input=valid)

    def test_partial_entry_uses_actual_holdings_after_no_fill(self):
        c=AdjustmentController('partial_6h')
        a=self.call(c);b=self.call(c,t=64800)
        self.assertEqual(a['target_qty'],[200.,0.])
        self.assertEqual(b['target_qty'],a['target_qty'])

    def test_risk_cap_is_not_delayed_by_smoothing(self):
        c=AdjustmentController('partial_6h')
        a=self.call(c,qty=(400.,0.),raw=(.02,0.),retention=(.5,1.))
        self.assertLessEqual(a['target_qty'][0],200.)

    def test_daily_policy_blocks_only_increases_outside_daily_boundary(self):
        a=self.call(AdjustmentController('partial_24h'),t=21600)
        self.assertEqual(a['target_qty'],[0.,0.])
        b=self.call(AdjustmentController('partial_24h'),t=21600,qty=(400.,0.),raw=(0.,0.))
        self.assertEqual(b['target_qty'],[0.,0.])
        self.assertEqual(self.call(AdjustmentController('partial_24h'))['target_qty'],[200.,0.])

    def test_hard_risk_reduction_ignores_schedule_and_smoothing(self):
        a=self.call(AdjustmentController('partial_24h'),t=3600,qty=(2000.,0.))
        self.assertTrue(a['risk_forced']);self.assertLessEqual(a['target_qty'][0],1350.)
        b=self.call(AdjustmentController('partial_24h'),t=3600,qty=(2000.,0.),halted=True)
        self.assertEqual(b['target_qty'],[0.,0.])

    def test_restart_restores_cursor_and_blocks_duplicate(self):
        c=AdjustmentController('band_edge');a=self.call(c)
        d=AdjustmentController('band_edge',c.snapshot())
        self.assertIsNone(self.call(d))
        self.assertEqual(a['target_qty'],[320.,0.])
        with self.assertRaises(ValueError):AdjustmentController('partial_6h',c.snapshot())

    def test_invalid_input_does_not_consume_decision(self):
        c=AdjustmentController('partial_6h');self.assertIsNone(self.call(c,valid=False))
        self.assertIsNotNone(self.call(c))

    def test_short_risk_caps_and_lot_rounding(self):
        a=self.call(AdjustmentController('band_edge'),qty=(-400.,0.),raw=(-.020001,0.),retention=(.5,1.))
        self.assertGreaterEqual(a['target_qty'][0],-200.01)
        self.assertLess(a['target_qty'][0],0)

    def test_official_boundary_excludes_midnight_noise(self):
        r=dict(snapshots=[dict(time=43200,equity=100.),dict(time=86400,equity=900.),
            dict(time=129600,equity=101.),dict(time=172800,equity=1.),dict(time=216000,equity=100.)],
            fills=[dict(time=46800)])
        out=official_metrics(r)
        self.assertEqual(out['completed_days'],2);self.assertEqual(out['return_fraction'],0.)
        self.assertAlmostEqual(out['daily_sample_std'],np.std([.01,100/101-1],ddof=1))
        self.assertTrue(out['eligible'])


if __name__=='__main__':unittest.main()
