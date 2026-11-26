import copy,json,unittest
from pathlib import Path
import numpy as np
from strategy.agent import TAIL
from experiments.par_direct_forecast_20261017.data import ts
from .engine import TargetState,aggregate,map_shadow,replay

HERE=Path(__file__).resolve().parent


class NettingTests(unittest.TestCase):
    def setUp(self):
        self.cfg=json.loads((HERE/'protocol.json').read_text())
        self.cfg['scenarios']={'base':dict(fee_bps=9,impact_bps=10,participation=.05,
            adverse_funding_bps_hour=0,funding_receipt_multiplier=1)}
        self.p=json.loads((TAIL/'baseline.json').read_text())['protocol']
        t=np.arange(ts('2026-10-08T12:00:00'),ts('2026-10-10T12:00:00')+300,300);n=len(t)
        self.d=dict(times=t,par_close=np.full((n,10),100.),reference_close=np.full((n,10),90.),
            par_volume=np.full((n,10),10000.),par_trades=np.full((n,10),50.),funding=np.zeros((n,10)),
            funding_observed=np.tile((t%3600==0)[:,None],(1,10)),sigma=np.tile(np.eye(10)*1e-7,(n,1,1)))
        self.base=np.zeros((n,10));self.pair=np.zeros((n,10))

    def run_account(self,share=.5):
        return replay(self.d,self.p,self.cfg,'base',int(self.d['times'][0]),int(self.d['times'][-1]),
            self.base,self.pair,share,self.d['times']//3600*3600)

    def test_opposing_targets_net_to_zero_with_no_virtual_pnl(self):
        self.base[:,0]=10;self.pair[:,0]=-10
        r=self.run_account();self.assertEqual(r['fills'],[]);self.assertEqual(r['metrics']['net_pnl'],0)
        self.assertFalse(r['metrics']['eligible'])

    def test_same_symbol_uses_single_inventory_and_single_funding_charge(self):
        self.base[:,0]=10;self.pair[:,0]=20
        self.d['funding'][self.d['times']%3600==0,0]=.0001
        r=self.run_account();self.assertEqual(len(r['fills']),1);self.assertEqual(r['fills'][0]['signed_qty'],15.)
        self.assertEqual(len(r['funding_events']),48)
        for e in r['funding_events']:self.assertAlmostEqual(e['signed_amount'],-15*90*.0001)
        self.assertAlmostEqual(r['metrics']['net_pnl'],-1.5-15*100.1*.0009-48*15*90*.0001,places=7)

    def test_unavailable_leg_is_not_treated_as_atomic_and_is_paid_unwind(self):
        self.pair[:,:2]=[10,-10];self.d['par_volume'][:,1]=0
        r=self.run_account(1.)
        self.assertEqual([f['signed_qty'] for f in r['fills']],[10.,-10.])
        self.assertGreater(r['metrics']['fees'],0);self.assertEqual(r['metrics']['residual_gross'],0)
        self.assertEqual(r['metrics']['suppressed_pair_events'],1)

    def test_exit_retries_after_zero_liquidity_beyond_deadline(self):
        self.pair[:12,:2]=[10,-10]
        self.d['par_volume'][12:24,:2]=0
        r=self.run_account(1.)
        exits=[f for f in r['fills'] if f['reduce_only']]
        self.assertEqual(len(exits),2);self.assertTrue(all(f['time']>=self.d['times'][25] for f in exits))
        self.assertEqual(r['metrics']['residual_gross'],0)

    def test_sign_reversal_pays_exit_before_new_entry(self):
        self.base[:12,0]=10;self.base[12:,0]=-10
        r=self.run_account(0.)
        a=r['fills'];self.assertEqual([f['signed_qty'] for f in a],[10.,-10.,-10.])
        self.assertTrue(a[1]['reduce_only']);self.assertFalse(a[2]['reduce_only'])
        self.assertGreaterEqual(a[2]['time'],a[1]['time']);self.assertTrue(all(f['fee']>0 for f in a))

    def test_actual_ledger_never_uses_future_source_or_same_bar_fill(self):
        self.base[:,0]=10;r=self.run_account(0.)
        self.assertTrue(all(f['source_available']<=f['decision_time']<f['time'] for f in r['fills']))
        bad=self.d['times']+300
        with self.assertRaises(AssertionError):
            replay(self.d,self.p,self.cfg,'base',int(self.d['times'][0]),int(self.d['times'][-1]),self.base,self.pair,0.,bad)

    def test_future_price_changes_cannot_change_earlier_trades(self):
        self.base[:,0]=10;a=self.run_account(0.)
        cut=288;self.d['par_close'][cut+1:,0]=200;b=self.run_account(0.)
        t=int(self.d['times'][cut])
        self.assertEqual([f for f in a['fills'] if f['time']<=t],[f for f in b['fills'] if f['time']<=t])
        self.assertEqual(a['snapshots'][:cut+1],b['snapshots'][:cut+1])

    def test_single_shadow_control_ignores_other_sleeve(self):
        self.base[:,0]=10;a=self.run_account(0.);self.pair[:,1]=10000;b=self.run_account(0.)
        self.assertEqual(a['fills'],b['fills']);self.assertEqual(a['snapshots'],b['snapshots'])

    def test_net_target_is_projected_inside_shared_risk_limits(self):
        self.base[:,:]=100000
        r=self.run_account(0.)
        self.assertTrue(r['fills']);self.assertEqual(r['metrics']['risk_breach_bars'],0)
        self.assertLess(r['metrics']['maximum_gross'],self.p['risk']['gross_fraction'])

    def test_source_mapping_is_asof_and_rejects_extrapolation(self):
        run={'snapshots':[{'time':100,'qty':[1,0]},{'time':200,'qty':[2,0]},{'time':300,'qty':[3,0]}]}
        q,t=map_shadow(np.array([100,150,200,250]),run)
        np.testing.assert_array_equal(q[:,0],[1,1,2,2]);np.testing.assert_array_equal(t,[100,100,200,200])
        shortened={'snapshots':run['snapshots'][:2]};a,_=map_shadow(np.array([100,150,200]),shortened)
        np.testing.assert_array_equal(a,q[:3])
        for invalid in ([99],[301]):
            with self.assertRaises(ValueError):map_shadow(np.array(invalid),run)

    def test_target_changes_do_not_postpone_completion_deadline(self):
        s=TargetState();z=np.zeros(2);p=np.array([10.,-10.]);px=np.ones(2)*100
        s.propose(0,z,z,p,1.,px,300,self.cfg);deadline=s.task['deadline']
        s.propose(300,z,z,p*2,1.,px,300,self.cfg);self.assertEqual(s.task['deadline'],deadline)
        target=s.propose(600,z,z,p*2,1.,px,300,self.cfg)
        np.testing.assert_array_equal(target,z);self.assertEqual(len(s.events),1)
        # Repeated snapshots of a failed source target cannot restart entry.
        np.testing.assert_array_equal(s.propose(900,z,z,p*2,1.,px,300,self.cfg),z)

    def test_capital_fractions_and_signed_quantity_netting(self):
        np.testing.assert_array_equal(aggregate([10,0],[-2,8],.5),[4,4])
        with self.assertRaises(ValueError):aggregate([10],[-10],1.1)


if __name__=='__main__':unittest.main()
