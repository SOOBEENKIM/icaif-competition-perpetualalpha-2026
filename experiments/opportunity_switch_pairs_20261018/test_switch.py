import copy
import json
import unittest
import numpy as np
from experiments.paid_rehedge_pairs_20261017 import test_rehedge as fixture
from experiments.decision_gate_pairs_20261017.portfolio import replay as previous_replay
from .policy import comparison,switch_plan,tighten_entry
from .portfolio import replay


class SwitchTests(unittest.TestCase):
    def setUp(self):
        f=fixture.PaidAdjustmentTests();f.setUp();self.p=copy.deepcopy(f.p);self.cfg=copy.deepcopy(f.cfg)
        n=25;self.t=1785585600+np.arange(n)*3600;shape=(n,10)
        self.d=dict(times=self.t,par_close=np.full(shape,100.),reference_close=np.full(shape,100.),
            par_volume=np.full(shape,1e6),par_trades=np.ones(shape),sigma=np.tile(np.eye(10)*1e-6,(n,1,1)),
            funding=np.zeros(shape),funding_observed=np.ones(shape,bool),known_hourly_funding=np.zeros(shape),
            first=np.zeros(shape),second=np.zeros(shape),retention=np.ones(n))
        self.de=dict(dense=np.array(True),pairs=np.array([[0,1],[0,2]]),
            signal_times=np.repeat(self.t[:,None],2,axis=1),directions=np.ones((n,2)),
            source_prices=np.full(shape,100.),beta=np.ones((n,2)),vol=np.full((n,2),.002),
            mu=np.column_stack([np.full(n,.02),np.r_[np.zeros(2),np.full(n-2,.08)]]),
            economic_available=np.ones((n,2),bool),economic_action=np.ones((n,2)))
        self.cfg.update(dense_pairs=[[self.p['symbols'][0],self.p['symbols'][1]],
                                    [self.p['symbols'][0],self.p['symbols'][2]]],
            pair_max_gross=1000.,signal_cadence_seconds=3600,entry_slippage_cap_bps=10,
            conditional_completion={'risk_aversion':0.},opportunity_switch={'minimum_holding_seconds':3600,'switch_slippage_cap_bps':10})
        self.cfg['dense_policy']['source_lifetime_hours']=1
        self.cfg['scenarios']['base']['participation']=.1

    def direct(self):
        pair=dict(dense=True,balanced=True,ids=[0,1],pair=self.cfg['dense_pairs'][0],pair_index=0,
            target=[5.,-5.],horizon=24,first_fill=int(self.t[1]),direction=1)
        qty=np.zeros(10);qty[:2]=[5.,-5.]
        return pair,qty

    def plan(self,pair=None,qty=None):
        if pair is None:pair,qty=self.direct()
        return switch_plan(pair,qty,2,self.d,self.de,self.cfg,self.p,self.cfg['scenarios']['base'],
                           qty.copy(),1e6,np.zeros(10),{}, {})

    def run_arm(self,engine=replay):
        return engine(self.d,self.de,None,self.cfg,self.p,'pair_dense_hourlyhedge_switch_24',
                      'base',int(self.t[0]),int(self.t[-1]))

    def test_costed_comparison_rejects_unfunded_switch(self):
        good=comparison(10,-2,13,100,100,10,5)
        bad=comparison(10,-2,11,100,100,10,5)
        self.assertTrue(good['accepted']);self.assertFalse(bad['accepted'])
        self.assertAlmostEqual(good['advantage'],.9)
        self.assertEqual(good['sunk_entry_costs_charged_again'],0)

    def test_switch_limit_never_widens_and_economic_value_is_preserved(self):
        self.cfg['entry_slippage_cap_bps']=10000
        plan=self.plan();self.assertTrue(plan['accepted'])
        proposal=plan['proposed_entry'];self.assertEqual(proposal['switch_limit_cap_bps'],10)
        self.assertGreater(proposal['minimum_expected_cash_at_limits'],proposal['buffer_cash'])
        again=tighten_entry(proposal,self.d['par_close'][2],self.p,self.cfg['scenarios']['base'],10)
        self.assertEqual(again['entry_limits'],proposal['entry_limits'])

    def test_future_market_and_forecast_values_cannot_change_current_plan(self):
        before=self.plan();self.assertTrue(before['accepted'])
        self.d['par_close'][3:]*=3;self.d['known_hourly_funding'][3:]=.2;self.de['mu'][3:]=-.7
        self.assertEqual(before,self.plan())

    def test_stale_held_or_different_maturity_cannot_be_compared(self):
        self.de['signal_times'][2,0]-=3600
        self.assertEqual(self.plan()['reason'],'held_forecast_unavailable_or_stale')
        self.de['signal_times'][2,0]+=3600;self.de['signal_times'][2,1]-=300
        self.assertEqual(self.plan()['reason'],'no_cost_qualified_alternative')

    def test_missing_source_close_capacity_cannot_be_assumed(self):
        self.d['par_volume'][2,1]=1
        self.assertEqual(self.plan()['reason'],'insufficient_source_close_capacity')

    def test_partial_close_risk_can_reject(self):
        self.p['risk']['net_fraction']=.0001
        self.assertEqual(self.plan()['reason'],'partial_close_global_risk')

    def test_busy_inventory_is_not_switched_and_identity_is_checked(self):
        pair,qty=self.direct();pair['completing']=True
        self.assertIsNone(self.plan(pair,qty))
        pair.pop('completing');pair['pair']=['WRONG','PAIR']
        with self.assertRaises(ValueError):self.plan(pair,qty)

    def test_disabled_policy_is_exact_old_engine(self):
        self.cfg.pop('opportunity_switch')
        def plain(x):
            return json.loads(json.dumps(x,default=lambda v:v.item() if isinstance(v,np.generic) else v.tolist()))
        self.assertEqual(plain(self.run_arm()),plain(self.run_arm(previous_replay)))

    def test_delayed_partial_close_must_finish_before_new_entry(self):
        self.d['par_volume'][3,1]=1
        r=self.run_arm();events=r['events']
        request=next(e for e in events if e['event']=='switch_exit_requested')
        flat=next(e for e in events if e['event']=='switch_flat_confirmed')
        self.assertGreaterEqual(flat['time']-request['time'],3*3600)
        self.assertFalse(any(e['event']=='pair_intent' and request['time']<e['time']<flat['time'] for e in events))
        recheck=next(e for e in events if e['event']=='switch_entry_rechecked')
        self.assertEqual(flat['time'],recheck['time'])
        snap=next(s for s in r['snapshots'] if s['time']==flat['time'])
        self.assertTrue(np.all(np.asarray(snap['pair_qty'])==0))
        later=next(e for e in events if e['event']=='pair_intent' and e['time']>=flat['time'])
        self.assertGreater(later['forecast_source'],request['proposed_entry']['forecast_source'])
        paid=[f for f in r['owner_fills']['pair'] if request['time']<f['time']<=flat['time']]
        self.assertTrue(paid);self.assertTrue(all(f['fee']>0 for f in paid))

    def test_opportunity_can_disappear_during_paid_close(self):
        self.d['par_volume'][3,1]=1
        self.de['economic_available'][3:]=False
        r=self.run_arm();flat=next(e for e in r['events'] if e['event']=='switch_flat_confirmed')
        self.assertFalse(any(e['event']=='pair_intent' and e['time']>=flat['time'] for e in r['events']))
        after=[s for s in r['snapshots'] if s['time']>=flat['time']]
        self.assertTrue(all(np.all(np.asarray(s['pair_qty'])==0) for s in after))
        self.assertGreater(r['metrics']['fees'],0)


if __name__=='__main__':unittest.main()
