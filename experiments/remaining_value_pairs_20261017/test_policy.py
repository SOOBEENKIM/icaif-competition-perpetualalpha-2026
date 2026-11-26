import copy
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from experiments.joint_leg_pairs_20261017.model import fit, rolling, targets_and_units
from .model import map_remaining, terminal_for_maturity
from .policy import remaining_check, renewal_plan, reconcile_adjustment


class RemainingTests(unittest.TestCase):
    def fixture(self):
        d = dict(times=np.array([0,3600]), par_close=np.array([[100.,50.],[100.,50.]]),
                 reference_close=np.array([[100.,50.],[100.,50.]]), known_hourly_funding=np.zeros((2,2)))
        state = dict(signal_times=np.zeros((2,1), dtype=int), holding_horizons=np.array([6,24]),
                     holding_terminal=np.tile(np.array([[[[90.,60.],[90.,60.]]]]),(2,1,1,1)),
                     holding_available=np.ones((2,1,2), bool), holding_fit_times=np.zeros((2,1,2), int))
        pair = dict(ids=[0,1],pair_index=0,balanced=True,target=[1.,-2.])
        cfg = dict(remaining_value=dict(mode='remaining',decision_cadence_hours=6,exit_buffer_bps=2.),
                   renewal_policy=dict(incremental_buffer_bps=2.))
        p = dict(execution=dict(decimals=[2,2]))
        sc = dict(fee_bps=10.,impact_bps=0.)
        return pair,d,state,cfg,p,sc

    def test_actual_inventory_cash_and_no_sunk_entry_recharge(self):
        pair,d,de,cfg,p,sc=self.fixture()
        state=dict(reason=None,maturity=21600)
        a=remaining_check(pair,np.array([1.,-2.]),0,d,de,cfg,p,sc,state)
        # Price loss -30; future gross 210 costs .21; close-now gross 200 costs .20.
        self.assertAlmostEqual(a['holding_advantage'],-30.01)
        self.assertTrue(a['close'])
        pair['entry_fees']=100000.
        b=remaining_check(pair,np.array([1.,-2.]),0,d,de,cfg,p,sc,state)
        self.assertEqual(a,b)
        short=remaining_check(pair,np.array([-1.,2.]),0,d,de,cfg,p,sc,state)
        self.assertFalse(short['close'])
        self.assertAlmostEqual(short['holding_advantage'],29.99)

    def test_original_exit_and_inflight_inventory_take_priority(self):
        pair,d,de,cfg,p,sc=self.fixture()
        self.assertIsNone(remaining_check(pair,np.array([1.,-2.]),0,d,de,cfg,p,sc,dict(reason='hard_holding_limit',maturity=21600)))
        for name in ['closing','completing','repairing','rehedging']:
            self.assertIsNone(remaining_check(dict(pair,**{name:True}),np.array([1.,-2.]),0,d,de,cfg,p,sc,dict(reason=None,maturity=21600)))

    def test_horizon_exactness_freshness_and_future_fit(self):
        _,_,de,_,_,_=self.fixture()
        self.assertEqual(terminal_for_maturity(de,0,0,0,21600)['horizon_hours'],6)
        self.assertIsNone(terminal_for_maturity(de,0,0,0,21601))
        self.assertIsNone(terminal_for_maturity(de,0,0,0,25*3600))
        self.assertIsNone(terminal_for_maturity(de,0,0,3600,21600))
        de['holding_fit_times'][0,0,0]=1
        self.assertIsNone(terminal_for_maturity(de,0,0,0,21600))

    def test_known_funding_only_and_receipt_haircut(self):
        pair,d,de,cfg,p,sc=self.fixture()
        de['holding_terminal'][:]=[100.,50.]
        d['known_hourly_funding'][0]=[.01,0.]
        a=remaining_check(pair,np.array([1.,-2.]),0,d,de,cfg,p,sc,dict(reason=None,maturity=21600))
        self.assertAlmostEqual(a['holding_advantage'],-6.)
        d['known_hourly_funding'][1]=[-1000.,1000.]
        self.assertEqual(a,remaining_check(pair,np.array([1.,-2.]),0,d,de,cfg,p,sc,dict(reason=None,maturity=21600)))
        d['known_hourly_funding'][0]=[-.01,0.]
        b=remaining_check(pair,np.array([1.,-2.]),0,d,de,cfg,p,sc,dict(reason=None,maturity=21600))
        self.assertAlmostEqual(b['holding_advantage'],3.)

    def test_mapping_preserves_every_original_entry_field(self):
        baseline=dict(signal_times=np.array([[0],[3600]]),source_prices=np.array([[100.,50.],[101.,49.]]),
                      pairs=np.array([[0,1]]),mu=np.array([[.1],[.2]]),economic_action=np.array([[1],[-1]]))
        predictions=np.array([[[[.01,-.02]]],[[[.02,-.01]]]])
        result=map_remaining(np.array([0,3600]),np.array([0,3600]),predictions,np.zeros((2,1,1),int),baseline,[6])
        for key in baseline:np.testing.assert_array_equal(result[key],baseline[key])
        np.testing.assert_allclose(result['holding_terminal'][0,0,0],[101.,49.])
        result['mu'][0]=0
        self.assertEqual(baseline['mu'][0,0],.1)

    def test_renewal_veto_uses_worst_sealed_cost(self):
        pair,d,de,cfg,p,sc=self.fixture();de['holding_terminal'][:]=[103.,49.]
        plan=dict(accepted=True,requires_adjustment=True,target=[2.,-4.],entry_limits={'0':120.,'1':40.},
                  new_maturity=86400,old_maturity=21600,buffer_cash=.04)
        with patch('experiments.remaining_value_pairs_20261017.policy.original_renewal',return_value=plan):
            got=renewal_plan(pair,np.array([1.,-2.]),0,d,de,cfg,p,sc,np.array([1.,-2.]),1e6)
        self.assertFalse(got['accepted'])
        self.assertNotIn('requires_adjustment',got)
        self.assertGreater(got['joint_hold_cash'],0)
        self.assertGreater(got['joint_worst_adjustment_cost'],40.)

    def test_unavailable_and_disabled_models_preserve_original(self):
        pair,d,de,cfg,p,sc=self.fixture()
        plan=dict(accepted=True,new_maturity=86400)
        with patch('experiments.remaining_value_pairs_20261017.policy.original_renewal',return_value=plan):
            cfg['remaining_value']['mode']=''
            self.assertIs(renewal_plan(pair,np.array([1.,-2.]),0,d,de,cfg,p,sc,np.zeros(2),1e6),plan)
            cfg['remaining_value']['mode']='renewal';de['holding_available'][:]=False
            got=renewal_plan(pair,np.array([1.,-2.]),0,d,de,cfg,p,sc,np.zeros(2),1e6)
            self.assertTrue(got['accepted']);self.assertFalse(got['remaining_model_available'])

    def test_arrival_unknown_blocks_and_filled_cost_is_sunk(self):
        pair,d,de,cfg,p,sc=self.fixture()
        pair['rehedge_plan']=dict(joint_terminal_prices=[101.,49.],new_maturity=86400,joint_worst_adjustment_cost=999.)
        ledger=SimpleNamespace(qty=np.array([1.,-2.]))
        unknown=dict(outcome='awaiting_reconciliation')
        with patch('experiments.remaining_value_pairs_20261017.policy.original_reconcile',return_value=unknown):
            self.assertIs(reconcile_adjustment(pair,ledger,0,d,de,cfg,p,sc,np.zeros(2),1e6,all_outcomes_observed=False),unknown)
        with patch('experiments.remaining_value_pairs_20261017.policy.original_reconcile',return_value=dict(outcome='renewed')):
            got=reconcile_adjustment(pair,ledger,0,d,de,cfg,p,sc,np.zeros(2),1e6,all_outcomes_observed=True)
        self.assertEqual(got['outcome'],'renewed');self.assertGreater(got['joint_arrival_advantage'],0)
        pair['rehedge_plan']['joint_terminal_prices']=[90.,60.]
        with patch('experiments.remaining_value_pairs_20261017.policy.original_reconcile',return_value=dict(outcome='renewed')):
            got=reconcile_adjustment(pair,ledger,0,d,de,cfg,p,sc,np.zeros(2),1e6,all_outcomes_observed=True)
        self.assertEqual(got['outcome'],'unwind');self.assertTrue(pair['closing'])

    def test_unmatured_training_label_rejected(self):
        spec=dict(minimum_training_rows=2,half_life_days=30,ridge_penalty=1,intercept_prior_rows=12,clip=4)
        with self.assertRaises(ValueError):fit(np.ones((2,3)),np.zeros((2,2)),np.array([1,2]),2,spec)

    def test_future_prices_cannot_change_issued_forecasts(self):
        rng=np.random.default_rng(4);times=np.arange(500)*3600
        prices=100*np.exp(np.cumsum(rng.normal(0,.01,(500,2)),axis=0))
        features=dict(x=rng.normal(size=(500,1,3)),fresh=np.ones((500,1),bool),pairs=np.array([[0,1]]))
        spec=dict(horizon=6,minimum_training_rows=6,half_life_days=30,ridge_penalty=1.,
                  intercept_prior_rows=12,clip=4,refit_hours=24,sample_hours=6,lookback_days=90)
        ref=dict(market_times=times,close=prices)
        y,m,u,t=targets_and_units(ref,features,6)
        a,_,records=rolling(times,features,y,m,u,t,spec)
        cut=360
        altered=copy.deepcopy(ref);altered['close'][cut+1:]*=np.linspace(2,4,len(times)-cut-1)[:,None]
        future=copy.deepcopy(features);future['x'][cut+1:]*=100
        y,m,u,t=targets_and_units(altered,future,6)
        b,_,_=rolling(times,future,y,m,u,t,spec)
        np.testing.assert_array_equal(a[:cut+1],b[:cut+1])
        self.assertTrue(records)
        self.assertTrue(all(r['last_maturity']<r['fit_time'] for r in records))


if __name__=='__main__':unittest.main()
