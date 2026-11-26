import copy
import json
import unittest
from pathlib import Path
import numpy as np
from experiments.inventory_renewal_pairs_20261017.test_renewal import RenewalTests
from experiments.paid_rehedge_pairs_20261017.portfolio import replay as original_replay
from experiments.paid_rehedge_pairs_20261017.policy import renewal_plan as original_plan
from experiments.shared_execution_pairs_20261017.account_v2 import SharedAccount
from experiments.shared_execution_pairs_20261017.test_account import AccountTests
from .portfolio import replay
from .policy import anchor_stop, held_state, renewal_plan, reconcile_adjustment, variance_charge


class RetainedSizingTests(unittest.TestCase):
    def setUp(self):
        f=RenewalTests();f.setUp();self.d,self.de,self.cfg,self.p=f.d,f.de,f.cfg,f.p
        folder=Path(__file__).parent
        self.cfg['rehedge_policy']=json.loads((folder.parent/'paid_rehedge_pairs_20261017/par_protocol.json').read_text())['rehedge_policy']
        spec=json.loads((folder/'protocol.json').read_text())
        self.cfg['retained_sizing']={k:spec[k] for k in ['gross_fractions','utility_buffer_bps']}
        self.cfg['retained_sizing']['risk_aversion']=64.
        self.t=self.d['times'];self.change=int(self.t[0])+18*3600;self.i=int(np.searchsorted(self.t,self.change))
        self.d['sigma'][:]=np.eye(10)*.0001
        self.de['mu'][self.t>=self.change]=.004
        self.p['initial']=10000.

    def direct(self):
        a=SharedAccount(10000.,self.p['symbols']);q=np.zeros(10);q[[1,0]]=[5.,-5.]
        for j in [1,0]:
            a.physical_fill(j,q[j],100.,9,int(self.t[0])+300)
            a.owner_fill('pair',j,q[j],100.,9,int(self.t[0])+300,'external_fill')
        pair=dict(renewable=True,dense=True,balanced=True,paid_rehedge_enabled=True,utility_renewal_enabled=True,
                  ids=[1,0],pair=['ETH','BTC'],pair_index=0,target=[5.,-5.],direction=1,
                  forecast_source=int(self.t[0]),horizon=24,first_fill=int(self.t[0])+300,
                  entry_vol=.002,risk_entry_prices=[100.,100.],risk_quantities=[5.,-5.])
        anchor_stop(pair,a.owners['pair'],self.d['par_close'][self.i]);return pair,a

    def plan(self,pair,a):
        return renewal_plan(pair,a.owners['pair'].qty,self.i,self.d,self.de,self.cfg,self.p,
             self.cfg['scenarios']['base'],a.physical.qty,a.physical.equity(self.d['par_close'][self.i]),ledger=a.owners['pair'])

    def run_arm(self,arm='pair_dense_utility64_bias_24',engine=replay):
        return engine(self.d,self.de,None,self.cfg,self.p,arm,'base',int(self.t[0]),int(self.t[-1]))

    def test_existing_controls_unchanged(self):
        for arm in ['first_strategy','second_strategy','first_dense_bias_24','pair_dense_renewhedge_bias_24']:
            a=self.run_arm(arm);b=self.run_arm(arm,original_replay)
            for k in ['fills','events','snapshots','metrics']:self.assertEqual(a[k],b[k],(arm,k))

    def test_hourly_covariance_and_baseline_cross_term(self):
        marks=np.array([100.,100.]);sigma=np.array([[.0001,.00004],[.00004,.0001]])
        owned=np.array([5.,-5.]);target=np.array([2.5,-2.5]);total=np.array([8.,-3.])
        b=(total-owned)*marks;v=target*marks
        expected=64*24*((b+v)@sigma@(b+v)-b@sigma@b)/(2*10000)
        got=variance_charge(target,owned,total,[0,1],marks,sigma,10000,24,64)
        self.assertAlmostEqual(got,expected)
        self.assertAlmostEqual(variance_charge(target,owned,total,[0,1],marks,sigma,10000,48,64),2*got)

    def test_weak_forecast_shrinks_and_limits_keep_cash_and_utility_buffer(self):
        pair,a=self.direct();plan=self.plan(pair,a)
        self.assertTrue(plan['requires_adjustment']);self.assertTrue(plan['utility_enabled'])
        self.assertAlmostEqual(plan['target_gross'],500.)
        self.assertGreater(plan['adjustment_cost_preview'],0)
        self.assertGreaterEqual(plan['minimum_advantage_at_limits'],plan['buffer_cash']-1e-8)
        self.assertGreaterEqual(plan['minimum_utility_advantage_at_limits'],plan['utility_buffer_cash']-1e-8)

    def test_unchanged_strong_forecast_does_not_manufacture_trade(self):
        pair,a=self.direct();self.de['mu'][self.i]=.02
        plan=self.plan(pair,a);probe=dict(pair);probe.pop('utility_renewal_enabled')
        old=original_plan(probe,a.owners['pair'].qty,self.i,self.d,self.de,self.cfg,self.p,
             self.cfg['scenarios']['base'],a.physical.qty,a.physical.equity(self.d['par_close'][self.i]),ledger=a.owners['pair'])
        self.assertEqual(plan,old);self.assertFalse(plan.get('requires_adjustment',False))

    def test_restore_reduced_inventory_is_capped_at_initial_budget(self):
        pair,a=self.direct()
        for j,z in [(1,-2.5),(0,2.5)]:
            a.physical_fill(j,z,100.,9,self.change-300)
            a.owner_fill('pair',j,z,100.,9,self.change-300,'external_fill')
        pair.update(target=[2.5,-2.5],has_rehedged=True)
        self.de['mu'][self.i]=.02
        plan=self.plan(pair,a)
        self.assertTrue(plan['requires_adjustment']);self.assertAlmostEqual(plan['current_gross'],500.)
        self.assertAlmostEqual(plan['target_gross'],1000.)
        self.assertEqual(plan['original_balanced_gross'],1000.)

    def test_flat_is_sticky_paid_close_not_fake_adjusted_inventory(self):
        self.d['sigma'][:]=np.eye(10)*.0004;self.p['risk']['daily_volatility']=.1
        self.de['mu'][self.t>=self.change]=.001
        r=self.run_arm();event=next(e for e in r['events'] if e['event']=='utility_close_requested')
        closed=next(e for e in r['events'] if e['event']=='closed')
        self.assertGreater(closed['time'],event['time']);self.assertGreater(r['metrics']['fees'],0)
        self.assertFalse(any(e['event']=='renewed' and event['time']<e['time']<closed['time'] for e in r['events']))

    def test_zero_fill_keeps_original_maturity(self):
        pair,a=self.direct();plan=self.plan(pair,a);pair.update(rehedging=True,rehedge_plan=plan)
        out=reconcile_adjustment(pair,a.owners['pair'],self.i+1,self.d,self.de,self.cfg,self.p,
            self.cfg['scenarios']['base'],a.physical.qty,9999.1,all_outcomes_observed=True)
        self.assertEqual(out['outcome'],'unfilled');self.assertNotIn('renewal_maturity',pair)

    def test_successful_reduction_preserves_cumulative_stop_anchor(self):
        pair,a=self.direct();plan=self.plan(pair,a);origin=pair['stop_price_origin'];pair.update(rehedging=True,rehedge_plan=plan)
        for j,z,px in zip(pair['ids'],plan['delta'],[99.9,100.1]):
            a.physical_fill(j,z,px,9,self.change+300)
            a.owner_fill('pair',j,z,px,9,self.change+300,'external_fill')
        out=reconcile_adjustment(pair,a.owners['pair'],self.i+1,self.d,self.de,self.cfg,self.p,
            self.cfg['scenarios']['base'],a.physical.qty,a.physical.equity(self.d['par_close'][self.i+1]),all_outcomes_observed=True)
        self.assertEqual(out['outcome'],'renewed');self.assertEqual(pair['stop_price_origin'],origin)
        self.assertEqual(pair['stop_initial_gross'],1000.)
        self.assertGreater(out['arrival_utility_advantage'],out['arrival_utility_buffer'])
        self.assertLess(a.reconcile(self.d['par_close'][self.i+1]),10000.)

    def test_adverse_arrival_risk_forces_paid_unwind(self):
        pair,a=self.direct();plan=self.plan(pair,a);pair.update(rehedging=True,rehedge_plan=plan)
        for j,z in zip(pair['ids'],plan['delta']):
            a.physical_fill(j,z,100.,9,self.change+300)
            a.owner_fill('pair',j,z,100.,9,self.change+300,'external_fill')
        self.d['sigma'][self.i+1]=np.eye(10)*.0004;self.p['risk']['daily_volatility']=.1
        out=reconcile_adjustment(pair,a.owners['pair'],self.i+1,self.d,self.de,self.cfg,self.p,
            self.cfg['scenarios']['base'],a.physical.qty,9999.,all_outcomes_observed=True)
        self.assertEqual(out['outcome'],'unwind');self.assertEqual(out['reason'],'arrival_retained_utility_insufficient')
        self.assertTrue(pair['closing']);self.assertTrue(np.any(a.owners['pair'].qty))

    def book_case(self,missing=False):
        pair,a=self.direct();plan=self.plan(pair,a);pair.update(rehedging=True,rehedge_plan=plan)
        target=a.owners['pair'].qty.copy();target[pair['ids']]=plan['target']
        sc=dict(self.cfg['scenarios']['base'],depth_fraction=1.,latency_ms=1000)
        packet=a.seal(dict(decision_time=self.change,source_close=self.change,source_available=self.change,
            source_prices=self.d['par_close'][self.i].tolist(),risk_forced=False,
            owner_targets={'base':np.zeros(10).tolist(),'pair':target.tolist()},owner_limits={'pair':plan['entry_limits']}),self.p,sc)
        book=AccountTests().book
        streams={'ETH':book(at=self.change*1000+1000,size=10.)}
        if not missing:streams['BTC']=book(at=self.change*1000+1000,size=.5)
        limits=dict(book_max_age_ms=15000,context_max_age_ms=3000,future_exit_observation_grace_ms=5000)
        a.execute_books(packet,streams,self.d['par_close'][self.i],self.d['sigma'][self.i],self.p,sc,limits)
        out=reconcile_adjustment(pair,a.owners['pair'],self.i+1,self.d,self.de,self.cfg,self.p,sc,
            a.physical.qty,a.physical.equity(self.d['par_close'][self.i+1]),all_outcomes_observed=not a.unknown)
        return pair,a,out,packet,sc,limits

    def test_received_depth_does_not_make_reduction_atomic(self):
        pair,a,out,_,_,_=self.book_case()
        self.assertEqual(out['outcome'],'unwind');self.assertTrue(pair['closing'])
        self.assertNotEqual(abs(out['actual_delta'][0]),abs(out['actual_delta'][1]))
        self.assertGreater(a.physical.fees,0);a.reconcile(self.d['par_close'][self.i+1])

    def test_unknown_book_outcome_keeps_pending_and_blocks_duplicate(self):
        pair,a,out,packet,sc,cfg=self.book_case(missing=True)
        self.assertEqual(out['outcome'],'awaiting_reconciliation');self.assertIn('rehedge_plan',pair)
        with self.assertRaises(RuntimeError):
            a.execute_books(packet,{},self.d['par_close'][self.i],self.d['sigma'][self.i],self.p,sc,cfg)

    def test_future_changes_cannot_change_prior_decisions_or_fills(self):
        a=self.run_arm();cut=self.change+3600
        self.d['par_close'][self.t>cut]*=1.2;self.d['sigma'][self.t>cut]*=10;self.de['mu'][self.t>cut]*=-2
        b=self.run_arm()
        for key in ['events','fills','snapshots']:
            self.assertEqual([e for e in a[key] if e['time']<=cut],[e for e in b[key] if e['time']<=cut],key)


if __name__=='__main__':unittest.main()
