import copy
import unittest
import numpy as np
from .policy import allocate
from .controller import Sleeves
from .controller import OpportunityController
from unittest.mock import patch
from .market import market_inputs
from .model import basis_adjustment,train,predict


class AllocationTests(unittest.TestCase):
    def args(self,n=1):
        return dict(base_qty=np.full(n,10.),old_extra=np.zeros(n),prices=np.full(n,100.),equity=10000.,
            covariance=np.eye(n)*1e-6,risk=dict(gross_fraction=.6,net_fraction=.4,asset_fraction=.15,
            asset_notional=7500.,daily_volatility=.05,sizing_headroom=.9),decimals=[3]*n,
            expected_return=np.full(n,.1),uncertainty=np.zeros(n),entry_cost=np.full(n,.0019),
            exit_cost=np.full(n,.0024),capacity=np.full(n,10000.),allowed=np.full(n,True),hold_allowed=np.full(n,True))

    def test_only_profitable_asset_expanded_and_short_symmetry(self):
        a=self.args(2);a['expected_return'][1]=-.02;r=allocate(**a)
        self.assertGreater(r['extra_qty'][0],0);self.assertEqual(r['extra_qty'][1],0)
        a=self.args();positive=allocate(**a);a['base_qty']*=-1;a['expected_return']*=-1
        self.assertAlmostEqual(positive['extra_qty'][0],-allocate(**a)['extra_qty'][0])

    def test_protection_and_closed_entry_do_not_allow_new_units(self):
        a=self.args();a['allowed'][:]=False;a['hold_allowed'][:]=False
        self.assertEqual(allocate(**a)['extra_qty'],[0.])

    def test_cost_and_uncertainty_can_remove_opportunity(self):
        for k in ['uncertainty','entry_cost','exit_cost']:
            a=self.args();a[k][:]=.2
            self.assertEqual(allocate(**a)['extra_qty'],[0.])

    def test_capacity_and_asset_caps_apply_after_rounding(self):
        a=self.args();a['capacity'][:]=10.;r=allocate(**a)
        self.assertLessEqual(abs(r['extra_qty'][0])*100,10.+1e-8)
        a=self.args();r=allocate(**a)
        self.assertLessEqual(r['target_qty'][0]*100,1350+1e-8)

    def test_negative_covariance_rejected_and_correlation_matters(self):
        a=self.args();a['covariance'][:]=-.1
        with self.assertRaises(ValueError):allocate(**a)
        a=self.args(2);a['risk']['daily_volatility']=.006
        a['covariance']=np.eye(2)*.00003
        separate=allocate(**a)
        a['covariance'][:]=.00003
        correlated=allocate(**a)
        self.assertLessEqual(sum(correlated['extra_qty']),sum(separate['extra_qty']))

    def test_base_is_never_changed_to_fund_extra(self):
        a=self.args(2);a['allowed'][0]=False;a['hold_allowed'][0]=False
        r=allocate(**a);self.assertEqual(r['target_qty'][0],a['base_qty'][0])

    def test_thesis_failure_closes_extra(self):
        a=self.args();a['old_extra'][:]=2.;a['expected_return'][:]=-.1
        self.assertEqual(allocate(**a)['extra_qty'],[0.])


class InventoryTests(unittest.TestCase):
    def test_partial_additions_and_reductions_reconcile(self):
        s=Sleeves(1);s.base_target[:]=10;s.extra_target[:]=5;s.last_intent=100
        s.reconcile([8]);np.testing.assert_array_equal(s.base,[8]);np.testing.assert_array_equal(s.extra,[0])
        s.reconcile([12]);np.testing.assert_array_equal(s.base,[10]);np.testing.assert_array_equal(s.extra,[2])
        self.assertEqual(s.opened[0],100)
        s.reconcile([9]);np.testing.assert_array_equal(s.base,[9]);np.testing.assert_array_equal(s.extra,[0])
        self.assertEqual(s.opened[0],-1)

    def test_short_reversal_and_close(self):
        s=Sleeves(1);s.base_target[:]=-10;s.extra_target[:]=-5;s.reconcile([-12])
        np.testing.assert_array_equal(s.base,[-10]);np.testing.assert_array_equal(s.extra,[-2])
        s.base_target[:]=3;s.reconcile([3]);np.testing.assert_array_equal(s.base,[3]);np.testing.assert_array_equal(s.extra,[0])
        s.reconcile([0]);np.testing.assert_array_equal(s.base+s.extra,[0])


class CausalityTests(unittest.TestCase):
    def dataset(self):
        rng=np.random.default_rng(21);n=24*120;times=np.arange(n)*3600
        close=np.exp(np.cumsum(rng.normal(0,.002,(n,2)),axis=0))*100
        return dict(market_times=times,close=close,volume=np.ones((n,2)))

    def test_future_changes_cannot_change_fit_or_past_prediction(self):
        d=self.dataset();end=24*70*3600;cal=24*100*3600
        fit=train(d,0,end,cal);a,b=predict(d,fit)
        other=copy.deepcopy(d);other['close'][other['market_times']>=cal]*=4
        self.assertEqual(fit,train(other,0,end,cal))
        aa,bb=predict(other,fit);cut=np.searchsorted(d['market_times'],cal)
        np.testing.assert_allclose(a[:cut],aa[:cut],equal_nan=True)
        self.assertLess(fit['last_training_maturity'],end);self.assertLess(fit['last_calibration_maturity'],cal)

    def test_constant_basis_does_not_create_convergence_profit(self):
        drift,se,n,rho=basis_adjustment(np.full(72,.1))
        self.assertEqual(drift,0);self.assertEqual(se,0);self.assertEqual(rho,1)

    def test_future_market_and_funding_observations_are_ignored(self):
        t=np.arange(1,180)*3600.;close=np.ones((len(t),1))*100
        d=dict(times=t*1000,close=close.copy(),high=close*1.001,low=close*.999,
               volume=np.ones_like(close),active=np.ones_like(close,bool),age_ms=np.zeros_like(close))
        ref=dict(market_times=t,close=close.copy(),volume=np.ones_like(close))
        funding=[[(float(v),.0001) for v in t]];at=t[100]
        a=market_inputs(at,[100],d,ref,funding)
        other=copy.deepcopy(d)
        for k in ['close','high','low','volume']:other[k][t>at]*=100
        ff=[[(v,rate if v+60<=at else -100) for v,rate in funding[0]]]
        b=market_inputs(at,[100],other,ref,ff)
        for k in a:np.testing.assert_equal(a[k],b[k])
        self.assertGreater(a['funding_24h'][0],0)


class ControllerTests(unittest.TestCase):
    def controller(self):
        class Base:
            times=np.array([0,21600,86400]);protected=np.zeros((3,1),bool)
            trace=[]
            forced=False
            def propose(self,*args,**kwargs):
                self.trace.append(dict(entry_filter_pass=[True]))
                return dict(target_qty=[10.],risk_forced=self.forced)
        base=Base();ref=dict(market_times=base.times)
        return OpportunityController(base,ref,np.full((3,1),.1),np.zeros((3,1)),{},[[]])

    def market(self):
        return dict(basis_log_change=np.array([0.]),basis_uncertainty=np.array([0.]),
                    funding_24h=np.array([0.]),fresh=np.array([True]),funding_ready=np.array([True]),
                    entry_cost=np.array([.0019]),exit_cost=np.array([.0024]),capacity_notional=np.array([10000.]))

    def call(self,c,t,qty):
        args=AllocationTests().args();protocol=dict(risk=args['risk'],execution=dict(decimals=[3]))
        return c.propose(t,np.array([100.]),np.array([qty]),10000.,np.array([.1]),
                         np.array([[1e-6]]),np.ones(1),np.ones(1),protocol)

    def test_extra_expiry_and_forced_reduction_override_positive_forecast(self):
        with patch('opportunity_sizing.controller.market_inputs',return_value=self.market()):
            c=self.controller();first=self.call(c,0,0);self.assertGreater(first['target_qty'][0],10)
            self.call(c,21600,12)
            self.assertEqual(self.call(c,86400,12)['target_qty'],[10.])
            c=self.controller();self.call(c,0,0);c.base.forced=True
            self.assertEqual(self.call(c,21600,12)['target_qty'],[10.])

    def test_protected_position_cannot_be_amplified(self):
        with patch('opportunity_sizing.controller.market_inputs',return_value=self.market()):
            c=self.controller();c.base.protected[:]=True
            self.assertEqual(self.call(c,0,0)['target_qty'],[10.])


if __name__=='__main__':unittest.main()
