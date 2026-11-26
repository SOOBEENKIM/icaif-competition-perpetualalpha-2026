import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
from experiments.cost_horizon_20261016.research import data,ts,SecondController
from experiments.performance_attribution_20261016.engine import simulate
from experiments.incumbent_pairs_20261017.model import fit,features
from .policy import IncumbentPairController,feasible_scale,cap_pair


class Invariants(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _,cls.arrays,cls.protocol,cls.market=data()
        cls.cfg=json.loads(Path(__file__).with_name('protocol.json').read_text())

    def test_turnover_feasible_interval_need_not_contain_zero(self):
        r=copy.deepcopy(self.protocol['risk']);r['normal_turnover_fraction']=.02
        units=np.array([.08,-.08]);qty=units*.9;prices=np.ones(2)
        scale=feasible_scale(np.zeros(2),units,[0,1],qty,prices,1.,np.eye(2)*1e-5,r)
        self.assertAlmostEqual(scale,1.)
        self.assertGreater(abs(qty).sum(),r['normal_turnover_fraction'])
        self.assertLessEqual(abs(units*scale-qty).sum(),r['normal_turnover_fraction'])

    def test_infeasible_transition_is_rejected(self):
        r=copy.deepcopy(self.protocol['risk']);r['normal_turnover_fraction']=.01
        scale=feasible_scale(np.zeros(2),np.array([.08,-.08]),[0,1],np.array([-.08,.08]),
                             np.ones(2),1.,np.eye(2)*1e-5,r)
        self.assertIsNone(scale)

    def test_one_leg_risk_cap_reduces_both_without_regrowing(self):
        actual=np.array([4.,-10.])
        out=cap_pair(actual,np.array([2.,100.]),np.array([True,False]))
        np.testing.assert_array_equal(out,[2.,-5.])
        np.testing.assert_array_equal(cap_pair(out,np.array([4.,100.]),np.array([True,False])),out)

    def controller(self,phase='holding'):
        n=100;m=SimpleNamespace(exact=lambda t:t//3600,times=np.arange(n)*3600)
        d=dict(pairs=[[0,1]],x=np.ones((n,1,1)),names=['correlation_168'],beta=np.ones((n,1)),vol=np.ones((n,1))*.01)
        v=dict(mode='standalone',model='opposed',horizon=24)
        c=IncumbentPairController(m,np.zeros((n,2)),np.ones((n,2)),d,np.zeros((n,1)),np.zeros((n,1)),self.cfg,v,{'impact_bps':10},99)
        c.states[0]=dict(phase=phase,planned=np.array([1.,-1.]),entry=4,entry_prices=np.array([100.,100.]),vol=.01)
        return c

    def call(self,c,i,qty):
        p=copy.deepcopy(self.protocol);p['execution']['decimals']=[4,4]
        return c.propose(i*3600,np.array([100.,100.]),np.array(qty),1e6,np.zeros(2),
                         np.eye(2)*1e-5,np.ones(2)*100,np.ones(2),p)

    def test_exit_decision_precedes_execution_horizon(self):
        c=self.controller()
        self.assertIsNone(self.call(c,26,[1.,-1.]))
        self.assertEqual(self.call(c,27,[1.,-1.])['target_qty'],[0.,0.])
        self.assertEqual(c.events[-1]['reason'],'planned_horizon')

    def test_single_leg_fill_is_closed_not_marked_balanced(self):
        c=self.controller('entering')
        self.assertEqual(self.call(c,4,[1.,0.])['target_qty'],[0.,0.])
        self.assertEqual(c.states[0]['phase'],'closing')
        self.assertEqual(c.events[-1]['event'],'unhedged_abort')

    def test_old_partial_risk_reduction_cannot_leave_broken_hedge(self):
        c=self.controller()
        self.assertEqual(self.call(c,12,[.2,-1.])['target_qty'],[0.,0.])
        self.assertEqual(c.states[0]['phase'],'closing')

    def test_inactive_overlay_exactly_matches_second_ledger(self):
        m,a,p=self.market,self.arrays,self.protocol;n=len(m.times)
        d=dict(pairs=[],x=np.empty((n,0,1)),names=['correlation_168'],beta=np.empty((n,0)),vol=np.empty((n,0)))
        w=a['candidate']*1.2958681757584685
        first,last=[m.exact(ts(t)) for t in ('2026-08-15T12:00:00','2026-08-17T12:00:00')]
        c=IncumbentPairController(m,w,a['retention'],d,np.empty((n,0)),np.empty((n,0)),self.cfg,
                                 dict(mode='overlay',model='opposed',horizon=24),p['scenarios']['base'],last)
        x=simulate(m,w,a['sigma'],p,first,last,'base',c)
        y=simulate(m,w,a['sigma'],p,first,last,'base',SecondController(m.times,a['retention']))
        self.assertEqual(x['fills'],y['fills'])
        self.assertEqual(x['snapshots'],y['snapshots'])

    def test_features_cannot_read_future_prices_or_model_outputs(self):
        m,a=self.market,self.arrays
        before=features(m,a,self.cfg)['calibration_features']
        split=m.exact(ts('2026-08-15T12:00:00'));mutated=copy.deepcopy(m);aa=copy.deepcopy(a)
        mutated.close[split+1:]*=np.linspace(1.,2.,10)
        for k in ('forecast','candidate','retention'):aa[k][split+1:]*=7
        after=features(mutated,aa,self.cfg)['calibration_features']
        np.testing.assert_allclose(before[:split+1],after[:split+1],equal_nan=True)

    def test_calibration_excludes_pre_model_training_and_unmatured_labels(self):
        times=np.arange(ts('2026-07-01'),ts('2026-09-01'),3600);n=len(times)
        cfg=copy.deepcopy(self.cfg);cfg['fit_dates']=['2026-08-15T12:00:00'];cfg['minimum_training_labels']=10
        rng=np.random.default_rng(7);x=rng.normal(size=(n,3,2));y=rng.normal(size=(n,3))
        maturity=times+25*3600
        d=dict(calibration_features=x,fresh=np.ones((n,3),bool),vol=np.ones((n,3))*.01,
               beta=np.ones((n,3)),x=np.ones((n,3,1)),names=['correlation_168'])
        at=ts(cfg['fit_dates'][0]);excluded=(times<ts(cfg['signal_training_cutoff']))|(maturity>=at)
        bady=y.copy();bady[excluded]=12345
        m=SimpleNamespace(times=times)
        with TemporaryDirectory() as folder,patch('experiments.incumbent_pairs_20261017.model.features',return_value=d):
            with patch('experiments.incumbent_pairs_20261017.model.labels',return_value=(y,maturity)):
                _,pred,_,records=fit(m,{},cfg,Path(folder)/'one')
            with patch('experiments.incumbent_pairs_20261017.model.labels',return_value=(bady,maturity)):
                _,changed,_,other=fit(m,{},cfg,Path(folder)/'two')
        np.testing.assert_allclose(pred,changed,equal_nan=True)
        self.assertGreaterEqual(records[0]['feature_first_time'],ts(cfg['signal_training_cutoff']))
        self.assertLess(records[0]['last_label_time'],at)
        self.assertEqual(records,other)

    def relative_entry(self,mu,stamp=3600):
        c=self.controller();c.states.clear();c.weights[:]=[.03,.01]
        c.scenario['participation']=.01
        c.predictions[:]=mu;c.stamps[:]=stamp
        p=copy.deepcopy(self.protocol);p['execution']['decimals']=[4,4]
        action=c.propose(21600,np.array([100.,100.]),np.zeros(2),1e6,np.zeros(2),
            np.eye(2)*1e-5,np.ones(2)*1e6,np.ones(2),p)
        return c,action

    def test_relative_forecast_can_hedge_two_same_direction_signals(self):
        c,action=self.relative_entry(-.02)
        self.assertIsNotNone(action)
        qty=np.array(action['target_qty'])
        self.assertLess(qty[0],0);self.assertGreater(qty[1],0)
        self.assertLessEqual(abs(qty).sum()*100/1e6,.04)
        self.assertEqual(c.events[-1]['direction'],-1)

    def test_cost_gate_still_blocks_insufficient_relative_return(self):
        c,action=self.relative_entry(.0039)
        self.assertIsNone(action)
        self.assertFalse(c.states)

    def test_future_fit_is_rejected_at_decision_time(self):
        c,action=self.relative_entry(.02,21601)
        self.assertIsNone(action)
        self.assertFalse(c.states)

    def transition_run(self,expire=False,mode='overlay',parent=False):
        n=42;times=np.arange(n)*3600
        m=SimpleNamespace(symbols=('ETH','BTC'),times=times,close=np.ones((n,2))*100,
            volume=np.ones((n,2))*1e6,trades=np.ones((n,2))*100,funding=np.zeros((n,2)),
            observed=np.ones((n,2),bool),exact=lambda t:t//3600)
        p=copy.deepcopy(self.protocol);p['execution']['decimals']=[4,4]
        d=dict(pairs=[[0,1]],x=np.ones((n,1,1)),names=['correlation_168'],beta=np.ones((n,1)),vol=np.ones((n,1))*.01)
        pred=np.zeros((n,1));pred[12:]=.02
        if expire:pred[13:]=0.
        weights=np.tile([.03,.01],(n,1));cov=np.tile(np.eye(2)*1e-5,(n,1,1))
        klass=IncumbentPairController
        if parent:
            from experiments.relative_incumbent_pairs_20261017.policy import IncumbentPairController as klass
        c=klass(m,weights,np.ones((n,2)),d,pred,np.ones((n,1)),self.cfg,
            dict(mode=mode,model='calibrated',horizon=24),p['scenarios']['base'],40)
        result=simulate(m,weights,cov,p,6,40,'base',c)
        return c,result

    def test_real_shared_executor_closes_then_opens_reversal(self):
        c,r=self.transition_run()
        by_time={s['time']//3600:s for s in r['snapshots']}
        self.assertTrue(np.all(np.array(by_time[12]['qty'])>0))
        self.assertEqual(by_time[13]['qty'],[0.,0.])
        self.assertGreater(by_time[14]['qty'][0],0)
        self.assertLess(by_time[14]['qty'][1],0)
        self.assertTrue(any(e['event']=='prepare_flat' for e in c.events))
        self.assertTrue(any(e['event']=='balanced_open' for e in c.events))
        self.assertFalse(any(e['event']=='unhedged_abort' for e in c.events))
        self.assertTrue(r['independent_accounting_audit']['passed'])

    def test_revalidated_edge_can_cancel_after_paid_flattening(self):
        c,r=self.transition_run(expire=True)
        self.assertTrue(any(e['event']=='preparation_cancelled_after_revalidation' for e in c.events))
        self.assertFalse(any(e['event']=='balanced_open' for e in c.events))
        self.assertGreater(r['metrics']['fees'],0.)
        self.assertFalse(any(f['signed_qty']<0 and not f['reduce_only'] for f in r['fills']))

    def test_flat_standalone_without_reversal_is_identical_to_parent(self):
        _,new=self.transition_run(mode='standalone')
        _,old=self.transition_run(mode='standalone',parent=True)
        self.assertEqual(new['fills'],old['fills'])
        self.assertEqual(new['snapshots'],old['snapshots'])


if __name__=='__main__':unittest.main()
