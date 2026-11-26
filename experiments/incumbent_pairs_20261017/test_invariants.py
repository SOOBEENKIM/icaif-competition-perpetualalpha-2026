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
from .model import fit,features
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


if __name__=='__main__':unittest.main()
