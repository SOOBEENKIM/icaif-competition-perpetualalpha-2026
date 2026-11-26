import copy,unittest
import numpy as np
from .model import feature_at,training_rows,fit,predict,map_state


class LongHistoryTests(unittest.TestCase):
    def setUp(self):
        self.spec=dict(minimum_scalar_events=12,minimum_ridge_events=48,prior_events=24,ridge_penalty=1.,feature_clip=8.,target_clip=8.)

    def rows(self,n=60):
        return dict(x=np.zeros((n,5)),y=np.ones(n),gaps=np.full(n,.01),price_targets=np.full(n,.005),source_times=np.arange(n),maturities=np.arange(n)+1)

    def test_small_pair_uses_declared_scalar_fallback(self):
        m=fit(self.rows(17),100,self.spec)
        self.assertIsNone(m['ridge']);p=predict(m,np.zeros(5),.01,.01,'ridge')
        self.assertEqual(p['method'],'scalar_fallback');self.assertGreater(p['ratio'],0)
        self.assertLess(p['ratio'],.5)
        self.assertEqual(predict(fit(self.rows(11),100,self.spec),np.zeros(5),.01,.01,'ridge')['method'],'unavailable')

    def test_ridge_intercept_has_zero_prior_and_does_not_use_other_pairs(self):
        m=fit(self.rows(),100,self.spec);p=predict(m,np.zeros(5),.01,.01,'ridge')
        self.assertAlmostEqual(p['raw_price_return'],60/84*.01)
        self.assertEqual(p['method'],'ridge');self.assertTrue(0<p['ratio']<1)

    def test_fit_rejects_unmatured_rows(self):
        with self.assertRaises(ValueError):fit(self.rows(),60,self.spec)

    def test_causal_features_do_not_move_with_future_prices(self):
        n=35;price=np.full((n,2),100.);price[24,0]=101
        ref=dict(market_times=np.arange(n)*3600,close=price)
        raw=dict(pairs=[[0,1]],beta=np.ones((n,1)),sd=np.full((n,1),.01),delayed=np.ones((n,1),int),z=np.full((n,1),-2.9),confirmed=np.ones((n,1),bool))
        a=feature_at(ref,raw,24,0);ref['close'][25:]*=10;raw['beta'][25:]*=2
        b=feature_at(ref,raw,24,0);np.testing.assert_array_equal(a[0],b[0]);self.assertEqual(a[1],b[1])

    def test_training_builder_excludes_same_time_and_censored_outcomes(self):
        n=35;ref=dict(market_times=np.arange(n)*3600,close=np.full((n,2),100.))
        raw=dict(pairs=[[0,1]],beta=np.ones((n,1)),sd=np.full((n,1),.01),delayed=np.ones((n,1),int),z=np.full((n,1),-2.9),confirmed=np.ones((n,1),bool))
        events=[dict(time=24*3600,pair_index=0,maturity=m,expected_full_convergence=.01,realized_price_return=v) for m,v in [(26*3600,.005),(27*3600,-999),(None,None)]]
        r=training_rows(ref,raw,events,27*3600);self.assertEqual(len(r['y']),1);self.assertAlmostEqual(r['y'][0],1.)

    def test_negative_prediction_suppresses_entry_and_positive_target_is_capped(self):
        rows=self.rows();rows['y'][:]=-2
        m=fit(rows,100,self.spec);self.assertEqual(predict(m,np.zeros(5),.01,.01,'ridge')['ratio'],0)
        rows['y'][:]=8;m=fit(rows,100,self.spec)
        self.assertEqual(predict(m,np.zeros(5),.01,.01,'ridge')['ratio'],1)

    def test_mapping_preserves_frozen_exit_state_and_source_anchor(self):
        n=26;ref=dict(market_times=np.arange(n)*3600,close=np.full((n,2),100.))
        raw=dict(pairs=[[0,1]],beta=np.ones((n,1)),sd=np.full((n,1),.01),delayed=np.ones((n,1),int),z=np.full((n,1),-2.9),confirmed=np.ones((n,1),bool))
        state=dict(directions=np.array([[1],[1]]),signal_times=np.array([[86400],[86400]]),terminal_prices=np.array([[[102.,98.]],[[102.,98.]]]),beta=np.ones((2,1)),mean=np.full((2,1),.02),sd=np.full((2,1),.01))
        m=dict(fit_time=1,events=20,last_maturity=0,scalar=dict(ratio=.5),ridge=None)
        a,audit=map_state(state,np.array([86400,86500]),ref,raw,{('BTC','ETH'):m},'scalar',[['BTC','ETH']])
        np.testing.assert_array_equal(a['terminal_prices'],np.array([[[101.,99.]],[[101.,99.]]]))
        for k in ['mean','sd','beta']:np.testing.assert_array_equal(a[k],state[k])
        self.assertEqual(len(audit),1)


if __name__=='__main__':unittest.main()
