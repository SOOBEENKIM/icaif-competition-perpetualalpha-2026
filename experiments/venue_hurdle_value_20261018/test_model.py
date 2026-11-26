import copy
import unittest
import numpy as np
from .features import activity,features,initial_state
from .model import fit,predict


class HurdleTests(unittest.TestCase):
    def setUp(self):
        self.spec=dict(training_window_days=7,half_life_days=3,minimum_rows=3,minimum_distinct_sources=3,
            minimum_head_rows=2,ridge_penalty=1.,intercept_prior_rows=12,target_scale=.01,target_clip=4.,
            tree=dict(max_depth=2,eta=.05,min_child_weight=1.,reg_lambda=10.,seed=0,nthread=1,tree_method='hist',num_boost_round=3))
        self.rows=[dict(sample_id=str(i),horizon=6,status='unfilled' if i%3==0 else 'closed',source=i*3600,
            initial_time=i*3600+300,initial_class=i%3,outcome_time=i*3600+(300 if i%3==0 else 7200),
            pair_index=0,features=[float(i),float(i%2)],net_pnl=0. if i%3==0 else float(i-2),intended_gross=1000.) for i in range(10)]

    def test_future_initial_and_cash_labels_do_not_change_fit(self):
        now=6*3600;first=fit(self.rows,now,6,self.spec);rows=copy.deepcopy(self.rows)
        for r in rows:
            if r['outcome_time']>=now:r['net_pnl']=1e9
            if r['initial_time']>=now:r.update(initial_class=2,features=[1e9,1e9])
        self.assertEqual(first,fit(rows,now,6,self.spec))

    def test_censored_cash_can_inform_only_observed_initial_state(self):
        rows=copy.deepcopy(self.rows);rows[1].update(status='censored',outcome_time=None,net_pnl=None)
        model=fit(rows,6*3600,6,self.spec)
        self.assertIn('1',model['initial_ids']);self.assertNotIn('1',model['cash_ids'])

    def test_activity_prefix_does_not_use_future_volume(self):
        rng=np.random.default_rng(0);d=dict(par_volume=rng.integers(0,4,(400,2)),par_trades=rng.integers(0,4,(400,2)))
        a,names=activity(d,[[0,1]]);changed=copy.deepcopy(d)
        changed['par_volume'][201:]=1000;changed['par_trades'][201:]=1000
        b,_=activity(changed,[[0,1]])
        np.testing.assert_array_equal(a[:201],b[:201]);self.assertEqual(len(names),a.shape[-1])

    def test_initial_classes_are_not_final_cash_labels(self):
        self.assertEqual(initial_state([0,0],[1,-2],.05),0)
        self.assertEqual(initial_state([1,0],[1,-2],.05),1)
        self.assertEqual(initial_state([.5,-1],[1,-2],.05),2)
        with self.assertRaises(ValueError):initial_state([np.nan,0],[1,-2],.05)

    def test_quantity_geometry_is_available_in_features(self):
        d=dict(par_close=np.array([[100.,50.]]),par_volume=np.array([[10.,20.]]))
        de=dict(x=np.array([[[1.,2.]]]),execution_x=np.zeros((1,1,19)))
        p=dict(initial=1e6,execution=dict(decimals=[2,2],minimum_order_value=10))
        proposal=dict(pair_index=0,ids=[0,1],direction=1,target=[1.,-2.],beta=1.)
        a,g=features(proposal,d,de,0,p);b,g2=features(dict(proposal,target=[2.,-4.]),d,de,0,p)
        self.assertEqual(g2,2*g);self.assertFalse(np.array_equal(a[-7:],b[-7:]))

    def test_mixture_has_zero_no_fill_value_and_probabilities_sum_to_one(self):
        model=fit(self.rows,6*3600,6,self.spec);value,parts=predict(model,np.array([2.,0.]),6*3600,'hurdle')
        self.assertEqual(parts['conditional_net_returns'][0],0.)
        self.assertAlmostEqual(sum(parts['initial_probabilities']),1.,places=6)
        self.assertAlmostEqual(value,np.dot(parts['initial_probabilities'],parts['conditional_net_returns']))

    def test_prediction_rejects_future_model(self):
        model=fit(self.rows,6*3600,6,self.spec)
        with self.assertRaises(ValueError):predict(model,np.array([2.,0.]),5*3600,'hurdle')


if __name__=='__main__':unittest.main()
