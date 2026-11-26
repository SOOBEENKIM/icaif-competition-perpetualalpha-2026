import copy
import unittest
import numpy as np
from .model import fit,adjusted_net,features
from .policy import held_state,completion_plan


class VenueCashTests(unittest.TestCase):
    def setUp(self):
        self.spec=dict(minimum_rows=3,minimum_distinct_sources=3,training_window_days=7,half_life_days=3,
                       ridge_penalty=1.,intercept_prior_rows=12,target_scale=.01,target_clip=4.)
        self.rows=[dict(sample_id=str(i),horizon=6,status='closed',source=i*3600,outcome_time=(i+2)*3600,
                       pair_index=0,features=[float(i),float(i%2)],net_pnl=float(i-2),intended_gross=1000.) for i in range(8)]

    def test_base_cost_is_not_subtracted_twice(self):
        sc=dict(fee_bps=9,impact_bps=10)
        self.assertEqual(adjusted_net(.01,sc,sc),.01)

    def test_stress_only_deducts_incremental_reserve(self):
        self.assertAlmostEqual(adjusted_net(.01,dict(fee_bps=9,impact_bps=10),dict(fee_bps=12,impact_bps=20)),.0074)

    def test_future_outcomes_cannot_change_fit(self):
        now=6*3600;a=fit(self.rows,now,6,self.spec);changed=copy.deepcopy(self.rows)
        for r in changed:
            if r['outcome_time']>=now:r.update(net_pnl=1e9,features=[1e9,1e9])
        self.assertEqual(a,fit(changed,now,6,self.spec));self.assertLess(a['last_outcome'],now)

    def test_censoring_and_not_admitted_are_not_zero_outcomes(self):
        rows=copy.deepcopy(self.rows)
        rows[0].update(status='censored',outcome_time=None,net_pnl=None)
        rows[1].update(status='not_admitted',outcome_time=None,net_pnl=None)
        model=fit(rows,20*3600,6,self.spec)
        self.assertNotIn('0',model['training_ids']);self.assertNotIn('1',model['training_ids'])

    def test_observed_unfilled_outcome_is_valid_zero(self):
        rows=copy.deepcopy(self.rows);rows[0].update(status='unfilled',net_pnl=0.)
        model=fit(rows,20*3600,6,self.spec)
        self.assertIn('0',model['training_ids']);self.assertEqual(model['zero_rows'],1)

    def test_direction_interactions_preserve_common_features(self):
        d=dict(par_close=np.array([[100.,50.]]));de=dict(x=np.array([[[2.,3.]]]))
        p=dict(pair_index=0,direction=1,target=[1.,-2.],ids=[0,1])
        a,g=features(p,d,de,0,1000.);b,_=features(dict(p,direction=-1,target=[-1.,2.]),d,de,0,1000.)
        np.testing.assert_array_equal(a[:2],b[:2]);np.testing.assert_array_equal(a[2:4],-b[2:4]);self.assertEqual(g,200.)

    def test_horizon_uses_actual_first_fill(self):
        pair=dict(venue_cash=True,first_fill=300,forecast_source=0,horizon=6,entry_vol=.01)
        d=dict(times=np.arange(0,22501,300))
        cfg=dict(dense_policy=dict(stop_sigma=3))
        before=held_state(pair,71,d,None,cfg);at=held_state(pair,72,d,None,cfg)
        self.assertIsNone(before['reason']);self.assertEqual(at['reason'],'fixed_horizon');self.assertEqual(at['maturity'],21900)

    def test_completion_deadline_cannot_be_extended(self):
        class Ledger:qty=np.array([1.,0.])
        pair=dict(venue_cash=True,ids=[0,1],target=[1.,-1.],completion_deadline=300)
        self.assertIsNone(completion_plan(pair,Ledger(),1,dict(times=np.array([0,300])),None,{}, {},{}))


if __name__=='__main__':unittest.main()
