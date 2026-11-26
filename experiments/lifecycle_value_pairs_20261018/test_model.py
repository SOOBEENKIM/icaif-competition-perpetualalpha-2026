import copy
import unittest
from pathlib import Path
import numpy as np
from .model import eligible,uniqueness,fit,predict,choose,feature


class LifecycleValueTests(unittest.TestCase):
    def setUp(self):
        self.spec=dict(minimum_rows=3,lookback_days=90,half_life_days=30,
                       ridge_penalty=1.,intercept_prior_rows=12,target_clip=4.)
        self.rows=[dict(sample_id=str(i),source=i*3600,outcome_time=(i+2)*3600,
                        pair_index=0,status='closed',features=[i,i%2],target=float(i-2)) for i in range(6)]

    def test_strict_outcome_boundary(self):
        self.assertEqual(len(eligible(self.rows,4*3600,self.spec)),2)

    def test_open_and_unattempted_not_zero_labels(self):
        rows=copy.deepcopy(self.rows)
        rows[0]['status']='censored';rows[1]['status']='not_admitted'
        self.assertEqual(len(eligible(rows,10*3600,self.spec)),4)

    def test_future_payoff_and_features_do_not_change_fit(self):
        now=6*3600;before=fit(self.rows,now,self.spec);changed=copy.deepcopy(self.rows)
        for r in changed:
            if r['outcome_time']>=now:r.update(features=[1e8,-1e9],target=1e10)
        self.assertEqual(before,fit(changed,now,self.spec))

    def test_duplicate_intervals_share_weight(self):
        r=dict(source=0,outcome_time=7200,pair_index=0)
        np.testing.assert_allclose(uniqueness([r,r,dict(r,pair_index=1)]),[.5,.5,1.])

    def test_partial_overlap_weight(self):
        rows=[dict(source=0,outcome_time=7200,pair_index=0),dict(source=3600,outcome_time=10800,pair_index=0)]
        np.testing.assert_allclose(uniqueness(rows),[.75,.75])

    def test_missing_model_preserves_order_and_two_pair_scope(self):
        choices=[dict(pair_index=2),dict(pair_index=1),dict(pair_index=0)]
        self.assertEqual(choose(choices,None,None,0,1,None,'rank'),choices)
        self.assertEqual(choose(choices,None,None,0,1,None,'filter'),choices[1:])

    def test_held_policy_and_execution_body_unchanged(self):
        parent=Path(__file__).resolve().parent.parent/'decision_gate_pairs_20261017/portfolio.py'
        local=Path(__file__).with_name('portfolio.py')
        self.assertEqual(parent.read_text().splitlines()[1:],local.read_text().splitlines()[1:])

    def test_forecast_values_and_quantities_not_replaced(self):
        d=dict(times=np.array([86400]),par_close=np.array([[100.,50.]]))
        state=dict(vol=np.ones((1,1))*.01,lifecycle_features=np.zeros((1,1,20)))
        p=dict(pair_index=0,ids=[0,1],target=[1.,-2.],horizon=24,direction=1,
               expected_net_bps=80.,expected_funding_bps=1.,source_beta=1.,pair=['A','B'],terminal_prices=[102.,49.])
        x,unit,gross=feature(p,d,state,0,1e6)
        model=dict(fit_time=0,center=np.zeros(len(x)),scale=np.ones(len(x)),coefficient=np.r_[1.,np.zeros(len(x))])
        result=choose([p],d,state,0,1e6,model,'filter')
        for k,v in p.items():self.assertEqual(result[0][k],v)
        self.assertAlmostEqual(result[0]['lifecycle_value'],unit*gross)
        model['coefficient'][0]=-1.
        self.assertEqual(choose([p],d,state,0,1e6,model,'filter'),[])


if __name__=='__main__':unittest.main()
