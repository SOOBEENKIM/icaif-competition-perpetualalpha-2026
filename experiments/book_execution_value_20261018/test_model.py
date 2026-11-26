import copy
import unittest
import numpy as np
from .model import fit,predict,score_plan,weights


class ExecutionModelTests(unittest.TestCase):
    def setUp(self):
        self.spec=dict(minimum_fit_rows=4,tree=dict(max_depth=2,eta=.05,min_child_weight=1.,reg_lambda=1.,seed=0,nthread=1,tree_method='hist',num_boost_round=3))
        self.rows=[dict(sample_id=str(i),tape='a',source=i*60,available_at=i*60+5,outcome=i%4,
            held_fraction=.5 if i%4==2 else 0.,entry_cost_return=None if i%4==3 else .001*i,
            features=[float(i),float(i%2)]) for i in range(12)]

    def test_future_outcomes_do_not_change_fit(self):
        original=fit(self.rows,['a'],500,self.spec);rows=copy.deepcopy(self.rows)
        for row in rows:
            if row['available_at']>=500:row.update(features=[1e8,1e8],outcome=2,entry_cost_return=1e8)
        self.assertEqual(original,fit(rows,['a'],500,self.spec))

    def test_unresolved_cash_is_not_trained_as_zero(self):
        model=fit(self.rows,['a'],1000,self.spec)
        self.assertEqual(model['rows'],12);self.assertEqual(model['cost_rows'],9)
        self.assertEqual(model['class_counts'][3],3)

    def test_source_replicas_share_weight(self):
        rows=[dict(source=0),dict(source=0),dict(source=60)]
        np.testing.assert_array_equal(weights(rows),[.5,.5,1.])

    def test_model_rejects_future_inference(self):
        model=fit(self.rows,['a'],1000,self.spec)
        with self.assertRaises(ValueError):predict(model,[1.,0.],999)

    def test_score_does_not_add_full_profit_to_zero_fill(self):
        plan=dict(intended_gross=1000.,expected_net_cash=10.,exit_cost_reserve=3.)
        estimate=dict(probabilities=[1.,0.,0.,0.],held_fraction_if_balanced=1.,extra_cost_bps_if_reconciled=0.)
        self.assertEqual(score_plan(plan,estimate)['value'],0.)
        estimate.update(probabilities=[0.,0.,.5,.5])
        self.assertEqual(score_plan(plan,estimate)['value'],3.5)


if __name__=='__main__':unittest.main()
