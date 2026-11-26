import copy
import unittest
from pathlib import Path
import numpy as np
from experiments.lifecycle_value_pairs_20261018.model import fit
from .model import intervals,overlaps,bundle,assess


def constant(value):
    return dict(center=[0.],scale=[1.],coefficient=[value,0.])


class StabilityTests(unittest.TestCase):
    def setUp(self):
        self.learning=dict(minimum_rows=3,lookback_days=90,half_life_days=30,
                           ridge_penalty=1.,intercept_prior_rows=12,target_clip=4.)
        self.spec=dict(block_days=7,block_anchor_unix=345600,minimum_blocks=4)
        self.rows=[dict(sample_id=str(i),source=i*86400,outcome_time=(i+2)*86400,
                        pair_index=i%2,status='closed',features=[float(i%3)],target=float(i%3-1)) for i in range(50)]

    def value_bundle(self,values):
        return dict(fit_time=100,last_outcome=99,full=constant(values[0]),complete=True,
                    blocks=[dict(model=constant(x)) for x in values[1:]])

    def test_one_positive_deleted_week_preserves_entry(self):
        result=assess(self.value_bundle([-1.,-2.,.01,-.5]),np.array([0.]),100)
        self.assertTrue(result['allowed']);self.assertEqual(result['upper'],.01)

    def test_every_negative_value_vetoes(self):
        self.assertFalse(assess(self.value_bundle([-.1,-.2,-.3]),np.array([0.]),100)['allowed'])

    def test_zero_is_not_evidence_of_negative_opportunity(self):
        self.assertTrue(assess(self.value_bundle([-.1,0.]),np.array([0.]),100)['allowed'])

    def test_incomplete_bundles_never_invent_confidence(self):
        b=self.value_bundle([-1.,-2.]);b['complete']=False
        self.assertTrue(assess(b,np.array([0.]),100)['allowed'])
        self.assertTrue(assess(None,np.array([0.]),100)['allowed'])

    def test_future_model_is_rejected(self):
        with self.assertRaises(ValueError):assess(self.value_bundle([-1.,-2.]),np.array([0.]),99)

    def test_cross_boundary_episode_and_closing_boundary_are_purged(self):
        self.assertTrue(overlaps(dict(source=0,outcome_time=10),10,20))
        self.assertTrue(overlaps(dict(source=5,outcome_time=25),10,20))
        self.assertFalse(overlaps(dict(source=20,outcome_time=30),10,20))

    def test_future_labels_do_not_change_blocks_or_models(self):
        now=45*86400;full=fit(self.rows,now,self.learning)
        before=bundle(self.rows,full,self.learning,self.spec);changed=copy.deepcopy(self.rows)
        for row in changed:
            if row['outcome_time']>=now:row.update(features=[1e9],target=1e8)
        self.assertEqual(before,bundle(changed,full,self.learning,self.spec))
        for block in before['blocks']:
            if block['model']:
                ids=set(block['model']['training_ids'])
                self.assertFalse(any(r['sample_id'] in ids and overlaps(r,block['left'],block['right']) for r in self.rows))

    def test_original_execution_and_holding_body_unchanged(self):
        folder=Path(__file__).resolve().parent
        parent=folder.parent/'lifecycle_value_pairs_20261018/portfolio.py'
        self.assertEqual(parent.read_text().splitlines()[1:],(folder/'portfolio.py').read_text().splitlines()[1:])


if __name__=='__main__':unittest.main()
