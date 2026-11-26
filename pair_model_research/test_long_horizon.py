import unittest
import numpy as np
from .long_horizon import fit_day,consensus


class LongHorizonTests(unittest.TestCase):
    def test_only_mature_labels_and_no_pair_whitelist(self):
        rng=np.random.default_rng(123);n=1700;t=np.arange(n)*3600;h=72;m=t+h*3600;source=1500*3600
        x=rng.normal(size=(n,45,3));y=rng.normal(size=(n,45));fresh=np.ones((n,45),bool)
        spec=dict(lookback_days=90,sample_hours=6,minimum_training_rows=200,half_life_days=30,ridge_penalty=1,intercept_prior_rows=12,target_clip=4,prediction_clip=4)
        a=fit_day(t,x,y,m,fresh,fresh,source,spec);self.assertEqual(len(a),45)
        x[m>=source]=np.nan;y[m>=source]=1e10
        b=fit_day(t,x,y,m,fresh,fresh,source,spec);self.assertEqual(a,b)
        self.assertTrue(all(z['last_maturity']<source for z in a.values()))

    def test_consensus_is_not_a_fake_observation(self):
        a=np.array([.01,-.01,.02]);b=np.array([.02,-.02,-.01]);mean,entry=consensus(a,b)
        np.testing.assert_allclose(mean,[.015,-.015,.005]);np.testing.assert_equal(entry,[True,True,False])


if __name__=='__main__':unittest.main()
