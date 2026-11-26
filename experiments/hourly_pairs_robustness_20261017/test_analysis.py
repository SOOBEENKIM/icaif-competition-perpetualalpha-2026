import unittest
import numpy as np
from .analysis import summary, circular_indices, paired_resampling, episode_paths


class AnalysisTests(unittest.TestCase):
    def test_compounded_return_and_risk_scale(self):
        x=np.array([.01,-.01,.02,-.02])
        result=summary(x)
        self.assertAlmostEqual(result['return_fraction'],1.01*.99*1.02*.98-1)
        self.assertAlmostEqual(summary(x*.4)['volatility_horizon'],result['volatility_horizon']*.4)
        self.assertLess(summary([-0.01,-0.02])['financial_score'],.1)

    def test_paired_identical_returns_have_zero_advantage(self):
        x=np.array([.01,-.02,.04,-.01,.003])
        rows=paired_resampling(x,{'same':x.copy()},dict(seed=10,samples=100,circular_block_days=[2,7],quantiles=[.025,.5,.975]))
        for row in rows:
            self.assertEqual(row['return_fraction_difference_quantiles'],[0.,0.,0.])
            self.assertEqual(row['financial_score_difference_quantiles'],[0.,0.,0.])

    def test_blocks_preserve_order_wrap_and_seed(self):
        a=circular_indices(7,3,10,5)
        np.testing.assert_array_equal(a,circular_indices(7,3,10,5))
        self.assertTrue(np.all((a[:,1:3]-a[:,:2])%7==1))
        self.assertTrue(np.all((a[:,4:6]-a[:,3:5])%7==1))

    def test_closing_fill_at_next_intent_belongs_to_old_episode(self):
        prices=np.array([[100.],[110.],[120.],[125.]])
        d=dict(times=np.array([0,3600,7200,10800]),par_close=prices,reference_close=prices,
               funding=np.array([[0.],[.01],[.01],[-.01]]))
        snaps=[dict(time=t,pair_qty=[q],qty=[q]) for t,q in [(0,0),(3600,1),(7200,0),(10800,1)]]
        # First entry costs 2 price impact +1 fee. At the next intent, the old
        # episode realizes +10 minus 1.2 funding, 1 exit impact and 1 exit fee.
        fills=[dict(time=3600,asset=0,signed_qty=1,price=112,fee=1),
               dict(time=7200,asset=0,signed_qty=-1,price=119,fee=1),
               dict(time=10800,asset=0,signed_qty=1,price=126,fee=1)]
        run=dict(events=[dict(event='pair_intent',time=0),dict(event='pair_intent',time=7200)],
                 owner_fills={'pair':fills},snapshots=snaps,metrics={'pair_owned_pnl':1.8})
        entries,cumulative,parts=episode_paths(run,d,{'funding_receipt_multiplier':.5})
        np.testing.assert_allclose(cumulative[:,-1],[3.8,-2.],atol=1e-10)
        self.assertAlmostEqual(parts[0,3],-1.2)
        self.assertEqual(len(entries),2)


if __name__=='__main__':unittest.main()
