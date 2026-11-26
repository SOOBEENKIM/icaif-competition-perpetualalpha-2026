import unittest
import numpy as np
from .ranking import rank


class RankingTests(unittest.TestCase):
    def test_ranking_keeps_prices_sizes_and_means(self):
        a=dict(pair=['A','B'],ids=[0,1],target=[30,-30],horizon=72,entry_vol=.01,expected_net_cash=90,mu=.02)
        b=dict(pair=['C','D'],ids=[2,3],target=[30,-30],horizon=72,entry_vol=.001,expected_net_cash=50,mu=.01)
        out=rank([a,b],np.full(4,100.))
        self.assertEqual(out[0]['pair'],b['pair'])
        for x in out:
            original=a if x['pair']==a['pair'] else b
            for key,value in original.items():self.assertEqual(x[key],value)
        self.assertNotIn('ranking_net_to_risk',a)


if __name__=='__main__':unittest.main()
