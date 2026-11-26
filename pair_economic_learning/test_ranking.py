import unittest
from unittest.mock import patch
import numpy as np
from .ranked_entry import proposals


class RankingTests(unittest.TestCase):
    def test_ranking_never_introduces_rejected_pair_or_changes_quantity(self):
        allowed=[dict(pair_index=0,pair=['A','B'],expected_cash=10.,target=[1.,-2.]),dict(pair_index=1,pair=['A','C'],expected_cash=4.,target=[3.,-4.])]
        state={'ranking_score':np.array([[.1,.9,100.]])}
        with patch('pair_economic_learning.ranked_entry.original',return_value=allowed):
            got=proposals(0,{},state,None,72,{},None,None,None,None,None,None,'test')
        self.assertEqual([x['pair_index'] for x in got],[1,0]);self.assertEqual(got[0]['target'],[3.,-4.])

    def test_missing_rank_cannot_create_order(self):
        with patch('pair_economic_learning.ranked_entry.original',return_value=[dict(pair_index=0,pair=['A','B'],expected_cash=10.)]):
            out=proposals(0,{},dict(ranking_score=np.array([[np.nan]])),None,72,{},None,None,None,None,None,None,'test')
        self.assertEqual(out,[])


if __name__=='__main__':unittest.main()
