import copy
import unittest
import numpy as np
from .book_completion import OriginalLimitSession
from experiments.published_pair_lifecycle_20261017.account import BookAccount


class BookCompletionTests(unittest.TestCase):
    def setUp(self):
        p=dict(execution=dict(decimals=[2,2],minimum_order_value=10.),
               risk=dict(gross_fraction=.6,net_fraction=.4,asset_fraction=.15,asset_notional=150000.,daily_volatility=.005))
        account=BookAccount(1e6,['A','B'])
        account.physical.fill(0,.5,100.,9.,1.)
        account.owners['pair'].fill(0,.5,100.,9.,1.)
        intent=dict(ids=[0,1],pair=['A','B'],target=[1.,-2.],entry_limits={'0':101.,'1':49.},forecast_source=0,horizon=6)
        self.s=OriginalLimitSession(account,intent,dict(pair_balance_tolerance=.05),p,dict(fee_bps=9.))
        self.s.target=np.array(intent['target'])
        self.quotes={s:dict(book=dict(levels=[[dict(px='50',sz='10')],[dict(px='51',sz='10')]])) for s in ['A','B']}
        self.marks=np.array([100.,50.]);self.sigma=np.eye(2)*1e-6

    def test_completion_preserves_original_limit_and_filled_leg(self):
        plan=self.s.completion(2,self.quotes,self.marks,self.sigma)
        np.testing.assert_array_equal(plan['target'],[.5,-1.])
        self.assertEqual(plan['limits'],{1:49.})

    def test_completion_does_not_chase_through_limit(self):
        self.quotes['B']['book']['levels'][0][0]['px']='48'
        self.assertIsNone(self.s.completion(2,self.quotes,self.marks,self.sigma))

    def test_missing_book_is_not_a_fabricated_completion(self):
        self.quotes['B']=None
        self.assertIsNone(self.s.completion(2,self.quotes,self.marks,self.sigma))

    def test_current_account_risk_is_respected(self):
        self.s.p=copy.deepcopy(self.s.p);self.s.p['risk']['asset_notional']=20.
        self.assertIsNone(self.s.completion(2,self.quotes,self.marks,self.sigma))


if __name__=='__main__':unittest.main()
