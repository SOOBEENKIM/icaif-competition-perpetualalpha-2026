import unittest
from . import test_account as account_fixture
from . import test_portfolio as portfolio_fixture
from . import test_book_session as session_fixture
from .account_v2 import SharedAccount
from .portfolio_v2 import replay
from .book_session import BookPairSession


class AccountV2Tests(account_fixture.AccountTests):
    def setUp(self):
        super().setUp();self.a=SharedAccount(1e6,self.p['symbols'])

    def test_virtual_reversal_is_split_and_conserved(self):
        self.a.execute_candle(self.intent(10,-10),self.market,1,self.p,self.sc)
        self.a.execute_candle(self.intent(-5,5,at=300),self.market,2,self.p,self.sc)
        self.assertEqual(self.a.physical.fills,[])
        self.assertEqual(self.a.owners['base'].qty[0],0);self.assertEqual(self.a.owners['pair'].qty[0],0)
        # Each owner keeps its original close-before-new-direction semantics.
        next_packet=self.intent(-5,5,at=600)
        self.assertEqual(next_packet['owner_targets']['base'][0],-5)
        self.a.reconcile(self.price)

    def test_net_reversal_for_new_pair_fills_full_request_and_pays_both_parts(self):
        self.a.execute_candle(self.intent(10,0),self.market,1,self.p,self.sc)
        fees=self.a.physical.fees
        orders=self.a.execute_candle(self.intent(10,-15,at=300),self.market,2,self.p,self.sc)
        self.assertEqual(self.a.physical.qty[0],-5);self.assertEqual(self.a.owners['base'].qty[0],10)
        self.assertEqual(self.a.owners['pair'].qty[0],-15)
        self.assertAlmostEqual(self.a.physical.fees-fees,15*99.9*.0009)
        self.assertEqual(orders[0]['requested_qty'],15);self.assertEqual(orders[0]['filled_qty'],15)
        self.assertFalse(orders[0]['reduce_only']);self.a.reconcile(self.price)

    def test_net_reversal_respects_shared_available_volume(self):
        self.a.execute_candle(self.intent(10,0),self.market,1,self.p,self.sc)
        self.market.volume[2,0]=120
        self.a.execute_candle(self.intent(10,-15,at=300),self.market,2,self.p,self.sc)
        self.assertEqual(self.a.physical.qty[0],-2);self.assertEqual(self.a.owners['pair'].qty[0],-12)

    def test_book_reversal_uses_one_depth_budget_and_counts_fees_once(self):
        self.a.execute_candle(self.intent(10,0),self.market,1,self.p,self.sc)
        fees=self.a.physical.fees
        self.a.execute_books(self.intent(10,-15,at=300),{'BTC':self.book(at=301000,size=12.)},self.price,self.sigma,self.p,self.sc,self.cfg)
        self.assertEqual(self.a.physical.qty[0],-2);self.assertEqual(self.a.owners['pair'].qty[0],-12)
        self.assertAlmostEqual(self.a.physical.fees-fees,12*99.99*.0009)


class PortfolioV2Tests(portfolio_fixture.PortfolioTests):
    def run_arm(self,arm):
        return replay(self.d,self.de,None,self.cfg,self.p,arm,'base',int(self.d['times'][0]),int(self.d['times'][-1]))


class BookSessionV2Tests(session_fixture.BookSessionTests):
    def setUp(self):
        super().setUp();self.account=SharedAccount(1e6,self.p['symbols'])
        self.session=BookPairSession(self.account,self.intent,self.cfg,self.p,self.sc)


if __name__=='__main__':unittest.main()
