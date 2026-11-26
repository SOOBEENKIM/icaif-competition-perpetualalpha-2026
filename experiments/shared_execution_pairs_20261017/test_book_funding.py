import unittest
import numpy as np
from . import test_book_session as fixture
from .account_v2 import SharedAccount
from .book_session import BookPairSession


class FundingBoundaryTests(unittest.TestCase):
    def setUp(self):
        fixture.BookSessionTests.setUp(self)
        self.account=SharedAccount(1e6,self.p['symbols'])
        self.account.physical.fill(0,10,100,0,3500);self.account.owner_fill('base',0,10,100,0,3500,'fixture')
        self.settled=False

    def streams_at(self,mapping):
        result={}
        for symbol,times in mapping.items():
            book=[(t,0,dict(time=t,levels=[[dict(px='99.99',sz='100')],[dict(px='100.01',sz='100')]])) for t in times]
            ctx=[(t,0,dict(oraclePx='100',funding='0')) for t in times]
            result[symbol]=dict(book=book,book_times=times,ctx=ctx,ctx_times=times,resets=[])
        return result

    def advance_funding(self,at,account):
        if at>=3600000 and not self.settled:
            rates=np.zeros(10);rates[:2]=.001
            account.charge_funding(3600,self.marks,rates,np.ones(10,bool));self.settled=True

    def test_settlement_before_same_timestamp_fill_uses_old_inventory(self):
        self.intent['forecast_source']=3599
        streams=self.streams_at({'BTC':[3599000,3600000],'ETH':[3599000,3600000]})
        session=BookPairSession(self.account,self.intent,self.cfg,self.p,self.sc,on_time=self.advance_funding)
        session.enter(3599,streams,self.marks,self.sigma)
        self.assertEqual(session.state,'open');self.assertAlmostEqual(self.account.physical.funding,-1.)
        self.assertAlmostEqual(self.account.owners['pair'].funding,0.);self.account.reconcile(self.marks)

    def test_two_arrival_times_charge_only_the_leg_actually_held_at_settlement(self):
        self.intent['forecast_source']=3598
        streams=self.streams_at({'BTC':[3598000,3599000,3600000],'ETH':[3598000,3600000]})
        session=BookPairSession(self.account,self.intent,self.cfg,self.p,self.sc,on_time=self.advance_funding)
        session.enter(3598,streams,self.marks,self.sigma)
        self.assertEqual(session.state,'open');self.assertAlmostEqual(self.account.physical.funding,-.5)
        self.assertAlmostEqual(self.account.owners['base'].funding,-1.)
        self.assertAlmostEqual(self.account.owners['pair'].funding,.5);self.account.reconcile(self.marks)


if __name__=='__main__':unittest.main()
