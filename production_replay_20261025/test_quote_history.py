import copy
import unittest
import numpy as np
from .quote_history import quote_history,filter_target


class QuoteHistoryTests(unittest.TestCase):
    def stream(self):
        points=[3600,14400];books=[];ctx=[]
        for t,px in zip(points,[100.,99.]):
            at=t*1000-1000
            books.append((at,1,dict(time=at,levels=[[dict(px=str(px-.01),sz='100')],[dict(px=str(px+.01),sz='100')]])))
            ctx.append((at,1,dict(markPx=str(px),oraclePx=str(px))))
        return dict(BTC=dict(book=books,ctx=ctx,book_times=[x[0] for x in books],ctx_times=[x[0] for x in ctx],resets=[]))

    def test_quote_history_does_not_need_recent_trade(self):
        p,e=quote_history(self.stream(),['BTC'],14400,dict(book_max_age_ms=15000,context_max_age_ms=3000))
        self.assertTrue(np.isnan(p[0,0]));self.assertEqual(p[1:,0].tolist(),[100.,99.])
        q,ok=filter_target([1.],[0.],p)
        self.assertEqual(q.tolist(),[1.]);self.assertTrue(ok[0])

    def test_future_quote_cannot_change_feature(self):
        a=self.stream();b=copy.deepcopy(a)
        b['BTC']['book'].append((14400001,1,dict(time=14400001,levels=[[dict(px='1000',sz='1')],[dict(px='1001',sz='1')]])))
        b['BTC']['book_times'].append(14400001)
        cfg=dict(book_max_age_ms=15000,context_max_age_ms=3000)
        np.testing.assert_array_equal(quote_history(a,['BTC'],14400,cfg)[0],quote_history(b,['BTC'],14400,cfg)[0])

    def test_missing_quote_cannot_be_replaced_with_future_or_trade_price(self):
        p=np.full((3,2),np.nan)
        q,ok=filter_target([2.,-2.],[1.,-1.],p)
        self.assertEqual(q.tolist(),[1.,-1.]);self.assertFalse(ok.any())

    def test_reductions_and_close_before_reversal_survive(self):
        q,_=filter_target([.5,-.5,-2.,2.],[1.,-1.,1.,-1.],np.full((3,4),np.nan))
        self.assertEqual(q.tolist(),[.5,-.5,0.,0.])

    def test_short_waits_for_pullback_in_same_symmetric_rule(self):
        p=np.array([[np.nan,np.nan],[100.,100.],[101.,99.]])
        q,ok=filter_target([-1.,-1.],[0.,0.],p)
        self.assertEqual(q.tolist(),[-1.,0.]);self.assertEqual(ok.tolist(),[True,False])


if __name__=='__main__':unittest.main()
