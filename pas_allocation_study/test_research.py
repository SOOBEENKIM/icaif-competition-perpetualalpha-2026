import unittest
import numpy as np
from . import model
from .venue import basis_forecast,net_forecast,executable_edge


class ResearchContracts(unittest.TestCase):
    def test_every_symbol_can_be_long_or_short_without_forced_trade(self):
        p=dict(entry_cost=.0038,entry_buffer=.0005,daily_target_risk=.001)
        vol=np.full(10,.03)
        long=model.allocate(np.full(10,.02),vol,np.zeros(10),p)
        short=model.allocate(np.full(10,-.02),vol,np.zeros(10),p)
        self.assertTrue(np.all(long>0));np.testing.assert_allclose(long,-short)
        np.testing.assert_array_equal(model.allocate(np.zeros(10),vol,long,p),np.zeros(10))
        np.testing.assert_array_equal(model.allocate(np.full(10,.002),vol,np.zeros(10),p),np.zeros(10))

    def test_funding_sign_and_current_entry_price(self):
        a,carry=net_forecast(.01,0,np.full(24,.0001),reference_price=100,venue_price=102)
        self.assertAlmostEqual(a,101/102-1-.0012)
        self.assertGreater(carry,0)
        b,_=net_forecast(.01,0,np.full(24,-.0001),reference_price=100,venue_price=102)
        self.assertGreater(b,a)

    def test_future_basis_cannot_change_forecast(self):
        p=dict(basis_lookback=96,basis_minimum_pairs=48,basis_ridge_prior_pairs=24)
        t=np.arange(120)*3600;b=.01*np.sin(np.arange(120)/12)
        x=basis_forecast(t,b,80*3600,p)
        b[81:]=1000
        self.assertEqual(x,basis_forecast(t,b,80*3600,p))
        gap=t.copy();gap[1::2]+=1
        self.assertEqual(basis_forecast(gap,b,80*3600,p)[0],0.)

    def test_quote_probe_uses_ask_for_buy_bid_for_sell(self):
        book={'levels':[[{'px':'99','sz':'2'}],[{'px':'101','sz':'3'}]]}
        r=executable_edge(0,100,0,book,.001,0)
        self.assertLess(r['long_edge'],0);self.assertLess(r['short_edge'],0)
        self.assertEqual(r['minimum_top_notional'],198)

    def test_label_maturity_and_source_anchor(self):
        n=24*10;t=np.arange(n)*3600
        close=np.exp(np.arange(n)[:,None]*.001)*np.ones((n,10))
        d=dict(market_times=t,close=close)
        f=dict(x=np.ones((n,10,7)),unit=np.ones((n,10)),fresh=np.ones((n,10),bool))
        i,j,y=model.label_rows(d,f,0,5*86400)
        self.assertTrue(np.all(t[i]+86400<5*86400))
        np.testing.assert_allclose(y,.024)


if __name__=='__main__':unittest.main()
