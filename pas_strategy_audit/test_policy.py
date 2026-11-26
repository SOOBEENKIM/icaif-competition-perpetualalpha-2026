import unittest
import numpy as np
from all_market_strategy.signals import ema_trend,channel_trend
from .policy import intraday_targets,sticky_conflict_guard

class DirectionTests(unittest.TestCase):
    def daily(self,side):
        t=1775001600+np.arange(24*100)*3600;p=100*np.exp(side*np.arange(len(t))*.012/24);p=np.repeat(p[:,None],10,axis=1)
        return dict(known=t,close=p,high=p*1.001,low=p*.999)
    def test_deployed_directional_components_have_no_bearish_state(self):
        a=self.daily(-1);p=dict(frame=24,fast=10,slow=60,confidence=0.,atr_multiple=3.,breakout=0,reentry=0,exit_cross=False,decision_hours=6,scale=1.)
        c,v=channel_trend(a,np.ones(10,bool));w,s=ema_trend(a,v,p,np.ones(10,bool))
        self.assertFalse((s<0).any());self.assertFalse((c<0).any())
    def test_symmetric_daily_enters_both_directions_in_all_ten_assets(self):
        for side in (-1,1):
            a=self.daily(side);p=dict(frame=24,fast=10,slow=60,confidence=0.,atr_multiple=3.,breakout=0,reentry=0,exit_cross=True,decision_hours=6,scale=1.)
            c,v=channel_trend(a,np.ones(10,bool),symmetric=True);w,s=ema_trend(a,v,p,np.ones(10,bool),symmetric=True)
            self.assertTrue(np.all(s[-1]==side));self.assertTrue(np.all(c[-1]*side>0))
    def minute(self,side=1,n=1000):
        p=100*np.exp(side*np.arange(n)*.00003);p=np.repeat(p[:,None],10,axis=1)
        return dict(times=1792411200000+np.arange(n)*60000,close=p,active=np.ones_like(p,bool),age_ms=np.zeros_like(p))
    def test_intraday_signed_all_markets(self):
        for side in (-1,1):
            w,_=intraday_targets(self.minute(side));self.assertTrue(np.all(w[-1]*side>0));self.assertAlmostEqual(abs(w[-1]).sum(),1.)
    def test_future_data_cannot_change_past_targets(self):
        a=self.minute();w,_=intraday_targets(a);short={k:v[:700].copy() for k,v in a.items()};p,_=intraday_targets(short)
        np.testing.assert_allclose(w[:700],p)
    def test_flat_and_unobserved_cannot_force_entries(self):
        a=self.minute(0);w,_=intraday_targets(a);self.assertFalse(np.any(w))
        a=self.minute();a['age_ms'][:]=1e10;a['active'][:]=False;w,_=intraday_targets(a);self.assertFalse(np.any(w))
    def test_price_inversion_reverses_direction_without_manual_override(self):
        a=self.minute();b={k:v.copy() for k,v in a.items()};b['close']=10000/a['close'];wa,_=intraday_targets(a);wb,_=intraday_targets(b)
        np.testing.assert_allclose(wa,-wb,atol=1e-10)
    def test_conflict_reduction_cannot_restore_or_inflate_other_assets(self):
        b=[dict(time=0,weights=[.1,-.1]),dict(time=100,weights=[.1,-.1])]
        r=[dict(time=10,weights=[-1.,-1.]),dict(time=20,weights=[1.,-1.])]
        events=sticky_conflict_guard(b,r)
        np.testing.assert_allclose(events[0]['weights'],[.02,-.02])
        np.testing.assert_allclose(events[1]['weights'],[0,-.02])
        self.assertEqual(events[2]['time'],100)
        np.testing.assert_allclose(events[2]['weights'],[.02,-.02])

if __name__=='__main__':unittest.main()
