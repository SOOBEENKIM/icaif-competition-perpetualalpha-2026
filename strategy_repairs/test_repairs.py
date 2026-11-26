import unittest
import numpy as np
from competition_live.profile import load_profile
from replay_lab.ema_rebuild.engine import Controller
from replay_lab.competition_rebuild.causal_risk import normalize_risk
from replay_lab.v10_revision.core import sizing
from all_market_strategy.signals import channel_trend,ema_trend
from .model import apply_gate_after_sizing
from .controller import RiskPreservingController


class RepairTests(unittest.TestCase):
    def test_original_two_stage_sizing_erases_uniform_risk_cut(self):
        times=1775001600+np.arange(1500)*3600
        base=np.zeros((1500,10));base[:,0]=.04
        sigma=np.repeat((np.eye(10)*1e-6)[None],1500,axis=0)
        reduced=base*.5
        old=sizing(normalize_risk(base,reduced,sigma,times),sigma,times)
        ungated=sizing(base,sigma,times)
        np.testing.assert_allclose(old[-1],ungated[-1])
        repaired=apply_gate_after_sizing(ungated,np.full_like(base,.5))
        np.testing.assert_allclose(repaired[-1],ungated[-1]*.5)

    def test_risk_cap_bypasses_twenty_percent_buffer_without_other_increases(self):
        p=load_profile()['trading'];t=1792411200
        prices=np.full(10,100.);qty=np.array([20.,10.]+[0.]*8)
        raw=np.array([.036,.02]+[0.]*8);sigma=np.eye(10)*1e-8
        args=(t,prices,qty,50000.,raw,sigma,np.ones(10)*1000,np.ones(10),p)
        old=Controller().propose(*args,buffer=.2)
        self.assertEqual(old['target_qty'][0],20.)
        c=RiskPreservingController([t],[raw],[[True]+[False]*9])
        new=c.propose(*args,buffer=.2)
        self.assertEqual(new['target_qty'][0],18.)
        self.assertEqual(new['target_qty'][1],10.)
        self.assertTrue(new['risk_forced'])

    def test_risk_cap_bypasses_turnover_limit_and_does_not_restore_cut(self):
        p=load_profile()['trading'];p['risk']['normal_turnover_fraction']=.001
        t=1792411200;raw=np.array([.01]+[0.]*9);qty=np.array([20.]+[0.]*9)
        args=(t,np.full(10,100.),qty,50000.,raw,np.eye(10)*1e-8,np.ones(10)*1000,np.ones(10),p)
        c=RiskPreservingController([t],[raw],[[True]+[False]*9])
        r=c.propose(*args,buffer=.2)
        self.assertEqual(r['target_qty'][0],5.)
        r=c.propose(t+60,*args[1:],buffer=.2)
        self.assertEqual(r['target_qty'][0],5.)

    def test_short_only_does_not_change_rising_long_path(self):
        n=24*100;t=1775001600+np.arange(n)*3600
        price=np.repeat((100*np.exp(np.arange(n)*.012/24))[:,None],10,axis=1)
        d=dict(known=t,close=price,high=price*1.001,low=price*.999)
        policy=dict(frame=24,fast=10,slow=60,confidence=0.,atr_multiple=3.,breakout=0,reentry=0,
                    exit_cross=False,decision_hours=6,scale=1.)
        allowed=np.arange(10)<3
        a,v=channel_trend(d,allowed);b,v2=channel_trend(d,allowed,symmetric=True)
        np.testing.assert_array_equal(a,b)
        wa,sa=ema_trend(d,v,policy,allowed);wb,sb=ema_trend(d,v2,policy,allowed,symmetric=True)
        np.testing.assert_array_equal(sa,sb);np.testing.assert_array_equal(wa,wb)
        d['close']=10000/d['close'];d['high']=d['close']*1.001;d['low']=d['close']*.999
        c,v=channel_trend(d,allowed,symmetric=True)
        w,s=ema_trend(d,v,policy,allowed,symmetric=True)
        self.assertTrue(np.all(s[-1,:3]==-1));self.assertFalse(np.any(s[:,3:]))

    def test_no_risk_contract_means_exact_old_controller_behavior(self):
        p=load_profile()['trading'];t=1792411200;raw=np.array([.04,-.03]+[0.]*8)
        args=(t,np.full(10,100.),np.zeros(10),50000.,raw,np.eye(10)*1e-8,np.ones(10)*1000,np.ones(10),p)
        a=Controller().propose(*args,buffer=.2)
        b=RiskPreservingController([t],[raw],np.zeros((1,10),bool)).propose(*args,buffer=.2)
        self.assertEqual(a,b)

    def test_cross_exit_is_a_separate_explicit_policy_change(self):
        n=24*140;t=1775001600+np.arange(n)*3600
        returns=np.r_[np.full(24*80,.01/24),np.full(24*60,-.01/24)]
        price=np.repeat((100*np.exp(returns.cumsum()))[:,None],10,axis=1)
        d=dict(known=t,close=price,high=price*1.2,low=price*.8)
        allowed=np.arange(10)<3
        _,v=channel_trend(d,allowed,symmetric=True)
        p=dict(frame=24,fast=10,slow=60,confidence=0.,atr_multiple=3.,breakout=0,
               reentry=0,exit_cross=False,decision_hours=6,scale=1.)
        _,old=ema_trend(d,v,p,allowed,symmetric=True)
        p['exit_cross']=True
        _,new=ema_trend(d,v,p,allowed,symmetric=True)
        self.assertTrue(np.all(old[-1,:3]==1))
        self.assertTrue(np.all(new[-1,:3]==-1))


if __name__=='__main__':
    unittest.main()
