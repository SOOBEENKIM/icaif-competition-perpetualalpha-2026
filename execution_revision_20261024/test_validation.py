"""Causality, price-limit parity diagnostics and portfolio safety regressions."""
import unittest
from unittest.mock import patch
import numpy as np
from competition_live.live_account import LiveAccount
from pas_strategy_audit import test_execution as fixture
from .venue_momentum import features,VenueMomentum
from .carry_allocation import CarryAllocation
from competition_live.profile import load_profile

class PriceLimitTests(unittest.TestCase):
    def test_real_live_path_blocks_buy_and_short_outside_mark_guard(self):
        for sign in [1,-1]:
            with self.subTest(side=sign):
                helper=fixture.ExecutionTests();a,p,t,orders,stream,_=helper.setup_case(LiveAccount)
                def wide(symbol):
                    s=stream(symbol)
                    s['book'][0][2]['levels']=[[dict(px='79840',sz='10')],[dict(px='80160',sz='10')]]
                    return s
                result=helper.execute(a,p,t,wide,[sign*.01]+[0.]*9,'wide')
                self.assertEqual(orders,[])
                self.assertEqual(result['orders'][0]['status'],'ECONOMIC_LIMIT')
                self.assertEqual(a.physical.qty[0],0.)

    def test_guard_is_not_applied_to_reduction_and_explicit_pair_limit_survives(self):
        helper=fixture.ExecutionTests();a,p,t,*_=helper.setup_case(LiveAccount)
        a.physical_fill(0,.01,80000.,0.,t[0]);a.owner_fill('base',0,.01,80000.,0.,t[0],'seed')
        targets=[0.]*10;pair=[0.]*10;pair[1]=.01
        packet=a.seal(dict(source_prices=[80000.]*10,owner_targets={'base':targets,'pair':pair},
            owner_limits={'pair':{'1':79900.}}),p.p,p.sc)
        self.assertIsNone(packet['sealed_owner_limits'][('base',0)])
        self.assertEqual(packet['sealed_owner_limits'][('pair',1)],79900.)

class CausalityTests(unittest.TestCase):
    def history(self):
        times=np.arange(100)*3600;price=np.exp(np.arange(100)[:,None]*np.linspace(-.001,.001,10)[None,:])*100
        age=np.zeros_like(price);sigma=np.eye(10)*1e-6
        return times,price,age,sigma

    def test_future_prices_cannot_change_past_signal(self):
        t,p,a,s=self.history();first=features(t[80],t,p,a,s)
        p2=p.copy();p2[81:]*=1000
        changed=features(t[80],t,p2,a,s)
        for x,y in zip(first,changed):np.testing.assert_array_equal(x,y)

    def test_missing_or_stale_history_is_not_valid_evidence(self):
        t,p,a,s=self.history();a[:]=7200
        mu,strength,count=features(t[80],t,p,a,s)
        np.testing.assert_array_equal(count,np.zeros(10))
        np.testing.assert_array_equal(mu,np.zeros(10))

    def test_signal_is_symmetric_and_uses_all_ten_assets(self):
        t,p,a,s=self.history();mu,q,n=features(t[80],t,p,a,s)
        opposite=features(t[80],t,10000/p,a,s)
        np.testing.assert_allclose(mu,-opposite[0],atol=1e-14)
        np.testing.assert_allclose(q,-opposite[1],atol=1e-12)
        self.assertEqual(len(mu),10);self.assertTrue(np.all(n==3))
        self.assertTrue(np.all(q[:5]<0));self.assertTrue(np.all(q[5:]>0))

    def test_halt_flattens_even_off_schedule(self):
        t,p,a,s=self.history();protocol=load_profile()['trading']
        c=VenueMomentum(None,0,t,p,a)
        result=c.propose(t[80]+1,p[80],np.ones(10),50000.,np.zeros(10),s,np.ones(10),np.ones(10),protocol,halted=True)
        np.testing.assert_array_equal(result['target_qty'],np.zeros(10))

    def test_short_and_long_targets_obey_same_hard_risk_budget(self):
        from replay_lab.common_comparison.engine import risk_breaches
        t,p,a,s=self.history();protocol=load_profile()['trading'];c=VenueMomentum(None,0,t,p,a)
        r=c.propose(t[78],p[78],np.zeros(10),50000.,np.zeros(10),s,np.ones(10),np.ones(10),protocol)
        self.assertFalse(risk_breaches(np.array(r['target_qty']),50000.,p[78],s,protocol['risk']))
        self.assertTrue(any(x<0 for x in r['target_qty']));self.assertTrue(any(x>0 for x in r['target_qty']))

if __name__=='__main__':unittest.main()
