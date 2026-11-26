import unittest
from unittest.mock import patch
import numpy as np
from competition_live.live_account import LiveAccount
from competition_live.profile import load_profile,START
from competition_live.candidate import build_candidate
from .execution import FreshLiveAccount

class ExecutionTests(unittest.TestCase):
    def setup_case(self,cls):
        profile=load_profile();clock=[START+130.];rows=[];orders=[];fetches=[]
        def stream(symbol):
            fetches.append(symbol);at=int(clock[0]*1000)-5
            b=dict(time=at,levels=[[dict(px='80000',sz='10')],[dict(px='80000',sz='10')]])
            return dict(book_times=[at],book=[(at,0,b)],ctx_times=[at],ctx=[(at,0,dict(markPx='80000',oraclePx='80000',funding='0'))],resets=[])
        class Venue:
            def market(self,coin):return dict(name=coin,szDecimals=profile['trading']['execution']['decimals'][profile['trading']['symbols'].index(coin[4:])]),dict(markPx='80000')
            def set_risk_state(self,*args):pass
            def submit(self,order):
                oid=len(orders)+1;orders.append(order);clock[0]+=4
                rows.append(dict(coin=order.coin,oid=oid,tid=oid,side='B' if order.is_buy else 'A',sz=order.size,px='80000',fee=str(float(order.size)*80000*.0009),feeToken='USDP',time=int(clock[0]*1000)))
                return {'filled':dict(oid=oid,totalSz=order.size,avgPx='80000')}
        kwargs=dict(refresh_market=stream) if cls is FreshLiveAccount else {}
        a=cls(profile,Venue(),lambda *x:None,lambda oid,*x:[r for r in rows if r['oid']==oid],**kwargs)
        p=build_candidate(profile,lambda *a:None,account=a)['portfolio']
        return a,p,clock,orders,stream,fetches
    def execute(self,a,p,clock,stream,target,packet):
        intent=a.seal(dict(packet_id=packet,decision_time=clock[0],source_close=clock[0],source_available=clock[0],source_prices=[80000.]*10,risk_forced=False,owner_targets={'base':target,'pair':[0.]*10}),p.p,p.sc)
        streams={s:stream(s) for s in p.p['symbols']}
        with patch('competition_live.live_account.time.time',side_effect=lambda:clock[0]):
            return a.execute_books(intent,streams,np.full(10,80000.),np.eye(10)*1e-8,p.p,p.sc,p.cfg)
    def test_original_loses_later_asset_after_confirmed_fill_wait(self):
        a,p,t,orders,stream,_=self.setup_case(LiveAccount)
        self.execute(a,p,t,stream,[.01,.01]+[0.]*8,'old')
        self.assertEqual(len(orders),1);self.assertEqual(a.physical.qty[1],0.)
    def test_candidate_refreshes_second_asset_and_keeps_exact_accounting(self):
        a,p,t,orders,stream,_=self.setup_case(FreshLiveAccount)
        self.execute(a,p,t,stream,[.01,.01]+[0.]*8,'new')
        self.assertEqual(len(orders),2);np.testing.assert_allclose(a.physical.qty[:2],[.01,.01])
        self.assertAlmostEqual(a.reconcile(np.full(10,80000.)),50000-a.physical.fees)
    def test_reversal_closes_confirmed_long_then_opens_short(self):
        a,p,t,orders,stream,_=self.setup_case(FreshLiveAccount)
        self.execute(a,p,t,stream,[.01]+[0.]*9,'open')
        self.execute(a,p,t,stream,[-.01]+[0.]*9,'close')
        self.assertEqual(a.physical.qty[0],0.);self.assertTrue(orders[-1].reduce_only)
        self.execute(a,p,t,stream,[-.01]+[0.]*9,'short')
        self.assertEqual(a.physical.qty[0],-.01);self.assertFalse(orders[-1].is_buy);self.assertFalse(orders[-1].reduce_only)
    def test_stale_refresh_cannot_create_order(self):
        a,p,t,orders,stream,_=self.setup_case(FreshLiveAccount)
        bad=stream('BTC');a.refresh_market=lambda symbol:bad;t[0]+=10
        self.execute(a,p,t,stream,[.01]+[0.]*9,'stale');self.assertEqual(orders,[])

if __name__=='__main__':unittest.main()
