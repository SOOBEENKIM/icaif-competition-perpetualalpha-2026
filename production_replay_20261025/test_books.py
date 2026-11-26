import copy,unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from competition_live.profile import load_profile
from bot.execution import OrderRejected
from .engine import build
from .book_replay import BookVenue
from .resting_book import RestingBook

class RecordedBookTests(unittest.TestCase):
    def case(self):
        p=load_profile();start=p['start_unix'];streams={}
        for s in p['trading']['symbols']:
            times=[(start+i)*1000 for i in range(11)]
            books=[(t,1,dict(coin='pas:'+s,time=t,levels=[[dict(px='99.99',sz='1')],[dict(px='100.01',sz='1')]])) for t in times]
            ctx=[(t,1,dict(markPx='100',oraclePx='100',funding='0')) for t in times]
            streams[s]=dict(book=books,book_times=times,ctx=ctx,ctx_times=times,resets=[])
        sc=dict(fee_bps=9.,depth_fraction=1.,latency_ms=1000,receipt_multiplier=.5)
        v=BookVenue(streams,p,sc,start,start+10,[]);v.stream=v.refresh
        port=build(v,'incumbent');return v,port,start

    def order(self,size=1.,price=100.1,side=True):
        return SimpleNamespace(coin='pas:BTC',is_buy=side,size=size,price=price)

    def test_limit_is_rechecked_against_arrival(self):
        v,p,t=self.case();b=v.streams['BTC']['book'][1][2];b['levels'][1][0]['px']='101'
        v.send(self.order());self.assertEqual(v.orders[-1]['filled'],0.)

    def test_unchanged_later_snapshot_does_not_replenish_simulated_consumption(self):
        v,p,t=self.case();v.send(self.order());v.send(self.order())
        self.assertEqual([r['filled'] for r in v.orders],[1.,0.])

    def test_only_observed_increase_replenishes_depth(self):
        v,p,t=self.case();v.send(self.order())
        v.streams['BTC']['book'][2][2]['levels'][1][0]['sz']='1.4'
        v.send(self.order());self.assertAlmostEqual(v.orders[-1]['filled'],.4)

    def test_disappearance_then_reappearance_is_new_observed_capacity(self):
        v,p,t=self.case();v.send(self.order())
        v.streams['BTC']['book'][2][2]['levels'][1]=[]
        v.now=t+2;v.send(self.order());self.assertEqual(v.orders[-1]['filled'],1.)

    def test_disconnected_context_cannot_be_used(self):
        v,p,t=self.case();v.streams['BTC']['resets']=[(t+1)*1000];v.now=t+1.5
        v.streams['BTC']['ctx_times']=[t*1000]
        v.streams['BTC']['ctx']=v.streams['BTC']['ctx'][:1]
        self.assertIsNone(v.mark('BTC'))

    def test_no_future_context_in_funding_and_receipt_haircut(self):
        v,p,t=self.case();a=p.account
        for ledger in [a.physical,a.owners['base']]:ledger.fill(0,1.,100.,0.,t-.1)
        v.funding=[dict(time=(t+1)*1000,coin='pas:BTC',fundingRate=-.001)]
        v.streams['BTC']['ctx'][2][2]['oraclePx']='10000'
        v.settle((t+2)*1000,a);v.settle((t+2)*1000,a)
        self.assertAlmostEqual(a.physical.funding,.05)
        self.assertAlmostEqual(a.owners['base'].funding,.05)

    def test_missing_funding_oracle_is_explicit_failure(self):
        v,p,t=self.case();a=p.account
        for ledger in [a.physical,a.owners['base']]:ledger.fill(0,1.,100.,0.,t-.1)
        v.funding=[dict(time=(t+10)*1000,coin='pas:BTC',fundingRate=.001)]
        v.streams['BTC']['ctx_times']=[t*1000];v.streams['BTC']['ctx']=v.streams['BTC']['ctx'][:1]
        v.settle((t+10)*1000,a)
        self.assertEqual(v.failures[0]['kind'],'missing_funding_oracle')

    def test_actual_account_path_with_observed_books(self):
        v,p,t=self.case();marks=np.full(10,100.);target=np.array([1.]+[0.]*9)
        with patch('competition_live.live_account.time',SimpleNamespace(time=lambda:v.now)):
            p.submit_baseline(t,target,v.streams,marks,np.eye(10)*1e-6,False)
        self.assertAlmostEqual(p.account.physical.qty[0],1.)
        self.assertAlmostEqual(p.account.owners['base'].qty[0],1.)
        self.assertAlmostEqual(p.account.reconcile(marks),50000-.01-.090009)
        self.assertFalse(p.account.pending_orders)

class RestingOrderTests(unittest.TestCase):
    def book(self):
        return RestingBook([dict(key='a:1',symbol='BTC',side='A',price=100.,size=10.,start=1000,end=10000,
                                 fills=[[6000,4.]])],1000,.5)
    def test_before_acceptance_latency_or_after_cancel_no_liquidity(self):
        b=self.book();self.assertFalse(b.levels('BTC',1999)[1]);self.assertFalse(b.levels('BTC',10000)[1])
    def test_public_partial_fills_and_simulation_both_reduce_capacity(self):
        b=self.book();self.assertEqual(b.sweep('BTC',2000,1,2,100,3)[0],2)
        self.assertEqual(b.sweep('BTC',6001,1,10,100,3)[0],1)
        self.assertEqual(b.sweep('BTC',7000,1,10,100,3)[0],0)
    def test_future_partial_fill_does_not_reduce_current_book(self):
        b=self.book();self.assertEqual(float(b.levels('BTC',5000)[1][0]['sz']),5.)
    def test_wrong_side_or_outside_limit_does_not_fill(self):
        b=self.book();self.assertEqual(b.sweep('BTC',2000,-1,10,100,3)[0],0.)
        self.assertEqual(b.sweep('BTC',2000,1,10,99.99,3)[0],0.)

if __name__=='__main__':unittest.main()
