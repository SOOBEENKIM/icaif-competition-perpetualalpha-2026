import copy
import json
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from bot.execution import OutcomeUnknown
from competition_live.profile import ROOT,load_profile,START
from competition_live.test_candidate import FakeExecutor
from .account import external_ledger,SharedWalletAccount,reconcile_shared,isolated_snapshot
from .runner import ProtectedExecutor


class IsolationTests(unittest.TestCase):
    def setUp(self):
        self.p=load_profile();self.raw=ROOT/'reports/incumbent_constraints_review_20261021/raw'
    def read(self,n):return json.loads((self.raw/(n+'.json')).read_text())['response']
    def reserve(self):
        return external_ledger(self.p,self.read('latest_own_fills'),self.read('latest_own_userFunding'),set(),self.read('latest_own_positions')['time'])

    def test_actual_external_history_rebuild_is_deduplicated_and_conserved(self):
        fills=self.read('latest_own_fills');funds=self.read('latest_own_userFunding');cut=self.read('latest_own_positions')['time']
        a=external_ledger(self.p,fills,funds,set(),cut)
        b=external_ledger(self.p,fills+fills,funds+funds,set(),cut)
        np.testing.assert_array_equal(a.qty,b.qty);self.assertEqual(a.cash,b.cash)
        self.assertAlmostEqual(a.qty[1],-.4064);self.assertAlmostEqual(a.funding,-.680848)
        marks=np.ones(10);marks[1]=1017.30048/.4064
        self.assertAlmostEqual(a.reconcile(marks)-50000,-21.106502)

    def test_unknown_history_or_own_eth_cannot_be_silently_adopted(self):
        fills=[r for r in self.read('latest_own_fills') if r['coin']=='pas:ETH']
        with self.assertRaises(OutcomeUnknown):external_ledger(self.p,fills[1:],[],set(),10**15)
        with self.assertRaises(OutcomeUnknown):external_ledger(self.p,fills,[],{fills[0]['cloid']},10**15)

    def test_exact_account_recovery_keeps_own_performance_separate(self):
        profile=load_profile()
        account=SharedWalletAccount(profile,None,lambda *a:None,lambda *a:[])
        for f in self.read('latest_own_fills'):
            if f['coin']=='pas:ETH':continue
            j=profile['trading']['symbols'].index(f['coin'].split(':')[1])
            q=float(f['sz'])*(1 if f['side']=='B' else -1);price=float(f['px'])
            fee=float(f['fee'])/(abs(q)*price)*10000
            account.physical_fill(j,q,price,fee,f['time']/1000)
            account.owner_fill('base',j,q,price,fee,f['time']/1000,'external_fill')
            account.applied_fills.add((f['coin'],f['tid']))
        account.reserved=self.reserve()
        account.apply_actual_funding(self.read('latest_own_userFunding'))
        snapshot=dict(mode='unifiedAccount',positions=self.read('latest_own_positions'),spot=self.read('latest_own_spot'),orders=self.read('latest_own_orders'))
        value=reconcile_shared(account,snapshot,profile)
        self.assertAlmostEqual(value['strategy_equity']-50000,51.236142)
        self.assertAlmostEqual(value['teammate_equity']-50000,-21.106502)
        self.assertAlmostEqual(value['ledger_team_equity'],100030.12964)
        snapshot['positions']['assetPositions'][0]['position']['szi']='.03'
        with self.assertRaises(OutcomeUnknown):reconcile_shared(account,snapshot,profile)

    def test_risk_view_deducts_teammate_pnl_without_modifying_original(self):
        snapshot=dict(mode='unifiedAccount',positions=self.read('latest_own_positions'),spot=self.read('latest_own_spot'),orders=[dict(coin='pas:ETH',oid=9)])
        before=copy.deepcopy(snapshot);view=isolated_snapshot(snapshot,self.p,self.reserve())
        self.assertEqual(snapshot,before)
        self.assertEqual(len(view['positions']['assetPositions']),1)
        self.assertEqual(view['orders'],[])
        usdp=next(b for b in view['spot']['balances'] if b['coin']=='USDP')
        self.assertAlmostEqual(float(usdp['total']),100051.236142)
        snapshot['positions']['assetPositions'][1]['position']['szi']='-.5'
        with self.assertRaises(OutcomeUnknown):isolated_snapshot(snapshot,self.p,self.reserve())

    def test_packet_with_protected_leg_is_rejected_before_any_leg(self):
        ex=FakeExecutor();a=SharedWalletAccount(self.p,ex,lambda *a:None,lambda *a:[]);a.reserved=self.reserve()
        target=[.01,-.5]+[0.]*8
        intent=dict(packet_id='protected',decision_time=START,owner_targets={'base':[0.]*10,'pair':target})
        result=a.execute_books(intent,None,None,None,None,None,None)
        self.assertEqual(ex.orders,[]);self.assertEqual(result['orders_sent'],0)
        self.assertAlmostEqual(a.reserved.qty[1],-.4064)
        self.assertTrue(np.all(a.physical.qty==0))

    def test_executor_cannot_send_or_cancel_eth_even_if_packet_guard_is_bypassed(self):
        from types import SimpleNamespace
        executor=object.__new__(ProtectedExecutor)
        with self.assertRaises(OutcomeUnknown):executor.submit(SimpleNamespace(coin='pas:ETH'))
        with self.assertRaises(OutcomeUnknown):executor.cancel('pas:ETH','0x1')

    def test_forced_btc_close_does_not_touch_teammate_eth(self):
        now=START+2000;at=now*1000
        class CloseExecutor(FakeExecutor):
            def submit(self,order):
                self.orders.append(order)
                return {'filled':dict(oid=17,totalSz=order.size,avgPx='80000')}
        ex=CloseExecutor()
        rows=[dict(coin='pas:BTC',oid=17,tid=18,side='A',sz='.001',px='80000',fee='.072',feeToken='USDP',time=at)]
        a=SharedWalletAccount(self.p,ex,lambda *a:None,lambda *a:rows);a.reserved=self.reserve()
        a.physical_fill(0,.001,80000.,9.,now-10);a.owner_fill('base',0,.001,80000.,9.,now-10,'external_fill')
        original_reserve=copy.deepcopy(a.reserved.__dict__)
        book=dict(coin='pas:BTC',time=at,levels=[[dict(px='80000',sz='10')],[dict(px='80001',sz='10')]])
        stream=dict(book_times=[at],ctx_times=[at],book=[(at,0,book)],ctx=[(at,0,dict(markPx='80000',oraclePx='80000',funding='0'))],resets=[])
        raw=dict(packet_id='close',decision_time=now,source_close=now,source_available=now,source_prices=[80000.]*10,
            risk_forced=True,owner_targets={'base':[0.]*10,'pair':[0.]*10})
        sc=dict(fee_bps=9.,impact_bps=0.,depth_fraction=1.)
        intent=a.seal(raw,self.p['trading'],sc)
        with patch('competition_live.live_account.time.time',return_value=now):
            result=a.execute_books(intent,dict(BTC=stream),np.full(10,80000.),np.eye(10)*1e-8,self.p['trading'],sc,dict(book_max_age_ms=15000,context_max_age_ms=3000))
        self.assertEqual(a.physical.qty[0],0.);self.assertEqual(len(ex.orders),1)
        self.assertEqual(ex.orders[0].coin,'pas:BTC');self.assertTrue(ex.orders[0].reduce_only)
        np.testing.assert_array_equal(a.reserved.qty,original_reserve['qty'])
        self.assertEqual(a.reserved.cash,original_reserve['cash'])
        self.assertEqual(a.reserved.funding,original_reserve['funding'])


if __name__=='__main__':unittest.main()
