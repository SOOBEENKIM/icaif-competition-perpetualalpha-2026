import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np

from .profile import load_profile,START
from .candidate import build_candidate
from .live_account import LiveAccount
from .runtime import save_coordinator,load_coordinator,private_load
from .runner import require_initial_account,reconcile_account,load_publication,LiveRunner
from .test_candidate import FakeExecutor
from bot.execution import OutcomeUnknown


def account_snapshot(profile,*,cash='100000',qty='0',entry='80000',pnl='0'):
    positions=[] if float(qty)==0 else [{'position':dict(coin='pas:BTC',szi=qty,entryPx=entry,unrealizedPnl=pnl)}]
    return dict(mode='unifiedAccount',orders=[],positions=dict(assetPositions=positions,time=int(START*1000),
        marginSummary={'accountValue':str(float(cash)+float(pnl))}),
        spot={'balances':[dict(coin='USDP',token=profile['collateral_token'],total=str(float(cash)+float(pnl)))]})


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.profile=load_profile();self.executor=FakeExecutor()
        self.account=LiveAccount(self.profile,self.executor,lambda *a:None,lambda *a:[])
        self.portfolio=build_candidate(self.profile,lambda *a:None,account=self.account)['portfolio']

    def test_initial_grant_must_match_and_account_be_flat(self):
        require_initial_account(account_snapshot(self.profile),self.profile)
        for s in [account_snapshot(self.profile,cash='0'),account_snapshot(self.profile,cash='1000000'),
                  account_snapshot(self.profile,qty='.1')]:
            with self.assertRaises(ValueError):require_initial_account(s,self.profile)

    def test_restart_preserves_confirmed_inventory_and_unresolved_orders_without_clients(self):
        self.account.physical.fill(0,.01,80000,9,START)
        self.account.owners['pair'].fill(0,.01,80000,9,START)
        self.account.applied_fills.add(('pas:BTC',99))
        self.account.pending_orders['unresolved']={'oid':100}
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'state.pkl';save_coordinator(path,self.portfolio,self.profile,'test')
            detached=private_load(path)['portfolio']
            self.assertIsNone(detached.account.executor);self.assertIsNone(detached.account.checkpoint)
            self.assertIsNone(detached.settlement)
            loaded=load_coordinator(path,self.profile,self.executor,lambda *a:None,lambda *a:[],lambda *a:None)
            self.assertAlmostEqual(loaded.account.physical.qty[0],.01)
            self.assertEqual(loaded.account.applied_fills,{('pas:BTC',99)})
            self.assertIn('unresolved',loaded.account.pending_orders)
            self.assertIs(loaded.account.executor,self.executor)
            changed=copy.deepcopy(self.profile);changed['trading']['risk']['asset_notional']=14000
            with self.assertRaises(ValueError):load_coordinator(path,changed,None,None,None,None)

    def test_account_reconciliation_checks_cash_entry_and_inventory(self):
        self.account.physical.fill(0,.01,80000,9,START)
        self.account.owners['pair'].fill(0,.01,80000,9,START)
        okay=account_snapshot(self.profile,cash='99999.28',qty='.01',pnl='123.45')
        reconcile_account(self.account,okay,self.profile)
        bad=copy.deepcopy(okay);bad['spot']['balances'][0]['total']='100001'
        with self.assertRaises(OutcomeUnknown):reconcile_account(self.account,bad,self.profile)
        bad=copy.deepcopy(okay);bad['positions']['assetPositions'][0]['position']['entryPx']='79000'
        with self.assertRaises(OutcomeUnknown):reconcile_account(self.account,bad,self.profile)
        bad=copy.deepcopy(okay);bad['positions']['assetPositions'][0]['position']['szi']='.02'
        with self.assertRaises(OutcomeUnknown):reconcile_account(self.account,bad,self.profile)

    def streams(self,now):
        at=int(now*1000)-5
        return {s:dict(book_times=[at],ctx_times=[at],book=[(at,0,dict(time=at,levels=[[],[]]))],
            ctx=[(at,0,dict(markPx='80000',oraclePx='80000',funding='0'))],resets=[])
            for s in self.profile['trading']['symbols']}

    def test_fresh_marks_value_inactive_markets_but_empty_books_cannot_fill(self):
        now=START+900.123
        marks,quotes=self.portfolio.prices(now,self.streams(now))
        self.assertEqual(marks.tolist(),[80000.]*10)
        self.assertTrue(all(q is None for q in quotes.values()))
        marks,_=self.portfolio.prices(now+4,self.streams(now))
        self.assertTrue(np.all(marks==0))
        self.assertEqual(self.portfolio.unpriced_flat_assets,self.profile['trading']['symbols'])

    def test_poll_jitter_one_entry_check_per_bucket_and_no_late_entry(self):
        now=START+900.123
        event={'received_ms':(now-1)*1000,'data':{'frame':dict(source_time=START,sigma=np.eye(10)*1e-8)}}
        self.portfolio.step(now,self.streams(now),event,[])
        self.assertEqual(self.portfolio.pair_bars,{now//900})
        self.portfolio.step(now+10,self.streams(now+10),event,[])
        self.assertEqual(len(self.portfolio.pair_bars),1)
        late=START+1831
        self.portfolio.step(late,self.streams(late),event,[])
        self.assertEqual(len(self.portfolio.pair_bars),1)

    def test_runner_restart_refuses_unresolved_order_before_sending(self):
        with tempfile.TemporaryDirectory() as temp:
            directory=Path(temp);self.account.pending_orders['unresolved']={'oid':100}
            save_coordinator(directory/'coordinator.pkl',self.portfolio,self.profile,'test')
            with self.assertRaises(OutcomeUnknown):LiveRunner(self.profile,directory,None,self.executor)
            self.assertEqual(self.executor.orders,[])

    def test_missing_or_wrong_signal_never_creates_an_entry(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'signal.json'
            self.assertEqual(load_publication(path,self.profile,START),(None,[]))
            path.write_text(json.dumps({'candidate':'wrong','profile_hash':'wrong'}))
            with self.assertRaises(ValueError):load_publication(path,self.profile,START)

    def test_selected_pair_entry_and_forced_exit_use_confirmed_pas_fills(self):
        now=START+900.123;rows=[];orders=[]
        profile=self.profile
        class FilledVenue:
            def set_risk_state(self,*args):pass
            def market(self,coin):
                j=profile['trading']['symbols'].index(coin.split(':')[1])
                return dict(name=coin,szDecimals=profile['trading']['execution']['decimals'][j]),dict(markPx='80000')
            def submit(self,order):
                orders.append(order);oid=len(orders)
                rows.append(dict(coin=order.coin,oid=oid,tid=oid,side='B' if order.is_buy else 'A',
                    sz=order.size,px='80000',fee=str(float(order.size)*80000*.0009),feeToken='USDP',time=int(now*1000)))
                return {'filled':dict(oid=oid,totalSz=order.size,avgPx='80000')}
        self.account.executor=FilledVenue()
        self.account.collect_fills=lambda oid,*a:[r for r in rows if r['oid']==oid]
        def streams():
            result=self.streams(now)
            for s in result.values():s['book'][0][2]['levels']=[[dict(px='80000',sz='10')],[dict(px='80000',sz='10')]]
            return result
        forecast=dict(model='decision_gate',economic_available=True,economic_action=1,pair=['BTC','ETH'],
            ids=[0,1],horizon=24,forecast_source=START,available_time=START+5,
            source_prices=[80000,80000],source_beta=1.,source_vol=.01,mu=.05,terminal_prices=[84000,76000])
        event={'received_ms':(now-1)*1000,'data':{'frame':dict(source_time=START,sigma=np.eye(10)*1e-8)}}
        with patch('competition_live.live_account.time.time',side_effect=lambda:now):
            self.portfolio.step(now,streams(),event,[forecast])
            self.assertIsNotNone(self.portfolio.pair)
            self.assertEqual(self.portfolio.pair.state,'open')
            self.assertEqual(len(orders),2)
            self.assertGreater(self.account.physical.qty[0],0)
            self.assertLess(self.account.physical.qty[1],0)
            self.portfolio.halted=True;now+=30
            self.portfolio.step(now,streams(),event,[forecast])
        self.assertTrue(np.all(self.account.physical.qty==0))
        self.assertEqual(self.portfolio.pair.state,'closed')
        self.assertEqual(len(orders),4)
        self.assertTrue(all(o.reduce_only for o in orders[2:]))
        self.assertEqual(len(self.account.physical.fills),4)
        self.assertAlmostEqual(self.account.reconcile(np.array([80000.]*10)),50000-self.account.physical.fees)


if __name__=='__main__':unittest.main()
