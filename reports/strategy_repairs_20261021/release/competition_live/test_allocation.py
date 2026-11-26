import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from hyperliquid.api import API
from .profile import load_profile,START,END
from .live_account import LiveAccount
from .candidate import build_candidate
from .runtime import save_coordinator,load_coordinator
from .runner import require_initial_account,reconcile_account,BalanceMismatch
from .execution import PASExchange,prepare_pas_order
from .venue import account_equity,phase_state
from .test_runtime import account_snapshot
from bot.execution import OutcomeUnknown


class AllocationTests(unittest.TestCase):
    def setUp(self):
        self.p=load_profile();self.a=LiveAccount(self.p,None,lambda *a:None,None)

    def test_team_grant_and_isolated_sleeve_budget(self):
        require_initial_account(account_snapshot(self.p),self.p)
        for cash in ['50000','0','1000000']:
            with self.assertRaises(ValueError):require_initial_account(account_snapshot(self.p,cash=cash),self.p)
        self.assertEqual(self.a.physical.initial,50000)
        self.assertEqual(self.a.reserved.cash,50000)
        self.assertEqual(self.a.team_equity(np.ones(10)),100000)
        self.assertEqual(self.p['trading']['risk']['asset_notional'],7500)
        self.assertEqual(self.p['configs']['reference']['pair_max_gross'],6000)

    def test_fee_loss_and_pnl_belong_to_user_reserve_is_conserved(self):
        self.a.physical.fill(0,.01,80000,9,START)
        self.a.owners['base'].fill(0,.01,80000,9,START)
        self.assertAlmostEqual(self.a.team_equity(np.full(10,81000.)),100009.28)
        self.assertEqual(self.a.reserved.cash,50000)
        s=account_snapshot(self.p,cash='99999.28',qty='.01',pnl='10')
        v=reconcile_account(self.a,s,self.p)
        self.assertAlmostEqual(v['equity'],100009.28)
        self.assertAlmostEqual(v['strategy_equity'],50009.28)
        self.a.reserved.cash-=1
        with self.assertRaises(OutcomeUnknown):reconcile_account(self.a,s,self.p)

    def test_unified_equity_brackets_observation_and_balance_error_is_separate(self):
        s=account_snapshot(self.p);s['positions_before']=copy.deepcopy(s['positions'])
        self.assertEqual(account_equity(s,self.p)['strategy_equity'],50000)
        s['spot']['balances'][0]['total']='99999'
        with self.assertRaises(BalanceMismatch):reconcile_account(self.a,s,self.p)

    def test_reserve_survives_restart_and_cannot_enter_pairs(self):
        c=build_candidate(self.p,lambda *a:None,account=self.a)['portfolio']
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'state.pkl';save_coordinator(path,c,self.p,'test')
            restored=load_coordinator(path,self.p,None,None,None,None)
            self.assertEqual(restored.account.reserved.cash,50000)
            self.assertEqual(restored.p['initial'],50000)
            self.assertFalse(restored.account.reserved.qty.any())

    def test_start_end_and_wrong_market_are_blocked_below_sdk(self):
        client=object.__new__(PASExchange);client.base_url=self.p['rest_url'];client.orders_enabled=True
        client.scored_profile=self.p;client.pas_asset_ids={2810000}
        order={'action':{'type':'order','orders':[{'a':2810000,'r':False}]}}
        with patch.object(API,'post',return_value={}) as send:
            for t in [START-1,END-1,END]:
                with patch('competition_live.execution.time.time',return_value=t):
                    with self.assertRaises(ValueError):client.post('/exchange',order)
            self.assertFalse(send.called)
            with patch('competition_live.execution.time.time',return_value=START):
                client.post('/exchange',order)
                bad=copy.deepcopy(order);bad['action']['orders'][0]['a']=2670000
                with self.assertRaises(ValueError):client.post('/exchange',bad)
            self.assertEqual(send.call_count,1)


if __name__=='__main__':unittest.main()
