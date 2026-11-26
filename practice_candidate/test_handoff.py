import copy
import json
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from hyperliquid.api import API
from .profile import ROOT, load_profile, END
from .live_account import LiveAccount
from .handoff import bootstrap
from .runner import reconcile_account
from .venue import account_equity, validate_metadata, phase_state
from .execution import PARExchange, prepare_par_order
from bot.execution import OutcomeUnknown


class HandoffTests(unittest.TestCase):
    def setUp(self):
        self.p=load_profile();folder=ROOT/'reports/practice_candidate_activation_20261018'
        self.snapshot=json.loads((folder/'preflight.json').read_text())
        self.history=json.loads((folder/'history.json').read_text())
        self.a=LiveAccount(self.p,None,lambda *a:None,None)

    def test_actual_handoff_preserves_inventory_costs_and_owner_conservation(self):
        bootstrap(self.a,self.history,self.p)
        np.testing.assert_allclose(self.a.physical.qty,[.4682,0,85.78,0,0,0,0,0,0,0])
        self.assertAlmostEqual(self.a.physical.fees,40.914225)
        self.assertAlmostEqual(self.a.physical.funding,209.185069)
        self.assertEqual(len(self.a.applied_fills),14)
        self.assertFalse(self.a.owners['pair'].qty.any())
        self.a.reconcile(np.full(10,80000.))
        value=reconcile_account(self.a,self.snapshot,self.p)
        self.assertAlmostEqual(value['equity'],999604.503034)
        self.a.apply_actual_funding(self.history['userFunding'])
        self.assertAlmostEqual(self.a.physical.funding,209.185069)

    def test_missing_fill_or_extra_transfer_cannot_be_bootstrapped(self):
        for kind in ['userFillsByTime','userNonFundingLedgerUpdates']:
            h=copy.deepcopy(self.history)
            if kind=='userFillsByTime':h[kind].pop(0)
            else:h[kind].append(h[kind][0])
            with self.assertRaises(ValueError):bootstrap(LiveAccount(self.p,None,lambda *a:None,None),h,self.p)

    def test_unified_equity_is_not_double_counted_and_snapshot_skew_is_bracketed(self):
        bootstrap(self.a,self.history,self.p)
        s=copy.deepcopy(self.snapshot);s['positions_before']=copy.deepcopy(s['positions'])
        s['positions']['assetPositions'][0]['position']['unrealizedPnl']=str(-356.4322+20)
        self.assertAlmostEqual(account_equity(s,self.p)['equity'],999604.503034)
        reconcile_account(self.a,s,self.p)
        s['spot']['balances'][0]['total']='999000'
        with self.assertRaises(OutcomeUnknown):reconcile_account(self.a,s,self.p)

    def test_par_metadata_and_pas_rejection_and_end_boundary(self):
        self.assertEqual(validate_metadata(self.snapshot,self.p)['collateral'],'USDPR')
        self.assertEqual(phase_state(END-901,self.p),'trading')
        self.assertEqual(phase_state(END-900,self.p),'closing_only')
        self.assertEqual(phase_state(END,self.p),'ended')
        with self.assertRaises(ValueError):prepare_par_order(self.p,'pas:BTC',True,.01,80000,
            {'name':'pas:BTC','szDecimals':5},{'markPx':'80000'},slot='wrong')
        client=object.__new__(PARExchange);client.base_url=self.p['rest_url'];client.orders_enabled=True
        client.scored_profile=self.p;client.par_asset_ids={2670000}
        with patch('practice_candidate.execution.time.time',return_value=END-60),patch.object(API,'post',return_value={}) as send:
            with self.assertRaises(ValueError):client.post('/exchange',{'action':{'type':'order','orders':[{'a':2670000,'r':False}]}})
            with self.assertRaises(ValueError):client.post('/exchange',{'action':{'type':'order','orders':[{'a':2810000,'r':True}]}})
            client.post('/exchange',{'action':{'type':'order','orders':[{'a':2670000,'r':True}]}})
            self.assertEqual(send.call_count,1)
        with patch('practice_candidate.execution.time.time',return_value=END):
            with self.assertRaises(ValueError):client.post('/exchange',{'action':{'type':'cancel','cancels':[{'a':2670000}]}})


if __name__=='__main__':unittest.main()
