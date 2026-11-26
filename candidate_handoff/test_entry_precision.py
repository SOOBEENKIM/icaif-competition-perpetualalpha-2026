"""Replay the real false-positive halt, while preserving missing-trade guards."""
import copy,json,unittest
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
from .bridge import HistoryBridge
from competition_live.profile import ROOT,load_profile
from joint_wallet.audit import reconstruct
from joint_wallet.book import ReconciliationError

CAPTURE=ROOT/'reports/pas_cashflow_research_20261023/entry_reconciliation_before/public'

def fixture():
    def read(name):return json.loads((CAPTURE/name).read_text())['response']
    p=load_profile();fills=read('017_userFillsByTime.json');funding=read('018_userFunding.json')
    # Public order IDs are fixed with this captured evidence, not future state.
    ids=json.loads((CAPTURE.parent/'ownership_ids.json').read_text())
    book=reconstruct(p['trading']['symbols'],{'ours':50000.,'teammate':50000.},fills,funding,set(ids['cloids']),set(ids['oids']))
    state=dict(mode=read('019_userAbstraction.json'),positions_before=read('020_clearinghouseState.json'),
        spot=read('021_spotClearinghouseState.json'),positions=read('022_clearinghouseState.json'),orders=read('023_openOrders.json'))
    market=read('024_metaAndAssetCtxs.json')
    api=SimpleNamespace(account=lambda:state,market=lambda:market)
    bridge=HistoryBridge(ROOT,ROOT/'.runtime/pas_budget_lifecycle',p,api)
    return bridge,book,state,market

class EntryPrecisionTests(unittest.TestCase):
    def test_reporting_precision_and_trailing_zeros(self):
        from .entry_precision import matches_reported_entry as matches
        self.assertTrue(matches('2393.3',2393.3077024407676,4))
        self.assertFalse(matches('2393.31',2393.3077024407676,4))
        self.assertTrue(matches('1.12178',1.121786629138672,1))
        self.assertFalse(matches('NaN',2393.3,4))
        self.assertFalse(matches('2393.3',float('inf'),4))

    def test_changed_market_precision_still_rejected(self):
        bridge,book,state,market=fixture();market[0]['universe'][1]['szDecimals']=3
        with patch('candidate_handoff.bridge.time.time',return_value=state['positions']['time']/1000):
            with self.assertRaisesRegex(ReconciliationError,'precision'):bridge.reconcile_exchange(book)

    def test_captured_complete_history_reconciles_without_changing_cost_basis(self):
        bridge,book,state,_=fixture();before=book.physical.entry.copy()
        with patch('candidate_handoff.bridge.time.time',return_value=state['positions']['time']/1000):
            checked=bridge.reconcile_exchange(book)
        np.testing.assert_array_equal(before,book.physical.entry)
        self.assertLess(abs(checked['equity_error']),.05)

    def test_one_full_reporting_unit_of_false_entry_is_still_rejected(self):
        bridge,book,state,_=fixture()
        for snap in [state['positions_before'],state['positions']]:
            row=next(x['position'] for x in snap['assetPositions'] if x['position']['coin']=='pas:ETH')
            row['entryPx']='2393.32'
        with patch('candidate_handoff.bridge.time.time',return_value=state['positions']['time']/1000):
            with self.assertRaisesRegex(ReconciliationError,'average entry'):bridge.reconcile_exchange(book)

    def test_missing_quantity_still_rejected(self):
        bridge,book,state,_=fixture();state['positions']['assetPositions'][0]['position']['szi']='0.01'
        with patch('candidate_handoff.bridge.time.time',return_value=state['positions']['time']/1000):
            with self.assertRaisesRegex(ReconciliationError,'inventory changed'):bridge.reconcile_exchange(book)

    def test_cash_mismatch_still_rejected(self):
        bridge,book,state,_=fixture()
        row=next(x for x in state['spot']['balances'] if x['coin']=='USDP');row['total']=str(float(row['total'])+1)
        with patch('candidate_handoff.bridge.time.time',return_value=state['positions']['time']/1000):
            with self.assertRaisesRegex(ReconciliationError,'Equity outside'):bridge.reconcile_exchange(book)

if __name__=='__main__':unittest.main()
