import json
import time
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np

from bot.execution import OutcomeUnknown
from .profile import load_profile,START
from .candidate import build_candidate
from .live_account import LiveAccount


class FakeExecutor:
    def __init__(self):self.orders=[]
    def market(self,coin):return {'name':coin,'szDecimals':5},{'markPx':'80000'}
    def set_risk_state(self,qty,sigma):pass
    def submit(self,order):
        self.orders.append(order)
        return {'filled':{'oid':77,'totalSz':'.001','avgPx':'80000'}}


class CandidateTests(unittest.TestCase):
    def setUp(self):
        self.profile=load_profile();self.executor=FakeExecutor();self.events=[]
        self.now=START+900
        at=self.now*1000
        book={'coin':'par:BTC','time':at,'levels':[[{'px':'79999','sz':'10'}],[{'px':'80000','sz':'10'}]]}
        ctx={'oraclePx':'80000','markPx':'80000','funding':'0'}
        self.streams={'BTC':{'book_times':[at],'ctx_times':[at],
            'book':[(at,0,book)],'ctx':[(at,0,ctx)],'resets':[]}}
        self.rows=[{'coin':'par:BTC','oid':77,'tid':123,'side':'B','sz':'.001',
            'px':'80000','fee':'.072','feeToken':'USDPR','time':at}]
        self.account=LiveAccount(self.profile,self.executor,lambda *a:self.events.append(a),lambda *a:self.rows)
        self.p=self.profile['trading'];self.sc={'fee_bps':9.,'impact_bps':0.,'depth_fraction':1.}
        self.cfg={'book_max_age_ms':15000,'context_max_age_ms':3000}
        raw=dict(packet_id='actual-test',decision_time=self.now,source_close=self.now,source_available=self.now,
            source_prices=[80000.]*10,risk_forced=False,owner_targets={'base':[0.]*10,'pair':[.002]+[0.]*9})
        self.intent=self.account.seal(raw,self.p,self.sc)

    def test_factory_preserves_selected_gate_and_new_phase(self):
        candidate=build_candidate(self.profile,lambda *a:None,account=self.account)
        self.assertEqual(candidate['portfolio'].model,'decision_gate')
        self.assertIs(candidate['portfolio'].account,self.account)
        self.assertEqual(candidate['plan']['paper_start_unix'],START)
        self.assertEqual(candidate['plan']['trading_protocol']['initial'],1000000)
        self.assertEqual(candidate['plan']['book_protocol']['pair_max_gross'],120000)
        self.assertEqual(candidate['signals'].gate_spec['horizon'],24)

    def run_packet(self):
        with patch('practice_candidate.live_account.time.time',return_value=self.now):
            return self.account.execute_books(self.intent,self.streams,np.array([80000.]*10),np.eye(10)*1e-8,
                self.p,self.sc,self.cfg)

    def test_partial_receipt_only_applies_real_fills_and_actual_fee(self):
        result=self.run_packet()
        self.assertEqual(result['orders'][0]['status'],'ACTUAL_PARTIAL')
        self.assertAlmostEqual(self.account.physical.qty[0],.001)
        self.assertAlmostEqual(self.account.owners['pair'].qty[0],.001)
        self.assertAlmostEqual(self.account.physical.fees,.072)
        self.assertAlmostEqual(self.account.reconcile(np.array([80000.]*10)),999999.928)
        self.assertTrue(all(o.coin.startswith('par:') for o in self.executor.orders))
        self.assertIn('before_order',[event[0] for event in self.events])
        with self.assertRaises(RuntimeError):self.run_packet()

    def test_missing_actual_fill_does_not_create_assumed_inventory(self):
        self.rows=[]
        with self.assertRaises(OutcomeUnknown):self.run_packet()
        self.assertEqual(self.account.physical.qty[0],0.)
        self.assertIn('BTC',self.account.unknown)
        self.assertTrue(self.account.pending_orders)

    def test_wrong_currency_fill_is_not_accepted(self):
        self.rows[0]['feeToken']='USDPRR'
        with self.assertRaises(OutcomeUnknown):self.run_packet()
        self.assertEqual(self.account.physical.qty[0],0.)

    def test_funding_uses_actual_par_payment_once(self):
        self.run_packet()
        row={'time':(self.now+1)*1000,'delta':{'coin':'par:BTC','szi':'.001','usdc':'-.01'}}
        unrelated={'time':row['time'],'delta':{'coin':'pas:BTC','szi':'1','usdc':'100'}}
        self.account.apply_actual_funding([unrelated,row,row])
        self.assertAlmostEqual(self.account.physical.funding,-.01)
        self.assertAlmostEqual(self.account.owners['pair'].funding,-.01)
        self.assertAlmostEqual(self.account.reconcile(np.array([80000.]*10)),999999.918)


if __name__=='__main__':unittest.main()
