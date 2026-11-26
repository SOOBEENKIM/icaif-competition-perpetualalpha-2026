import copy,unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from production_replay_20261025.test_books import RecordedBookTests
from .account import PairOpeningAccount,candidates
from .coordinator import Portfolio

class OpeningTests(unittest.TestCase):
 def fixture(self,side=1):
  v,p,t=RecordedBookTests().case();a=p.account;a.__class__=PairOpeningAccount
  marks=np.full(10,100.);sigma=np.eye(10)*1e-6
  for l in [a.physical,a.owners['base']]:l.fill(0,-side*.2,100.,0.,t-1)
  base=a.owners['base'].qty.copy();pair=np.zeros(10);pair[0]=side;pair[1]=-side
  raw=dict(packet_id='pair:test:1',decision_time=t,source_close=t,source_available=t,source_prices=marks,
   owner_targets=dict(base=base.tolist(),pair=pair.tolist()),owner_limits={'pair':{'0':100.1 if side>0 else 99.9,'1':99.9 if side>0 else 100.1}},risk_forced=False,sigma=sigma)
  return v,p,t,marks,sigma,a.seal(raw,p.p,p.sc)
 def run_packet(self,v,p,marks,sigma,intent):
  with patch('competition_live.live_account.time',SimpleNamespace(time=lambda:v.now)):
   return p.account.execute_books(intent,v.streams,marks,sigma,p.p,p.sc,p.cfg,on_time=v.settle)
 def test_long_and_short_reversal_complete_with_actual_three_receipts(self):
  for side in [1,-1]:
   v,p,t,m,s,intent=self.fixture(side);r=self.run_packet(v,p,m,s,intent)
   self.assertEqual(len(v.orders),3);self.assertTrue(r['bounded_zero_continuation'])
   np.testing.assert_allclose(p.account.owners['pair'].qty[:2],[side,-side],atol=1e-9)
   np.testing.assert_allclose(p.account.physical.qty[:2],[.8*side,-side],atol=1e-9)
   self.assertAlmostEqual(p.account.owners['base'].qty[0],-.2*side)
   self.assertGreater(p.account.physical.fees,0);p.account.reconcile(m)
 def test_partial_physical_close_is_not_retried(self):
  v,p,t,m,s,intent=self.fixture()
  for row in v.streams['BTC']['book']:row[2]['levels'][1][0]['sz']='0.1'
  r=self.run_packet(v,p,m,s,intent);self.assertFalse(r.get('bounded_zero_continuation',False))
  self.assertEqual(len(v.orders),1);self.assertAlmostEqual(p.account.physical.qty[0],-.1)
  self.assertEqual(p.account.owners['pair'].qty[1],0.)
 def test_continuation_preserves_original_price_limit(self):
  v,p,t,m,s,intent=self.fixture()
  for row in v.streams['BTC']['book'][2:]:row[2]['levels'][1][0]['px']='101'
  self.run_packet(v,p,m,s,intent)
  # The continuation is submitted but has zero fills at the original limit.
  self.assertEqual(len(v.orders),3)
  self.assertEqual(v.orders[1]['limit'],100.1)
  self.assertEqual(v.orders[1]['filled'],0.)
  self.assertEqual(sum(o['filled']>0 for o in v.orders),2)
  np.testing.assert_allclose(p.account.owners['pair'].qty[:2],[.2,-.2])
 def test_ambiguous_receipt_stops_without_second_packet(self):
  v,p,t,m,s,intent=self.fixture();p.account.collect_fills=lambda *args:[]
  with self.assertRaises(Exception):self.run_packet(v,p,m,s,intent)
  self.assertEqual(len(v.orders),1);self.assertTrue(p.account.unknown)
 def test_base_and_close_packets_cannot_request_opening_continuation(self):
  v,p,t,m,s,intent=self.fixture();before=p.account.physical.qty.copy();pair=p.account.owners['pair'].qty.copy()
  for change in [dict(packet_id='base:1'),dict(risk_forced=True)]:
   self.assertEqual(candidates(dict(intent,**change),before,pair),[])

class CadenceTests(unittest.TestCase):
 def test_offset45_gets_one_check_per_bucket(self):
  v,seed,t=RecordedBookTests().case();p=Portfolio('first_plus_decision_gate','base',seed.plan,v.settle)
  p.account=seed.account;p.last_decision=t;p.unpriced_flat_assets=[]
  p.prices=lambda *args:(np.full(10,100.),{})
  event=dict(received_ms=t*1000,data=dict(frame=dict(source_time=t,sigma=np.eye(10)*1e-6)))
  f=dict(model='decision_gate',pair=['BTC','ETH'],forecast_source=t,available_time=t,horizon=24,
         economic_available=True,economic_action=0)
  p.step(t+45,{},event,[f]);count=len([e for e in p.events if e['kind']=='entry_check'])
  self.assertEqual(count,2)
  p.step(t+135,{},event,[f]);self.assertEqual(len([e for e in p.events if e['kind']=='entry_check']),2)

if __name__=='__main__':unittest.main()
