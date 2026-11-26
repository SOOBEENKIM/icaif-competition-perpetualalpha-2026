"""One bounded continuation after a CONFIRMED physical zero crossing.

The unchanged live adapter still handles receipts, precision, risk, limits and
unknown-order locking. Only a new pair's opening leg may get this continuation.
No incomplete or ambiguous receipt is retried. Never used by the live service.
"""
import copy
import numpy as np
from competition_live.live_account import LiveAccount
from competition_live import live_account as clock_source
from experiments.shared_execution_pairs_20261017.account_v2 import sum_targets

def candidates(intent,before,pair_before):
 if intent.get('risk_forced') or not str(intent.get('packet_id','')).startswith('pair:'):return []
 target=sum_targets(intent['owner_targets']);pair=np.asarray(intent['owner_targets']['pair'])
 return [j for j in range(len(before)) if before[j]*target[j]<0 and pair_before[j]==0
         and before[j]*pair[j]<0]

class PairOpeningAccount(LiveAccount):
 def execute_books(self,intent,streams,marks,sigma,p,sc,cfg,*,on_time=None):
  before=self.physical.qty.copy();pair_before=self.owners['pair'].qty.copy()
  eligible=candidates(intent,before,pair_before)
  first=super().execute_books(intent,streams,marks,sigma,p,sc,cfg,on_time=on_time)
  if not eligible or self.unknown or self.pending_orders or first['unknown_symbols']:return first
  eligible=[j for j in eligible if abs(self.physical.qty[j])<1e-12 and
      abs(intent['owner_targets']['pair'][j]-self.owners['pair'].qty[j])>1e-10]
  now=clock_source.time.time()
  if not eligible or now-intent['source_close']>cfg['intent_deadline_seconds']:return first
  # Rebuild only the interrupted pair legs; do not retry unrelated base orders.
  targets={name:ledger.qty.copy() for name,ledger in self.owners.items()}
  for j in eligible:targets['pair'][j]=intent['owner_targets']['pair'][j]
  raw=dict(packet_id=str(intent['packet_id'])+':confirmed_zero_continuation',decision_time=now,
      source_close=intent['source_close'],source_available=intent['source_available'],
      source_prices=np.asarray(marks).tolist(),owner_targets={k:v.tolist() for k,v in targets.items()},
      owner_limits=copy.deepcopy(intent.get('owner_limits',{})),risk_forced=False,sigma=sigma)
  # Every opening leg retains its ORIGINAL economic price ceiling/floor.
  for j in eligible:
   bound=intent['sealed_owner_limits'].get(('pair',j))
   if bound is None:return first
   raw['owner_limits'].setdefault('pair',{})[str(j)]=bound
  second=super().execute_books(self.seal(raw,p,sc),streams,marks,sigma,p,sc,cfg,on_time=on_time)
  self.checkpoint('research_confirmed_zero_continuation',dict(assets=eligible))
  return dict(orders=first['orders']+second['orders'],unknown_symbols=second['unknown_symbols'],
      orders_sent=first['orders_sent']+second['orders_sent'],bounded_zero_continuation=True)
