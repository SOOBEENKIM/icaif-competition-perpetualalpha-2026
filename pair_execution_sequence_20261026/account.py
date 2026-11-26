"""Close physical sign-crossings BEFORE sending the opposite hedge leg.

This is an offline sequence change. It never assumes two legs fill atomically.
Unknown/partial close, stale intent or invalid limits leave actual inventory
for the unchanged pair completion/unwind state machine. No generic retry.
"""
import copy
import numpy as np
from replay_lab.common_comparison.engine import rounded
from competition_live.live_account import LiveAccount
from competition_live import live_account as clock_source
from pair_execution_repair_20261026.account import candidates

class PairOpeningAccount(LiveAccount):
 def execute_books(self,intent,streams,marks,sigma,p,sc,cfg,*,on_time=None):
  before=self.physical.qty.copy();held={k:v.qty.copy() for k,v in self.owners.items()}
  eligible=candidates(intent,before,held['pair'])
  eligible=[j for j in eligible if abs(intent['owner_targets']['base'][j]-held['base'][j])<1e-10]
  if not eligible:return super().execute_books(intent,streams,marks,sigma,p,sc,cfg,on_time=on_time)
  # Separate, bounded close receipts first. Other legs have not been opened yet.
  close=copy.deepcopy(held)
  for j in eligible:close['pair'][j]-=before[j]
  limits=copy.deepcopy(intent.get('owner_limits',{}))
  for j in eligible:
   bound=intent['sealed_owner_limits'].get(('pair',j))
   if bound is None:return super().execute_books(intent,streams,marks,sigma,p,sc,cfg,on_time=on_time)
   limits.setdefault('pair',{})[str(j)]=bound
  raw=dict(packet_id=str(intent['packet_id'])+':physical_close_first',decision_time=clock_source.time.time(),
   source_close=intent['source_close'],source_available=intent['source_available'],source_prices=np.asarray(marks).tolist(),
   owner_targets={k:v.tolist() for k,v in close.items()},owner_limits=limits,risk_forced=False,sigma=sigma)
  first=super().execute_books(self.seal(raw,p,sc),streams,marks,sigma,p,sc,cfg,on_time=on_time)
  if self.unknown or self.pending_orders or first['unknown_symbols']:return first
  if any(abs(self.physical.qty[j])>1e-12 for j in eligible):return first
  now=clock_source.time.time()
  if now-intent['source_close']>cfg['intent_deadline_seconds']:return first
  def packet(targets,suffix):
   raw=dict(packet_id=str(intent['packet_id'])+suffix,decision_time=clock_source.time.time(),
    source_close=intent['source_close'],source_available=intent['source_available'],source_prices=np.asarray(marks).tolist(),
    owner_targets={k:v.tolist() for k,v in targets.items()},owner_limits=copy.deepcopy(intent.get('owner_limits',{})),
    risk_forced=False,sigma=sigma)
   for (owner,j),bound in intent['sealed_owner_limits'].items():
    if bound is not None:raw['owner_limits'].setdefault(owner,{})[str(j)]=bound
   return self.seal(raw,p,sc)
  # Complete the interrupted opening leg before sizing its hedge from receipts.
  leader=eligible[0];opening={k:v.qty.copy() for k,v in self.owners.items()}
  opening['pair'][leader]=intent['owner_targets']['pair'][leader]
  second=super().execute_books(packet(opening,':open_after_confirmed_close'),streams,marks,sigma,p,sc,cfg,on_time=on_time)
  orders=first['orders']+second['orders'];sent=first['orders_sent']+second['orders_sent']
  if self.unknown or self.pending_orders or clock_source.time.time()-intent['source_close']>cfg['intent_deadline_seconds']:
   return dict(orders=orders,unknown_symbols=sorted(self.unknown),orders_sent=sent)
  fraction=min(1.,abs(self.owners['pair'].qty[leader]/intent['owner_targets']['pair'][leader]))
  hedge={k:v.qty.copy() for k,v in self.owners.items()}
  for j,want in enumerate(intent['owner_targets']['pair']):
   if j!=leader:hedge['pair'][j]=rounded(want*fraction,p['execution']['decimals'][j])
  third=super().execute_books(packet(hedge,':hedge_actual_first_leg'),streams,marks,sigma,p,sc,cfg,on_time=on_time)
  self.checkpoint('research_physical_close_before_pair_open',dict(assets=eligible))
  return dict(orders=orders+third['orders'],unknown_symbols=third['unknown_symbols'],
   orders_sent=sent+third['orders_sent'],bounded_zero_continuation=True)
