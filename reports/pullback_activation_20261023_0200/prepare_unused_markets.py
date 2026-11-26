"""Authorized PAS testnet setup only: cross 1x on unused markets; no orders."""
import json,time
from pathlib import Path
from candidate_handoff import producer
from candidate_handoff.public import PublicPAS
from candidate_handoff.bridge import HistoryBridge
from candidate_handoff.locks import WriterLease
from candidate_handoff.migration import load_existing
from candidate_handoff.runner import require_contract
from competition_live.profile import ROOT,load_profile,verify_deployment
from competition_live.runner import fresh_preflight,explicit_signer
from competition_live.execution import build_order_client
from competition_live.runtime import json_save
from joint_wallet.pas_gateway import PASGateway

p=load_profile();verify_deployment();d=ROOT/'.runtime/pas_budget_lifecycle';require_contract(d,p)
out=ROOT/'reports/pullback_activation_20261023_0200/unused_market_setup.json'
public=PublicPAS(p);results=[]
with WriterLease(ROOT):
 load_existing(d/'coordinator.pkl',p)
 bridge=HistoryBridge(ROOT,d,p,public)
 initial=bridge.refresh();bridge.reconcile_exchange(initial)
 snapshot=fresh_preflight(public,p)
 signer=explicit_signer(ROOT,p,snapshot);client=build_order_client(p,signer,enable_orders=True)
 gateway=PASGateway(p,public,client)
 for symbol in ['SP500','USDEUR','USDJPY']:
  book=bridge.refresh();bridge.reconcile_exchange(book);j=book.symbols.index(symbol)
  if any(abs(owner.qty[j])>1e-12 for owner in book.owners.values()):
   raise RuntimeError('An owner is exposed; refuse leverage setup: '+symbol)
  coin='pas:'+symbol;active=public.post(dict(type='activeAssetData',user=p['team_wallet'],coin=coin))
  before=active.get('leverage');confirmed=gateway.leverage_for_entry(coin,book,active)
  results.append(dict(coin=coin,before=before,after=confirmed['leverage'],changed=before!=confirmed['leverage'],
   own_qty=float(book.owners['ours'].qty[j]),teammate_qty=float(book.owners['teammate'].qty[j])))
  json_save(out,dict(at=time.time(),markets=results,orders_sent=0,existing_position_settings_changed=False))
print(json.dumps(dict(markets=results,orders_sent=0)))
