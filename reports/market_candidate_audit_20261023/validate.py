"""Accounting, chronology and source-prefix regression checks for this audit."""
import json,hashlib,sys,datetime as dt,csv
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));OUT=Path(__file__).resolve().parent
from pair_model_research.adaptive_hedge import fit_pair
from auto_pair_selection.selection import IDS
from auto_pair_selection.study import verify

def main():
 d=json.loads((OUT/'analysis.json').read_text());s=d['summary'];a=json.loads((OUT/'candidate_audit.json').read_text());m=json.loads((OUT/'raw/manifest.json').read_text());checks={}
 checks['all_granted_wallets_collected']=set(x['address'] for x in d['wallets'])==set(m['grant_wallets']) and len(d['wallets'])==47
 checks['latest_funder_roster_verified']=m['latest_grants_match_prior'] and m['funder_history_under_page_limit']
 checks['all_histories_and_inventory_reconcile']=not s['issues'];checks['no_unknown_own_fills']=not s['unknown_own_fills'];checks['owner_ledgers_within_one_cent']=abs(s['owner_reconciliation_error'])<.01
 checks['all45_candidates_reported']=len(a['quote_scenarios'])==45 and len({r['pair'] for r in a['quote_scenarios']})==45
 checks['official_score_recalculated']=all(abs(.8*t['q']-t['score'])<1e-12 for t in d['official'] if t['q'] is not None)
 for p in (OUT/'raw/wallets').glob('*.json'):
  w=json.loads(p.read_text())
  assert all(x['time']<=w['history_end_ms'] for x in w['fills']['rows'])
  assert all(w[k]['pagination_complete'] for k in ('fills','funding','ledger'))
 checks['histories_end_at_declared_cutoff']=True
 markouts=list(csv.DictReader((OUT/'matured_pas_markouts.csv').open(encoding='utf-8-sig')))
 for r in markouts:
  assert int(r['maturity'])-int(r['source'])==72*3600
  for symbol in r['pair'].split('/'):
   raw=json.loads((OUT/'raw/markets'/f'pas_{symbol}.json').read_text());received=dt.datetime.fromisoformat(raw['candles']['received_kst']).timestamp()
   assert int(r['maturity'])<=received
 checks['no_partial_candle_labels']=True
 for r in a['quote_scenarios']:
  if 'oldest_book_ms' in r:assert r['oldest_book_ms']/1000>=r['source_time']
 checks['current_quote_not_before_forecast_source']=True
 inp=Path(a['input']);assert hashlib.sha256(inp.read_bytes()).hexdigest()==a['input_sha256'];data=dict(np.load(inp));t=data['market_times'];i=int(np.searchsorted(t,1792454400))
 logs=np.log(data['close']);act=(data['volume']>0)&(data['trades']>0);changed=logs.copy();changed[i+1:]+=np.linspace(-5,5,10)
 for ids in IDS:
  x=fit_pair(logs,act,i,ids);y=fit_pair(changed,act,i,ids);z=fit_pair(logs[:i+1],act[:i+1],i,ids)
  assert x==y==z
 checks['future_prices_do_not_change_past_pair_fit']=True
 verify();checks['frozen_live_source_integrity']=True
 assert all(checks.values()),checks
 out=dict(checks=checks,counts=dict(granted_wallets=len(d['wallets']),traded_wallets=s['verified_traded_wallets'],fills=sum(r['fills'] for r in d['wallets']),positions=len(d['positions']),candidate_pairs=45,completed_pas_markouts=len(markouts)),orders_sent=0,live_changes=False,limits=['Snapshot prices are sequential, not simultaneous.','Public fills cannot identify private models or prove causality.','Overlapping72h diagnostics are not a portfolio backtest or independent test.'])
 (OUT/'validation.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
if __name__=='__main__':main()
