"""Matured label diagnostics. These are NOT hypothetical executable fills."""
import json,time
from collections import Counter
import numpy as np
from .protocol import *

def main():
 from pas_strategy_audit.data import load
 rows_done=[json.loads(p.read_text()) for p in (OUT/'rows').glob('*_development.json')]
 if len(rows_done)!=len(ARMS)*3:raise RuntimeError('All development runs must finish before selection')
 if any(r['pair_fills'] for r in rows_done):
  from .study import select
  return select()
 cfg,d=load(OUT/'development_data');results=[]
 funding=json.loads((OUT/'development_data/funding.json').read_text())
 for arm in ARMS[1:]:
  fs=json.loads((OUT/'development_data'/(arm+'_forecasts.json')).read_text());rows=[]
  for group in fs.values():
   for f in group:
    source=f['forecast_source'];end=source+f['horizon']*3600;side=f['economic_action']
    if end>=SPLIT or not side:continue
    ei=np.searchsorted(d['times'],end*1000,side='right')-1;ids=f['ids'];beta=f['source_beta'];start=np.array(f['source_prices'])
    if np.any(d['age_ms'][ei,ids]>3600000):continue
    q=side*np.array([1.,-beta])/(1+beta)/start;price=float(q@(d['close'][ei,ids]-start));paid=received=0.
    for event in funding:
     j=SYMBOLS.index(event['coin'][4:]);at=event['time']/1000
     if j not in ids or not source<at<=end:continue
     ti=np.searchsorted(d['times'],event['time'],side='right')-1;unit=event.get('cash_per_unit')
     if unit is None:unit=-d['close'][ti,j]*event['rate']
     value=q[ids.index(j)]*unit
     if value>0:received+=value
     else:paid+=value
    rows.append(dict(source=source,end=end,pair=f['pair'],direction=side,price=price,received=received,paid=paid,
        base_net=price+received+paid-.0038,stress_net=price+.5*received+paid-.0064))
  days=len(set(r['source']//86400 for r in rows))
  r=dict(arm=arm,signals=len(rows),source_days=days,overlapping_labels=True,
     base_mean=float(np.mean([r['base_net'] for r in rows])) if rows else None,
     stress_mean=float(np.mean([r['stress_net'] for r in rows])) if rows else None,
     not_executable_pnl=True,rows=rows)
  results.append(r)
 save(OUT/'forecast_diagnostic.json',results)
 eligible=[r for r in results if r['source_days']>=3]
 chosen=max(eligible,key=lambda r:min(r['base_mean'],r['stress_mean']))['arm'] if eligible else 'ema_basis_blend'
 save(OUT/'selection.json',dict(at=time.time(),candidate=chosen,eligible=False,
  reason='Economic label diagnostic only; native OHLC execution does not establish tradable advantage',
  source_days_required=3,used_diagnostic=bool(eligible),all_candidates=[{k:v for k,v in r.items() if k!='rows'} for r in results],
  selected_before_later_outcome_evaluation=True,no_independence_claim=True))
 print(chosen,flush=True)

if __name__=='__main__':main()
