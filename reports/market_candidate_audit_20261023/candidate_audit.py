"""Frozen candidate diagnostics on observed PAS data; no execution or fitting to labels."""
import hashlib,json,sys,time,datetime as dt
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));OUT=Path(__file__).resolve().parent;RAW=OUT/'raw'
from pair_model_research.candidate import snapshot
from pair_model_research.adaptive_hedge import fit_pair,forecasts
from auto_pair_selection.selection import SYMBOLS,PAIRS,IDS
from practice_live.refresh import REFERENCES
from analyze import save,csvout

def walk(levels,qty):
 left=qty;cash=0.
 for row in levels:
  take=min(left,float(row['sz']));cash+=take*float(row['px']);left-=take
  if left<1e-10:return cash/qty
 return None

def main():
 pointer=json.loads((ROOT/'practice_live/latest.json').read_text());path=ROOT/pointer['inputs'];d=dict(np.load(path));times=d['market_times'];source=int(times[-1]);received=time.time()
 ss=snapshot(d,source,received);save('candidate_current_snapshot',ss)
 meta,ctxs=json.loads((RAW/'current/market.json').read_text())['response'];ctx={u['name'].split(':')[1]:c for u,c in zip(meta['universe'],ctxs)};books={s:json.loads((RAW/'markets'/f'pas_{s}.json').read_text()) for s in SYMBOLS}
 for s in SYMBOLS:books[s]['book']=json.loads((RAW/'current'/f'pas_{s}.json').read_text())
 logs=np.log(d['close']);active=(d['volume']>0)&(d['trades']>0);index=int(np.searchsorted(times,source//86400*86400));rows=[]
 for aud in ss['pair_audit']:
  names=aud['pair'];a,b=names;r=dict(aud,pair='/'.join(names))
  f=next((f for f in ss['forecasts'] if f['pair']==names),None)
  if f is None:rows.append(r);continue
  ids=f['ids'];model=fit_pair(logs,active,index,ids);mids=np.array([(float(books[s]['book']['response']['levels'][0][0]['px'])+float(books[s]['book']['response']['levels'][1][0]['px']))/2 for s in names]);anchor=np.array(f['source_prices']);beta=f['source_beta'];sign=f['economic_action']
  r.update(source_time=source,fit_time=ss['model_source'],source_mu_bps=f['mu']*1e4,source_beta=beta,half_life_hours=f['estimated_half_life'],ar_tstat=model['ar_tstat'],level_correlation=model['level_correlation'],mid_basis_vs_source_bps=((mids/anchor-1)*1e4).tolist(),diagnostic_only=True)
  # Auditable economics at captured books; asynchronous books never promoted to an executable order.
  if not sign:rows.append(r);continue
  ub=beta*anchor[0]/anchor[1];q0=6000/(mids[0]+ub*mids[1]);q=sign*np.array([q0,-ub*q0]);terminal=np.array(f['terminal_prices']);entry=[];exit_=[];sidebands=[]
  for s,qq,mid in zip(names,q,mids):
   levels=books[s]['book']['response']['levels'];side=1 if qq>0 else 0
   entry.append(walk(levels[side],abs(qq)));exit_.append(walk(levels[1-side],abs(qq)))
   best=float(levels[side][0]['px']);sidebands.append(sum(float(x['sz']) for x in levels[side] if abs(float(x['px'])/best-1)<=.0025)*.25/abs(qq))
  r['displayed_depth_fraction_for_target']=min(sidebands)
  if None in entry or None in exit_:r['quote_net_reason']='insufficient_displayed_entry_or_exit';rows.append(r);continue
  entry=np.array(entry);exit_=np.array(exit_);gross=float(abs(q)@mids);fee=.0009
  pricecash=float(q@(terminal-entry));fees=float(abs(q)@entry)*fee+gross*fee
  exitreserve=float(q@(mids-exit_))+gross*.001 # 10bp adverse exit move, separately reported.
  rate=np.array([float(ctx[s]['funding']) for s in names]);oracle=np.array([float(ctx[s]['oraclePx']) for s in names]);hours=max(0,72-(received-source)/3600)
  cash=-q*oracle*rate*hours;carry=float(np.where(cash>0,cash*.5,cash).sum());carrycost=float(np.minimum(cash,0).sum())
  r.update(direction=f'{a} '+('롱' if sign>0 else '숏')+f' / {b} '+('숏' if sign>0 else '롱'),gross=gross,quantities=q.tolist(),entry_vwap=entry.tolist(),exit_vwap_now=exit_.tolist(),predicted_price_cash=pricecash,roundtrip_fee_estimate=fees,exit_spread_and_10bp_reserve=exitreserve,current_rate_carry_discounted=carry,current_rate_carry_cost_only=carrycost,net_with_constant_current_carry=pricecash-fees-exitreserve+carry,net_no_funding=pricecash-fees-exitreserve,net_carry_cost_only=pricecash-fees-exitreserve+carrycost,buffer_5bp=.0005*gross,quote_net_reason='scenario_only_not_admitted_live',oldest_book_ms=min(books[s]['book']['response']['time'] for s in names))
  rows.append(r)
 csvout('candidate_quotes',rows)
 # Real PAS completed 1h trade prices aligned to completed reference bars.
 close=np.full((len(times),10),np.nan);ntrades=np.zeros_like(close);start=1792411200
 for j,s in enumerate(SYMBOLS):
  for c in books[s]['candles']['response']:
   end=(int(c['t'])//3600000+1)*3600;i=int(np.searchsorted(times,end))
   if i<len(times) and times[i]==end and end<=source and c['n']>0 and int(c['T'])<int(dt.datetime.fromisoformat(books[s]['candles']['received_kst']).timestamp()*1000):close[i,j]=float(c['c']);ntrades[i,j]=c['n']
 mu,betas,vol,strong,recs=forecasts(d,start//86400*86400);hist=[]
 for i,t in enumerate(times):
  if t<start or i+72>=len(times):continue
  for k,(names,ids) in enumerate(zip(PAIRS,IDS)):
   if not np.isfinite(mu[i,k]) or abs(mu[i,k])<=.0038:continue
   prices=close[[i,i+72]][:,list(ids)]
   if not np.isfinite(prices).all():continue
   beta=betas[i,k];sgn=np.sign(mu[i,k]);weights=np.array([1.,-beta])/(1+beta);real=float(weights@(prices[1]/prices[0]-1));ref=d['close'][[i,i+72]][:,list(ids)];refret=float(weights@(ref[1]/ref[0]-1))
   hist.append(dict(source=int(t),maturity=int(times[i+72]),pair='/'.join(names),mu=float(mu[i,k]),beta=float(beta),real_pas_relative_return=real,real_reference_relative_return=refret,direction_correct=bool(mu[i,k]*real>0),directed_pas_gross_return=float(sgn*real),directed_pas_less18bp=float(sgn*real-.0018),note='72h gross trade-close markout, not a fillable strategy replay. Fee-only number excludes spread, depth, funding, lifecycle and allocation. Strongly overlapping outcomes.'))
 csvout('matured_pas_markouts',hist)
 grouped=[]
 for names in PAIRS:
  rs=[r for r in hist if r['pair']=='/'.join(names)]
  if not rs:continue
  grouped.append(dict(pair='/'.join(names),overlapping_observations=len(rs),direction_accuracy=sum(x['direction_correct'] for x in rs)/len(rs),mean_forecast_bps=np.mean([x['mu'] for x in rs])*1e4,mean_directed_pas_bps=np.mean([x['directed_pas_gross_return'] for x in rs])*1e4,mean_fee_only_bps=np.mean([x['directed_pas_less18bp'] for x in rs])*1e4,mean_directed_reference_bps=np.mean([np.sign(x['mu'])*x['real_reference_relative_return'] for x in rs])*1e4))
 csvout('matured_pas_by_pair',grouped)
 audit=dict(created_utc=dt.datetime.now(dt.timezone.utc).isoformat(),input=str(path),input_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),input_source=source,source_mapping=REFERENCES,all_pairs=45,forecast_available=len(ss['forecasts']),source_cost_gate_pass=sum(f['economic_action']!=0 for f in ss['forecasts']),quote_scenarios=rows,history_summary=dict(observations=len(hist),distinct_pairs=len(grouped),distinct_source_hours=len({x['source'] for x in hist}),first_source=min((x['source'] for x in hist),default=None),last_source=max((x['source'] for x in hist),default=None),accuracy=sum(x['direction_correct'] for x in hist)/len(hist) if hist else None,mae_bps=np.mean([abs(x['mu']-x['real_pas_relative_return']) for x in hist])*1e4 if hist else None,no_change_mae_bps=np.mean([abs(x['real_pas_relative_return']) for x in hist])*1e4 if hist else None,mean_directed_pas_bps=np.mean([x['directed_pas_gross_return'] for x in hist])*1e4 if hist else None,mean_fee_only_bps=np.mean([x['directed_pas_less18bp'] for x in hist])*1e4 if hist else None),history_by_pair=grouped,limitations=['Candidate selected after viewing reused histories: descriptive development diagnostic, not untouched test.','72h labels overlap strongly; 20 hourly forecasts are not 20 independent 72h trials.','Quote snapshots sequential, old compared with source if refreshed late, fixed72h funding extrapolation is a scenario only.','Source AR mean reversion is estimated on mainnet reference prices, not proven for PAS.','Current pair leg allocation still assumes opposite-sided two-leg hedge; beta is log-level standard-deviation ratio, not statistically validated cointegration coefficient.','Existing own and teammate inventory, integer lots, actual account risk and independent fill handling are NOT approved by this arithmetic quote screen.'],orders_sent=0,live_changed=False)
 save('candidate_audit',audit);print(json.dumps({k:v for k,v in audit.items() if k not in ('quote_scenarios','history_by_pair')},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
