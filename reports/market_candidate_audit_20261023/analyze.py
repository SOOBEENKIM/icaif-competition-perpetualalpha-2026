"""Read-only comparison of timestamped public observations, no live imports."""
import csv,datetime as dt,html,importlib.util,json,statistics
from collections import Counter,defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent;RAW=OUT/'raw';USER='0x125778065a578d598d88bddaa3fdd635280d0e47'
spec=importlib.util.spec_from_file_location('old_analysis',ROOT/'reports/team_comparison_20261021/analyze.py');old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
def save(n,d): (OUT/(n+'.json')).write_text(json.dumps(d,ensure_ascii=False,indent=2))
def csvout(n,rs):
 if not rs:return
 fs=list(dict.fromkeys(k for r in rs for k in r))
 with (OUT/(n+'.csv')).open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fs);w.writeheader();w.writerows({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(list,dict)) else v for k,v in r.items()} for r in rs)
def plain(v):
 if v is None:return '—'
 if isinstance(v,float):return f'{v:,.6f}'
 if isinstance(v,(dict,list)):return json.dumps(v,ensure_ascii=False)
 return str(v)
def table(rows,columns):
 return '<div class="scroll"><table><thead><tr>'+''.join('<th>'+html.escape(label)+'</th>' for _,label in columns)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(plain(r.get(k)))+'</td>' for k,_ in columns)+'</tr>' for r in rows)+'</tbody></table></div>'
def main():
 allrows=[];events=[];positions=[];assets=[];fundingrows=[];ledgerrows=[];openorders=[]
 for path in sorted((RAW/'wallets').glob('*.json')):
  w=json.loads(path.read_text());ex=json.loads((RAW/'extra'/path.name).read_text());r,fs,ps,ss,_=old.summarize(w,ex)
  r['collected_to']=w['completed_kst'];cut=w['history_end_ms']-86400000;ff=[f for f in fs if f['time']>=cut]
  r['recent24h_fills']=len(ff);r['recent24h_orders']=len({f['oid'] for f in ff});r['recent24h_volume']=sum(f['notional'] for f in ff)
  r['recent24h_fees']=sum(float(f['fee']) for f in ff);r['recent24h_closed_pnl']=sum(float(f['closedPnl']) for f in ff)
  r['price_pnl']=r['closed_pnl']+r['unrealized'];r['cost_drag']=r['fees']-r['funding'];r['fee_bps']=r['fees']/r['volume']*10000 if r['volume'] else None
  allrows.append(r);events+=fs;positions+=ps;assets+=ss
  fundingrows += [dict(address=w['address'],**x) for x in w['funding']['rows'] if x['delta'].get('coin','').startswith('pas:')]
  ledgerrows += [dict(address=w['address'],**x) for x in w['ledger']['rows'] if x['delta'].get('token')=='USDP']
  openorders += [dict(address=w['address'],**o) for o in w['orders']['response']]
 rankable=sorted([r for r in allrows if r['fills'] and r['return_pct'] is not None and not r['history_issues']],key=lambda r:-r['return_pct'])
 for i,r in enumerate(rankable,1):r['observed_active_wallet_pnl_rank']=i
 leaderboard=json.loads((RAW/'leaderboard.json').read_text())['response'];chart=json.loads((RAW/'chart.json').read_text())['response'];latest={t['name']:t for t in chart['teams']}
 teams=sorted(leaderboard['teams'],key=lambda t:(t['rank'] is None,t['rank'] or 999,t['name']))
 for t in teams:
  t['return_pct']=t['return']*100 if t['return'] is not None else None;t['daily_std_pct']=t['volatility']*100 if t['volatility'] is not None else None;t['period_volatility_pct']=t['volatility_horizon']*100 if t['volatility_horizon'] is not None else None
  t['live_pnl_pct']=latest[t['name']]['pnl']*100 if t['name'] in latest else None;t['live_top10_position']=latest.get(t['name'],{}).get('position');t['live_at']=latest.get(t['name'],{}).get('last_at')
  if t['q'] is not None:
   q=(max(t['return'],0)+.005)/(abs(t['return'])+1.5*t['volatility_horizon']+.05);assert abs(q-t['q'])<1e-12;t['implied_J']=(t['score']-.8*q)/.2
 market=[];meta,ctxs=json.loads((RAW/'market.json').read_text())['response'];recent=[]
 for u,c in zip(meta['universe'],ctxs):
  coin=u['name'];f=RAW/'markets'/(coin.replace(':','_')+'.json')
  if not f.exists():continue
  d=json.loads(f.read_text());bid,ask=d['book']['response']['levels'];bestbid=float(bid[0]['px']) if bid else None;bestask=float(ask[0]['px']) if ask else None;mid=(bestbid+bestask)/2 if bid and ask else None
  cs=sorted(d['candles']['response'],key=lambda x:x['t']);end=int(dt.datetime.fromisoformat(d['candles']['received_kst']).timestamp()*1000);complete=[x for x in cs if x['T']<end and x['n']>0]
  def change(hours):
   if not complete:return None
   last=complete[-1];prior=[x for x in complete if x['t']<=last['t']-hours*3600000]
   return (float(last['c'])/float(prior[-1]['c'])-1)*100 if prior else None
  def vwap(side,amount):
   rem=amount;qty=0
   for r in side:
    price=float(r['px']);take=min(rem,float(r['sz'])*price);qty+=take/price;rem-=take
    if rem<1e-8:return amount/qty
   return None
  fh=d['funding_history']['response'];rates=[float(x['fundingRate']) for x in fh]
  row=dict(coin=coin,book_received_kst=d['book']['received_kst'],mark=float(c['markPx']),oracle=float(c['oraclePx']),mid=mid,oracle_basis_pct=(float(c['markPx'])/float(c['oraclePx'])-1)*100,funding_hourly_pct=float(c['funding'])*100,funding_phase_sum_pct=sum(rates)*100,funding_positive_hours=sum(r>0 for r in rates),funding_negative_hours=sum(r<0 for r in rates),day_volume=float(c['dayNtlVlm']),open_interest_usdp=float(c['openInterest'])*float(c['markPx']),close_return_3h_pct=change(3),close_return_24h_pct=change(24),first_to_last_trade_pct=(float(complete[-1]['c'])/float(complete[0]['o'])-1)*100 if complete else None,last_completed_candle_kst=old.stamp(complete[-1]['T']) if complete else None,bid=bestbid,ask=bestask,spread_bps=(bestask-bestbid)/mid*10000 if mid else None,best_bid_notional=float(bid[0]['px'])*float(bid[0]['sz']) if bid else 0,best_ask_notional=float(ask[0]['px'])*float(ask[0]['sz']) if ask else 0,buy_3000_vwap=vwap(ask,3000),sell_3000_vwap=vwap(bid,3000),source_candles=len(cs),nonzero_completed_candles=len(complete))
  for bps in (10,25,50):
   row[f'ask_depth_{bps}bp']=sum(float(r['px'])*float(r['sz']) for r in ask if float(r['px'])<=bestask*(1+bps/10000)) if ask else 0;row[f'bid_depth_{bps}bp']=sum(float(r['px'])*float(r['sz']) for r in bid if float(r['px'])>=bestbid*(1-bps/10000)) if bid else 0
  market.append(row);recent+=d['recent_trades']['response']
 sides=defaultdict(dict)
 for f in events:sides[(f['coin'],f['time'],f['tid'])][f['side']]=f['address']
 recentidx={(f['coin'],f['tid']):[x.lower() for x in f.get('users',[])] for f in recent}
 for f in events:
  f['counterparty']=sides[(f['coin'],f['time'],f['tid'])].get('A' if f['side']=='B' else 'B');us=recentidx.get((f['coin'],f['tid']),[])
  if not f['counterparty'] and len(us)==2 and f['address'] in us:f['counterparty']=us[1] if us[0]==f['address'] else us[0]
 counts=[]
 for m in market:
  ps=[p for p in positions if p['coin']==m['coin']];counts.append(dict(coin=m['coin'],long_wallets=sum(p['qty']>0 for p in ps),short_wallets=sum(p['qty']<0 for p in ps),long_notional=sum(p['notional'] for p in ps if p['qty']>0),short_notional=sum(p['notional'] for p in ps if p['qty']<0)))
 own=next(r for r in allrows if r['address']==USER);ownfs=[f for f in events if f['address']==USER];ownps=[p for p in positions if p['address']==USER]
 ownership=json.loads((OUT/'runtime/ownership.json').read_text());known_owner={e['row']['oid']:e['owner'] for e in ownership['events'] if e['kind']=='fill'}
 dispatch=json.loads((OUT/'runtime/dispatch.json').read_text())
 for x in dispatch['completed'].values():known_owner[x['oid']]=x['owner']
 for f in ownfs:
  if f['coin']=='pas:HYPE' and f['time']<1792670400000:known_owner[f['oid']]='ours'
 virtual={o:dict(realized=0.,fees=0.,funding=0.,unrealized=0.,gross=0.,quantities={},entries={}) for o in ('ours','teammate')};unknown=[]
 for f in ownfs:
  owner=known_owner.get(f['oid']);f['owner']=owner
  if owner not in virtual:unknown.append(f);continue
  v=virtual[owner];c=f['coin'];q=float(f['sz'])*(1 if f['side']=='B' else -1);px=float(f['px']);oldqty=v['quantities'].get(c,0);entry=v['entries'].get(c,0);new=oldqty+q
  if oldqty*q>=0:v['entries'][c]=(abs(oldqty)*entry+abs(q)*px)/abs(new) if new else 0
  elif oldqty*new<0:v['entries'][c]=px
  close=min(abs(q),abs(oldqty)) if oldqty*q<0 else 0;v['realized']+=close*(px-entry)*(1 if oldqty>0 else -1);v['fees']+=float(f['fee']);v['quantities'][c]=new
 for r in fundingrows:
  if r['address']!=USER:continue
  c=r['delta']['coin'];qs=defaultdict(float)
  for f in ownfs:
   if f['coin']==c and f['time']<=r['time'] and f.get('owner') in virtual:qs[f['owner']]+=float(f['sz'])*(1 if f['side']=='B' else -1)
  net=sum(qs.values())
  if net:
   for o,q in qs.items():virtual[o]['funding']+=float(r['delta']['usdc'])*q/net
 for v in virtual.values():
  for c,q in v['quantities'].items():
   if abs(q)<1e-10:continue
   p=next((x for x in ownps if x['coin']==c),None);mark=p['mark'] if p else next(x['mark'] for x in market if x['coin']==c)
   v['unrealized']+=q*(mark-v['entries'][c]);v['gross']+=abs(q)*mark
  v['net_pnl']=v['realized']-v['fees']+v['funding']+v['unrealized'];v['return_on_virtual_50k_pct']=v['net_pnl']/50000*100;v['gross_on_virtual_50k_pct']=v['gross']/50000*100
 reconciliation=sum(v['net_pnl'] for v in virtual.values())-own['trade_net_pnl']
 if not unknown:assert abs(reconciliation)<.01,reconciliation
 active=[r for r in allrows if r['fills']]
 summary=dict(collected_wallets=len(allrows),expected_previously_granted_wallets=len(json.loads((RAW/'manifest.json').read_text())['grant_wallets']),verified_traded_wallets=len(active),nonempty_positions=sum(r['positions']>0 for r in allrows),directions=dict(Counter(r['current'] for r in active)),historical_directions=dict(Counter(r['historical'] for r in active)),official_generated_at=leaderboard['generated_at'],official_listed=len(teams),official_ranked=sum(t['eligible'] for t in teams),official_sfinx=next(t for t in teams if t['name']=='SFINX'),sfinx=own,virtual_owner_accounting=virtual,unknown_own_fills=unknown,owner_reconciliation_error=reconciliation,issues=[dict(address=r['address'],issues=r['history_issues']) for r in allrows if r['history_issues']],runtime=json.loads((OUT/'runtime/status.json').read_text()),median_active_gross_pct=statistics.median(r['gross_equity_pct'] for r in active if r['gross_equity_pct'] is not None),name_address_mapping='Only SFINX verified; peers not assigned names',snapshot_note='Sequential observations; spot/mark timing differences retained. Wallet rank is not official Score rank. Same wallet is not necessarily one human or one strategy.')
 save('analysis',dict(summary=summary,wallets=allrows,official=teams,market=market,market_counts=counts,positions=positions,assets=assets));save('summary',summary)
 for n,rs in [('wallets',allrows),('official_teams',teams),('positions',positions),('fills',events),('by_market',assets),('market',market),('market_counts',counts),('funding',fundingrows),('transfers',ledgerrows),('open_orders',openorders)]:csvout(n,rs)
 print(json.dumps({'collected':len(allrows),'active':len(active),'own':own,'virtual':virtual,'unknown':len(unknown),'issues':summary['issues']},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
