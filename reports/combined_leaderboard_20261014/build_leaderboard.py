import bisect, collections, csv, datetime, html, itertools, json, math, pathlib, statistics
from decimal import Decimal as D

HERE=pathlib.Path(__file__).resolve().parent
SRC=HERE/'raw' if (HERE/'raw'/'wallets').is_dir() else pathlib.Path('/tmp/par-leaderboard-20261014')
OUT=HERE if SRC==HERE/'raw' else pathlib.Path('/home/ssd-990/soobeenkim/icaif_competition_2026/reports/combined_leaderboard_20261014')
OUT.mkdir(parents=True,exist_ok=True)
UTC=datetime.timezone.utc;KST=datetime.timezone(datetime.timedelta(hours=9));Z=D(0)
START=int(datetime.datetime(2026,10,5,tzinfo=UTC).timestamp()*1000)
market=json.load(open(str(SRC/'market_now.json')))
END=market['queried_at_ms'];DAY=86400000
oracle={a['name']:D(c['oraclePx']) for a,c in zip(market['response'][0]['universe'],market['response'][1])}
mark={a['name']:D(c['markPx']) for a,c in zip(market['response'][0]['universe'],market['response'][1])}
def dec(x):return D(str(x or 0))
def fmt(x,n=4):return '—' if x is None or x=='' else format(dec(x),',.%df'%n)
def date(t,tz=KST):return datetime.datetime.fromtimestamp(t/1000,tz).strftime('%Y-%m-%d %H:%M:%S')
def short(a):return a[:10]+'…'+a[-6:]
def csvout(name,rows):
 if not rows:return
 cols=list(dict.fromkeys(k for r in rows for k in r))
 with open(str(OUT/name),'w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=cols);w.writeheader();w.writerows({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v for k,v in r.items()} for r in rows)
def table(cols,rows):
 return '<div class="scroll"><table><thead><tr>'+''.join('<th>'+html.escape(c)+'</th>' for c in cols)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in r)+'</tr>' for r in rows)+'</tbody></table></div>'

candles={}
for p in (SRC/'candles').glob('*.json'):
 rows=json.load(open(str(p))).get('response',[])
 candles['par:'+p.stem]=sorted(rows,key=lambda r:r['T'])
def prices_at(t,current=False):
 if current:return oracle,{c:0 for c in oracle}
 px={};ages={}
 for c,rs in candles.items():
  idx=bisect.bisect_left([r['T'] for r in rs],t)-1
  if idx>=0:px[c]=dec(rs[idx]['c']);ages[c]=(t-rs[idx]['T'])/3600000
 return px,ages

def dietz(e0,e1,flows,t0,t1):
 if t1<=t0 or e0<=0:return None
 fs=[(t,v) for t,v in flows if t0<t<=t1]
 den=e0+sum((v*D(t1-t)/D(t1-t0) for t,v in fs),Z)
 if den<=0:return None
 return (e1-e0-sum((v for t,v in fs),Z))/den

def cashflows(d):
 a=d['address'];fs=[];cost=[];issues=[]
 for r in d['ledger']['rows']:
  if r['time']>END:continue
  de=r['delta'];token=de.get('token')
  if token!='USDPR':continue
  typ=de.get('type');amount=dec(de.get('amount'))
  if typ=='spotGenesis':fs.append((r['time'],amount,'genesis'));continue
  s=de.get('user','').lower();dst=de.get('destination','').lower()
  if s==a and dst==a:pass
  elif dst==a:fs.append((r['time'],amount,'external_deposit'))
  elif s==a:fs.append((r['time'],-amount,'external_withdrawal'))
  else:issues.append('unclassified USDPR ledger '+str(typ))
  if s==a and de.get('feeToken')=='USDPR':cost.append((r['time'],dec(de.get('fee'))))
 for f in d['fills']['rows']:
  if f['time']>END:continue
  if f['coin']=='@2900':
   # Verified spotMeta: @2900 = USDPR / USDC.
   v=(dec(f['sz']) if f['side']=='B' else -dec(f['sz']))-(dec(f['fee']) if f['feeToken']=='USDPR' else Z)
   fs.append((f['time'],v,'USDPR_spot_conversion'))
 return sorted(fs),sorted(cost),issues

def ordered_fills(fills):
 # Same-millisecond fills may be returned in reverse execution order.
 # Follow the position transitions supplied by the exchange, per coin/time.
 out=[];current=collections.defaultdict(D);errors=[]
 for tm,group in itertools.groupby(sorted(fills,key=lambda f:f['time']),key=lambda f:f['time']):
  grouped=collections.defaultdict(list)
  for f in group:grouped[f['coin']].append(f)
  for coin,rs in grouped.items():
   edges=collections.defaultdict(list)
   for f in rs:edges[dec(f['startPosition'])].append(f)
   stack=[(current[coin],None)];path=[]
   while stack:
    node,_=stack[-1]
    if edges[node]:
     f=edges[node].pop();q=dec(f['sz'])*(1 if f['side']=='B' else -1);stack.append((node+q,f))
    else:
     _,f=stack.pop()
     if f is not None:path.append(f)
   path.reverse()
   if len(path)!=len(rs):errors.append('incomplete fill position chain '+coin);path=rs
   for f in path:
    if dec(f['startPosition'])!=current[coin]:errors.append('fill position transition mismatch '+coin)
    current[coin]+=dec(f['sz'])*(1 if f['side']=='B' else -1);out.append(f)
 return out,sorted(set(errors))

def equity(t,fills,funding,flows,cost,px):
 capital=sum((v for tm,v,_ in flows if tm<=t),Z)
 cash=Z;pos=collections.defaultdict(D);entry=collections.defaultdict(D)
 for f in fills:
  if f['time']>t:break
  coin=f['coin'];q=dec(f['sz'])*(1 if f['side']=='B' else -1);p=dec(f['px']);before=pos[coin];after=before+q
  closed=dec(f.get('closedPnl'))
  if not before:entry[coin]=p
  elif before*q>0:entry[coin]=(abs(before)*entry[coin]+abs(q)*p)/abs(after)
  elif after and before*after>0:
   # Infer the exchange's actual basis from its settled closing PnL;
   # do not replace official settlement with unrounded theoretical cash PnL.
   closing=min(abs(before),abs(q));entry[coin]=p-closed/closing*(1 if before>0 else -1)
  elif after:entry[coin]=p
  else:entry[coin]=Z
  pos[coin]=after;cash+=closed
  if f['feeToken']=='USDPR':cash-=dec(f['fee'])
 for r in funding:
  if r['time']<=t:cash+=dec(r['delta']['usdc'])
 cash-=sum((v for tm,v in cost if tm<=t),Z)
 missing=[c for c,q in pos.items() if q and c not in px]
 if missing:return None,{'missing_prices':missing,'capital':str(capital)}
 pnl=cash+sum((q*(px[c]-entry[c]) for c,q in pos.items() if q),Z)
 return capital+pnl,{'capital':str(capital),'pnl':str(pnl),'settled_pnl_after_costs':str(cash),'positions':{c:str(q) for c,q in pos.items() if q},'reconstructed_entry_prices':{c:str(entry[c]) for c,q in pos.items() if q}}

def sanity():
 # A pure cash injection must never become a trading return.
 assert dietz(D(100),D(1100),[(50,D(1000))],0,100)==0
 assert dietz(D(100),D(110),[],0,100)==D('.1')
 f=[{'time':1,'coin':'par:X','side':'B','sz':'2','px':'10','fee':'1','feeToken':'USDPR','closedPnl':'0'}]
 e,_=equity(2,f,[],[(0,D(100),'deposit')],[],{'par:X':D(12)})
 assert e==103
 f.append({'time':3,'coin':'par:X','side':'A','sz':'2','px':'12','fee':'1','feeToken':'USDPR','closedPnl':'4'})
 e,_=equity(4,f,[],[(0,D(100),'deposit')],[],{})
 assert e==102
sanity()

wallets=[json.load(open(str(p))) for p in sorted((SRC/'wallets').glob('*.json'))]
old_ids=json.load(open(str(SRC/'wallet_ids.json'))) if (SRC/'wallet_ids.json').exists() else {d['address']:d['id'] for d in [json.load(open(str(p))) for p in pathlib.Path('/home/ssd-990/soobeenkim/icaif_competition_2026/reports/wallet_review_20261014/raw').glob('W*.json')]}
ids=dict(old_ids);i=max([int(v[1:]) for v in ids.values()]+[0])+1
for d in wallets:
 if d['par_fill_count'] and d['address'] not in ids:ids[d['address']]='W%02d'%i;i+=1

trade_map=collections.defaultdict(lambda:collections.defaultdict(set))
for d in wallets:
 for f in d['fills']['rows']:
  if f['coin'].startswith('par:'):trade_map[(f['coin'],f['time'],f['tid'])][f['side']].add(d['address'])
for fname in ['recent_trades_refresh.json']:
 p=SRC/fname
 if p.exists():
  for v in json.load(open(str(p))).values():
   for t in v.get('response',[]):
    trade_map[(t['coin'],t['time'],t['tid'])]['B'].add(t['users'][0]);trade_map[(t['coin'],t['time'],t['tid'])]['A'].add(t['users'][1])

summaries=[];daily=[];inventory=[];fillrows=[];otherfills=[];positionrows=[];orderrows=[];fundingrows=[];ledgerrows=[];cashrows=[];historyrows=[];balances=[];detail={};unfilled_active=[]
for d in wallets:
 a=d['address'];wid=ids.get(a,'');allfs=d['fills']['rows'];fills,chain_issues=ordered_fills([f for f in allfs if f['coin'].startswith('par:') and f['time']<=END])
 practice=[f for f in fills if f['time']>=START]
 pos=d['positions'].get('response',{}).get('assetPositions',[]);orders=d['orders'].get('response',[])
 inventory.append({'wallet_id':wid,'address':a,'discovery_sources':'; '.join(d['discovery_sources']),'lifetime_par_fills_to_cutoff':len(fills),'practice_par_fills_to_cutoff':len(practice),'positions_at_query':len(pos),'orders_at_query':len(orders) if isinstance(orders,list) else '', 'history_pagination_complete':d['fills']['pagination_complete'],'query_completed_utc':d['completed_at_utc']})
 if not fills:
  for p in pos:positionrows.append({'wallet_id':wid,'address':a,'snapshot_utc':d['positions']['received_at_utc'],**p['position']})
  for o in orders if isinstance(orders,list) else []:orderrows.append({'wallet_id':wid,'address':a,'snapshot_utc':d['orders']['received_at_utc'],**o})
  if pos or orders:unfilled_active.append({'address':a,'positions':pos,'orders':orders,'reason':'No par fills found before ranking cutoff; excluded from performance ranking'})
  continue
 funding=[r for r in d.get('funding',{}).get('rows',[]) if r['delta'].get('coin','').startswith('par:') and r['time']<=END]
 flows,cost,issues=cashflows(d)
 issues.extend(chain_issues)
 if not d['fills']['pagination_complete']:issues.append('fill query incomplete')
 if not d.get('funding',{}).get('pagination_complete'):issues.append('funding query incomplete')
 if not d['ledger']['pagination_complete']:issues.append('ledger query incomplete')
 for coin in {f['coin'] for f in fills}:
  cf=[f for f in fills if f['coin']==coin];ft=min(f['time'] for f in cf)
  if not any(dec(f['startPosition'])==0 for f in cf if f['time']==ft):issues.append('earlier fill history missing for '+coin)
 if len(allfs)>=10000 and not d['fills'].get('cached_history_included'):issues.append('10000-fill retention limit may truncate history')
 if any(f['feeToken']!='USDPR' for f in fills):issues.append('non-USDPR trading fee')
 firstfund=min((tm for tm,v,k in flows if v>0),default=None)
 start=max(START,firstfund) if firstfund is not None else START
 basepx,baseages=prices_at(start)
 e0,b0=equity(start,fills,funding,flows,cost,basepx)
 e1,b1=equity(END,fills,funding,flows,cost,oracle)
 if e0 is None:issues.append('start valuation has missing historical prices')
 if e1 is None:issues.append('end valuation has missing prices')
 cf=[(tm,v) for tm,v,k in flows]
 ret=dietz(e0,e1,cf,start,END) if e0 is not None and e1 is not None else None
 step_returns=[];chart=[];max_age=0.0
 firstday=((start+DAY-1)//DAY)*DAY
 for t0 in range(firstday,(END//DAY)*DAY,DAY):
  t1=t0+DAY;px0,age0=prices_at(t0);px1,age1=prices_at(t1)
  v0,s0=equity(t0,fills,funding,flows,cost,px0);v1,s1=equity(t1,fills,funding,flows,cost,px1)
  r=dietz(v0,v1,cf,t0,t1) if v0 is not None and v1 is not None else None
  used=set(s0.get('positions',{}))|set(s1.get('positions',{}))
  age=max([age0.get(c,0) for c in used]+[age1.get(c,0) for c in used]+[0]);max_age=max(max_age,age)
  if r is not None:step_returns.append(float(r));chart.append([date(t1,UTC)[:10],float(v1),float(r)*100])
  daily.append({'wallet_id':wid,'address':a,'start_utc':date(t0,UTC),'end_utc':date(t1,UTC),'start_equity_USDPR':str(v0) if v0 is not None else '', 'end_equity_USDPR':str(v1) if v1 is not None else '', 'daily_return_percent_proxy':str(r*100) if r is not None else '', 'max_candle_age_hours':age,'valuation':'last completed par 1h trade candle close before UTC midnight; not historical mark/oracle'})
 sigma=statistics.pstdev(step_returns)*100 if len(step_returns)>=2 else None
 nav=peak=1.0;mdd=0.0
 for r in step_returns:nav*=1+r;peak=max(peak,nav);mdd=max(mdd,(peak-nav)/peak)
 if ret is None:issues.append('positive invested-capital denominator unavailable')
 if sigma is None:issues.append('fewer than two complete daily returns')
 if not practice:issues.append('no par fills in practice comparison window')
 gross=sum((dec(f['closedPnl']) for f in practice),Z);fees=sum((dec(f['fee']) for f in practice if f['feeToken']=='USDPR'),Z)
 fund=sum((dec(r['delta']['usdc']) for r in funding if r['time']>=start),Z)
 volume=sum((dec(f['px'])*dec(f['sz']) for f in practice),Z);maker=sum(not f['crossed'] for f in practice)
 realized_period=gross-fees
 adj_profit=e1-e0-sum((v for tm,v,k in flows if start<tm<=END),Z) if e0 is not None and e1 is not None else None
 capital_in=sum((v for tm,v,k in flows if k=='external_deposit'),Z)
 one=[v for tm,v,k in flows if k=='external_deposit']
 pattern='100만 USDPR 1회 지급 패턴' if one==[D(1000000)] else '기타 자금 지급 패턴'
 if a=='0x24574c94ab1e08638eb55db0af3687b5ebaae211':pattern='par DEX 배포자 — 공개 메타데이터 확인'
 top=collections.Counter()
 for f in practice:top[f['coin']]+=float(dec(f['px'])*dec(f['sz']))
 s={'wallet_id':wid,'address':a,'funding_pattern':pattern,'registered_team_identity':'not verified','rank70':None,'score70_proxy':None,'return_score40':None,'risk_score30_proxy':None,'jury_score30':None,'official_total_score100':None,'comparison_start_kst':date(start),'comparison_end_kst':date(END),'period_return_percent':float(ret*100) if ret is not None else None,'daily_volatility_percent_proxy':sigma,'daily_return_observations':len(step_returns),'max_drawdown_percent_daily_proxy':mdd*100 if step_returns else None,'period_profit_USDPR':str(adj_profit) if adj_profit is not None else None,'reconstructed_end_equity_USDPR':str(e1) if e1 is not None else None,'start_equity_USDPR':str(e0) if e0 is not None else None,'external_deposits_total_USDPR':str(capital_in),'lifetime_par_fill_count':len(fills),'practice_fill_count':len(practice),'practice_order_count_in_fills':len({f['oid'] for f in practice}),'practice_traded_coins':len({f['coin'] for f in practice}),'practice_notional_USDPR':str(volume),'practice_maker_fill_percent':100*maker/len(practice) if practice else None,'practice_gross_closed_pnl_USDPR':str(gross),'practice_trade_fees_USDPR':str(fees),'practice_net_closed_pnl_excluding_funding_USDPR':str(realized_period),'period_funding_net_USDPR':str(fund),'open_position_count_at_query':len(pos),'open_order_count_at_query':len(orders) if isinstance(orders,list) else None,'snapshot_positions_utc':d['positions']['received_at_utc'],'top_markets_by_notional':', '.join(c.replace('par:','') for c,_ in top.most_common(3)),'history_issues':'; '.join(sorted(set(issues))),'max_daily_candle_age_hours':max_age,'return_method':'Modified Dietz; start and daily valuations use trade candle close; final valuation uses observed oracle','status':'eligible for assumption-based practice score' if not issues else 'metrics incomplete; not assigned composite rank'}
 summaries.append(s)
 for hr in d.get('historical_orders',{}).get('response',[]):
  o=hr.get('order',{})
  if o.get('coin','').startswith('par:'):historyrows.append({'wallet_id':wid,'address':a,'status':hr.get('status'),'status_timestamp':hr.get('statusTimestamp'),**o})
 for b in d.get('spot',{}).get('response',{}).get('balances',[]):
  if dec(b.get('total')) or b.get('coin')=='USDPR':balances.append({'wallet_id':wid,'address':a,'account_mode':d.get('mode',{}).get('response'),'snapshot_utc':d.get('spot',{}).get('received_at_utc'),**b})
 for f in allfs:
  if f['time']>END:continue
  base={'wallet_id':wid,'address':a,'time_kst':date(f['time']),'time_ms':f['time'],**f}
  if f['coin'].startswith('par:'):
   peers=trade_map[(f['coin'],f['time'],f['tid'])]['A' if f['side']=='B' else 'B'];base['counterparty_address']=next(iter(peers)) if len(peers)==1 else '';fillrows.append(base)
  else:otherfills.append(base)
 for p in pos:positionrows.append({'wallet_id':wid,'address':a,'snapshot_utc':d['positions']['received_at_utc'],**p['position']})
 for o in orders if isinstance(orders,list) else []:orderrows.append({'wallet_id':wid,'address':a,'snapshot_utc':d['orders']['received_at_utc'],**o})
 for r in funding:fundingrows.append({'wallet_id':wid,'address':a,'time_kst':date(r['time']),**r['delta'],'time_ms':r['time']})
 for r in d['ledger']['rows']:ledgerrows.append({'wallet_id':wid,'address':a,'time_kst':date(r['time']),**r['delta'],'time_ms':r['time']})
 for tm,v,k in flows:cashrows.append({'wallet_id':wid,'address':a,'time_ms':tm,'time_kst':date(tm),'amount_USDPR':str(v),'classification':k})
 detail[wid]={'positions':[p['position'] for p in pos],'orders':orders,'recent_fills':sorted([f for f in fillrows if f['address']==a],key=lambda f:f['time'],reverse=True)[:20],'daily':chart,'ending_replay':b1,'issues':issues}

eligible=[r for r in summaries if not r['history_issues']]
if eligible:
 lo=min(r['period_return_percent'] for r in eligible);hi=max(r['period_return_percent'] for r in eligible)
 vl=min(r['daily_volatility_percent_proxy'] for r in eligible);vh=max(r['daily_volatility_percent_proxy'] for r in eligible)
 for r in eligible:
  r['return_score40']=40*(r['period_return_percent']-lo)/(hi-lo) if hi>lo else 20
  r['risk_score30_proxy']=30*(vh-r['daily_volatility_percent_proxy'])/(vh-vl) if vh>vl else 15
  r['score70_proxy']=r['return_score40']+r['risk_score30_proxy']
 eligible.sort(key=lambda r:(-r['score70_proxy'],r['address']))
 for i,r in enumerate(eligible):r['rank70']=i+1 if i==0 or abs(r['score70_proxy']-eligible[i-1]['score70_proxy'])>1e-10 else eligible[i-1]['rank70']
summaries.sort(key=lambda r:(r['rank70'] is None,r['rank70'] or 0,-(r['period_return_percent'] or 0)))
for name,rows in [('leaderboard.csv',summaries),('daily_returns.csv',daily),('wallet_inventory.csv',inventory),('fills.csv',fillrows),('other_market_fills.csv',otherfills),('positions.csv',positionrows),('open_orders.csv',orderrows),('historical_orders.csv',historyrows),('spot_balances.csv',balances),('funding.csv',fundingrows),('ledger.csv',ledgerrows),('capital_flows.csv',cashrows)]:csvout(name,rows)
method={'status':'ASSUMPTION-BASED PRACTICE LEADERBOARD, NOT OFFICIAL RANKINGS','practice_window_start_utc':date(START,UTC),'valuation_cutoff_kst':date(END),'weights':{'net_return':40,'daily_volatility':30,'jury':30},'jury':'unknown; not zero; no 100-point total calculated','return':'Modified Dietz to neutralize external capital changes. Ordinary single-deposit accounts reduce to standard return. USDPR spot conversions treated as capital transfers, not par trading profits. Genesis and external transfers included in capital. Internal spot/par transfers excluded. USDPR transfer fees deducted. Non-par trades excluded.','volatility_proxy':'Population standard deviation (ddof=0) of complete UTC calendar-day Modified Dietz returns. At least two daily observations. Historical equity valued using last available completed par hourly TRADE candle close, not mark or oracle. Final equity valued using actual oracle snapshot. This substitution can materially change rank.','normalization':'Min-max across all calculable active wallets in this report, including bots and unknown identities. Return weight 40 and reversed volatility weight 30. Equal values receive equal half-weight scores. Incomplete rows do not participate in normalization.','discovery':'USDPR distribution-wallet recipients, transitive USDPR ledger graph, existing and refreshed recent par/spot trade addresses. Not an exhaustive verified census; API history limits and unobserved spot buyers may omit wallets.','sources':['https://perpetual-alpha26.github.io/about','https://perpetual-alpha26.github.io/','https://app.notion.com/p/3eb73946edee810b995de6f2fcfa2608','https://hyperliquid.gitbook.io/hyperliquid-docs/for-developers/api/info-endpoint'],'ranked_count':len(eligible),'trading_wallet_count':len(summaries),'candidate_count':len(inventory)}
method['orders_only_wallet_count']=len(unfilled_active)
method['sources'].insert(0,'https://icaif2026.org/competitions.html')
method['source_interpretation']='ICAIF overview describes realized PnL at 40%; competition About specifies total net return marked to oracle at the end. This analysis follows the more detailed competition About definition, using the ICAIF 40/30/30 weights.'
method['pnl_basis']='Exchange-reported closedPnl is accumulated as settled trading PnL. Open positions and entry basis are replayed using startPosition chains, weighted entries, and implied basis on partial closes. Non-trading protocol revenue is not inferred.'
coverage=json.load(open(str(SRC/'candle_coverage_validation.json'))) if (SRC/'candle_coverage_validation.json').exists() else {}
method['candle_count_coverage']={k:v for k,v in coverage.items() if k!='checks'}
method['excluded_nontrading_income']='Model reconstructs par trade/funding PnL and USDPR transfer costs. Deployment-fee income and non-par trading income are not inferred from total wallet balance. Observed spot balances and raw ledgers are exported separately.'
(OUT/'analysis.json').write_text(json.dumps({'methodology':method,'leaderboard':summaries,'details':detail,'unfilled_active':unfilled_active},ensure_ascii=False,indent=2))

body='''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Perpetual Alpha · 통합 연습 참고 순위</title><style>body{font:15px/1.65 system-ui,sans-serif;margin:30px auto;max-width:1550px;padding:0 22px;background:#f7f9fc;color:#172434}h1{font-size:27px}h2{font-size:21px}h3{font-size:17px}.warning{background:#fff0d5;border-left:5px solid #d48b06;padding:18px;margin:20px 0}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;font-size:13px;white-space:nowrap}td,th{padding:9px 11px;border-bottom:1px solid #dce2eb;text-align:right}th{background:#e9eff7}td:first-child,th:first-child{text-align:left}a{color:#145cb7}section,details{background:white;border:1px solid #dbe3ee;border-radius:10px;padding:18px;margin:20px 0}summary{cursor:pointer;font-weight:650}code{font-size:12px}input{padding:10px;width:340px;max-width:90%;border:1px solid #a8b8cc;border-radius:6px}small{color:#526479}p{max-width:1250px}</style><h1>Perpetual Alpha · 모든 발견 거래 지갑의 연습 참고 순위</h1>'''
body+='<p>기준 시각: <b>'+date(END)+' KST</b> · 비교 시작: 2026-10-05 00:00 UTC(한국시간 09:00). 참가팀·봇·운영 지갑을 역할로 제외하지 않았습니다. 계정 소유자를 추측하지 않습니다.</p>'
body+='<div class="warning"><b>공식 대회 리더보드가 아닙니다.</b> 공식 채점은 10월 19일~11월 6일이며 현재는 연습 기간입니다. 과거 일별 오라클·마크 가격을 확보하지 못해 <b>과거 체결 종가로 평가한 일별 변동성</b>을 사용했습니다. 이는 공식 위험 지표를 정확히 재현한 값이 아니며 순위가 달라질 수 있습니다. 심사위원 평가 30점은 미평가이며 0점으로 대체하지 않았습니다. 표의 점수는 가정을 둔 공개 거래 성과 70점입니다.</div>'
body+='<p>조사한 후보 '+str(len(inventory))+'개 · 기준 시각까지 체결한 지갑 '+str(len(summaries))+'개 · 참고 종합점수 산출 '+str(len(eligible))+'개. 초기 자금과 거래 시작일이 다르므로 이 순위로 참가팀의 공식 성적을 단정할 수 없습니다.</p>'
if coverage:
 body+='<p><b>수집 검산:</b> 공개 API가 제공한 완료 시간봉 '+str(coverage.get('all_completed_coin_hours'))+'개의 체결 수와 수집한 고유 체결 '+str(coverage.get('all_observed_trades'))+'건이 모두 일치했습니다. 연습 비교 구간만으로는 '+str(coverage.get('practice_completed_coin_hours'))+'개 종목별 시간봉, '+str(coverage.get('practice_observed_trades'))+'건입니다. 진행 중인 마지막 시간봉과 API가 제공하지 않은 과거 구간의 완전성까지 보장하는 것은 아닙니다. 지갑별 체결 CSV에는 매수자·매도자가 각각 기록되어 같은 체결이 두 번 등장합니다.</p>'
body+='<p>'+ ' · '.join('<a href="'+n+'">'+label+'</a>' for n,label in [('leaderboard.csv','순위·전체 지표 CSV'),('fills.csv','모든 수집 체결'),('daily_returns.csv','일별 수익률·가격시차'),('positions.csv','포지션'),('open_orders.csv','미체결 주문'),('historical_orders.csv','조회된 주문 이력'),('spot_balances.csv','조회 시점 잔액'),('wallet_inventory.csv','모든 후보 지갑'),('capital_flows.csv','자금 이동'),('funding.csv','펀딩비')])+'</p>'
body+='<section><h2>참고 순위 — 70점</h2><p>수익률점수 = 40 × (R − 최저 R) / (최고 R − 최저 R)<br>위험점수 = 30 × (최고 σ − σ) / (최고 σ − 최저 σ)<br>참고점수 = 수익률점수 + 위험점수. 심사 30점은 별도 미평가입니다. σ는 아래 가정을 둔 일별 변동성입니다.</p><input id="filter" placeholder="지갑 ID 또는 주소 검색" aria-label="지갑 검색"><div id="ranking">'
body+=table(['순위','지갑 ID · 주소','기간 수익률 %','일별 변동성 % (대용)','수익률 / 40','위험 / 30 (대용)','합계 / 70 (참고)','기간 손익 USDPR','체결 수','일별 표본 수'],[[r['rank70'] or '미산출',r['wallet_id']+' '+short(r['address']),fmt(r['period_return_percent'],8),fmt(r['daily_volatility_percent_proxy'],8),fmt(r['return_score40'],6),fmt(r['risk_score30_proxy'],6),fmt(r['score70_proxy'],6),fmt(r['period_profit_USDPR'],2),r['practice_fill_count'],r['daily_return_observations']] for r in summaries])+'</div><p>소수점 이하 점수 차이가 매우 작으면 순위 차이를 실질적 성과 차이로 해석할 수 없습니다. 특히 상위 소액 체결 지갑들의 순서는 평가가격·일별 기준 가정에 민감합니다.</p></section>'
body+='<details><summary>계산 가정과 데이터 범위</summary><p>수익률은 10월 5일 UTC 이후부터 기준 시각까지 계산합니다. 그 이후 자금을 받은 지갑은 최초 자금 취득 시각부터 시작합니다. 기간 중 입출금이 있으면 금액과 시점을 반영한 Modified Dietz 수익률을 사용합니다. 한 번 입금하고 추가 입출금이 없으면 일반 수익률과 같습니다.</p><p>일별 자산 = USDPR 순투입 자금 + 누적 매매 현금흐름 − 거래 수수료 + 펀딩 순수취액 − USDPR 이체 수수료 + 보유 수량 × 평가가격. 다른 시장 거래와 현물 취득은 par 매매 이익으로 계산하지 않습니다. USDPR 현물 취득·처분은 외부 자본 변동으로 처리합니다.</p><p>매일 UTC 00:00 전에 완결된 가장 최근 par 1시간 캔들의 종가를 평가가격으로 사용합니다. 종가를 미래에서 가져오거나 시점 이후의 미완결 캔들을 사용하지 않습니다. UTC 00:00, 체결 종가, 모집단 표준편차(ddof=0)는 이 보고서의 가정이며 공식 확정 규정이 아닙니다. 매일 가격이 얼마나 오래됐는지 CSV에 표시했습니다. 마지막 시점은 조회한 실제 오라클 가격을 사용합니다.</p><p>정규화 대상은 이 보고서에서 계산 가능한 모든 거래 지갑입니다. 실제 대회는 등록·자격 요건을 만족한 참가팀으로 정규화하므로 점수와 순위가 다릅니다. 기술보고서 점수는 공개 거래로 추정하지 않았습니다. 관측 일수가 다른 지갑도 있어 초기 단계 순위는 특히 불안정합니다.</p><p>이전 13개 지갑 보고서는 거래 시작 이후 누적 성과였고, 이 보고서는 10월 5일부터의 공통 연습 구간입니다. 평가기간·마지막 가격·입출금 보정 방식이 달라 이전 숫자와 다를 수 있습니다.</p></details>'
for r in summaries:
 wid=r['wallet_id'];de=detail[wid]
 body+='<details id="'+wid+'"><summary>'+wid+' · '+r['address']+' · 참고순위 '+str(r['rank70'] or '미산출')+'</summary>'
 body+='<p>'+html.escape(r['funding_pattern'])+' · 참가팀 등록·실명 대응 미확인. '+html.escape(r['history_issues'] or '수집한 이력으로 가정 기반 지표를 산출했습니다.')+'</p>'
 body+=table(['총 수집 체결','기간 체결','기간 거래대금','메이커 체결 비율','주요 종목','종가 최대 시차(시간)'],[[r['lifetime_par_fill_count'],r['practice_fill_count'],fmt(r['practice_notional_USDPR'],2),fmt(r['practice_maker_fill_percent'],2)+'%',r['top_markets_by_notional'],fmt(r['max_daily_candle_age_hours'],2)]])
 body+=table(['기간 체결 실현손익','기간 거래 수수료','펀딩 순수취액','일별 최대낙폭 대용 %','기준 시각 재구성 자산'],[[fmt(r['practice_gross_closed_pnl_USDPR'],2),fmt(r['practice_trade_fees_USDPR'],2),fmt(r['period_funding_net_USDPR'],2),fmt(r['max_drawdown_percent_daily_proxy'],6),fmt(r['reconstructed_end_equity_USDPR'],2)]])
 body+='<h3>현재 포지션 — 개별 조회 시점 '+html.escape(r['snapshot_positions_utc'])+'</h3>'+table(['종목','수량(−:숏)','평균 진입가','설정 레버리지','미실현손익','청산가'],[[p['coin'],p['szi'],p['entryPx'],str(p['leverage']['value'])+' '+p['leverage']['type'],p['unrealizedPnl'],p['liquidationPx']] for p in de['positions']])
 body+='<h3>미체결 주문</h3>'+table(['종목','매수/매도','주문유형','지정가','수량','Reduce-only'],[[o['coin'],'매수' if o['side']=='B' else '매도',o.get('orderType'),o['limitPx'],o['sz'],o.get('reduceOnly')] for o in de['orders']])
 body+='<h3>최근 체결 20건 — 전체는 CSV</h3>'+table(['한국시간','종목','진입/종료','가격','수량','실현손익','수수료','역할','확인된 상대 주소'],[[f['time_kst'],f['coin'],f['dir'],f['px'],f['sz'],f['closedPnl'],f['fee'],'테이커' if f['crossed'] else '메이커',f.get('counterparty_address') or '이번 수집 자료에서 미확인'] for f in de['recent_fills']])
 body+='<h3>일별 재구성</h3>'+table(['UTC 날짜','체결 종가 평가 자산 USDPR','입출금 조정 일별 수익률 %'],[[t,fmt(e,2),fmt(v,6)] for t,e,v in de['daily']])+'</details>'
body+='<section><h2>출처</h2>'+''.join('<p><a href="'+html.escape(u)+'">'+html.escape(u)+'</a></p>' for u in method['sources'])+'<p>원자료는 raw/ 폴더, 재현 계산은 build_leaderboard.py, 계산법 및 세부 결과는 analysis.json에 있습니다. 공개 체결 이력은 API 보존 한도에 따라 과거 내역이 빠질 수 있습니다. 상대 주소는 수집한 양측 체결 또는 공개 시장 체결에서 일치하는 경우만 기록했습니다.</p></section>'
body+='<section><h2>체결 없이 주문만 확인된 지갑</h2>'+table(['주소','현재 포지션 수','미체결 주문 수','순위 제외 사유'],[[x['address'],len(x['positions']),len(x['orders']),'기준 시각까지 par 체결 없음'] for x in unfilled_active])+'<p>이 계정의 주문도 open_orders.csv에 포함했습니다. par 거래 밖에서 발생한 배포 수수료 수입·다른 시장의 손익은 거래 성과로 추정하지 않았습니다. 실제 조회 자산과 재구성한 par 거래 자산은 그러한 항목 때문에 다를 수 있습니다.</p></section>'
body+='<p>출처 해석: ICAIF 소개는 수익 항목을 realized P&amp;L로 표현하지만 대회 About은 마지막 오라클 가격으로 평가한 총 순수익률을 명시합니다. 이 보고서는 상세한 대회 About의 정의에 따라 최종 평가에 오라클 가격을 사용하고, ICAIF에 명시된 40/30/30 가중치를 적용했습니다.</p>'
body+='<script>const walletSearch='+json.dumps([r['wallet_id']+' '+r['address'] for r in summaries])+';document.querySelectorAll("#ranking tbody tr").forEach((r,i)=>r.dataset.search=walletSearch[i].toLowerCase());document.getElementById("filter").addEventListener("input",e=>{const q=e.target.value.toLowerCase();document.querySelectorAll("#ranking tbody tr").forEach(r=>r.hidden=!r.dataset.search.includes(q))});</script></html>'
body=body.replace('누적 매매 현금흐름','거래소가 기록한 누적 실현손익').replace('보유 수량 × 평가가격','보유 수량 × (평가가격 − 재구성 평균 진입가)')
(OUT/'report.html').write_text(body)
print(json.dumps({'cutoff_kst':date(END),'candidate_count':len(inventory),'trading_wallet_count':len(summaries),'ranked_count':len(eligible),'fills_records':len(fillrows),'output':str(OUT)},ensure_ascii=False))
for r in summaries:print(json.dumps({k:r[k] for k in ['rank70','wallet_id','address','period_return_percent','daily_volatility_percent_proxy','score70_proxy','practice_fill_count','history_issues']},ensure_ascii=False))
