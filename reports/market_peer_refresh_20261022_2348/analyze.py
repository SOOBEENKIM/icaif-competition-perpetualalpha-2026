"""Snapshot accounting and comparison. Reads data only; never imports a live runner."""
import csv,datetime as dt,html,importlib.util,json,statistics
from collections import Counter,defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent;RAW=OUT/'raw';USER='0x125778065a578d598d88bddaa3fdd635280d0e47'
spec=importlib.util.spec_from_file_location('old_analysis',ROOT/'reports/team_comparison_20261021/analyze.py');old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
KST=dt.timezone(dt.timedelta(hours=9))
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
 return '<table><thead><tr>'+''.join('<th>'+html.escape(label)+'</th>' for _,label in columns)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(plain(r.get(k)))+'</td>' for k,_ in columns)+'</tr>' for r in rows)+'</tbody></table>'
def main():
 allrows=[];events=[];positions=[];assets=[];fundingrows=[];ledgerrows=[];openorders=[]
 for path in sorted((RAW/'wallets').glob('*.json')):
  w=json.loads(path.read_text());ex=json.loads((RAW/'extra'/path.name).read_text());r,fs,ps,ss,_=old.summarize(w,ex)
  r['collected_to']=w['completed_kst'];r['fills_since_20261021_21KST']=sum(f['time']>=1792584000000 for f in fs)
  allrows.append(r);events+=fs;positions+=ps;assets+=ss
  fundingrows += [dict(address=w['address'],**r) for r in w['funding']['rows'] if r['delta'].get('coin','').startswith('pas:')]
  ledgerrows += [dict(address=w['address'],**r) for r in w['ledger']['rows'] if r['delta'].get('token')=='USDP']
  openorders += [dict(address=w['address'],**o) for o in w['orders']['response']]
 rankable=sorted([r for r in allrows if r['return_pct'] is not None and not r['history_issues']],key=lambda r:-r['return_pct'])
 for i,r in enumerate(rankable,1):r['observed_wallet_pnl_rank']=i
 leaderboard=json.loads((RAW/'leaderboard.json').read_text())['response'];chart=json.loads((RAW/'chart.json').read_text())['response']
 previous=json.loads((ROOT/'reports/pas_performance_research_20261022/peers/leaderboard.json').read_text())['response'];prev={t['name']:t for t in previous['teams']};latest={t['name']:t for t in chart['teams']}
 teams=sorted(leaderboard['teams'],key=lambda t:(t['rank'] is None,t['rank'] or 999,t['name']))
 for t in teams:
  t['return_pct']=t['return']*100 if t['return'] is not None else None
  t['daily_std_pct']=t['volatility']*100 if t['volatility'] is not None else None
  t['period_volatility_pct']=t['volatility_horizon']*100 if t['volatility_horizon'] is not None else None
  t['previous_rank']=prev.get(t['name'],{}).get('rank');t['previous_return_pct']=100*prev[t['name']]['return'] if prev.get(t['name'],{}).get('return') is not None else None
  t['live_pnl_pct']=latest[t['name']]['pnl']*100 if t['name'] in latest else None;t['live_top10_position']=latest.get(t['name'],{}).get('position');t['live_at']=latest.get(t['name'],{}).get('last_at')
  if t['q'] is not None:
   q=(max(t['return'],0)+.005)/(abs(t['return'])+1.5*t['volatility_horizon']+.05);assert abs(q-t['q'])<1e-12;t['implied_J']=(t['score']-.8*q)/.2
 market=[];meta,ctxs=json.loads((RAW/'market.json').read_text())['response'];recent=[]
 for u,c in zip(meta['universe'],ctxs):
  coin=u['name'];f=RAW/'markets'/(coin.replace(':','_')+'.json')
  if not f.exists():continue
  d=json.loads(f.read_text());b=d['book']['response'];bid,ask=b['levels'];bestbid=float(bid[0]['px']) if bid else None;bestask=float(ask[0]['px']) if ask else None;mid=(bestbid+bestask)/2 if bid and ask else None
  cs=sorted(d['candles']['response'],key=lambda x:x['t']);end=int(dt.datetime.fromisoformat(d['candles']['received_kst']).timestamp()*1000);complete=[x for x in cs if x['T']<end]
  def change(hours):
   if not complete:return None
   last=complete[-1];prior=[x for x in complete if x['t']<=last['t']-hours*3600000]
   return (float(last['c'])/float(prior[-1]['c'])-1)*100 if prior else None
  def vwap(side,amount):
   rem=amount;qty=0
   for r in side:
    price=float(r['px']);available=float(r['sz'])*price;take=min(rem,available);qty+=take/price;rem-=take
    if rem<1e-8:return amount/qty
   return None
  row=dict(coin=coin,meta_received_kst=json.loads((RAW/'market.json').read_text())['received_kst'],book_received_kst=d['book']['received_kst'],mark=float(c['markPx']),oracle=float(c['oraclePx']),oracle_basis_pct=(float(c['markPx'])/float(c['oraclePx'])-1)*100,funding_hourly_pct=float(c['funding'])*100,day_volume=float(c['dayNtlVlm']),open_interest_usdp=float(c['openInterest'])*float(c['markPx']),prevDayPx=float(c['prevDayPx']),mark_vs_prev_day_pct=(float(c['markPx'])/float(c['prevDayPx'])-1)*100,close_return_3h_pct=change(3),close_return_24h_pct=change(24),last_completed_candle_kst=old.stamp(complete[-1]['T']) if complete else None,bid=bestbid,ask=bestask,spread_bps=(bestask-bestbid)/mid*10000 if mid else None,best_bid_notional=float(bid[0]['px'])*float(bid[0]['sz']) if bid else 0,best_ask_notional=float(ask[0]['px'])*float(ask[0]['sz']) if ask else 0,buy_1000_vwap=vwap(ask,1000),sell_1000_vwap=vwap(bid,1000))
  for bps in (10,25,50):
   row[f'ask_depth_{bps}bp']=sum(float(r['px'])*float(r['sz']) for r in ask if float(r['px'])<=bestask*(1+bps/10000)) if ask else 0
   row[f'bid_depth_{bps}bp']=sum(float(r['px'])*float(r['sz']) for r in bid if float(r['px'])>=bestbid*(1-bps/10000)) if bid else 0
  market.append(row);recent += d['recent_trades']['response']
 # Counterparties only from explicit public IDs, never infer a team name.
 sides=defaultdict(dict)
 for f in events:sides[(f['coin'],f['time'],f['tid'])][f['side']]=f['address']
 for f in events:
  f['counterparty']=sides[(f['coin'],f['time'],f['tid'])].get('A' if f['side']=='B' else 'B')
  if not f['counterparty']:
   for t in recent:
    us=[x.lower() for x in t.get('users',[])]
    if t['coin']==f['coin'] and t['tid']==f['tid'] and f['address'] in us and len(us)==2:f['counterparty']=us[1] if us[0]==f['address'] else us[0]
 counts=[]
 for m in market:
  ps=[p for p in positions if p['coin']==m['coin']];counts.append(dict(coin=m['coin'],long_wallets=sum(p['qty']>0 for p in ps),short_wallets=sum(p['qty']<0 for p in ps),long_notional=sum(p['notional'] for p in ps if p['qty']>0),short_notional=sum(p['notional'] for p in ps if p['qty']<0)))
 own=next((r for r in allrows if r['address']==USER),None)
 ownfs=[f for f in events if f['address']==USER];ownps=[p for p in positions if p['address']==USER]
 # Only confirmed teammate opening oid is assigned to teammate. All subsequent fills are matched to local dispatch.
 ownership=json.loads((OUT/'runtime/ownership.json').read_text());known_owner={e['row']['oid']:e['owner'] for e in ownership['events'] if e['kind']=='fill'}
 dispatch=json.loads((OUT/'runtime/dispatch.json').read_text())
 for x in dispatch['completed'].values():known_owner[x['oid']]=x['owner']
 # HYPE exit belongs to earlier own strategy; inventory was exclusively ours then.
 for f in ownfs:
  if f['coin']=='pas:HYPE' and f['time']<1792670400000:known_owner[f['oid']]='ours'
 virtual={o:dict(realized=0.,fees=0.,funding=0.,unrealized=0.,gross=0.,quantities={},entries={}) for o in ('ours','teammate')};unknown=[]
 for f in ownfs:
  owner=known_owner.get(f['oid']);f['owner']=owner
  if owner is None:unknown.append(f);continue
  v=virtual[owner];c=f['coin'];q=float(f['sz'])*(1 if f['side']=='B' else -1);px=float(f['px']);oldqty=v['quantities'].get(c,0);entry=v['entries'].get(c,0);new=oldqty+q
  if oldqty*q>=0:v['entries'][c]=(abs(oldqty)*entry+abs(q)*px)/abs(new) if new else 0
  elif oldqty*new<0:v['entries'][c]=px
  close=min(abs(q),abs(oldqty)) if oldqty*q<0 else 0;v['realized']+=close*(px-entry)*(1 if oldqty>0 else -1);v['fees']+=float(f['fee']);v['quantities'][c]=new
 for r in fundingrows:
  if r['address']!=USER:continue
  c=r['delta']['coin'];qs=defaultdict(float)
  for f in ownfs:
   if f['coin']==c and f['time']<=r['time'] and f.get('owner'):qs[f['owner']]+=float(f['sz'])*(1 if f['side']=='B' else -1)
  net=sum(qs.values())
  if net:
   for o,q in qs.items():virtual[o]['funding']+=float(r['delta']['usdc'])*q/net
 for v in virtual.values():
  for c,q in v['quantities'].items():
   if not q:continue
   p=next(x for x in ownps if x['coin']==c);v['unrealized']+=q*(p['mark']-v['entries'][c]);v['gross']+=abs(q)*p['mark']
  v['net_pnl']=v['realized']-v['fees']+v['funding']+v['unrealized'];v['return_on_virtual_50k_pct']=v['net_pnl']/50000*100
 if own:assert abs(sum(v['net_pnl'] for v in virtual.values())-own['trade_net_pnl'])<.01
 summary=dict(collected_wallets=len(allrows),expected_previously_granted_wallets=len(json.loads((RAW/'manifest.json').read_text())['grant_wallets']),verified_traded_wallets=sum(r['fills']>0 for r in allrows),nonempty_positions=sum(r['positions']>0 for r in allrows),directions=dict(Counter(r['current'] for r in allrows)),official_generated_at=leaderboard['generated_at'],official_listed=len(teams),official_ranked=sum(t['eligible'] for t in teams),official_sfinx=next(t for t in teams if t['name']=='SFINX'),sfinx=own,virtual_owner_accounting=virtual,unknown_own_fills=unknown,issues=[dict(address=r['address'],issues=r['history_issues']) for r in allrows if r['history_issues']],runtime=json.loads((OUT/'runtime/status.json').read_text()),name_address_mapping='Only SFINX verified; peers not assigned names',accounting_note='Spot and position snapshots are sequential; balance-vs-fill PnL differences are retained explicitly; valuation timing can contribute, and they are not forced to zero. Current wallet ranking is not official daily Score ranking.')
 save('analysis',dict(summary=summary,wallets=allrows,official=teams,market=market,market_counts=counts,positions=positions,assets=assets));save('summary',summary)
 for n,rs in [('wallets',allrows),('official_teams',teams),('positions',positions),('fills',events),('by_market',assets),('market',market),('market_counts',counts),('funding',fundingrows),('transfers',ledgerrows),('open_orders',openorders)]:csvout(n,rs)
 body='<h1>PAS 시장·지갑·순위 재확인</h1><p>수집은 지갑별 순차 진행. 각 행 시각은 CSV/원자료 참조. 공식 Score는 21:00 KST 일별 평가, 장중 차트 및 지갑 수익률은 별도입니다. 전략·주문·서비스 변경 없음.</p>'
 body+='<h2>실행 상태</h2><p>현재 실행자는 10/22 21:02:33 KST API 429 오류로 종료. 신호 생산자는 동작하지만 자동 주문 및 이 실행자의 위험 감시는 멈췄습니다. 정상 HOLD가 아닙니다. 21:02의 6개 주문·9개 체결은 공개 API와 로컬 주문 ID를 대조했습니다. 재개 전 계좌 및 주문·소유 장부 대조가 필요합니다.</p>'
 body+='<h2>공식 순위 (전체)</h2>'+table(teams,[('rank','Score 순위'),('name','팀'),('return_pct','일별평가 PnL %'),('daily_std_pct','일별 표준편차 %'),('period_volatility_pct','평가기간 σ %'),('q','Q'),('score','Score'),('previous_rank','직전 평가순위'),('live_top10_position','장중차트 순위'),('live_pnl_pct','장중차트 PnL %')])
 body+='<h2>SFINX 계좌 성과</h2>'+table([own],[('snapshot_kst','포지션 조회시각'),('equity','USDP 총자산'),('balance_net_pnl','잔액 기준 순손익'),('return_pct','10만 대비 %'),('closed_pnl','실현손익'),('unrealized','미실현손익'),('fees','수수료'),('funding','펀딩'),('accounting_snapshot_difference','순차 조회 손익 차이'),('gross_equity_pct','자산 대비 명목 노출 %')])
 body+='<p>잔액과 포지션 가격은 동시에 조회할 수 없으므로 차이를 별도로 기록했습니다. 시점 차이 외의 원인을 모두 배제한 것은 아닙니다. 개별 손익 분해는 포지션 시각의 평가가격, 계좌 순수익률은 USDP 잔액 조회값을 사용합니다.</p>'
 body+='<h2>SFINX 현재 보유</h2>'+table(ownps,[('coin','종목'),('side','방향'),('qty','수량'),('entry','계좌 평균진입가'),('mark','평가가격'),('notional','명목 USDP'),('unrealized','미실현손익'),('leverage','설정레버리지')])
 body+='<p>ETH는 팀원 −0.4064와 우리 신규 −0.1549의 합계입니다. 공식 SFINX 성과는 양쪽을 합친 팀 계좌 성과입니다. 아래 가상 장부는 확인된 주문 ID와 시간별 펀딩을 소유 수량에 따라 분리 재계산한 연구 사본이며, 실행자 상태를 수정하지 않았습니다.</p>'+table([dict(owner=o,**v) for o,v in virtual.items()],[('owner','소유'),('realized','실현손익'),('unrealized','미실현'),('funding','펀딩'),('fees','수수료'),('net_pnl','재구성 순손익'),('gross','명목 노출'),('return_on_virtual_50k_pct','50k 대비 %')])
 body+='<h2>SFINX 전체 PAS 체결</h2>'+table(ownfs,[('time_kst','한국시간'),('coin','종목'),('dir','체결방향'),('px','가격'),('sz','수량'),('closedPnl','실현손익'),('fee','수수료'),('owner','소유'),('oid','주문ID')])
 body+='<h2>10종목 시장</h2><p>3h·24h 수익률은 완료된 1시간 체결 캔들 종가. 호가는 별도 현재 스냅샷. 펀딩은 현재 시간당 비율이며 이후에도 고정됨을 뜻하지 않습니다. 양수이면 롱 지급, 음수이면 롱 수취. Mark/Oracle 괴리는 독립적인 차익 보장이 아닙니다.</p>'+table(market,[('coin','종목'),('close_return_3h_pct','3h %'),('close_return_24h_pct','24h %'),('mark','Mark'),('oracle','Oracle'),('oracle_basis_pct','Mark/Oracle 괴리 %'),('funding_hourly_pct','시간당 펀딩 %'),('spread_bps','스프레드 bp'),('best_bid_notional','최우선 매수호가 금액'),('best_ask_notional','최우선 매도호가 금액'),('day_volume','24h 체결대금')])
 body+='<h2>공개 지갑 비교</h2><p>이전 확인된 100,000 USDP 지급 지갑 47개가 대상입니다. 공식 팀명과 지갑 매핑은 SFINX 외에는 미확인입니다. 아래 순위는 수집된 지갑의 장중 수익률 비교이며 공식 Score 순위가 아닙니다. 거래 없는 계좌도 포함. 추가 전송 여부는 전송 원장으로 검증. 설정 레버리지와 실제 명목노출/자본은 서로 다릅니다.</p>'+table(sorted(allrows,key=lambda r:r.get('observed_wallet_pnl_rank',999)),[('observed_wallet_pnl_rank','지갑 PnL 순위'),('address','주소'),('return_pct','장중 수익률 %'),('balance_net_pnl','자산 기준 순손익'),('funding','누적 펀딩'),('fees','누적 수수료'),('fills','체결 조각'),('filled_orders','체결된 주문'),('positions','보유 종목'),('current','현재 방향'),('configured_leverages','설정 레버리지'),('gross_equity_pct','명목노출/자산 %'),('maker_notional_pct','메이커 비중 %'),('open_orders','미체결 주문'),('collected_to','수집종료')])
 body+='<h2>전체 지갑 보유 포지션</h2>'+table(positions,[('address','주소'),('coin','종목'),('side','방향'),('qty','수량'),('entry','진입가'),('mark','평가가격'),('unrealized','미실현'),('leverage','설정 레버리지'),('notional','명목금액')])
 body+='<h2>종목별 보유 분포</h2>'+table(counts,[('coin','종목'),('long_wallets','롱 지갑'),('short_wallets','숏 지갑'),('long_notional','롱 명목금액'),('short_notional','숏 명목금액')])
 body+='<h2>검증 범위</h2><pre>'+html.escape(json.dumps(summary['issues'],ensure_ascii=False,indent=2))+'</pre><p>공개 포지션과 체결만으로 모델, 사전 예측력, 전략의 인과적 우열을 알아낼 수 없습니다. 과거 체결 자료는 이전 완료 수집본과 이번 증가분을 중복 제거해 합쳤습니다. 기간별 체결·펀딩·전송 원본과 수집 시각을 raw에 보관했습니다.</p><p><a href="https://perpetual-alpha26.github.io/leaderboard">공식 리더보드</a> · <a href="wallets.csv">지갑 CSV</a> · <a href="fills.csv">체결 CSV</a> · <a href="market.csv">시장 CSV</a></p>'
 (OUT/'report.html').write_text('<!doctype html><html lang="ko"><meta charset="utf-8"><title>PAS 현황 재확인</title><style>body{font:15px/1.6 system-ui;margin:36px;color:#162332}h1,h2{margin-top:30px}table{border-collapse:collapse;white-space:nowrap;margin:15px 0}td,th{border:1px solid #dbe2e8;padding:7px;text-align:right}th{background:#eaf1f7}td:first-child,th:first-child{text-align:left}p{max-width:1200px}pre{white-space:pre-wrap}</style>'+body+'</html>')
 print(json.dumps(dict(wallets=len(allrows),sfinx=own,virtual=virtual,market=market,issues=summary['issues']),ensure_ascii=False,indent=2))
if __name__=='__main__':main()
