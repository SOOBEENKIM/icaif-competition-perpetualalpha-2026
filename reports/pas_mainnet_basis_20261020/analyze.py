"""Read-only analysis of public API receipts; no trading dependencies/signers."""
import json, math, html
from pathlib import Path
from decimal import Decimal as D
from datetime import datetime,timezone,timedelta
P=Path(__file__).resolve().parent
r=json.loads((P/'snapshot.json').read_text())
M={'BTC':'BTC','ETH':'ETH','HYPE':'HYPE','SP500':'xyz:SP500','EWY':'xyz:EWY','EWJ':'xyz:EWJ','BRENT':'xyz:BRENTOIL','GOLD':'xyz:GOLD','USDEUR':'xyz:EUR','USDJPY':'xyz:JPY'}
ctx={};meta={}
for key in ('pas_context','native_context','xyz_context'):
    a,b=r[key]['data']
    for x,y in zip(a['universe'],b):ctx[x['name']]=y;meta[x['name']]=x
F=.0009

def sweep(levels,qty):
    remain=qty;value=0.
    for row in levels:
        take=min(remain,float(row['sz']));value+=take*float(row['px']);remain-=take
        if remain<=1e-10:break
    got=qty-remain
    return {'requested_qty':qty,'filled_qty':got,'vwap':value/got if got else None,'notional':value,'complete':remain<=1e-10}
rows=[]
for s,main in M.items():
    pc,mc=ctx['pas:'+s],ctx[main]
    pb,mb=r['pas_book_'+s]['data'],r['main_book_'+s]['data']
    bid,ask=[float(x[0]['px']) for x in pb['levels']]
    mainbid,mainask=[float(x[0]['px']) for x in mb['levels']];mid=(mainbid+mainask)/2
    direction=1 if (mid-ask)/ask >= (bid-mid)/bid else -1
    entry=ask if direction==1 else bid
    # Hypothetical PAS exit at current mainnet midpoint. No guarantee of convergence;
    # excludes future exit spread/impact/funding, not simultaneous cross-network arbitrage.
    gross=direction*(mid-entry)/entry
    fee=F*(1+mid/entry)
    unit=10**meta['pas:'+s]['szDecimals']
    qty=math.floor(1000/entry*unit)/unit
    fill=sweep(pb['levels'][1 if direction==1 else 0],qty)
    net_1k=direction*(mid-fill['vwap'])/fill['vwap']-F*(1+mid/fill['vwap']) if fill['complete'] and fill['vwap'] else None
    rows.append({'symbol':s,'mainnet_coin':main,'pas_bid':bid,'pas_ask':ask,'mainnet_bid':mainbid,'mainnet_ask':mainask,'mainnet_mid':mid,'pas_mark':float(pc['markPx']),'mainnet_mark':float(mc['markPx']),'pas_oracle':float(pc['oraclePx']),'mainnet_oracle':float(mc['oraclePx']),'pas_mark_vs_main_mark_bps':(float(pc['markPx'])/float(mc['markPx'])-1)*1e4,'pas_ask_vs_main_mid_bps':(ask/mid-1)*1e4,'pas_bid_vs_main_mid_bps':(bid/mid-1)*1e4,'pas_spread_bps':(ask-bid)/((ask+bid)/2)*1e4,'pas_funding_hourly':float(pc['funding']),'mainnet_funding_hourly':float(mc['funding']),'hypothetical_direction':'long' if direction==1 else 'short','hypothetical_convergence_gross_bps':gross*1e4,'hypothetical_two_taker_fees_bps':fee*1e4,'hypothetical_convergence_net_bps_ex_funding_exit_impact':(gross-fee)*1e4,'hypothetical_1000_notional_fill':fill,'hypothetical_1000_net_bps_ex_funding_exit_impact':net_1k*1e4 if net_1k is not None else None,'pas_top_bid_notional':bid*float(pb['levels'][0][0]['sz']),'pas_top_ask_notional':ask*float(pb['levels'][1][0]['sz']),'book_timestamp_gap_ms':abs(pb['time']-mb['time'])})
fills=[x for x in r['fills']['data'] if x['coin'].startswith('pas:')]
funding=[x for x in r['funding']['data'] if x['delta']['coin'].startswith('pas:')]
fees=sum(D(x['fee']) for x in fills);received=sum(D(x['delta']['usdc']) for x in funding)
realized=sum(D(x['closedPnl']) for x in fills)
eq=D(next(x['total'] for x in r['spot']['data']['balances'] if x['coin']=='USDP'))
pos=[];upnl=D(0);reference_upnl=D(0);close_cash=D(0)
for p0 in r['positions']['data']['assetPositions']:
    p=p0['position'];s=p['coin'].split(':')[1];q=D(p['szi']);fs=[x for x in fills if x['coin']==p['coin']]
    assert all(x['side']=='B' and D(x['closedPnl'])==0 for x in fs), 'Analysis requires unchanged long-only positions'
    cost=sum(D(x['sz'])*D(x['px']) for x in fs)
    assert q==sum(D(x['sz']) for x in fs)
    mainmark=D(ctx[M[s]]['markPx']);paspnl=D(p['unrealizedPnl'])
    reference_pnl=q*mainmark-cost;upnl+=paspnl;reference_upnl+=reference_pnl
    close=sweep(r['pas_book_'+s]['data']['levels'][0],float(q))
    close_cash+=D(str(close['notional']))*(D(1)-D(str(F)))-cost
    minute=fs[0]['time']//60000*60000
    candle=next(x for x in r['main_entry_candles_'+s]['data'] if x['t']==minute)
    pos.append({'symbol':s,'quantity':float(q),'fill_vwap_exact':float(cost/q),'unrealized_pnl':float(paspnl),'mainnet_mark_revaluation_pnl':float(reference_pnl),'current_mark_difference_impact':float(paspnl-reference_pnl),'entry_minute_mainnet_candle':candle,'entry_minute_reference_move_pnl_range':sorted([float(q*(mainmark-D(candle['h']))),float(q*(mainmark-D(candle['l'])))]),'hypothetical_close':close})
net=realized+upnl-fees+received
assert abs((eq-D(100000))-net)<D('0.000002'), (eq,net)
summary={'snapshot_utc':r['collected_utc'],'snapshot_kst':datetime.fromisoformat(r['collected_utc']).astimezone(timezone(timedelta(hours=9))).isoformat(),'equity_usdp':float(eq),'pnl_usdp':float(net),'return_percent':float(net/D(100000)*100),'unrealized_pnl_usdp':float(upnl),'realized_price_pnl_usdp':float(realized),'fees_paid_usdp':float(fees),'funding_received_usdp':float(received),'filled_orders':len(set(x['oid'] for x in fills)),'fill_fragments':len(fills),'mainnet_mark_hypothetical_pnl_usdp':float(reference_upnl+realized-fees+received),'current_mark_difference_impact_usdp':float(upnl-reference_upnl),'hypothetical_full_close_pnl_usdp_at_displayed_depth':float(close_cash-fees+received),'positions':pos,'markets':rows,'limitations':['Snapshot only: does not establish future convergence or strategy profitability.','Mainnet book midpoint is a reference, not a price executable on PAS.','Cross-network accounts and collateral are distinct; these are PAS convergence scenarios.','Entry minute trade candles provide a range approximation, not exact simultaneous historical quotes or oracles.','Hypothetical full close assumes displayed depth remains available and 9 bp taker exit fees.','Fee-adjusted convergence calculations exclude future funding, exit spread/impact and latency.','Current-mark revaluation is an accounting counterfactual, not a full historical basis attribution.']}
(P/'analysis.json').write_text(json.dumps(summary,indent=2))

def num(v,n=4):return f'{v:,.{n}f}'
def table(headers,data):
    return '<table><thead><tr>'+''.join('<th>'+html.escape(str(x))+'</th>' for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in row)+'</tr>' for row in data)+'</tbody></table>'
parts=['<!doctype html><html lang="ko"><meta charset="utf-8"><title>SFINX PAS·메인넷 가격 괴리 분석</title><style>body{max-width:1250px;margin:36px auto;font:16px/1.65 system-ui;color:#172635;padding:0 20px}table{border-collapse:collapse;width:100%;font-size:14px}th,td{padding:9px;border:1px solid #dce3e8;text-align:right}th:first-child,td:first-child{text-align:left}h1,h2{line-height:1.3}code{background:#eef2f6}</style><h1>PAS·메인넷 가격 괴리와 SFINX 손익</h1>',f'<p>조회: {html.escape(summary["snapshot_kst"])}. 공개 API 29개 응답을 저장했다. 주문 및 운용 코드 변경 없음.</p>',f'<p><b>계좌 순손익 {num(float(net),6)} USDP ({num(summary["return_percent"],6)}%).</b> 현재 두 포지션의 평가가격만 메인넷 마크로 바꿔도 {num(summary["mainnet_mark_hypothetical_pnl_usdp"],6)} USDP다. 현재 PAS 마크 차이는 {num(summary["current_mark_difference_impact_usdp"],6)} USDP를 설명한다. 전체 손실을 현재 가격 괴리로 설명할 수 없다.</p>','<h2>실제 손익 대조</h2>',table(['항목','USDP'],[['BTC·HYPE 미실현 가격손익',num(float(upnl),6)],['실현 가격손익',num(float(realized),6)],['지급 수수료',num(-float(fees),6)],['수령 펀딩',num(float(received),6)],['합계',num(float(net),6)],['현재 호가 전량 청산 가정 순손익(추가 수수료 포함)',num(summary['hypothetical_full_close_pnl_usdp_at_displayed_depth'],6)]]),'<p>두 매수 주문, 세 체결 조각이며 청산은 없었다. 통합 계좌 USDP 총액은 평가손익을 포함하므로 미실현손익을 다시 더하지 않았다. 전량 청산값은 호가 유지와 전량 체결을 가정하며 실제 체결 결과가 아니다.</p>','<h2>10종목 동시점에 가까운 호가 비교</h2><p>메인넷 가격은 매수·매도 최우선 호가 중간값. PAS bid는 지금 파는 가격, ask는 지금 사는 가격. 원천의 시각 차이는 최대 '+str(max(x['book_timestamp_gap_ms'] for x in rows))+'ms.</p>',table(['종목','메인넷 mid','PAS bid','PAS ask','ask/mid 괴리(bp)','PAS 마크/메인넷 마크(bp)','PAS 펀딩(%/h)'],[[x['symbol'],num(x['mainnet_mid'],5),num(x['pas_bid'],5),num(x['pas_ask'],5),num(x['pas_ask_vs_main_mid_bps'],2),num(x['pas_mark_vs_main_mark_bps'],2),num(x['pas_funding_hourly']*100,6)] for x in rows]),'<h2>괴리가 줄어들 때의 가상 거래</h2><p>PAS에서 진입 후 현재 메인넷 중간가격에 PAS에서 청산된다고 가정한다. 미래 가격, 수렴, 체결을 보장하지 않는다. 양쪽 테이커 수수료는 실제 체결과 같은 9bp씩 적용. 미래 청산 스프레드·충격·펀딩은 제외했다. 1,000 명목금액은 현재 20레벨 호가를 직접 소진해 평균가격을 계산했다. 메인넷에서 주문하는 시나리오가 아니다.</p>',table(['종목','방향','단위수량 괴리 이익(bp)','왕복 비용 후(bp)','1,000 명목 왕복 비용 후(bp)','진입 최우선 수량 가치'],[[x['symbol'],x['hypothetical_direction'],num(x['hypothetical_convergence_gross_bps'],2),num(x['hypothetical_convergence_net_bps_ex_funding_exit_impact'],2),num(x['hypothetical_1000_net_bps_ex_funding_exit_impact'],2) if x['hypothetical_1000_net_bps_ex_funding_exit_impact'] is not None else '호가 부족',num(x['pas_top_ask_notional'] if x['hypothetical_direction']=='long' else x['pas_top_bid_notional'],2)] for x in rows]),'<p>이 순간에는 10종목 모두 위 단순 가정에서 테이커 왕복 비용 차감 후 음수다. 메이커 체결, 펀딩 수령, 이후 가격 변화는 별도 모델링 대상이다. 특히 펀딩은 변하며 현재 요율을 장기간 고정 수익으로 간주할 수 없다.</p>','<h2>진입 시점과 기초가격 하락</h2>',table(['종목','실제 평균 매수가','진입 분 메인넷 저가~고가','현재 메인넷 마크','PAS 미실현손익','현재 PAS 마크차 영향'],[[x['symbol'],num(x['fill_vwap_exact'],5),x['entry_minute_mainnet_candle']['l']+' ~ '+x['entry_minute_mainnet_candle']['h'],num(ctx[M[x['symbol']]]['markPx'] and float(ctx[M[x['symbol']]]['markPx']),5),num(x['unrealized_pnl'],6),num(x['current_mark_difference_impact'],6)] for x in pos]),'<p>진입 분(10월 19일 21:02 KST)의 메인넷 1분 체결 캔들 범위와 비교해 두 기초가격 모두 하락했다. 분봉은 정확한 진입 순간 호가/오라클이 아니므로 역사적 괴리 변화의 정확한 분해로 주장하지 않는다.</p>','<h2>운용 코드 확인</h2><p>실거래는 PAS·USDP에 연결되어 있다. 기존 baseline은 메인넷 기준시장 신호를 사용하고, submit_baseline은 PAS 마크를 source_prices로 넣는다. SharedAccount.seal은 그 가격 주위 10bp 실행 한도를 만들며 LiveAccount가 현재 호가·수량을 확인한다. 따라서 실행 한도는 있으나 baseline의 메인넷/PAS 괴리·향후 수렴·왕복 비용 대비 기대수익을 직접 검사하지 않는다. pair overlay에는 별도의 비용 판단이 있지만 현재 pair_state는 null이고 최근 진입 판단은 decision_gate_rejected였다. 현재 보유 두 롱의 성과를 페어거래 실적이라고 부를 수 없다.</p>','<p>관련 코드: competition_live/coordinator.py submit_baseline, experiments/shared_execution_pairs_20261017/account_v2.py seal, competition_live/live_account.py execute_books. 개선 검증 대상은 방향 예측 + 괴리 수렴 기대 + 보유 중 펀딩 − 진입·청산 비용을 일관되게 평가하는 PAS 전용 진입·유지·청산 판단이다.</p>','<h2>증거와 한계</h2><p><a href="snapshot.json">원본 API 응답</a> · <a href="analysis.json">계산 결과</a> · <a href="analyze.py">재계산 코드</a> · <a href="https://perpetual-alpha26.github.io/docs/">공식 Participant Guide</a> · <a href="https://perpetual-alpha26.github.io/">대회 개요</a></p><ul>'+''.join('<li>'+html.escape(s)+'</li>' for s in summary['limitations'])+'</ul></html>']
(P/'report.html').write_text(''.join(parts))
print(json.dumps({k:v for k,v in summary.items() if k not in ('positions','markets','limitations')},indent=2))
for x in rows:print(x['symbol'],x['hypothetical_direction'],'net_bps',round(x['hypothetical_convergence_net_bps_ex_funding_exit_impact'],3),'net_1k_bps',round(x['hypothetical_1000_net_bps_ex_funding_exit_impact'],3) if x['hypothetical_1000_net_bps_ex_funding_exit_impact'] is not None else None,'depth',round(x['pas_top_ask_notional'] if x['hypothetical_direction']=='long' else x['pas_top_bid_notional'],2))
