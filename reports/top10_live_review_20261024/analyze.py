"""Public snapshot tables with explicit timestamps and no guessed team mapping."""
import csv
import html
import importlib.util
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
RAW=OUT/'raw'
USER='0x125778065a578d598d88bddaa3fdd635280d0e47'
spec=importlib.util.spec_from_file_location('public_wallet_summary',ROOT/'reports/team_comparison_20261021/analyze.py')
old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)

def save(name,data):
    (OUT/(name+'.json')).write_text(json.dumps(data,ensure_ascii=False,indent=2))

def csvout(name,rows):
    if not rows:return
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with (OUT/(name+'.csv')).open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,keys);writer.writeheader()
        writer.writerows({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v for k,v in r.items()} for r in rows)

def fmt(v):
    if v is None:return '—'
    if isinstance(v,float):return f'{v:,.6f}'
    if isinstance(v,(dict,list)):return json.dumps(v,ensure_ascii=False)
    return str(v)

def table(rows,columns):
    return '<div class="scroll"><table><thead><tr>'+''.join('<th>'+html.escape(v)+'</th>' for _,v in columns)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(fmt(r.get(k)))+'</td>' for k,_ in columns)+'</tr>' for r in rows)+'</tbody></table></div>'

def kst(s):
    return datetime.fromisoformat(s.replace('Z','+00:00')).astimezone(ZoneInfo('Asia/Seoul')).isoformat()

def main():
    wallet_rows=[];fills=[];positions=[];assets=[];orders=[];historical=[];funding=[];scan=[];sources={}
    for path in sorted((RAW/'wallets').glob('*.json')):
        w=json.loads(path.read_text());a=w['address'];sources[a]=w
        scan.append(dict(address=a,at=w['scan_completed_kst'],**w['scan']))
        if 'completed_kst' not in w:continue
        ex=json.loads((RAW/'extra'/path.name).read_text())
        r,fs,ps,ss,hs=old.summarize(w,ex)
        r['long_share_pct']=r['long_notional']/r['gross_notional']*100 if r['gross_notional'] else None
        r['short_share_pct']=r['short_notional']/r['gross_notional']*100 if r['gross_notional'] else None
        r['margin_equity_pct']=r['margin_used']/r['equity']*100 if r['equity'] else None
        r['current_longs']=[p['coin'][4:] for p in ps if p['qty']>0]
        r['current_shorts']=[p['coin'][4:] for p in ps if p['qty']<0]
        r['position_count_long']=len(r['current_longs']);r['position_count_short']=len(r['current_shorts'])
        r['balance_snapshot_kst']=w['spot']['received_kst'];r['collected_to']=w['completed_kst']
        r['close_long_realized_pnl']=sum(float(f['closedPnl']) for f in fs if float(f['startPosition'])>0 and f['side']=='A')
        r['close_short_realized_pnl']=sum(float(f['closedPnl']) for f in fs if float(f['startPosition'])<0 and f['side']=='B')
        r['fees_by_token']=dict(Counter())
        tokenfees=defaultdict(float)
        for f in fs:tokenfees[f['feeToken']]+=float(f['fee'])
        r['fees_by_token']=dict(tokenfees)
        # Table order follows our own sequential public wallet observations, never an inferred name.
        r['name_mapping']='confirmed SFINX' if a==USER else 'not published; no inferred name'
        wallet_rows.append(r);fills+=fs;positions+=ps;assets+=ss
        orders += [dict(address=a,**o) for o in w['orders']['response']]
        historical += [dict(address=a,**h) for h in hs]
        funding += [dict(address=a,**f) for f in w['funding']['rows'] if f['delta'].get('coin','').startswith('pas:')]
    eligible=[r for r in wallet_rows if r['single_grant_100k'] and r['return_pct'] is not None and r['fills']]
    eligible.sort(key=lambda r:-r['return_pct'])
    for n,r in enumerate(eligible,1):r['observed_pnl_rank']=n
    top=eligible[:10]
    peers=defaultdict(dict)
    for f in fills:peers[(f['coin'],f['time'],f['tid'])][f['side']]=f['address']
    recent={}
    for p in (RAW/'recent_trades').glob('*.json'):
        for t in json.loads(p.read_text())['response']:
            recent[(t['coin'],t['tid'])]=t
    for f in fills:
        f['counterparty']=peers[(f['coin'],f['time'],f['tid'])].get('A' if f['side']=='B' else 'B')
        if not f['counterparty']:
            t=recent.get((f['coin'],f['tid']),{});users=[a.lower() for a in t.get('users',[])]
            if len(users)==2 and f['address'] in users:f['counterparty']=users[1] if users[0]==f['address'] else users[0]
    chart=json.loads((OUT/'chart.json').read_text());official=json.loads((OUT/'leaderboard.json').read_text())
    daily={t['name']:t for t in official['response']['teams']}
    teams=[]
    for t in sorted(chart['response']['teams'],key=lambda t:t['position']):
        d=daily.get(t['name'],{});p=t['pnl']
        ts=dict(name=t['name'],live_pnl_rank=t['position'],live_return_pct=p*100,sample_kst=kst(t['last_at']),
                daily_score_rank=d.get('rank'),daily_return_pct=d.get('return',0)*100,
                daily_std_pct=d.get('volatility',0)*100,period_sigma_pct=d.get('volatility_horizon',0)*100,q=d.get('q'),score=d.get('score'))
        if t['name']=='SFINX':ts['confirmed_address']=USER
        teams.append(ts)
    market=[]
    meta,ctxs=json.loads((RAW/'market.json').read_text())['response']
    for u,c in zip(meta['universe'],ctxs):
        market.append(dict(coin=u['name'],mark=float(c['markPx']),oracle=float(c['oraclePx']),
                           mark_oracle_pct=(float(c['markPx'])/float(c['oraclePx'])-1)*100,
                           mark_vs_prev_day_pct=(float(c['markPx'])/float(c['prevDayPx'])-1)*100,
                           hourly_funding_pct=float(c['funding'])*100,day_volume=float(c['dayNtlVlm']),
                           open_interest_notional=float(c['openInterest'])*float(c['markPx'])))
    own=next(r for r in wallet_rows if r['address']==USER)
    manifest=json.loads((RAW/'manifest.json').read_text())
    checks=dict(scanned_wallets=len(scan),granted_wallets=len(manifest['grant_wallets']),detailed_wallets=len(wallet_rows),
                official_chart_has_addresses=any(any('address' in k.lower() or 'wallet' in k.lower() for k in t) for t in chart['response']['teams']),
                all_scanned_single_grant=all(r['single_grant_100k'] for r in scan),
                max_identity_error=max(abs(r['trade_net_pnl']-(r['closed_pnl']-r['fees']+r['funding']+r['unrealized'])) for r in wallet_rows),
                history_issues=[dict(address=r['address'],issues=r['history_issues']) for r in wallet_rows if r['history_issues']],
                accounting_residuals=[dict(address=r['address'],difference=r['accounting_snapshot_difference']) for r in wallet_rows],
                chart_generated_kst=kst(chart['response']['generated_at']),collection_started=manifest['started_kst'],collection_completed=manifest.get('completed_kst'))
    for t in daily.values():
        if t.get('q') is not None:
            q=(max(t['return'],0)+.005)/(abs(t['return'])+1.5*t['volatility_horizon']+.05)
            assert abs(q-t['q'])<1e-11
    assert len(scan)==len(manifest['grant_wallets'])
    assert checks['max_identity_error']<1e-6
    owner_file=OUT/'sfinx_owner_split.json'
    owner_split=json.loads(owner_file.read_text()) if owner_file.exists() else None
    if owner_split:
        for r in owner_split['owners']:
            r['return_pct']=r['pnl']/r['capital']*100
            r['gross_capital_pct']=r['gross']/r['capital']*100
    result=dict(validation=checks,official_top10=teams,wallet_top10=top,wallets=wallet_rows,positions=positions,assets=assets,market=market,sfinx=own,sfinx_owner_split=owner_split)
    save('analysis',result);save('validation',checks)
    for name,rows in [('official_top10',teams),('wallet_top10',top),('wallets',wallet_rows),('wallet_scan',scan),('positions',positions),('fills',fills),('by_market',assets),('open_orders',orders),('historical_orders',historical),('funding',funding),('market',market)]:csvout(name,rows)
    body='<h1>PAS 실시간 PnL 상위권·공개 지갑 비교</h1>'
    body+='<p>공식 차트 생성 '+html.escape(checks['chart_generated_kst'])+' / 지갑 수집 '+html.escape(str(checks['collection_started']))+' ~ '+html.escape(str(checks['collection_completed']))+'. 모든 주문·전략은 변경하지 않고 읽기 전용으로 조회했습니다.</p>'
    body+='<p><b>공식 팀명과 지갑 주소의 대응은 SFINX만 확인됩니다.</b> 공식 공개 데이터에는 주소 필드나 팀별 지갑 링크가 없습니다. 아래 두 순위는 별도이며, 같은 순위·비슷한 PnL을 근거로 이름을 붙이지 않았습니다. SFINX 수익은 본인과 팀원을 합친 계좌 성과입니다.</p>'
    body+='<h2>공식 실시간 그래프 Top 10</h2>'+table(teams,[('live_pnl_rank','장중 PnL 순위'),('name','팀'),('live_return_pct','장중 PnL %'),('sample_kst','표본 KST'),('daily_score_rank','별도 일별 Score 순위'),('daily_return_pct','일별 PnL %'),('daily_std_pct','일별 표준편차 %'),('period_sigma_pct','평가기간 σ %'),('q','일별 Q'),('score','일별 Score')])
    body+='<h2>공개 지급 지갑 상위 10개</h2><p>100,000 USDP 지급 지갑 전체를 확인하고, 거래 이력이 있으며 전송을 검증한 지갑을 비교했습니다. 외부 입출금을 손익에서 제외했으며 지갑 조회는 순차적입니다. 실효 레버리지 = (롱 명목액 + 숏 명목액) / 계좌자산. 롱·숏 비중은 보유 명목액 기준입니다.</p>'
    body+=table(top,[('observed_pnl_rank','조회 PnL 순위'),('address','지갑'),('name','확인된 팀명'),('return_pct','계좌 수익률 %'),('balance_net_pnl','자산 기준 순손익 USDP'),('long_share_pct','롱 비중 %'),('short_share_pct','숏 비중 %'),('configured_leverages','현재 설정 배수'),('effective_gross_leverage','실효 총 레버리지'),('gross_equity_pct','총 명목노출/자산 %'),('margin_equity_pct','증거금/자산 %'),('current_longs','현재 롱'),('current_shorts','현재 숏'),('balance_snapshot_kst','조회 KST')])
    body+='<h2>거래·비용·성과 분해</h2>'+table(top,[('observed_pnl_rank','조회 순위'),('address','주소'),('closed_pnl','청산 실현손익'),('unrealized','현재 미실현'),('funding','누적 펀딩 수취(+)/지급(-)'),('fees','누적 수수료'),('trade_net_pnl','체결 재구성 순손익'),('accounting_snapshot_difference','잔액-재구성 차이'),('fills','체결 조각'),('filled_orders','체결된 주문 수'),('volume','누적 체결대금'),('maker_notional_pct','메이커 대금 비중 %'),('open_orders','미체결 주문 수'),('last_fill','마지막 체결 KST')])
    if owner_split:
        body+='<h2>SFINX 내부 소유권 원장</h2><p>공식 팀 성과는 두 전략 합계입니다. 아래는 주문 ID로 귀속된 내부 장부이며 각 50,000 USDP를 기준으로 합니다. 거래소의 합산 평균진입가를 수량 비례로 나눈 것이 아닙니다. 포지션 평가 시각 '+html.escape(owner_split['position_snapshot_kst'])+'.</p>'+table(owner_split['owners'],[('owner','ours=본인/teammate=팀원'),('capital','배정자본'),('realized','실현손익'),('unrealized','미실현손익'),('fees','수수료'),('funding','펀딩'),('pnl','순손익'),('return_pct','50k 대비 수익률 %'),('gross','명목노출'),('gross_capital_pct','50k 대비 명목노출 %')])
        csvout('sfinx_owner_split',owner_split['owners'])
    body+='<h2>개별 지갑 상세</h2><p>현재 레버리지 설정은 과거 설정 이력을 의미하지 않습니다. 아래 체결 성향은 관찰 결과이며 상대 팀의 모델·예측 규칙을 확인한 것은 아닙니다.</p><input id="search" placeholder="주소 또는 SFINX 검색" style="width:450px;padding:10px">'
    for r in eligible:
        a=r['address'];ps=[p for p in positions if p['address']==a];ss=[p for p in assets if p['address']==a];fs=[f for f in fills if f['address']==a];os=[o for o in orders if o['address']==a]
        body+='<details class="wallet" id="'+a+'" data-search="'+html.escape(a+' '+r['name'])+'"><summary>'+str(r['observed_pnl_rank'])+' · '+html.escape(r['name'] or a)+' · '+f'{r["return_pct"]:+.6f}%'+'</summary><p>전체 주소 '+a+' / '+html.escape(r['snapshot_kst'])+'</p>'
        body+=table(ps,[('coin','종목'),('side','방향'),('qty','부호 있는 수량'),('entry','평균진입가'),('mark','마크가격'),('notional','명목금액'),('unrealized','미실현손익'),('leverage','설정 레버리지'),('margin_type','증거금 방식'),('margin_used','증거금'),('liquidation_price','청산가격')])
        body+=table(ss,[('coin','종목'),('closed_pnl','청산 실현손익'),('fees','수수료'),('funding','펀딩'),('unrealized','미실현'),('net_pnl','재구성 순손익'),('open_long_notional','누적 롱 진입대금'),('open_short_notional','누적 숏 진입대금'),('close_long_notional','누적 롱 청산대금'),('close_short_notional','누적 숏 청산대금'),('median_completed_position_hours','완결 포지션 보유시간 중앙값(h)')])
        body+='<p>최근 100개 체결. 전체 체결은 <a href="fills.csv">fills.csv</a>에 있습니다.</p>'+table(sorted(fs,key=lambda f:f['time'],reverse=True)[:100],[('time_kst','KST'),('coin','종목'),('dir','진입·청산'),('side','B=매수/A=매도'),('px','가격'),('sz','수량'),('closedPnl','실현손익'),('fee','수수료'),('crossed','True=테이커'),('oid','주문번호'),('counterparty','직접 확인한 상대 주소')])
        body+='<p>현재 미체결 주문</p>'+table(os,[('coin','종목'),('side','방향'),('limitPx','주문가격'),('sz','잔여수량'),('orderType','유형'),('reduceOnly','감축 전용'),('oid','주문번호')])
        body+='<p>최근 주문 이력의 유형 '+html.escape(fmt(r['order_types']))+' / 상태 '+html.escape(fmt(r['order_statuses']))+' / TIF '+html.escape(fmt(r['time_in_force']))+'. 이력 API의 최근 2,000개 한도가 적용되므로 전기간 취소·거절 건수가 아닐 수 있습니다.</p>'
        body+='<p>검증 경고: '+html.escape(fmt(r['history_issues']))+'</p></details>'
    body+='<h2>같은 시점의 PAS 시장</h2>'+table(market,[('coin','종목'),('mark','마크'),('oracle','오라클'),('mark_vs_prev_day_pct','전일가격 대비 %'),('mark_oracle_pct','마크/오라클 괴리 %'),('hourly_funding_pct','현재 시간당 펀딩 %'),('day_volume','24h 체결대금'),('open_interest_notional','미결제약정 명목액')])
    body+='<h2>검증 범위</h2><p>계좌 모드별로 unifiedAccount USDP 잔액의 미실현손익을 중복 합산하지 않았습니다. 일반 계좌는 현물 USDP와 PAS 계좌가치를 합산했습니다. 체결은 startPosition에 따라 진입/청산을 나누고 부분 체결을 주문번호별로도 집계했습니다. 이전 원본과 신규 체결·펀딩을 시간 중첩 후 중복 제거해 연결했고, 포지션 수량과 대조했습니다. 잔액과 체결 재구성의 차이는 조회 시각 차이 등을 포함하므로 그대로 공개합니다. 누적 비용은 본대회 시작 이후 PAS 데이터입니다. API 체결 보존·페이지 한도에 걸리거나 수량이 맞지 않으면 경고를 표시합니다. 보유시간은 0에서 진입해 다시 0이 되거나 방향이 바뀐 포지션 구간 기준입니다.</p><pre>'+html.escape(json.dumps(checks,ensure_ascii=False,indent=2))+'</pre>'
    body+='<p>출처: <a href="https://perpetual-alpha26.github.io/leaderboard">공식 리더보드</a> · <a href="https://d23m2l9bjnbgul.cloudfront.net/public/chart.json">공식 실시간 차트 JSON</a> · <a href="https://hyperliquid.gitbook.io/hyperliquid-docs/for-developers/api/info-endpoint">Hyperliquid 공개 Info API</a>. 요청·응답·조회 시각은 raw/에 보존했습니다.</p>'
    body+='<p>'+ ' · '.join('<a href="'+n+'.csv">'+n+'.csv</a>' for n in ('official_top10','wallet_top10','wallets','wallet_scan','positions','fills','by_market','funding','open_orders','historical_orders','market'))+'</p>'
    (OUT/'report.html').write_text('<!doctype html><html lang="ko"><meta charset="utf-8"><title>PAS 실시간 Top 10 조사</title><style>body{font:15px/1.6 system-ui;color:#172634;margin:30px}h1,h2{margin-top:30px}.scroll{overflow-x:auto}table{border-collapse:collapse;white-space:nowrap;margin:14px 0}td,th{padding:7px 10px;border:1px solid #d8e0e8;text-align:right}th{background:#eaf2f8}td:first-child{text-align:left}details{border:1px solid #ccd5df;padding:12px;margin:16px 0}summary{cursor:pointer;font-weight:bold}pre{white-space:pre-wrap}</style>'+body+'<script>document.getElementById("search").addEventListener("input",function(){const q=this.value.toLowerCase();document.querySelectorAll(".wallet").forEach(e=>e.hidden=!e.dataset.search.toLowerCase().includes(q))});if(location.hash){const e=document.getElementById(location.hash.slice(1));if(e)e.open=true;}</script></html>')
    print(json.dumps(dict(validation=checks,official_top10=teams,top10=[{k:r[k] for k in ('observed_pnl_rank','address','return_pct','balance_net_pnl','current_longs','current_shorts','configured_leverages','gross_equity_pct','long_share_pct','short_share_pct','closed_pnl','unrealized','funding','fees','filled_orders','fills','volume','maker_notional_pct','open_orders','history_issues')} for r in top]),ensure_ascii=False,indent=2))

if __name__=='__main__':main()
