"""First trading-day report from existing equity observations and public flows.

The precise anniversary mark/oracle was not archived. Report the first actual
account observation after it, with explicit offset; do not interpolate prices.
"""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
from decimal import Decimal as D
from pathlib import Path
from zoneinfo import ZoneInfo
import hashlib,html,json,sqlite3,subprocess,urllib.request

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
KST=ZoneInfo('Asia/Seoul')

def save(name,x):
    (OUT/name).write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False))
def iso(ms):return datetime.fromtimestamp(ms/1000,KST).isoformat(timespec='milliseconds')
def info(payload):
    sent=datetime.now(timezone.utc).isoformat()
    req=urllib.request.Request('https://api.hyperliquid-testnet.xyz/info',data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=20) as r:value=json.load(r)
    return {'request':payload,'sent_at_utc':sent,'received_at_utc':datetime.now(timezone.utc).isoformat(),'response':value}

def main():
    if (OUT/'metrics.json').exists():raise FileExistsError('Completed first-day report exists')
    db=sqlite3.connect('file:'+str(ROOT/'practice_live/live_runtime/state.sqlite3')+'?mode=ro',uri=True)
    db.execute('BEGIN')
    anchor=json.loads(db.execute("select value from kv where key='anchor'").fetchone()[0])
    cached=[json.loads(r) for r, in db.execute("select raw from records where kind='fills'")]
    db.close()
    first=min(f['time'] for f in cached);boundary=first+86400000
    start_query=datetime.fromtimestamp((boundary-120000)/1000,KST).strftime('%Y-%m-%d %H:%M:%S')
    end_query=datetime.fromtimestamp((boundary+120000)/1000,KST).strftime('%Y-%m-%d %H:%M:%S')
    lines=subprocess.check_output(['journalctl','--user','-u','icaif-par-practice.service','--since',start_query,'--until',end_query,
         '-o','json','--no-pager','--output-fields=__REALTIME_TIMESTAMP,MESSAGE'],text=True)
    records=[]
    for line in lines.splitlines():
        try:
            envelope=json.loads(line);row=json.loads(envelope.get('MESSAGE',''))
            if not all(k in row for k in ['at','equity','fills','mode']):continue
            if row['mode']!='live_testnet':continue
            row['record_ms']=int(datetime.fromisoformat(row['at']).timestamp()*1000)
            records.append(row)
        except (ValueError,TypeError):pass
    records.sort(key=lambda r:r['record_ms'])
    before=max((r for r in records if r['record_ms']<=boundary),key=lambda r:r['record_ms'])
    after=min((r for r in records if r['record_ms']>=boundary),key=lambda r:r['record_ms'])
    if after['record_ms']-boundary>60000:raise ValueError('No close enough post-anniversary observation')
    save('equity_boundary_records.json',{'before':before,'after':after,'records':records})
    end=after['record_ms'];user=anchor['team_wallet'];initial=D(str(anchor['initial_equity']))
    kinds=['userFillsByTime','userFunding','userNonFundingLedgerUpdates']
    payloads=[dict(type=k,user=user,startTime=first,endTime=end) for k in kinds]
    with ThreadPoolExecutor(max_workers=3) as pool:responses=list(pool.map(info,payloads))
    for k,r in zip(kinds,responses):
        if not isinstance(r['response'],list) or len(r['response'])>=(2000 if k=='userFillsByTime' else 500):raise ValueError('Incomplete public history')
        save(k+'.json',r)
    fills=[f for f in responses[0]['response'] if first<=f['time']<=end and f['coin'].startswith('par:')]
    funding=[f for f in responses[1]['response'] if first<=f['time']<=end and f['delta'].get('coin','').startswith('par:')]
    transfers=responses[2]['response']
    if transfers:raise ValueError('Capital events require classification before returning a daily ROI')
    assert len(fills)==after['fills']
    assert {(f['coin'],f['tid']) for f in fills}=={(f['coin'],f['tid']) for f in cached if first<=f['time']<=end and f['coin'].startswith('par:')}
    assert all(f['feeToken']=='USDPR' for f in fills)
    fees=sum((D(f['fee']) for f in fills),D(0));closed=sum((D(f['closedPnl']) for f in fills),D(0))
    funding_net=sum((D(f['delta']['usdc']) for f in funding),D(0))
    equity=D(str(after['equity']));pnl=equity-initial;unrealized=pnl-closed+fees-funding_net
    qty={}
    for f in fills:qty[f['coin']]=qty.get(f['coin'],D(0))+D(f['sz'])*(1 if f['side']=='B' else -1)
    assert closed+unrealized-fees+funding_net==pnl
    status=json.loads((ROOT/'practice_live/live_runtime/status.json').read_text())
    metric={'wallet':user,'initial_equity':str(initial),'first_fill_kst':iso(first),
        'exact_24h_boundary_kst':iso(boundary),'valuation_record_kst':iso(end),
        'elapsed_seconds':(end-first)/1000,'offset_from_24h_seconds':(end-boundary)/1000,
        'mark_equity':str(equity),'mark_net_pnl':str(pnl),'first_day_return_percent_approx':str(pnl/initial*100),
        'external_flows':'0','fees':str(fees),'funding_net':str(funding_net),'reported_realized_gross_pnl':str(closed),
        'implied_unrealized_mark_pnl':str(unrealized),'fills':len(fills),'submitted_order_ids':len({f['oid'] for f in fills}),
        'positions':{k:str(v) for k,v in qty.items()},'complete_24h_intervals':1,
        'daily_std_sample_percent':None,'daily_std_explanation':'One first-day return; sample STD requires at least two returns, and meaningful estimation needs a longer series.',
        'exact_anniversary_valuation_available':False,'oracle_equity_at_anniversary':None,
        'valuation_source':'Existing live bot account-equity journal, mark-price basis. Public API fills/funding/transfers cross-checked; no price interpolation.',
        'precision_note':'The value is an approximate first-24h return, using the first actual account observation after the anniversary. The log timestamp is not a separately archived exact exchange-price timestamp.',
        'official_daily_cutoff_verified':False,'official_rule_url':'https://perpetual-alpha26.github.io/about',
        'current_bot_status':{k:status[k] for k in ['updated_at_utc','state','mode','profile','next_normal_decision_kst']},
        'orders_sent_by_report':0,'strategy_changed':False}
    save('metrics.json',metric)
    rows=[('시작',metric['first_fill_kst']),('정확한 24시간 경계',metric['exact_24h_boundary_kst']),
       ('사용한 계좌 기록',metric['valuation_record_kst']),('경계와 차이',str(metric['offset_from_24h_seconds'])+'초 뒤'),
       ('시작 자산',str(initial)+' USDPR'),('기록상 자산',str(equity)+' USDPR'),('순손익',str(pnl)+' USDPR'),
       ('첫 24시간 수익률 (근사)',metric['first_day_return_percent_approx']+'%'),('수수료',str(fees)+' USDPR'),
       ('펀딩 순수취',str(funding_net)+' USDPR'),('청산 실현손익',str(closed)+' USDPR'),('미실현 평가손익',str(unrealized)+' USDPR'),
       ('일별 변동성','계산 불가: 하루 수익률 1개')]
    page='<!doctype html><html lang="ko"><meta charset="utf-8"><title>첫 24시간 거래 수익률</title><style>body{font:16px/1.7 system-ui;max-width:850px;margin:36px auto;padding:0 20px;color:#17243a}table{border-collapse:collapse;width:100%}td,th{padding:10px;border-bottom:1px solid #ddd;text-align:left}</style><h1>첫 24시간 거래 수익률</h1><p>기존판 실제 거래 기록. 수수료·펀딩 반영, 외부 입출금 없음.</p><table>'+''.join('<tr><th>'+html.escape(k)+'</th><td>'+html.escape(v)+'</td></tr>' for k,v in rows)+'</table><p><strong>정확한 24시간 경계의 시세는 따로 저장되지 않았습니다.</strong> 약 10.8초 뒤 계좌 기록으로 계산한 근사 수익률입니다. Mark 평가이며, 해당 시각 oracle 평가액은 확인하지 못했습니다. 현재 웹 화면과 평가 기준·조회 시각이 다를 수 있습니다.</p><p>이 측정은 첫 체결부터 하루입니다. <a href="https://perpetual-alpha26.github.io/about">공식 규정</a>은 일별 수익률 변동성을 평가하지만, 이 공개 페이지에는 하루의 집계 경계 시각이 명시되지 않았습니다. 거래 전의 0% 수익률을 보태 일별 변동성을 만들지 않았습니다.</p></html>'
    (OUT/'report.html').write_text(page)
    save('manifest.json',{'created_at_utc':datetime.now(timezone.utc).isoformat(),'files_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.glob('*') if p.is_file() and p.name!='manifest.json'}})
    print(json.dumps(metric,indent=2,ensure_ascii=False))

if __name__=='__main__':main()
