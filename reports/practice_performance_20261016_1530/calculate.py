"""Finalize cutoff metrics from public receipts and sampled actual-account logs."""
from collections import defaultdict
from datetime import datetime, timezone
from decimal import Decimal as D
import json
from pathlib import Path
import sqlite3
import subprocess
import time
from zoneinfo import ZoneInfo

import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
RAW = OUT / 'raw'
KST = ZoneInfo('Asia/Seoul')


def read(name):
    return json.loads((RAW/name).read_text())


def iso(ms):
    return datetime.fromtimestamp(ms/1000, KST).isoformat(timespec='milliseconds')


def save(name, value):
    (OUT/name).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))


def exposure_and_drawdown(rows, flows, start, end, initial, terminal):
    """Sampled mark-equity path. All held positions must be long for gross=cash gap."""
    rows = [dict(x) for x in rows if start <= x['time'] <= end]
    path = [{'time': start, 'equity': initial, 'initial_before_fill': True}] + rows
    path.append({'time': end, 'equity': terminal})
    path.sort(key=lambda r:r['time'])
    cash, cursor, peak, mdd = initial, 0, initial, 0.
    samples = []
    for row in path:
        t, equity = row['time'], row['equity']
        if row.get('initial_before_fill'):
            gross = 0.
        else:
            while cursor < len(flows) and flows[cursor]['time'] <= t:
                cash += flows[cursor]['cash_delta'];cursor += 1
            gross = (equity-cash)/equity
            if gross < -1e-8:
                raise ValueError('Invalid inferred long-only gross exposure')
        peak = max(peak,equity)
        mdd = max(mdd,1-equity/peak)
        samples.append({'time':t,'equity':equity,'gross':max(0.,gross),'drawdown':1-equity/peak})
    weight = sum((b['time']-a['time'])*a['gross'] for a,b in zip(samples,samples[1:]))/(end-start)
    hourly = {}
    for boundary in range((start//3600000+1)*3600000,end+1,3600000):
        eligible = [x for x in samples if x['time'] <= boundary and boundary-x['time'] <= 60000]
        if eligible:
            hourly[boundary]=eligible[-1]
    returns = [dict(end_ms=t,return_=hourly[t]['equity']/hourly[t-3600000]['equity']-1,
                    start_sample_ms=hourly[t-3600000]['time'],end_sample_ms=hourly[t]['time'])
               for t in sorted(hourly) if t-3600000 in hourly]
    return {'sampled_mdd_percent':mdd*100,'sampled_average_gross_percent':weight*100,
            'sampled_maximum_gross_percent':max(x['gross'] for x in samples)*100,
            'equity_observations':len(rows),'maximum_sample_gap_seconds':max(b['time']-a['time'] for a,b in zip(samples,samples[1:]))/1000,
            'hourly_returns':returns,'hourly_sample_std_percent':float(np.std([x['return_'] for x in returns],ddof=1)*100) if len(returns)>=2 else None,
            'equity_series':samples}


def main():
    measurement=read('measurement.json');start,end=measurement['start_ms'],measurement['end_ms']
    if time.time()*1000 < end:
        raise ValueError('The requested reporting cutoff has not occurred')
    if read('capture_health.json')['state']!='complete':
        raise ValueError('Cutoff capture has not completed')
    initial=D(str(measurement['initial_equity']))
    all_fills=read('fills.json')['rows']
    fills=sorted([x for x in all_fills if start<=int(x['time'])<=end and x['coin'].startswith('par:')],key=lambda x:x['time'])
    funding=[x for x in read('funding.json')['rows'] if start<=int(x['time'])<=end and x['delta']['coin'].startswith('par:')]
    transfers=[x for x in read('transfers.json')['rows'] if start<=int(x['time'])<=end]
    # Any capital movement needs explicit classification before finalizing this report.
    if transfers:
        raise ValueError('Unexpected within-period capital events; review before publishing')
    ctx=read('cutoff_contexts.json')
    qty=defaultdict(D);cash=initial;fees=closed=notional=D(0);flows=[]
    by_oid=defaultdict(list)
    for f in fills:
        if f['feeToken']!='USDPR':raise ValueError('Unexpected fee currency')
        q=D(f['sz'])*(1 if f['side']=='B' else -1)
        paid=D(f['fee']);change=-q*D(f['px'])-paid
        qty[f['coin']]+=q;cash+=change;fees+=paid;closed+=D(f['closedPnl']);notional+=abs(q)*D(f['px'])
        flows.append({'time':f['time'],'cash_delta':float(change)})
        by_oid[int(f['oid'])].append(f)
    net_funding=sum((D(x['delta']['usdc']) for x in funding),D(0))
    cash+=net_funding
    flows.extend({'time':x['time'],'cash_delta':float(D(x['delta']['usdc']))} for x in funding)
    flows.sort(key=lambda x:x['time'])
    if any(q<0 for q in qty.values()):
        raise ValueError('Sampled exposure reconstruction requires the observed long-only book')
    mark_equity=cash+sum(q*D(ctx[c]['data']['ctx']['markPx']) for c,q in qty.items())
    oracle_equity=cash+sum(q*D(ctx[c]['data']['ctx']['oraclePx']) for c,q in qty.items())
    mark_pnl=mark_equity-initial;oracle_pnl=oracle_equity-initial
    unrealized=mark_pnl-closed+fees-net_funding
    oracle_unrealized=oracle_pnl-closed+fees-net_funding
    if any(end-ctx[c]['received_ms']>3000 or ctx[c]['received_ms']>end for c,q in qty.items() if q):
        raise ValueError('Missing sufficiently fresh pre-cutoff mark/oracle')
    states=read('account_cutoff_window.json')
    before=[s for s in states if isinstance(s.get('response'),dict) and int(s['response']['time'])<=end
            and int(datetime.fromisoformat(s['received_at_utc']).timestamp()*1000)<=end]
    if not before:raise ValueError('No pre-cutoff exchange account snapshot')
    account=max(before,key=lambda x:int(x['response']['time']))['response']
    if end-int(account['time'])>5000:raise ValueError('Stale pre-cutoff account snapshot')
    actual={p['position']['coin']:D(p['position']['szi']) for p in account['assetPositions']}
    assert all(q==actual.get(c,D(0)) for c,q in qty.items()) and not(set(actual)-set(qty))
    # Independent exchange unrealized-PnL check, transported from the account
    # snapshot's mark to the captured cutoff mark without rounded entry prices.
    exchange_unrealized_at_cutoff=sum((D(p['position']['unrealizedPnl'])
        + qty[p['position']['coin']]*D(ctx[p['position']['coin']]['data']['ctx']['markPx'])
        - D(p['position']['positionValue']) for p in account['assetPositions']),D(0))
    if abs(exchange_unrealized_at_cutoff-unrealized)>D('.01'):
        raise ValueError('Exchange unrealized PnL does not reconcile to the fill ledger')
    orders_api=read('open_orders_after_cutoff.json')
    if not isinstance(orders_api.get('response'),list):raise ValueError('Open order snapshot unavailable')
    db=sqlite3.connect('file:'+str(ROOT/'practice_live/live_runtime/state.sqlite3')+'?mode=ro',uri=True)
    orders=[{'status':s,'plan':json.loads(p),'result':json.loads(r) if r else None}
            for s,p,r in db.execute('select status,plan,result from orders')]
    rounds=[json.loads(v) for v, in db.execute("select value from kv where key like 'round:%'")]
    db.close()
    orders=[o for o in orders if int(o['result'].get('order',{}).get('order',{}).get('timestamp',end+1))<=end]
    submitted_by_oid={int(o['result']['order']['order']['oid']):o for o in orders}
    assert set(by_oid)<=set(submitted_by_oid)
    partial=sum(sum(D(f['sz']) for f in matches)<D(submitted_by_oid[oid]['plan']['size']) for oid,matches in by_oid.items())
    rejected=sum('reject' in o['status'].lower() for o in orders)
    # A report of zero open orders is supported by terminal status of every known
    # submitted order and the independent public snapshot immediately after cutoff.
    if any(o['status'] not in ('filled','canceled','rejected') for o in orders) or orders_api['response']:
        raise ValueError('Open orders around cutoff require individual status timing review')
    raw=subprocess.check_output(['journalctl','--user','-u','icaif-par-practice.service','--since',
        datetime.fromtimestamp(start/1000, KST).strftime('%Y-%m-%d %H:%M:%S'),'--until',
        datetime.fromtimestamp(end/1000,KST).strftime('%Y-%m-%d %H:%M:%S'),'-o','json','--no-pager',
        '--output-fields=__REALTIME_TIMESTAMP,MESSAGE'],text=True)
    logs=[]
    for line in raw.splitlines():
        try:
            row=json.loads(line);value=json.loads(row.get('MESSAGE',''))
            if 'equity' in value and 'at' in value:
                ms=int(datetime.fromisoformat(value['at']).timestamp()*1000)
                if start<=ms<=end:logs.append({'time':ms,'equity':float(value['equity']),'fills':value['fills']})
        except (ValueError,TypeError):pass
    if not logs:raise ValueError('No actual equity observations')
    logs.sort(key=lambda x:x['time'])
    save('actual_equity_logs.json',logs)
    risk=exposure_and_drawdown(logs,flows,start,end,float(initial),float(mark_equity))
    save('sampled_path.json',risk.pop('equity_series'))
    positions=[]
    for p in account['assetPositions']:
        p=p['position'];coin=p['coin'];q=qty[coin]
        coin_fills=[f for f in fills if f['coin']==coin]
        # All current positions are still in their initial long holding cycle.
        assert all(f['dir']=='Open Long' for f in coin_fills)
        vwap=sum(D(f['sz'])*D(f['px']) for f in coin_fills)/q
        mark=D(ctx[coin]['data']['ctx']['markPx']);oracle=D(ctx[coin]['data']['ctx']['oraclePx'])
        positions.append({'coin':coin,'side':'long','qty':str(q),'entry_vwap':str(vwap),
                          'mark':str(mark),'oracle':str(oracle),'mark_value':str(q*mark),
                          'unrealized_mark':str(q*(mark-vwap)),'unrealized_oracle':str(q*(oracle-vwap)),
                          'leverage':p['leverage'],'price_received_ms':ctx[coin]['received_ms']})
    assert abs(sum(D(p['unrealized_mark']) for p in positions)-unrealized)<D('.000001')
    detail={
        'measurement':measurement,'start_kst':iso(start),'end_kst':iso(end),'elapsed_seconds':(end-start)/1000,
        'initial_equity':str(initial),'net_capital_flows':'0','mark_equity':str(mark_equity),'mark_net_pnl':str(mark_pnl),
        'mark_return_percent':str(mark_pnl/initial*100),'oracle_equity':str(oracle_equity),'oracle_net_pnl':str(oracle_pnl),
        'oracle_return_percent':str(oracle_pnl/initial*100),'realized_gross_pnl':str(closed),
        'unrealized_mark':str(unrealized),'unrealized_oracle':str(oracle_unrealized),'fees':str(fees),'net_funding':str(net_funding),
        'complete_daily_returns':0,'daily_std_percent':None,
        'daily_evaluation':'UTC 00:00 (= KST 09:00); only one boundary in the window, no full daily-return pair.',
        'trading_days_kst':len({datetime.fromtimestamp(f['time']/1000,KST).date() for f in fills}),
        'trading_days_utc':len({datetime.fromtimestamp(f['time']/1000,timezone.utc).date() for f in fills}),
        'orders':len(orders),'fills':len(fills),'completed_trades':0,'partial_orders':partial,'rejected_orders':rejected,
        'trade_notional':str(notional),'maker_notional_percent':str(sum((D(f['sz'])*D(f['px']) for f in fills if not f['crossed']),D(0))/notional*100),
        'effective_fee_bps':str(fees/notional*10000),
        'positions':positions,'gross_notional':str(sum(abs(D(p['mark_value'])) for p in positions)),
        'net_notional':str(sum(D(p['mark_value']) for p in positions)),
        'margin_used_exchange_snapshot':account['marginSummary']['totalMarginUsed'],'margin_snapshot_ms':account['time'],
        'open_orders':0,'open_orders_snapshot_utc':orders_api['received_at_utc'],
        'normal_rounds':len([r for r in rounds if r['decision_time']*1000<=end]),
        'risk':risk,'reconciliation':'cash flows + marked actual quantities = initial + closed + unrealized - fees + funding',
        'exchange_unrealized_reconciliation_difference':str(exchange_unrealized_at_cutoff-unrealized),
        'daily_metrics_unavailable_reason':'Measurement interval is shorter than 24 hours and contains no completed UTC daily-return pair.',
        'official_scores':'Not available: practice phase; no official cross-team normalization or jury score.',
        'orders_sent_by_report':0,'strategy_changed':False,
    }
    save('metrics.json',detail)
    print(json.dumps({k:v for k,v in detail.items() if k not in ('risk','positions','measurement')},indent=2,ensure_ascii=False))


if __name__=='__main__':main()
