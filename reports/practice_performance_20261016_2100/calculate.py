"""Read-only reconciliation for the first strategy, ending exactly 21:00 KST.

Prices at the exact cutoff were not archived. Use the last actual equity
observation before it, and disclose its age. Never reuse the 15:30 price snapshot
or post-cutoff funding, and never inspect the reserved forward experiment.
"""
from collections import defaultdict
from datetime import datetime, timezone
from decimal import Decimal as D
import hashlib
import importlib.util
import json
from pathlib import Path
import sqlite3
from zoneinfo import ZoneInfo

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
KST = ZoneInfo('Asia/Seoul')
SOURCE = ROOT/'reports/practice_performance_first24h_20261016'


def ms(s):
    return int(datetime.fromisoformat(s).timestamp()*1000)


def iso(t):
    return datetime.fromtimestamp(t/1000, KST).isoformat(timespec='milliseconds')


def save(name, value):
    (OUT/name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def main():
    end = ms('2026-10-16T21:00:00+09:00')
    receipts = {name: json.loads((SOURCE/(name+'.json')).read_text()) for name in
                ['userFillsByTime', 'userFunding', 'userNonFundingLedgerUpdates']}
    for receipt in receipts.values():
        assert receipt['request']['endTime'] >= end
    fills = sorted((f for f in receipts['userFillsByTime']['response']
                    if f['time'] <= end and f['coin'].startswith('par:')), key=lambda f:f['time'])
    start = fills[0]['time']
    funding = [f for f in receipts['userFunding']['response']
               if start <= f['time'] <= end and f['delta']['coin'].startswith('par:')]
    assert not [f for f in receipts['userNonFundingLedgerUpdates']['response'] if start <= f['time'] <= end]
    assert all(f['feeToken']=='USDPR' and f['dir']=='Open Long' and f['side']=='B' for f in fills)
    logs = json.loads((OUT/'actual_equity_logs.json').read_text())
    logs = [dict(r,time=ms(r['at'])) for r in logs if start <= ms(r['at']) <= end]
    logs.sort(key=lambda r:r['time'])
    last = logs[-1]
    assert 0 <= end-last['time'] < 60_000
    # The last mark valuation and all included cash flows cover the same ledger.
    assert not [f for f in fills+funding if last['time'] < f['time'] <= end]
    initial = D('1000000')
    fees = sum((D(f['fee']) for f in fills),D(0))
    closed = sum((D(f['closedPnl']) for f in fills),D(0))
    net_funding = sum((D(f['delta']['usdc']) for f in funding),D(0))
    notional = sum((D(f['sz'])*D(f['px']) for f in fills),D(0))
    equity = D(str(last['equity']))
    pnl = equity-initial
    unrealized = pnl-closed+fees-net_funding
    gross = notional+unrealized
    assert closed+unrealized-fees+net_funding == pnl
    qty = defaultdict(D)
    flows=[]
    for f in fills:
        qty[f['coin']] += D(f['sz'])
        flows.append({'time':f['time'],'cash_delta':float(-D(f['sz'])*D(f['px'])-D(f['fee']))})
    flows.extend({'time':f['time'],'cash_delta':float(D(f['delta']['usdc']))} for f in funding)
    flows.sort(key=lambda f:f['time'])
    risk_module = ROOT/'reports/practice_performance_20261016_1530/calculate.py'
    spec=importlib.util.spec_from_file_location('cutoff_risk',risk_module)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    risk=module.exposure_and_drawdown(logs,flows,start,end,float(initial),float(equity))
    save('sampled_path.json',risk.pop('equity_series'))
    database=ROOT/'experiments/performance_attribution_20261016/deployment/state_before_switch.sqlite3'
    with sqlite3.connect(database.resolve().as_uri()+'?mode=ro',uri=True) as db:
        anchor=json.loads(db.execute("select value from kv where key='anchor'").fetchone()[0])
        orders=[dict(status=s,plan=json.loads(p),result=json.loads(r)) for s,p,r in db.execute('select status,plan,result from orders')]
        rounds=[json.loads(v) for v, in db.execute("select value from kv where key like 'round:%'")]
    orders=[o for o in orders if o['result']['order']['order']['timestamp'] <= end]
    assert all(o['status']=='filled' and o['result']['order']['statusTimestamp'] <= end for o in orders)
    assert {o['result']['order']['order']['oid'] for o in orders} == {f['oid'] for f in fills}
    assert len(orders)==last['orders']==4 and len(fills)==last['fills']==11
    partial=sum(sum(D(f['sz']) for f in fills if f['oid']==o['result']['order']['order']['oid']) < D(o['plan']['size']) for o in orders)
    positions=[]
    for coin,q in qty.items():
        value=sum((D(f['sz'])*D(f['px']) for f in fills if f['coin']==coin),D(0))
        positions.append({'coin':coin,'qty':str(q),'entry_vwap':str(value/q),'side':'long'})
    m={'wallet':anchor['team_wallet'],'strategy':'default / trend + 7-day reversion, tail calibrated',
       'first_fill_kst':iso(start),'cutoff_kst':iso(end),'valuation_record_kst':iso(last['time']),
       'valuation_age_seconds':(end-last['time'])/1000,'elapsed_seconds_to_cutoff':(end-start)/1000,
       'initial_equity':str(initial),'net_capital_flows':'0','mark_equity':str(equity),
       'mark_net_pnl':str(pnl),'mark_return_percent':str(pnl/initial*100),
       'realized_gross_pnl':str(closed),'unrealized_mark':str(unrealized),'fees':str(fees),'net_funding':str(net_funding),
       'funding_event_count':len(funding),'funding_last_kst':iso(max(f['time'] for f in funding)),
       'post_cutoff_funding_excluded':[f for f in receipts['userFunding']['response'] if end < f['time'] <= end+1000],
       'normal_rounds_completed':sum(ms(r['completed_at_utc'])<=end for r in rounds),
       'trading_days_kst':len({datetime.fromtimestamp(f['time']/1000,KST).date() for f in fills}),
       'trading_days_utc':len({datetime.fromtimestamp(f['time']/1000,timezone.utc).date() for f in fills}),
       'orders':len(orders),'fills':len(fills),'completed_trades':0,'partial_orders':partial,'rejected_orders':0,
       'trade_notional':str(notional),'maker_notional_percent':'0', 'effective_fee_bps':str(fees/notional*10000),
       'positions':positions,'gross_notional':str(gross),'net_notional':str(gross),
       'open_orders':0,'open_orders_basis':'All four submitted orders terminally filled before cutoff; original wallet executor ledger.',
       'complete_24h_intervals':0,'daily_std_percent':None,'oracle_equity':None,'individual_terminal_mark_or_oracle':None,
       'risk':risk,'valuation_note':'Last actual mark-equity log before the cutoff, not an exact 21:00 price; no interpolation. Held constant for the final approximately 28.8 seconds only for time-weighted exposure.',
       'orders_sent_by_report':0,'strategy_changed':False}
    save('metrics.json',m)
    save('audit_manifest.json',{'created_at_utc':datetime.now(timezone.utc).isoformat(),
         'input_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [*(SOURCE/(k+'.json') for k in receipts),OUT/'actual_equity_logs.json',database,risk_module]},
         'calculation_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    print(json.dumps({k:v for k,v in m.items() if k not in ('risk','post_cutoff_funding_excluded')},ensure_ascii=False,indent=2))
    print(json.dumps({k:v for k,v in risk.items() if k!='hourly_returns'},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
