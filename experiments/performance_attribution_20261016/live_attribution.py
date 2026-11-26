"""Explain observed PnL without inventing counterfactual fills or volatility."""
import json
from collections import defaultdict
from datetime import datetime,timezone
from .collect import ROOT,OUT,PEERS

def read(path):return json.loads(path.read_text())
def save(path,x):path.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False))
def contexts(snapshot):return {u['name']:c for u,c in zip(snapshot['response'][0]['universe'],snapshot['response'][1])}
def signed(f):return float(f['sz'])*(1 if f['side']=='B' else -1)
def valuation(w,ctx,end):
    fs=[f for f in w['fills']['rows'] if f['coin'].startswith('par:') and f['time']<=end]
    qty=defaultdict(float);cash=fees=funding=0.
    for f in fs:qty[f['coin']]+=signed(f);cash-=signed(f)*float(f['px']);fees+=float(f['fee'])
    for e in w['funding']['rows']:
        if e['time']<=end and e['delta'].get('coin','').startswith('par:'):funding+=float(e['delta']['usdc'])
    values={kind:cash-fees+funding+sum(q*float(ctx[c][kind+'Px']) for c,q in qty.items()) for kind in ['oracle','mark']}
    return dict(pnl=values,qty=dict(qty),fees=fees,funding=funding,fills=len(fs),
                gross_oracle=sum(abs(q)*float(ctx[c]['oraclePx']) for c,q in qty.items()))

def main():
    current=read(OUT/'public_snapshot.json');before=read(ROOT/'reports/wallet_leaderboard_20261016/raw/market_now.json')
    start=before['queried_at_ms'];end=current['end_ms'];oldctx=contexts(before);ctx=contexts(current['market'])
    peers=[]
    for w in current['wallets']:
        old=valuation(w,oldctx,start);new=valuation(w,ctx,end)
        atquery={p['position']['coin']:float(p['position']['szi']) for p in w['positions']['response']['assetPositions']}
        # Account query follows common cutoff. Reconcile only if no intervening fill.
        errors={c:atquery.get(c,0)-new['qty'].get(c,0) for c in set(atquery)|set(new['qty'])}
        if any(abs(x)>1e-7 for x in errors.values()):raise ValueError(('Inventory changed across cutoff or incomplete history',w['address'],errors))
        events=[f for f in w['fills']['rows'] if start<f['time']<=end and f['coin'].startswith('par:')]
        notional=sum(float(f['sz'])*float(f['px']) for f in events)
        flows=[e for e in w['ledger']['rows'] if start<e['time']<=end]
        peers.append({'address':w['address'],'prior':old,'current':new,
          'same_period_pnl':new['pnl']['oracle']-old['pnl']['oracle'],
          'same_period_return_percent':100*(new['pnl']['oracle']-old['pnl']['oracle'])/(1e6+old['pnl']['oracle']),
          'same_period_fills':len(events),'same_period_notional':notional,
          'maker_notional_percent':100*sum(float(f['sz'])*float(f['px']) for f in events if not f.get('crossed',True))/notional if notional else None,
          'ledger_events_in_period':len(flows),'reconciliation_qty_max_error':max(map(abs,errors.values()),default=0),
          'actual_daily_std':None,'reason_no_std':'Common interval shorter than one day; no padding before trading.'})
    user=next(w for w in current['wallets'] if w['address']==PEERS[0]);grouped=defaultdict(list)
    for f in user['fills']['rows']:
        if f['coin'].startswith('par:') and f['time']<=end:grouped[f['oid']].append(f)
    batches=[]
    prior_perf=read(ROOT/'reports/practice_performance_20261016_1530/metrics.json')
    cutoff_prices={p['coin']:float(p['oracle']) for p in prior_perf['positions']}
    for oid,fs in sorted(grouped.items(),key=lambda x:min(f['time'] for f in x[1])):
        if not all(f['dir']=='Open Long' for f in fs):raise ValueError('New lifecycle needs generalized batch attribution')
        coin=fs[0]['coin'];q=sum(float(f['sz']) for f in fs);cost=sum(float(f['sz'])*float(f['px']) for f in fs)
        fee=sum(float(f['fee']) for f in fs);funding=0.
        for e in user['funding']['rows']:
            d=e['delta']
            if d.get('coin')==coin and min(f['time'] for f in fs)<e['time']<=end:
                funded=float(d['szi']);funding+=float(d['usdc'])*q/funded
        batches.append({'coin':coin,'oid':oid,'time':min(f['time'] for f in fs),'qty':q,'vwap':cost/q,'notional':cost,
           'fees':fee,'funding':funding,'oracle_price_pnl':q*float(ctx[coin]['oraclePx'])-cost,
           'oracle_net_pnl':q*float(ctx[coin]['oraclePx'])-cost-fee+funding,
           'oracle_price_pnl_at_1530':q*cutoff_prices[coin]-cost})
    account=next(p for p in peers if p['address']==PEERS[0])['current']
    assert abs(sum(b['oracle_net_pnl'] for b in batches)-account['pnl']['oracle'])<1e-6
    oldvol=read(ROOT/'reports/wallet_volatility_20261016/analysis.json')
    vol_notes=[{k:w[k] for k in ('wallet_id','address','daily_std_sample_percent','observed_days','exposure_days','fills_in_period','common_period_return_percent')}
        for w in oldvol['wallets'] if w['address'] in PEERS]
    result={'same_period_start_utc':datetime.fromtimestamp(start/1000,timezone.utc).isoformat(),
            'same_period_end_utc':datetime.fromtimestamp(end/1000,timezone.utc).isoformat(),
            'oracle_context_received_utc':current['market']['received_at_utc'],
            'snapshot_timing_note':'Oracle contexts are receipt-time snapshots, not a synchronized historical oracle series. Account queries follow cutoff; inventories reconciled.',
            'peers':peers,'user_batches':batches,'prior_volatility_method_caveats':vol_notes,
            'prior_volatility_caveat':'Prior 3-day table included pre-trading/flat days and lagged portfolio boundaries; user exposure only 1 of 3 days. Do not present it as user active-trading daily volatility.',
            'counterfactual_fills_assumed':False,'live_orders_sent':0}
    save(OUT/'live_attribution.json',result)
    for p in peers:print(p['address'],round(p['same_period_pnl'],6),'interval fills',p['same_period_fills'],'gross',round(p['current']['gross_oracle'],2),flush=True)
    print('user batches',json.dumps(batches,ensure_ascii=False))

if __name__=='__main__':main()
