"""Observed PAS transactions/settlements, not reconstructed historical books."""
import json
from collections import defaultdict
import numpy as np
from .prepare import ROOT,OUT,START,END
from pas_adaptive_development.study import save,sha


def prepare_tape():
    folder=ROOT/'reports/team_comparison_20261021/raw/wallets'
    trades={};funding=defaultdict(list);files=[];wallets=0;incomplete=[]
    for file in sorted(folder.glob('*.json')):
        r=json.loads(file.read_text());files.append(file);wallets+=1
        if r.get('history_end_ms',0)<END*1000:continue
        if not r.get('fills',{}).get('pagination_complete'):incomplete.append(file.name)
        for f in r.get('fills',{}).get('rows',[]):
            if not f['coin'].startswith('pas:') or not START*1000<=f['time']<=END*1000:continue
            key=f['coin'],f['tid'];row=dict(time=f['time'],coin=f['coin'],tid=f['tid'],price=float(f['px']),qty=float(f['sz']))
            if key in trades:
                if trades[key]!=row:raise ValueError('Conflicting two-sided trade identity')
            else:trades[key]=row
        for f in r.get('funding',{}).get('rows',[]):
            d=f['delta'];coin=d.get('coin','')
            if not coin.startswith('pas:') or not START*1000<f['time']<=END*1000:continue
            qty=float(d['szi']);cash=float(d['usdc'])
            if qty:funding[(coin,f['time'])].append(dict(qty=qty,cash=cash,unit=cash/qty,rate=float(d['fundingRate'])))
    rows=sorted(trades.values(),key=lambda r:r['time'])
    settled=[]
    for (coin,t),vv in sorted(funding.items(),key=lambda r:r[0][1]):
        # Largest observed position minimizes the per-unit cash-rounding bound.
        v=max(vv,key=lambda v:abs(v['qty']))
        settled.append(dict(time=t,coin=coin,cash_per_unit=v['unit'],rate=v['rate'],
            qty_observed=v['qty'],cash_observed=v['cash'],rounding_unit_bound=1e-6/abs(v['qty']),
            source='actual wallet funding payment / settled signed quantity'))
    # Same-time transaction IDs have no ordering semantics. A conservative
    # side-dependent worst price is used for the combined timestamp batch.
    groups=defaultdict(list)
    for r in rows:groups[(r['time'],r['coin'])].append(r)
    batches=[dict(time=t,coin=c,qty=sum(r['qty'] for r in v),buy_price=max(r['price'] for r in v),
        sell_price=min(r['price'] for r in v),vwap=sum(r['qty']*r['price'] for r in v)/sum(r['qty'] for r in v))
        for (t,c),v in groups.items()]
    batches.sort(key=lambda r:r['time'])
    candle_dir=ROOT/'reports/pas_allocation_study_20261021/pas'
    symbols=json.loads((candle_dir/'dataset.json').read_text())['symbols']
    coverage=[]
    for s in symbols:
        bars=json.loads((candle_dir/'raw'/f'pas_candles_{s}.json').read_text())['data']
        minutevol={b['t']:float(b['v']) for b in bars if START*1000<=b['t']<END*1000}
        observed=defaultdict(float)
        for r in rows:
            if r['coin']=='pas:'+s:observed[r['time']//60000*60000]+=r['qty']
        total=sum(minutevol.values());got=sum(observed.values())
        coverage.append(dict(symbol=s,candle_volume=total,observed_volume=got,
            fraction=got/total if total else None,missing_active_minutes=sum(v>0 and observed.get(t,0)<v-1e-8 for t,v in minutevol.items()),
            funding_events=sum(r['coin']=='pas:'+s for r in settled)))
    save(OUT/'trade_tape.json',batches);save(OUT/'funding_tape.json',settled)
    save(OUT/'tape_coverage.json',dict(wallets=wallets,unique_trades=len(rows),batches=len(batches),
        incomplete_wallet_histories=incomplete,assets=coverage))
    save(OUT/'tape_sources.json',{str(f.relative_to(ROOT)):sha(f) for f in files})
    spec=dict(created_before_replay=True,primary='base_1pct',
        prices='Contemporaneous observed trade batch worst-side price plus cost scenario. Trade price is NOT a resting quote.',
        capacity='Fixed share of subsequent observed transaction batch; same conditions for both strategies. Missing/unobserved trades give no fill.',
        order_model='Conditional marketable IOC proxy, refreshed at each observed trade while an intent remains valid for15min. Bounded5% from signal-time trade-price proxy. This deliberately estimates executable-market uncertainty, not historical L2 queue fills.',
        scenarios=['base_1pct','stress_025pct','base_10pct','optimistic_100pct'],
        timing='Signals processed at original publication time+5s. Actual trade timestamps strictly after decisions. No input or limit derived from future transactions.',
        pair_scope='All96 pair gate actions from48 sources are zero; no new pair can open in either arm. Thus base ledger is the complete intended strategy for this specific interval.',
        funding='Use actual observed wallet funding per signed unit. Unobserved market-hour falls back to prior trade-close times actual fundingRate and is counted explicitly.',
        score='12UTC daily samples, trade-price valuation proxy. Teammate50000 fixed cash in both arms; actual teammate ETH trades are not attributed to our model.',
        selection='No candidate/threshold/horizon changes after PAS outcome. No borrowing risk capital or calling results untouched holdout.',
        orders_sent=0)
    save(OUT/'tape_protocol.json',spec)
    print(json.dumps(dict(wallets=wallets,trades=len(rows),batches=len(batches),coverage=coverage)))


if __name__=='__main__':prepare_tape()
