"""Inventory/funding-aware passive quotes on observed PAS aggressor prints.

Conditional development screen, NOT reconstructed post-only acceptance/queue.
Quotes strictly trade through after latency; no touch fills and no rebates.
"""
import bisect,gzip,json,math
from collections import Counter
from concurrent.futures import ProcessPoolExecutor,as_completed
from datetime import datetime,timezone
import numpy as np
import pandas as pd
from experiments.shared_execution_pairs_20261017.account_v2 import SharedAccount
from replay_lab.common_comparison.engine import rounded,risk_breaches
from competition_live.profile import load_profile
from competition_live.scoring import evaluate
from pas_strategy_audit.data import load
from .protocol import ROOT,OUT,DATA,save,sha

DEST=OUT/'passive_inventory'
SCENARIOS=dict(base=dict(maker_fee=3.,taker_fee=9.,impact=10.,participation=.01,latency=1000,receipt=1.),
    stress=dict(maker_fee=6.,taker_fee=12.,impact=20.,participation=.0025,latency=3000,receipt=.5),
    no_receipts=dict(maker_fee=6.,taker_fee=12.,impact=20.,participation=.0025,latency=3000,receipt=0.))

def trade_tape():
    result={};records=0
    for file in (ROOT/'reports/execution_revision_20261024/peer_consensus/raw').glob('*.json'):
        for r in json.loads(file.read_text())['rows']:
            if not r['coin'].startswith('pas:'):continue
            records+=1;key=(r['coin'],r['time'],r['tid'])
            item=dict(time=r['time'],coin=r['coin'],tid=r['tid'],price=float(r['px']),qty=float(r['sz']),
                taker_side=(r['side'] if r['crossed'] else ('A' if r['side']=='B' else 'B')))
            if key in result and result[key]!=item:raise ValueError('Conflicting public trade identity')
            result[key]=item
    return sorted(result.values(),key=lambda r:(r['time'],r['coin'],r['tid'])),records

def simulate(scenario,funding_skew=True):
    sc=SCENARIOS[scenario];cfg,d=load(DATA);profile=load_profile();p=profile['trading'];symbols=p['symbols'];risk=p['risk']
    start,end=cfg['start_ms'],cfg['cutoff_ms'];account=SharedAccount(50000.,symbols);ledger=account.physical
    funding=json.loads((DATA/'funding.json').read_text());signals=json.loads((DATA/'events.json').read_text());tape,_=trade_tape()
    # Past-completed return volatility; no bfill, no whole-test standardization.
    vol=pd.DataFrame(np.log(d['close'])).diff().ewm(span=1440,min_periods=60,adjust=False).std().to_numpy()
    events=[]
    for i,t in enumerate(d['times']):
        if start<=t<=end:events.append((int(t),2,'minute',i))
    for r in signals:
        if start<=r['time']<end:events.append((int(r['time']),3,'signal',r))
    for r in funding:
        if start<r['time']<=end:
            events.append((r['time'],0,'funding',r));events.append((r['time']+60000,4,'rate_known',r))
    for r in tape:
        if start<r['time']<=end:events.append((r['time'],1,'trade',r))
    m=np.zeros(10);last=np.zeros(10);rates=np.zeros(10);sigma=np.eye(10)*risk['hourly_sigma_floor']**2
    queues={j:[] for j in range(10)};snapshots=[];fills=[];stats=Counter();peak=50000.;halted=False;breaches=[];last_minute=None
    initial=np.searchsorted(d['times'],start,side='right')-1
    m=np.nan_to_num(d['close'][initial],nan=0.);last=np.where(m>0,start-d['age_ms'][initial],0.)
    def value(at):
        eq=account.reconcile(m);notional=float(abs(ledger.qty*m).sum())
        return dict(time=at,equity=eq+50000.,own_equity=eq,qty=ledger.qty.tolist(),prices=m.tolist(),
            fees=ledger.fees,funding=ledger.funding,notional=notional,valuation_age_seconds=float(np.max(np.where(ledger.qty!=0,(at-last)/1000,0))))
    def fill(j,side,amount,px,fee,at,kind,origin):
        account.physical_fill(j,side*amount,px,fee,at/1000,decision_time=origin/1000,simulated=True)
        account.owner_fill('base',j,side*amount,px,fee,at/1000,'conditional_passive_fill')
        fills.append(dict(time=at,asset=symbols[j],side=side,qty=amount,price=px,fee_bps=fee,kind=kind,decision=origin))
    for at,priority,kind,r in sorted(events,key=lambda x:x[:2]):
        if kind=='funding':
            j=symbols.index(r['coin'][4:]);unit=r.get('cash_per_unit')
            if unit is None and ledger.qty[j] and r['rate']:raise ValueError('Nonzero funding cannot be proxied')
            unit=unit or 0.
            if ledger.qty[j]*unit>0:unit*=sc['receipt']
            fr=np.zeros(10)
            if m[j]:fr[j]=-unit/m[j]
            account.charge_funding(at/1000,m,fr,np.ones(10,bool))
        elif kind=='rate_known':
            j=symbols.index(r['coin'][4:]);rates[j]=.8*rates[j]+.2*r['rate']
        elif kind=='signal':sigma=np.asarray(r['sigma'])
        elif kind=='trade':
            j=symbols.index(r['coin'][4:]);m[j]=r['price'];last[j]=at
            eligible=[q for q in queues[j] if q['ready']<at<q['expiry']]
            if not eligible:continue
            q=eligible[-1];side=1 if r['taker_side']=='A' else -1;key='bid' if side>0 else 'ask'
            # A resting maker bid needs a sell aggressor strictly below bid;
            # ask needs buy aggressor strictly above ask. Touch is never filled.
            px=q[key];remaining=q[key+'_remaining']
            if remaining<=0 or side*(r['price']-px)>=0:continue
            stats['trade_through']+=1
            amount=abs(rounded(min(remaining,r['qty']*sc['participation']),p['execution']['decimals'][j]))
            amount=account.feasible_size(j,side,amount,px,m,sigma,p,sc['maker_fee']) if amount else 0.
            if not amount:continue
            fill(j,side,amount,px,sc['maker_fee'],at,'maker_proxy',q['created']);q[key+'_remaining']-=amount
        else:
            i=r;known=np.isfinite(d['close'][i])&(d['close'][i]>0);m[known]=d['close'][i,known]
            last=np.maximum(last,np.where(d['active'][i],at-d['age_ms'][i],last))
            eq=account.reconcile(m);peak=max(peak,eq);halted|=eq/peak-1<=-risk['drawdown_halt']
            bad=risk_breaches(ledger.qty,eq,m,sigma,risk)
            if bad:breaches.append(dict(time=at,limits=bad))
            if at==end:snapshots.append(value(at));break
            # Conservative common inventory bound also protects gross/net if
            # all markets accumulate the same sign. Absolute covariance bounds
            # every signed portfolio in this box, not just an equal-risk guess.
            cap=min(risk['asset_fraction'],risk['asset_notional']/eq,risk['net_fraction']/10,risk['gross_fraction']/10)*risk['sizing_headroom']
            maxvar=float(np.full(10,cap)@abs(sigma)@np.full(10,cap))*24
            if maxvar>risk['daily_volatility']**2:cap*=risk['daily_volatility']/math.sqrt(maxvar)
            for j in range(10):
                queues[j]=[q for q in queues[j] if q['expiry']>at]
                if not m[j] or d['age_ms'][i,j]>3600000 or not np.isfinite(vol[i,j]):continue
                weight=ledger.qty[j]*m[j]/eq
                # Fee floor is fixed from nominal3bp+2bp edge, NOT optimized
                # independently to make stress easier. Stress charges6bp.
                half=max(.0005,float(vol[i,j])*.5)
                skew=-5.*weight*max(sigma[j,j],1e-12)*24
                if funding_skew:skew-=rates[j]*6*.5
                bid=m[j]*(1+min(skew-half,-.0005));ask=m[j]*(1+max(skew+half,.0005))
                step=min(cap*eq/3,7500.)
                buy=min(step,max(0.,cap*eq-ledger.qty[j]*m[j]))/bid
                sell=min(step,max(0.,cap*eq+ledger.qty[j]*m[j]))/ask
                buy=abs(rounded(buy,p['execution']['decimals'][j]));sell=abs(rounded(sell,p['execution']['decimals'][j]))
                if buy*bid<10:buy=0.
                if sell*ask<10:sell=0.
                if halted or bad:
                    # No new maker risk when the portfolio is over limits.
                    buy=sell=0.;stats['risk_disabled_quotes']+=1
                queues[j].append(dict(created=at,ready=at+sc['latency'],expiry=at+60000+sc['latency'],bid=bid,ask=ask,
                    bid_remaining=buy,ask_remaining=sell))
                stats['quote_pairs']+=1
            snapshots.append(value(at))
    daily=[dict(time=start//1000,equity=100000.)]+[dict(time=s['time']//1000,equity=s['equity']) for s in snapshots if s['time']>start and s['time']%86400000==43200000]
    score=evaluate(daily,trading_days=len({r['time']//86400000 for r in fills}))
    curve=np.array([r['equity'] for r in snapshots]);final=snapshots[-1];reserve=final['notional']*(sc['taker_fee']+sc['impact'])/10000
    row=dict(arm='funding_inventory_maker' if funding_skew else 'inventory_maker_control',scenario=scenario,
        net_pnl=final['equity']-100000.,**score,fees=ledger.fees,funding=ledger.funding,
        price_pnl=final['equity']-100000.+ledger.fees-ledger.funding,exit_reserve_pnl=final['equity']-100000.-reserve,
        mean_notional=float(np.mean([s['notional'] for s in snapshots])),end_notional=final['notional'],
        max_drawdown=float(np.max(1-curve/np.maximum.accumulate(curve))),fills=len(fills),risk_breaches=len(breaches),
        accounting_error=max(account.maximum_error,ledger.max_error),conditional=True,orders_sent=0)
    r=dict(row=row,snapshots=snapshots,fills=fills,stats=dict(stats),breaches=breaches,daily=daily)
    with gzip.open(DEST/f"{row['arm']}_{scenario}.json.gz",'wt') as f:json.dump(r,f,allow_nan=False,default=lambda x:x.tolist())
    return row

def main():
    assert not (DEST/'protocol.json').exists()
    tape,records=trade_tape()
    save(DEST/'protocol.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),primary='funding_inventory_maker',
        control='inventory_maker_control',source_records=records,unique_observed_trades=len(tape),
        rule='All10markets; quote refresh60s; halfspread=max(nominal3bpfee+2bpbuffer,0.5past-minute-vol). Inventory skew=-5*weight*hourlyvariance*24. Funding skew=-0.5*pastEWMAhourlyfunding*6h. No fitting on full-windowreturns. No parametergrid.',
        execution='Strict future trade-through, matching aggressor side,1s/3s latency,1%/.25% of observedtradequantity, quoted limit fill price. Dedup matched maker/taker records. No fill on touch. This lacks historical L2 for post-only acceptance,queue and counterfactual flow: screen not executable proof.',
        risk='Unchanged50k/globalcaps; symmetric10-asset inventory box bounded by bothnet/gross and absolute covariance. Riskbreach disqualifies run; no new quotes afterbreach.',
        sources=['https://arxiv.org/html/2605.06405v1','https://math.nyu.edu/inmemoriam/avellaneda/HighFrequencyTrading.pdf'],
        provenance='Inventory/funding state inspired by these papers; this heuristic does NOT reproduce their HJB or claim their returns. PASmakerfee3bp, not paper1.5bp. All exposed-history development.',
        files={str(__import__('pathlib').Path(__file__).relative_to(ROOT)):sha(__file__)},orders_sent=0))
    rows=[]
    with ProcessPoolExecutor(max_workers=2) as pool:
        for f in as_completed([pool.submit(simulate,s,skew) for s in SCENARIOS for skew in [True,False]]):
            row=f.result();rows.append(row);save(DEST/'results.json',rows)
            print({k:row[k] for k in ['arm','scenario','net_pnl','Q','price_pnl','fees','funding','fills','risk_breaches']},flush=True)

if __name__=='__main__':main()
