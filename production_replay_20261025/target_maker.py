"""Passive execution of unchanged incumbent targets, not two-sided speculation.

All target generation uses target_from_frame. Only entries rest; reductions use
paid taker attempts. Quote fills are conservative trade-through proxies. L2
acceptance/queue are unresolved, explicitly preventing an exact-fill claim.
"""
import gzip,json,math
from collections import Counter
from concurrent.futures import ProcessPoolExecutor,as_completed
from datetime import datetime,timezone
import numpy as np
import pandas as pd
from competition_live.profile import load_profile
from competition_live.scoring import evaluate
from experiments.shared_execution_pairs_20261017.account_v2 import SharedAccount,difference
from replay_lab.common_comparison.engine import rounded,risk_breaches
from candidate_handoff.portfolio import target_from_frame
from candidate_handoff.producer import validate
from .maker_study import trade_tape,SCENARIOS
from .protocol import ROOT,OUT,DATA,save,sha

DEST=OUT/'target_maker'
SWITCH=1792767600

def run(scenario,ttl,switch):
    from pas_strategy_audit.data import load
    cfg,d=load(DATA);p=load_profile();protocol=p['trading'];symbols=protocol['symbols'];risk=protocol['risk'];sc=SCENARIOS[scenario]
    start,end=cfg['start_ms'],cfg['cutoff_ms'];boundary=SWITCH*1000 if switch else start
    a=SharedAccount(50000.,symbols);ledger=a.physical
    signals=json.loads((OUT/'packets.json').read_text());funding=json.loads((DATA/'funding.json').read_text());tape,_=trade_tape()
    events=[]
    for i,t in enumerate(d['times']):
        if start<=t<=end:events.append((int(t),1,'minute',i))
    for s in signals:
        at=int(s['body']['published_at']*1000)
        if start<=at<end:events.append((at,3,'signal',s))
    for f in funding:
        if start<f['time']<=end:events.append((int(f['time']),0,'funding',f))
    for r in tape:
        if start<r['time']<end:events.append((r['time'],2,'trade',r))
    parent=ROOT/'reports/execution_revision_20261024/carry_allocation/runs'/f'incumbent_continuous120h_{scenario}.json.gz'
    with gzip.open(parent,'rt') as file:baseline=json.load(file)
    if switch:
        for f in baseline['fills']:
            if f['time']*1000<boundary:events.append((int(f['time']*1000),2,'baseline_fill',f))
    i0=np.searchsorted(d['times'],start,side='right')-1;m=np.nan_to_num(d['close'][i0],nan=0.)
    last=np.where(m>0,start-d['age_ms'][i0],0.);sigma=np.eye(10)*risk['hourly_sigma_floor']**2
    queues={j:[] for j in range(10)};snapshots=[];fills=[];decisions=[];stats=Counter();task=None;frame=None;caps=None
    last_decision=-1;peak=50000.;halt=False;breaches=[];adopted=False
    def record(at):
        eq=a.reconcile(m);notional=float(abs(ledger.qty*m).sum())
        return dict(time=at,equity=eq+50000.,notional=notional,fees=ledger.fees,funding=ledger.funding,
            qty=ledger.qty.tolist(),prices=m.tolist(),valuation_age_seconds=float(np.max(np.where(ledger.qty!=0,(at-last)/1000,0))))
    def fill(j,dq,px,fee,at,kind,origin):
        a.physical_fill(j,dq,px,fee,at/1000,decision_time=origin/1000,simulated=True)
        a.owner_fill('base',j,dq,px,fee,at/1000,'conditional_target_execution')
        fills.append(dict(time=at,symbol=symbols[j],signed_qty=dq,price=px,fee_bps=fee,kind=kind,decision_time=origin))
    for at,_,kind,r in sorted(events,key=lambda x:x[:2]):
        if switch and at>=boundary and not adopted:
            task=None;queues={j:[] for j in range(10)};adopted=True
            expected=next(x for x in baseline['snapshots'] if x['time']==boundary)
            # Adoption accounting must preserve the exact baseline inventory
            # and cost state. Mark prices can update on the same millisecond.
            np.testing.assert_allclose(ledger.qty,expected['qty'],atol=1e-12,rtol=0)
            assert abs(ledger.fees-expected['fees'])<1e-7 and abs(ledger.funding-expected['funding'])<1e-7
        if kind=='funding':
            j=symbols.index(r['coin'][4:]);unit=r.get('cash_per_unit')
            if unit is None and ledger.qty[j] and r['rate']:raise ValueError('Missing nonzero funding unit')
            unit=unit or 0.
            if ledger.qty[j]*unit>0:unit*=sc['receipt']
            fr=np.zeros(10)
            if m[j]:fr[j]=-unit/m[j]
            a.charge_funding(at/1000,m,fr,np.ones(10,bool))
        elif kind=='baseline_fill':
            j=r['asset'];fee=r['fee']/abs(r['signed_qty']*r['price'])*10000
            fill(j,r['signed_qty'],r['price'],fee,at,'baseline_prefix',r['decision_time']*1000)
        elif kind=='signal':
            body=validate(r,p,at/1000);frame=body['frame'];sigma=np.asarray(frame['sigma']);eq=a.reconcile(m)
            caps=np.divide(np.abs(frame['anchor_weights'])*eq,m,out=np.zeros(10),where=m>0)
            if at<boundary:continue
            if task:
                mask=np.array(frame['protected']);task['target']=np.where(mask,np.sign(task['target'])*np.minimum(abs(task['target']),caps),task['target'])
            source=frame['source_time']
            if source%21600==0 and source>last_decision and at/1000-source<=900 and np.all(m>0):
                action,trace=target_from_frame(frame,m,ledger.qty,eq,protocol)
                if action:
                    task=dict(target=np.array(action['target_qty']),expires=(source+ttl)*1000,source=source,price=m.copy())
                    last_decision=source;decisions.append(dict(time=at,target=task['target'].tolist(),trace=trace))
        elif kind=='trade':
            j=symbols.index(r['coin'][4:]);m[j]=r['price'];last[j]=at
            if at<boundary:continue
            qs=[q for q in queues[j] if q['ready']<at<q['expiry']]
            if not qs:continue
            q=qs[-1];side=q['side'];delta=difference(q['target'],ledger.qty[j])
            if not delta or np.sign(delta)!=side:continue
            request=min(abs(delta),q['remaining'])
            if q['reducing']:
                request=min(request,abs(ledger.qty[j]));px=r['price']*(1+side*sc['impact']/10000);fee=sc['taker_fee']
            else:
                if r['taker_side']!=('A' if side>0 else 'B') or side*(r['price']-q['price'])>=0:continue
                px=q['price'];fee=sc['maker_fee']
            amount=abs(rounded(min(request,r['qty']*sc['participation']),protocol['execution']['decimals'][j]))
            amount=a.feasible_size(j,side,amount,px,m,sigma,protocol,fee) if amount else 0.
            if amount:
                fill(j,side*amount,px,fee,at,'taker_reduction' if q['reducing'] else 'maker_entry',q['created']);q['remaining']-=amount
        else:
            i=r;valid=np.isfinite(d['close'][i])&(d['close'][i]>0);m[valid]=d['close'][i,valid]
            last=np.maximum(last,np.where(d['active'][i],at-d['age_ms'][i],last))
            eq=a.reconcile(m);peak=max(peak,eq);halt|=eq/peak-1<=-risk['drawdown_halt']
            bad=risk_breaches(ledger.qty,eq,m,sigma,risk)
            if bad:breaches.append(dict(time=at,limits=bad))
            if at==end:snapshots.append(record(at));break
            if at>=boundary:
                desired=ledger.qty.copy();forced=np.zeros(10,bool)
                fresh=frame is not None and 0<=at/1000-frame['source_time']<21600
                if fresh and task is not None:
                    if at<task['expires']:desired=task['target'].copy()
                    else:
                        old=ledger.qty;target=task['target']
                        desired=np.where(old*target>0,np.sign(old)*np.minimum(abs(old),abs(target)),0.)
                if fresh and caps is not None:
                    mask=np.asarray(frame['protected']);desired=np.where(mask,np.sign(desired)*np.minimum(abs(desired),caps),desired)
                if not fresh or halt or bad:desired[:]=0.;forced[:]=True
                for j in range(10):
                    queues[j]=[q for q in queues[j] if q['expiry']>at]
                    if not m[j] or d['age_ms'][i,j]>3600000:continue
                    delta=difference(desired[j],ledger.qty[j]);side=int(np.sign(delta))
                    if not side:continue
                    reducing=ledger.qty[j]*delta<0
                    if not reducing and (task is None or at>=task['expires']):continue
                    price=m[j]*(1-side*.0005)
                    if not reducing:
                        bound=task['price'][j]*(1+side*.001)
                        price=min(price,bound) if side>0 else max(price,bound)
                    request=min(abs(delta),7500/price)
                    if reducing:request=min(request,abs(ledger.qty[j]))
                    request=abs(rounded(request,protocol['execution']['decimals'][j]))
                    if request*price<10 and not (reducing and request==abs(ledger.qty[j])):continue
                    queues[j].append(dict(created=at,ready=at+sc['latency'],expiry=at+60000+sc['latency'],
                        price=price,remaining=request,side=side,target=desired[j],reducing=reducing))
                    stats['taker_reduction' if reducing else 'maker_quote']+=1
            snapshots.append(record(at))
    daily=[dict(time=start//1000,equity=100000.)]+[dict(time=s['time']//1000,equity=s['equity']) for s in snapshots if s['time']>start and s['time']%86400000==43200000]
    score=evaluate(daily,trading_days=len({f['time']//86400000 for f in fills}));curve=np.array([x['equity'] for x in snapshots]);last=snapshots[-1]
    first=next(x for x in snapshots if x['time']==SWITCH*1000)
    row=dict(arm='target_maker_'+str(ttl),scenario=scenario,switch=switch,net_pnl=last['equity']-100000.,**score,
        fees=ledger.fees,funding=ledger.funding,exit_reserve_pnl=last['equity']-100000.-last['notional']*(sc['taker_fee']+sc['impact'])/10000,
        tail_pnl=last['equity']-first['equity'],mean_notional=float(np.mean([s['notional'] for s in snapshots])),
        max_drawdown=float(np.max(1-curve/np.maximum.accumulate(curve))),fills=len(fills),risk_breaches=len(breaches),
        accounting_error=max(a.maximum_error,ledger.max_error),baseline_net_pnl=baseline['metrics']['net_pnl'],
        baseline_tail_pnl=baseline['snapshots'][-1]['equity']-next(x for x in baseline['snapshots'] if x['time']==SWITCH*1000)['equity'],
        conditional=True,actual_orders_sent=0)
    with gzip.open(DEST/f"{row['arm']}_{scenario}_{'switch' if switch else 'full'}.json.gz",'wt') as file:
        json.dump(dict(row=row,snapshots=snapshots,fills=fills,decisions=decisions,stats=dict(stats),breaches=breaches),file,allow_nan=False,default=lambda x:x.tolist())
    return row

def main():
    assert not (DEST/'protocol.json').exists()
    save(DEST/'protocol.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),primary='target_maker_3600',diagnostic='target_maker_900',
        rule='Same incumbent target_from_frame, frozenmodel and protectedcaps. Only new desired exposure posts5bp away from completed trade price, bounded by fixeddecisionmark+/-10bp; cancel/replace60s; new-entry deadline1h(primary)/15min(diagnostic). Risk reductions are paid taker, held inventory tracked. All10symbols. No alpha parameter change.',
        comparison='Common50k/risk; both fullpath and exact baseline prefix sameinventoryswitch Oct23 15UTC. All historical data exposed. Maker fills strict trade-through with aggressor side and1%/.25% volume. Missing L2/queue means this remains conditional research.',
        sources=['https://arxiv.org/html/2605.06405v1'],files={str(__import__('pathlib').Path(__file__).relative_to(ROOT)):sha(__file__)},orders_sent=0))
    rows=[]
    tasks=[(s,ttl,w) for ttl in [3600,900] for s in SCENARIOS for w in [False,True]]
    with ProcessPoolExecutor(max_workers=2) as pool:
        for f in as_completed([pool.submit(run,*x) for x in tasks]):
            row=f.result();rows.append(row);save(DEST/'results.json',rows)
            print({k:row[k] for k in ['arm','scenario','switch','net_pnl','Q','tail_pnl','baseline_tail_pnl','fees','fills','risk_breaches']},flush=True)

if __name__=='__main__':main()
