"""Executable-entry/exit opportunity probes on allowlisted old public tapes.

This is NOT a capital-scaled account backtest. Each independent arm trades at
most one position per asset. Missing exit observations remain unresolved.
"""
import bisect,gzip,json,time
from collections import Counter
import numpy as np
from .collect import HERE,ROOT,OUT,save,sha

HOUR=3600000

def walk(levels,quantity,fraction=1.):
    remaining=quantity;value=0.
    for row in levels:
        q=min(remaining,float(row['sz'])*fraction);value+=q*float(row['px']);remaining-=q
        if remaining<=1e-10:return value/quantity
    return None

def sweep(levels,quantity,fraction,side,decimals,limit=None):
    remaining=quantity;value=0.;filled=0.;factor=10**decimals
    for row in levels:
        px=float(row['px'])
        if limit is not None and side*(px-limit)>1e-10:break
        available=np.floor(float(row['sz'])*fraction*factor+1e-9)/factor
        q=min(remaining,available);value+=q*px;filled+=q;remaining-=q
        if remaining<=1e-10:break
    return float(filled),float(value/filled) if filled>0 else None

def funding_events_between(entry,exit):
    return list(range((entry//HOUR+1)*HOUR,(exit//HOUR)*HOUR+1,HOUR))

def expected_net(side,entry,oracle,mid,halfspread,rate,event_count,fee_bps,reversion,haircut):
    expected_mid=oracle+(1-reversion)*(mid-oracle)
    expected_exit=expected_mid-side*halfspread
    price=side*(expected_exit-entry)
    carry=-side*oracle*rate*event_count
    if carry>0:carry*=haircut
    fees=fee_bps/10000*(entry+expected_exit)
    return (price+carry-fees)/entry*10000

def load_tape(path):
    stream={};first=10**20;last=0;epoch=0;counts=Counter();resets=[];asset_resets={}
    with (gzip.open(path,'rt') if str(path).endswith('.gz') else open(path)) as f:
        for line in f:
            e=json.loads(line);m=e.get('message',e)
            if e.get('venue','par')!='par':continue
            t=e['received_ms'];first=min(first,t);last=max(last,t);channel=m.get('channel');counts[channel]+=1
            if channel in ('connected','disconnected'):epoch+=1;resets.append(t)
            d=m.get('data',{});coin=d.get('coin') if isinstance(d,dict) else None
            if channel=='subscriptionResponse':
                sub=d.get('subscription',{})
                if sub.get('type')=='l2Book':asset_resets.setdefault(sub['coin'].split(':')[-1],[]).append(t)
            if channel not in ('l2Book','activeAssetCtx') or not coin:continue
            symbol=coin.split(':')[-1];s=stream.setdefault(symbol,dict(book=[],ctx=[]))
            field='book' if channel=='l2Book' else 'ctx';s[field].append((t,epoch,d if field=='book' else d['ctx']))
    for symbol,s in stream.items():
        s['resets']=sorted(resets+asset_resets.get(symbol,[]))
        for field in ('book','ctx'):s[field].sort(key=lambda e:e[0]);s[field+'_times']=[r[0] for r in s[field]]
    return stream,dict(path=str(path.relative_to(ROOT)),start_ms=first,end_ms=last,hours=(last-first)/HOUR,counts=dict(counts))

def asof(stream,t,config):
    values={};epochs=[];received={}
    for field,age in [('book',config['book_max_age_ms']),('ctx',config['context_max_age_ms'])]:
        i=bisect.bisect_right(stream[field+'_times'],t)-1
        if i<0:return None
        at,epoch,value=stream[field][i]
        if t-at>age:return None
        values[field]=value;epochs.append(epoch);received[field]=at
    if epochs[0]!=epochs[1]:return None
    r=bisect.bisect_right(stream.get('resets',[]),t)-1
    if r>=0 and any(at<stream['resets'][r] for at in received.values()):return None
    # Received timestamp alone is insufficient: reject a stale server book.
    if t-values['book']['time']>config['book_max_age_ms'] or values['book']['time']>t+1000:return None
    if not all(values['book']['levels']):return None
    bid=float(values['book']['levels'][0][0]['px']);ask=float(values['book']['levels'][1][0]['px'])
    if bid<=0 or ask<bid:return None
    values.update(mid=(bid+ask)/2,halfspread=(ask-bid)/2,oracle=float(values['ctx']['oraclePx']),received=received)
    return values

def delayed_snapshot(stream,earliest,config):
    # Use the first newly RECEIVED book after the order's latency. Never fill
    # against the pre-order stale snapshot even when it lies within the age cap.
    i=bisect.bisect_left(stream['book_times'],earliest)
    while i<len(stream['book_times']) and stream['book_times'][i]<=earliest+config['future_exit_observation_grace_ms']:
        t=stream['book_times'][i];v=asof(stream,t,config)
        if v is not None:return t,v
        i+=1
    return None

def decompose(side,qty,entry,exit,oracle_entry,oracle_exit,funding,fee_bps):
    direction=side*qty*(oracle_exit-oracle_entry)
    basis=side*qty*((exit-oracle_exit)-(entry-oracle_entry))
    fees=qty*(entry+exit)*fee_bps/10000
    net=side*qty*(exit-entry)+funding-fees
    assert abs(net-(direction+basis+funding-fees))<1e-8
    return dict(oracle_direction_pnl=direction,basis_pnl=basis,funding=funding,fees=fees,net_pnl=net)

def run_arm(stream,cfg,scenario,arm,symbol,decimals,funding):
    starts=stream['book_times'];lo=starts[0];hi=starts[-1];horizon=cfg['horizon_minutes']*60000
    step=cfg['decision_interval_seconds']*1000;next_free=0;trials=[];counts=Counter()
    for t in range(((lo+step-1)//step)*step,hi+1,step):
        if t<next_free:continue
        now=asof(stream,t,cfg)
        if now is None:counts['invalid_decision_book_or_context']+=1;continue
        counts['valid_decisions']+=1
        quantity=np.floor(cfg['target_notional']/now['oracle']*10**decimals)/10**decimals
        if quantity<=0:continue
        candidates=[]
        count=len(funding_events_between(t+scenario['latency_ms'],t+scenario['latency_ms']+horizon))
        for side in (1,-1):
            preview=walk(now['book']['levels'][1 if side==1 else 0],quantity,scenario['depth_fraction'])
            if preview is None:continue
            fraction=1. if arm=='full_convergence_diagnostic' else cfg['basis_reversion_fraction']
            rate=0. if arm=='basis_only' else float(now['ctx']['funding'])
            score=expected_net(side,preview,now['oracle'],now['mid'],now['halfspread'],rate,count,scenario['fee_bps'],fraction,cfg['positive_funding_forecast_haircut'])
            candidates.append((score,side))
        if not candidates:counts['no_visible_entry_depth']+=1;continue
        score,side=max(candidates)
        if score<cfg['minimum_forecast_net_bps']:counts['predicted_edge_below_cost_buffer']+=1;continue
        counts['entry_signals']+=1
        row=dict(decision_ms=t,symbol=symbol,side=side,qty=float(quantity),predicted_net_bps=score,arm=arm)
        # The fixed tape end is an observation boundary, not an oracle exit.
        if t+scenario['latency_ms']+horizon>hi:
            row['status']='insufficient_future_observation';trials.append(row);next_free=t+horizon;continue
        event=delayed_snapshot(stream,t+scenario['latency_ms'],cfg)
        if event is None:row['status']='entry_unobserved';trials.append(row);continue
        entered,eb=event
        # A bounded IOC limit is sealed from the decision book. No chasing far
        # into the book following a delayed price jump.
        preview=walk(now['book']['levels'][1 if side==1 else 0],quantity,scenario['depth_fraction'])
        limit=preview*(1+side*.001)
        filled,entry=sweep(eb['book']['levels'][1 if side==1 else 0],quantity,scenario['depth_fraction'],side,decimals,limit)
        if not filled:row['status']='entry_no_fill_within_limit';trials.append(row);continue
        row['requested_qty']=float(quantity);row['qty']=filled;row['entry_partial']=bool(filled<quantity-1e-10);quantity=filled
        next_free=entered+horizon
        row.update(entry_ms=entered,entry_price=entry,entry_oracle=eb['oracle'])
        event=delayed_snapshot(stream,entered+horizon,cfg)
        if event is None:row['status']='unresolved_exit_observation';trials.append(row);continue
        exited,xb=event;closed,exit=sweep(xb['book']['levels'][0 if side==1 else 1],quantity,scenario['depth_fraction'],-side,decimals)
        if closed<quantity-1e-10:
            row.update(status='unresolved_exit_depth',closed_qty=closed,remaining_qty=quantity-closed,partial_exit_price=exit);trials.append(row);continue
        cash=0.;valid=True;settlements=[]
        for bucket in funding_events_between(entered,exited):
            fr=funding.get(bucket)
            if fr is None:valid=False;break
            settled,rate=fr;context=asof(stream,settled,cfg)
            if context is None:valid=False;break
            amount=-side*quantity*context['oracle']*rate;cash+=amount
            settlements.append(dict(time=settled,rate=rate,oracle=context['oracle'],amount=amount))
        if not valid:row['status']='unresolved_funding_oracle';trials.append(row);continue
        row.update(status='completed_probe',exit_ms=exited,exit_price=exit,exit_oracle=xb['oracle'],funding_settlements=settlements,
                   **decompose(side,quantity,entry,exit,eb['oracle'],xb['oracle'],cash,scenario['fee_bps']))
        row['net_bps']=row['net_pnl']/(quantity*entry)*10000
        trials.append(row)
    counts.update(r['status'] for r in trials)
    return trials,dict(counts)

def main():
    dest=OUT/'quote_probe';dest.mkdir(exist_ok=True)
    if (dest/'frozen.json').exists():raise FileExistsError('Quote experiment already frozen')
    cfg=json.loads((HERE/'protocol.json').read_text());meta=json.loads((OUT/'raw/metadata.json').read_text())['response'][0]
    decimals={r['name'].split(':')[-1]:r['szDecimals'] for r in meta['universe']}
    save(dest/'frozen.json',dict(at_ms=int(time.time()*1000),source_sha256=sha(__file__),protocol_sha256=sha(HERE/'protocol.json'),
         inputs={p:sha(ROOT/p) for p in cfg['tapes']},sealed_streams_read=False,orders_sent=0))
    rates={}
    for symbol in cfg['symbols']:
        rows=json.loads((OUT/'raw'/f'funding_{symbol}.json').read_text())['rows']
        rates[symbol]={r['time']//HOUR*HOUR:(r['time'],float(r['fundingRate'])) for r in rows}
    trials=[];coverage=[];summaries=[]
    for path in cfg['tapes']:
        stream,metadata=load_tape(ROOT/path);coverage.append(metadata)
        for symbol,s in stream.items():
            if not s['book'] or not s['ctx']:continue
            for scenario,sc in cfg['scenarios'].items():
                for arm in cfg['arms']:
                    tt,cc=run_arm(s,cfg,sc,arm,symbol,decimals[symbol],rates[symbol])
                    for row in tt:row.update(tape=path,scenario=scenario)
                    trials.extend(tt);summaries.append(dict(tape=path,symbol=symbol,scenario=scenario,arm=arm,counts=cc))
    aggregates=[]
    for scenario in cfg['scenarios']:
        for arm in cfg['arms']:
            selected=[r for r in trials if r['scenario']==scenario and r['arm']==arm];complete=[r for r in selected if r['status']=='completed_probe']
            aggregates.append(dict(scenario=scenario,arm=arm,signals=len(selected),statuses=dict(Counter(r['status'] for r in selected)),
                completed=len(complete),mean_completed_net_bps=float(np.mean([r['net_bps'] for r in complete])) if complete else None,
                totals={k:float(sum(r[k] for r in complete)) for k in ('net_pnl','oracle_direction_pnl','basis_pnl','funding','fees')},
                positive_completed=sum(r['net_pnl']>0 for r in complete),missing_paths_are_not_zero_profit=True))
    save(dest/'results.json',dict(coverage=coverage,trials=trials,summaries=summaries,aggregates=aggregates,
         orders_sent=0,independent=False,account_return_or_daily_std_estimated=False,
         limitations='Displayed-depth entry and timed exit probes. Missing future or exit observations remain unresolved; 6-24h performance is not tested.'))
    print(json.dumps(aggregates,indent=2),flush=True)

if __name__=='__main__':main()
