"""Conditional paired entry and paid exits on the observed old quote tapes."""
import json
from collections import Counter
import time
import numpy as np
from experiments.basis_carry_20261017.probe import load_tape, asof, delayed_snapshot, walk
from experiments.cost_horizon_20261016.research import data
from experiments.execution_retry_20261016.engine import lot_floor
from experiments.pairs_research_20261017.research import save,sha
from .probe import HERE, ROOT, HISTORY, settled_funding


def evaluate(fill, pair, beta, streams, decimals, rates, horizon, cfg, execution):
    lead=fill['coin'].split(':')[-1]; other=pair[1] if pair[0]==lead else pair[0]
    row=dict(pair=pair,lead=lead,hedge=other,beta=float(beta),horizon=horizon,
             lead_fill=fill,status=None,unhedged_notional=fill['qty']*fill['price'])
    lead_now=asof(streams[lead],fill['time'],cfg)
    hedge_now=asof(streams[other],fill['time'],cfg)
    if lead_now is None or hedge_now is None:row['status']='missing_hedge_decision';return row
    multiplier=beta if pair[0]==lead else 1/beta
    q=lot_floor(fill['qty']*lead_now['oracle']*multiplier/hedge_now['oracle'],decimals[other])
    side=-fill['side']
    hedge=delayed_snapshot(streams[other],fill['time']+execution['latency_ms'],cfg)
    if hedge is None:row['status']='missing_hedge_arrival';return row
    hedge_time,hedge_snap=hedge
    hedge_price=walk(hedge_snap['book']['levels'][1 if side>0 else 0],q,1.) if q else None
    if hedge_price is None:row['status']='insufficient_hedge_depth';return row
    if q*hedge_price<execution['minimum_order_notional']:row['status']='hedge_below_minimum';return row
    positions=[dict(symbol=lead,qty=fill['qty']*fill['side'],entry_time=fill['time'],entry=fill['price'],fee=execution['maker_fee_bps']),
               dict(symbol=other,qty=q*side,entry_time=hedge_time,entry=hedge_price,fee=execution['taker_fee_bps'])]
    row['positions']=positions; row['unhedged_notional']=0.
    for position in positions:
        symbol=position['symbol']; quantity=abs(position['qty']); signed=np.sign(position['qty'])
        exit=delayed_snapshot(streams[symbol],hedge_time+horizon*1000+execution['latency_ms'],cfg)
        if exit is None:row['status']='missing_exit';return row
        at,snapshot=exit
        price=walk(snapshot['book']['levels'][0 if signed>0 else 1],quantity,1.)
        if price is None:row['status']='insufficient_exit_depth';return row
        if price*quantity<execution['minimum_order_notional']:row['status']='exit_below_minimum';return row
        funding=0.
        for hour in range((position['entry_time']//3600000+1)*3600000,at+1,3600000):
            context=asof(streams[symbol],hour,cfg)
            if context is None or hour not in rates[symbol]:row['status']='missing_funding_or_oracle';return row
            funding-=position['qty']*context['oracle']*rates[symbol][hour]
        position.update(exit_time=at,exit=price,funding=funding,
                        fees=quantity*(position['entry']*position['fee']+price*execution['taker_fee_bps'])/10000)
    fees=sum(p['fees'] for p in positions);funding=sum(p['funding'] for p in positions)
    gross=sum(p['qty']*(p['exit']-p['entry']) for p in positions)
    notional=sum(abs(p['qty'])*p['entry'] for p in positions)
    row.update(status='closed',gross_price_pnl=gross,fees=fees,funding=funding,
               net_pnl=gross+funding-fees,net_bps=(gross+funding-fees)/notional*10000,notional=notional)
    return row


def main():
    out=HERE/'paired_artifacts';out.mkdir(exist_ok=False)
    cfg=json.loads((HERE/'protocol.json').read_text());execution=json.loads((HERE/'paired_protocol.json').read_text())
    prior=json.loads((HERE/'artifacts/results.json').read_text())
    save(out/'frozen.json',dict(at=time.time(),sources={str(p.relative_to(ROOT)):sha(p) for p in [HERE/'paired.py',HERE/'paired_protocol.json',HERE/'artifacts/results.json']},historical_reuse=True,orders_sent=0))
    metadata=json.loads((HISTORY/'metadata.json').read_text())['response'][0]['universe']
    decimals={r['name'].split(':')[-1]:r['szDecimals'] for r in metadata}
    _,_,_,market=data()
    predictions=np.load(ROOT/'experiments/pairs_forecast_20261017/artifacts/models/predictions.npz',allow_pickle=False)
    np.testing.assert_array_equal(predictions['times'],market.times)
    tapes={r['tape'] for r in prior['quotes'] if r['fills']}
    rows=[]
    for tape in sorted(tapes):
        streams,_=load_tape(ROOT/tape);rates={s:settled_funding(s) for s in streams}
        for quote in prior['quotes']:
            if quote['tape']!=tape:continue
            i=int(np.searchsorted(market.times,quote['decision']//1000,side='right')-1)
            if i<0 or quote['decision']/1000-market.times[i]>3600:continue
            for k,pair in enumerate(cfg['pairs']):
                if quote['symbol'] not in pair or any(s not in streams for s in pair):continue
                beta=predictions['beta'][i,k]
                for fill in quote['fills']:
                    for horizon in execution['holding_seconds']:
                        r=evaluate(fill,pair,beta,streams,decimals,rates,horizon,cfg,execution)
                        r.update(tape=tape,quote_decision=quote['decision'],ttl_seconds=quote['ttl_seconds'],
                                 queue_model=quote['queue_model'],filter_rejects=quote['filter_rejects'])
                        rows.append(r)
    summaries=[]
    for queue in cfg['queue_models']:
        for ttl in cfg['ttl_seconds']:
            for filtered in (False,True):
                for horizon in execution['holding_seconds']:
                    rs=[r for r in rows if r['queue_model']==queue and r['ttl_seconds']==ttl and r['horizon']==horizon and not(filtered and r['filter_rejects'])]
                    closed=[r for r in rs if r['status']=='closed'];notional=sum(r['notional'] for r in closed)
                    summaries.append(dict(queue_model=queue,ttl_seconds=ttl,flow_filtered=filtered,horizon=horizon,
                        observations=len(rs),statuses=dict(Counter(r['status'] for r in rs)),closed=len(closed),
                        conditional_diagnostic_net_pnl=sum(r['net_pnl'] for r in closed),
                        weighted_net_bps=sum(r['net_pnl'] for r in closed)/notional*10000 if notional else None,
                        missing_outcomes_not_zero_profit=True))
    save(out/'results.json',dict(summaries=summaries,trials=rows,historical_independence=False,
        account_return=None,financial_score=None,live_orders_sent=0,
        limitation='Conditional, overlapping microstructure probes. Cannot sum into a feasible portfolio or compare account Q; missing and unhedged outcomes require further observation.'))
    print(json.dumps(summaries,indent=2),flush=True)


if __name__=='__main__':main()
