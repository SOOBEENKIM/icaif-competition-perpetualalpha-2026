"""Allowlisted old public tapes; passive queue, flow and subsequent execution.

These are independent microstructure probes, NOT a portfolio backtest.
"""
import bisect
from collections import Counter
import gzip
import json
from pathlib import Path
import time
import numpy as np
from experiments.basis_carry_20261017.probe import load_tape, asof, delayed_snapshot, walk
from experiments.execution_retry_20261016.engine import lot_floor
from experiments.pairs_research_20261017.research import save, sha
from experiments.complementary_alpha_20261017.research import protected
from .core import FlowFilter, PassiveOrder, queue_at_arrival

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / 'artifacts'
HISTORY = ROOT / 'experiments/pairs_research_20261017/par_history/20261016T173824Z'


def trades_and_flow(path, cfg):
    opener = gzip.open if path.suffix == '.gz' else open
    trades, seen, flows, filters = {}, set(), {}, {}
    # A public transaction may match multiple makers. Use one side observation
    # per coin/time/hash/side group rather than treating each split fill as a
    # separate independent market order. Parent orders are not identifiable.
    groups = set()
    with opener(path, 'rt') as f:
        for line in f:
            row = json.loads(line); msg = row.get('message', row)
            if row.get('venue','par') != 'par' or msg.get('channel') != 'trades': continue
            received = row['received_ms']
            for tr in sorted(msg['data'],key=lambda t:(t['time'],t['tid'])):
                identity = tr['time'],tr['coin'],tr['tid']
                if identity in seen: continue
                seen.add(identity); symbol = tr['coin'].split(':')[-1]
                trades.setdefault(symbol, []).append((received,tr))
                group = tr['coin'],tr['time'],tr.get('hash'),tr['side']
                if group in groups: continue
                groups.add(group)
                ff = cfg['flow_filter']
                filters.setdefault(symbol, FlowFilter(ff['hazard_per_trade'],ff['prior_alpha'],ff['prior_beta'],ff['maximum_run_length']))
                prediction = filters[symbol].observe(int(tr['side']=='B'))
                flows.setdefault(symbol,[]).append((received,prediction))
    for values in trades.values(): values.sort(key=lambda p:(p[0],p[1]['time'],p[1]['tid']))
    return trades,flows


def settled_funding(symbol):
    raw = json.loads((HISTORY/f'funding_{symbol}.json').read_text())
    return {int(r['time'])//3600000*3600000:float(r['fundingRate']) for r in raw['rows']}


def evaluate_fill(fill, stream, horizon, cfg, funding):
    desired = fill['time'] + horizon*1000
    exit_snapshot = delayed_snapshot(stream,desired+cfg['latency_ms'],cfg)
    if exit_snapshot is None: return dict(horizon=horizon,status='missing_future_exit')
    at,snap = exit_snapshot; side=fill['side']; quantity=fill['qty']
    price = walk(snap['book']['levels'][0 if side>0 else 1],quantity,1.)
    if price is None: return dict(horizon=horizon,status='insufficient_exit_depth')
    payment=0.
    for hour in range((fill['time']//3600000+1)*3600000,at+1,3600000):
        ctx=asof(stream,hour,cfg)
        if hour not in funding or ctx is None: return dict(horizon=horizon,status='missing_funding_or_oracle')
        payment-=side*quantity*ctx['oracle']*funding[hour]
    markout=side*(snap['mid']-fill['price'])/fill['price']*10000
    fees=quantity*(fill['price']*cfg['maker_fee_bps']+price*9)/10000
    net=side*quantity*(price-fill['price'])+payment-fees
    return dict(horizon=horizon,status='closed',exit_time=at,exit_price=price,
        gross_markout_bps=markout,funding=payment,fees=fees,net_pnl=net,
        net_bps=net/(quantity*fill['price'])*10000,notional=quantity*fill['price'])


def probe(stream,trades,symbol,decision,side,ttl,queue_model,decimals,cfg,flow,funding):
    initial=asof(stream,decision,cfg)
    if initial is None: return None
    price=float(initial['book']['levels'][0 if side>0 else 1][0]['px'])
    quantity=lot_floor(cfg['quote_notional']/price,decimals)
    if quantity*price<10: return None
    row=dict(symbol=symbol,decision=decision,side=side,ttl_seconds=ttl,queue_model=queue_model,
        requested_price=price,requested_qty=quantity,spread_bps=initial['halfspread']*2/initial['mid']*10000,
        flow=flow,fills=[],markouts=[])
    row['filter_rejects']=(flow['observations']>=cfg['flow_filter']['minimum_seen'] and
        (flow['buy_probability']<1-cfg['flow_filter']['adverse_probability_threshold'] if side>0 else flow['buy_probability']>cfg['flow_filter']['adverse_probability_threshold']))
    arrival=delayed_snapshot(stream,decision+cfg['latency_ms'],cfg)
    if arrival is None: row['status']='no_arrival_book';return row
    at, snapshot=arrival
    ahead,status=queue_at_arrival(snapshot['book'],price,side)
    row.update(arrival=at,status=status)
    if status!='accepted':return row
    effective=at+ttl*1000+cfg['cancel_latency_ms']
    o=PassiveOrder('par:'+symbol,side,price,quantity,ahead if queue_model.startswith('all_') else 0.,at,effective,decimals)
    end=stream['book_times'][-1]
    resets=[t for t in stream['resets'] if at<t<effective]
    stop=min([effective,end+1,*resets])
    row.update(queue_ahead=ahead,cancel_effective=effective,
               status='data_gap' if resets else 'right_censored' if effective>end else 'expired')
    received_times=[r[0] for r in trades]
    for received,tr in trades[bisect.bisect_right(received_times,at):]:
        if received>=stop:break
        o.trade(received,tr)
        if sum(f['qty'] for f in o.fills)>=quantity-1e-10:row['status']='filled';break
    row['fills']=o.fills
    row['fill_fraction']=sum(f['qty'] for f in o.fills)/quantity
    for fill in o.fills:
        for h in cfg['markout_seconds']:
            row['markouts'].append(dict(fill=fill,**evaluate_fill(fill,stream,h,cfg,funding)))
    return row


def main():
    OUT.mkdir(exist_ok=False)
    cfg=json.loads((HERE/'protocol.json').read_text())
    verification=json.loads((HERE/'trade_side_verification.json').read_text())
    assert verification['verified']
    tapes=json.loads((ROOT/'experiments/basis_carry_20261017/protocol.json').read_text())['tapes']
    save(OUT/'frozen.json',dict(at=time.time(),sources={str(p.relative_to(ROOT)):sha(p) for p in [*HERE.glob('*.py'),HERE/'protocol.json']},
        tapes={p:sha(ROOT/p) for p in tapes},protected=protected(),orders_sent=0,historical_reuse=True))
    metadata=json.loads((HISTORY/'metadata.json').read_text())['response'][0]['universe']
    decimals={m['name'].split(':')[-1]:m['szDecimals'] for m in metadata}
    rows,coverage=[],[]
    for tape in tapes:
        stream,receipt=load_tape(ROOT/tape); coverage.append(receipt)
        trades,flows=trades_and_flow(ROOT/tape,cfg)
        receipt['unique_trade_counts']={s:len(ts) for s,ts in trades.items()}
        receipt['within_capture_trade_counts']={s:sum(receipt['start_ms']<=t['time']<=receipt['end_ms'] for _,t in ts) for s,ts in trades.items()}
        for symbol,market in stream.items():
            history=trades.get(symbol,[]); beliefs=flows.get(symbol,[]); belief_times=[b[0] for b in beliefs]
            funding=settled_funding(symbol)
            step=cfg['decision_seconds']*1000
            for now in range((receipt['start_ms']//step+1)*step,receipt['end_ms']+1,step):
                fi=bisect.bisect_left(belief_times,now)-1
                flow=beliefs[fi][1] if fi>=0 else dict(buy_probability=.5,signed_flow=0.,expected_run_length=0.,observations=0)
                for side in (-1,1):
                    for ttl in cfg['ttl_seconds']:
                        for queue in cfg['queue_models']:
                            r=probe(market,history,symbol,now,side,ttl,queue,decimals[symbol],cfg,flow,funding)
                            if r is not None:r['tape']=tape;rows.append(r)
    summaries=[]
    for queue in cfg['queue_models']:
        for ttl in cfg['ttl_seconds']:
            for filtered in (False,True):
                rs=[r for r in rows if r['queue_model']==queue and r['ttl_seconds']==ttl and not(filtered and r.get('filter_rejects'))]
                summary=dict(queue_model=queue,ttl_seconds=ttl,flow_filtered=filtered,quotes=len(rs),
                    statuses=dict(Counter(r['status'] for r in rs)),quotes_with_fill=sum(bool(r['fills']) for r in rs),
                    filled_notional=sum(f['qty']*f['price'] for r in rs for f in r['fills']),markouts=[])
                for h in cfg['markout_seconds']:
                    obs=[m for r in rs for m in r['markouts'] if m['horizon']==h]
                    closed=[m for m in obs if m['status']=='closed'];notional=sum(m['notional'] for m in closed)
                    summary['markouts'].append(dict(horizon=h,observed=len(closed),missing=len(obs)-len(closed),
                        diagnostic_net_pnl=sum(m['net_pnl'] for m in closed),
                        notional_weighted_net_bps=sum(m['net_pnl'] for m in closed)/notional*10000 if notional else None,
                        mean_gross_markout_bps=float(np.mean([m['gross_markout_bps'] for m in closed])) if closed else None))
                summaries.append(summary)
    save(OUT/'results.json',dict(summaries=summaries,quotes=rows,coverage=coverage,protected_after=protected(),
        independent=False,account_return=None,financial_score=None,orders_sent=0,
        interpretation='Independent passive quote probes and paid exits, not an account strategy. No known actual queue/cancellation positions. Missing outcomes are excluded and counted, not assigned zero PnL.'))
    print(json.dumps(summaries,indent=2),flush=True)


if __name__=='__main__':main()
