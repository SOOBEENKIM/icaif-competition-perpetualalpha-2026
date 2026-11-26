"""Actual-book costs, candle proxy errors, and conditional paid pair markouts."""
import bisect,json,time
from collections import Counter
from pathlib import Path
import numpy as np
from experiments.basis_carry_20261017.probe import load_tape,asof,delayed_snapshot,sweep,walk
from experiments.execution_retry_20261016.engine import lot_floor
from experiments.complementary_alpha_20261017.research import protected
from experiments.par_direct_forecast_20261017.data import sha,save
from experiments.par_direct_forecast_20261017.model import design
from strategy.agent import TAIL

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];OUT=HERE/'artifacts'
HISTORY=ROOT/'experiments/pairs_research_20261017/par_history/20261016T173824Z'

def quantiles(x):
    a=np.asarray(x,float)
    return dict(n=len(a),q05=float(np.quantile(a,.05)),median=float(np.median(a)),q95=float(np.quantile(a,.95))) if len(a) else dict(n=0)

def extension(fill,pair,beta,streams,decimals,hours,impact,cfg,d,symbols):
    lead=fill['coin'].split(':')[-1];other=pair[1] if pair[0]==lead else pair[0]
    row=dict(pair=pair,lead=lead,hedge=other,beta=float(beta),hours=hours,impact_bps=impact,lead_fill=fill)
    sn=asof(streams[lead],fill['time'],cfg);hn=asof(streams[other],fill['time'],cfg)
    if sn is None or hn is None:return dict(row,status='missing_hedge_decision')
    mult=beta if pair[0]==lead else 1/beta
    qty=lot_floor(fill['qty']*sn['oracle']*mult/hn['oracle'],decimals[other]);side=-fill['side']
    if qty<=0:return dict(row,status='hedge_below_minimum')
    arrival=delayed_snapshot(streams[other],fill['time']+1000,cfg)
    if arrival is None:return dict(row,status='missing_hedge_arrival')
    ht,hs=arrival;hp=walk(hs['book']['levels'][1 if side>0 else 0],qty,1.)
    if hp is None:return dict(row,status='insufficient_hedge_depth')
    if hp*qty<10:return dict(row,status='hedge_below_minimum')
    positions=[dict(symbol=lead,qty=fill['side']*fill['qty'],entry=fill['price'],entry_time=fill['time'],entry_fee_bps=3),
               dict(symbol=other,qty=side*qty,entry=hp,entry_time=ht,entry_fee_bps=9)]
    desired=ht/1000+hours*3600;ix=int(np.searchsorted(d['times'],desired,side='left'))
    if ix>=len(d['times']):return dict(row,status='future_candle_unavailable')
    if d['times'][ix]-desired>300:raise ValueError('Unexpected candle gap')
    for pos in positions:
        j=symbols.index(pos['symbol']);sign=np.sign(pos['qty'])
        # A zero-volume bar is not an executable exit. Do not pretend a stale
        # close liquidates the pair; count it as unresolved in this diagnostic.
        if d['par_volume'][ix,j]<=0:return dict(row,status='no_exit_trade_observation')
        exit=float(d['par_close'][ix,j]*(1-sign*impact/10000))
        mask=(d['times']*1000>pos['entry_time'])&(d['times']<=d['times'][ix])&(d['times']%3600==0)
        funding=-pos['qty']*float(np.sum(d['reference_close'][mask,j]*d['funding'][mask,j]))
        fees=abs(pos['qty'])*(pos['entry']*pos['entry_fee_bps']+exit*cfg['taker_fee_bps'])/10000
        pos.update(exit=exit,exit_time=int(d['times'][ix]*1000),funding=funding,fees=fees,
                   gross_pnl=pos['qty']*(exit-pos['entry']))
    gross=sum(p['gross_pnl'] for p in positions);funding=sum(p['funding'] for p in positions);fees=sum(p['fees'] for p in positions)
    notional=sum(abs(p['qty'])*p['entry'] for p in positions);net=gross+funding-fees
    return dict(row,status='modeled_close',positions=positions,gross_pnl=gross,funding=funding,fees=fees,notional=notional,net_pnl=net,net_bps=net/notional*10000)

def main():
    OUT.mkdir(exist_ok=True)
    if (OUT/'frozen.json').exists():raise FileExistsError('Calibration already run')
    cfg=json.loads((HERE/'protocol.json').read_text());tapes=json.loads((ROOT/'experiments/basis_carry_20261017/protocol.json').read_text())['tapes']
    old=ROOT/'experiments/par_direct_forecast_20261017/artifacts'
    sources={str(q.relative_to(ROOT)):sha(q) for q in [*HERE.glob('*.py'),HERE/'protocol.json',old/'inputs.npz',old/'predictions.npz']}
    sources.update({s:sha(ROOT/s) for s in tapes})
    passive_file=ROOT/'experiments/pairs_maker_20261017/artifacts/results.json';sources[str(passive_file.relative_to(ROOT))]=sha(passive_file)
    save(OUT/'frozen.json',dict(at_unix=time.time(),sources=sources,protected=protected(),historical_reuse=True,orders_sent=0))
    with np.load(old/'inputs.npz') as z:d={k:z[k].copy() for k in z.files}
    with np.load(old/'predictions.npz') as z:pred={k:z[k].copy() for k in z.files}
    p=json.loads((TAIL/'baseline.json').read_text())['protocol'];symbols=p['symbols']
    forecast_cfg=json.loads((old.parent/'protocol.json').read_text());de=design(d,forecast_cfg,symbols)
    meta=json.loads((HISTORY/'metadata.json').read_text())['response'][0]['universe'];decimals={r['name'].split(':')[-1]:r['szDecimals'] for r in meta}
    raw=old/'raw';candles={}
    for s in symbols:
        rows=json.loads((raw/f'par_{s}_5m.json').read_text())['response']
        candles[s]={int(r['t'])//1000:r for r in rows}
    passive=json.loads(passive_file.read_text())['quotes'];cost_rows=[];long_rows=[];coverage=[]
    for ti,path in enumerate(tapes):
        streams,receipt=load_tape(ROOT/path);coverage.append(receipt);phase='calibration' if ti<4 else 'later_tape'
        for symbol,stream in streams.items():
            if symbol not in symbols:continue
            step=cfg['sample_seconds']*1000
            for now in range((receipt['start_ms']//step+1)*step,receipt['end_ms']+1,step):
                snap=asof(stream,now,cfg)
                if snap is None:continue
                sec=now//1000;closed_open=sec//300*300-300;prior=candles[symbol].get(closed_open)
                nextbar=candles[symbol].get(sec//300*300)
                if prior is None:continue
                for notional in cfg['notionals']:
                    q=lot_floor(notional/snap['mid'],decimals[symbol])
                    if q*snap['mid']<10:continue
                    for side in (-1,1):
                        top=float(snap['book']['levels'][1 if side>0 else 0][0]['px'])
                        limit=top*(1+side*cfg['ioc_limit_bps']/10000)
                        for latency in cfg['latencies_ms']:
                            arrival=delayed_snapshot(stream,now+latency,cfg)
                            if arrival is None:
                                cost_rows.append(dict(phase=phase,tape=path,symbol=symbol,time=now,notional=notional,side=side,latency_ms=latency,status='missing_arrival'))
                                continue
                            at,ss=arrival
                            for fraction in cfg['depth_fractions']:
                                got,px=sweep(ss['book']['levels'][1 if side>0 else 0],q,fraction,side,decimals[symbol],limit)
                                r=dict(phase=phase,tape=path,symbol=symbol,time=now,notional=notional,side=side,latency_ms=latency,depth_fraction=fraction,
                                    status='filled' if got>=q-1e-10 else 'partial' if got else 'unfilled',requested=q,filled=got,
                                    prior_5m_volume=float(prior['v']),volume_10pct_capacity=float(prior['v'])*.1,
                                    source_spread_bps=snap['halfspread']*2/snap['mid']*10000,arrival=at)
                                if got:
                                    r.update(price=px,cost_from_decision_mid_bps=side*(px/snap['mid']-1)*10000,
                                        cost_from_arrival_mid_bps=side*(px/ss['mid']-1)*10000)
                                    if nextbar and now%300000==0 and float(nextbar['v'])>0:
                                        r.update(next_bar_open_error_bps=side*(float(nextbar['o'])/px-1)*10000,
                                                 next_bar_close_error_bps=side*(float(nextbar['c'])/px-1)*10000)
                                cost_rows.append(r)
        quotes=[q for q in passive if q['tape']==path and q['queue_model']==cfg['passive_queue_model'] and q['ttl_seconds']==cfg['passive_ttl_seconds']]
        for quote in quotes:
            i=int(np.searchsorted(d['times'],quote['decision']//1000,side='right')-1)
            if i<0 or quote['decision']/1000-d['times'][i]>300:continue
            for k,pair in enumerate(forecast_cfg['pairs']):
                if quote['symbol'] not in pair or any(s not in streams for s in pair):continue
                be=de['beta'][i,k]
                if not np.isfinite(be) or not .1<=be<=10:continue
                for fill in quote['fills']:
                    for h in cfg['long_markout_hours']:
                        for impact in cfg['exit_impact_bps']:
                            r=extension(fill,pair,be,streams,decimals,h,impact,cfg,d,symbols)
                            direction=fill['side'] if fill['coin'].split(':')[-1]==pair[0] else -fill['side']
                            f={name:float(pred[f'{name}_{h}'][i,k]) for name in ('ridge','tree','blend')}
                            r.update(phase=phase,tape=path,quote_decision=quote['decision'],forecast=f,
                                forecast_aligned={name:bool(np.isfinite(mu) and direction*mu>0) for name,mu in f.items()},pair_direction=direction)
                            # Nonfinite forecasts are missing, not zero predictions.
                            r['forecast']={name:v if np.isfinite(v) else None for name,v in f.items()};long_rows.append(r)
        print('tape',phase,receipt['hours'],'cost rows',len(cost_rows),'long probes',len(long_rows),flush=True)
    cost_summary=[]
    for phase in ('calibration','later_tape'):
        for latency in cfg['latencies_ms']:
            for notional in cfg['notionals']:
                for frac in cfg['depth_fractions']:
                    rr=[r for r in cost_rows if r['phase']==phase and r['latency_ms']==latency and r['notional']==notional and r.get('depth_fraction')==frac]
                    filled=[r for r in rr if r['status']=='filled'];zero=[r for r in rr if r['prior_5m_volume']==0]
                    errors=[r for r in filled if 'next_bar_open_error_bps' in r]
                    cost_summary.append(dict(phase=phase,latency_ms=latency,notional=notional,depth_fraction=frac,observations=len(rr),statuses=dict(Counter(r['status'] for r in rr)),
                        actual_depth_after_zero_volume=dict(cases=len(zero),full_fill=sum(r['status']=='filled' for r in zero)),
                        decision_mid_cost_bps=quantiles([r['cost_from_decision_mid_bps'] for r in filled]),
                        arrival_mid_cost_bps=quantiles([r['cost_from_arrival_mid_bps'] for r in filled]),
                        absolute_next_open_error_bps=quantiles([abs(r['next_bar_open_error_bps']) for r in errors]),
                        absolute_next_close_error_bps=quantiles([abs(r['next_bar_close_error_bps']) for r in errors])))
    long_summary=[]
    for h in cfg['long_markout_hours']:
        for impact in cfg['exit_impact_bps']:
            for condition in ('all','ridge','tree','blend'):
                rr=[r for r in long_rows if r['hours']==h and r['impact_bps']==impact and (condition=='all' or r['forecast_aligned'][condition])]
                closed=[r for r in rr if r['status']=='modeled_close'];notional=sum(r['notional'] for r in closed)
                long_summary.append(dict(hours=h,exit_impact_bps=impact,condition=condition,observations=len(rr),statuses=dict(Counter(r['status'] for r in rr)),
                    modeled_closed=len(closed),unique_quote_decisions=len({(r['tape'],r['quote_decision'],r['lead']) for r in closed}),
                    conditional_weighted_net_bps=sum(r['net_pnl'] for r in closed)/notional*10000 if notional else None,
                    conditional_notional=notional,not_account_pnl_or_Q=True))
    for path,h in sources.items():assert sha(ROOT/path)==h,path
    protected()
    save(OUT/'results.json',dict(cost_summary=cost_summary,long_summary=long_summary,cost_rows=cost_rows,long_rows=long_rows,coverage=coverage,
        historical_independence=False,missing_outcomes_not_zero=True,orders_sent=0,account_return=None,financial_score=None,protected_sources_unchanged=True))
    print('DONE',len(cost_rows),'cost observations',len(long_rows),'conditional probes',flush=True)

if __name__=='__main__':main()
