"""Controller-injected copy of the independently audited common replay loop.

Only extension: caller supplies the controller. Equality to the retained replay
is asserted before candidate results are used.
"""
import numpy as np
from replay_lab.common_comparison.models import HOUR
from replay_lab.common_comparison.engine import Ledger,execute_intent,risk_values,risk_breaches
from experiments.risk_preserving_20261016.audit import audit

def simulate(m,weights,sigma,protocol,first,last,scenario_name,control):
    ledger=Ledger(protocol['initial'],m.symbols);scenario=protocol['scenarios'][scenario_name]
    pending=None;peak=ledger.initial;mdd=0.;halted=False;streak=max_streak=0
    snapshots=[];decisions=[];orders=[];daily_values={};violations=[]
    for i in range(first,last+1):
        t=int(m.times[i]);prices=m.close[i];cov=sigma[i]
        if i>first:ledger.charge_funding(t,prices,m.funding[i],m.observed[i],scenario['adverse_funding_bps_hour'])
        if pending is not None:
            pending['sigma']=cov;orders.extend(execute_intent(ledger,pending,m,i,protocol,scenario));pending=None
        equity=ledger.reconcile(prices);peak=max(peak,equity);mdd=min(mdd,equity/peak-1)
        halted|=equity/peak-1<=-protocol['risk']['drawdown_halt']
        breach=risk_breaches(ledger.qty,equity,prices,cov,protocol['risk'])
        streak=streak+1 if breach else 0;max_streak=max(streak,max_streak)
        if breach:violations.append({'time':t,'constraints':breach})
        if t%86400==0:daily_values[t]=equity
        if i<last:
            action=control.propose(t,prices,ledger.qty,equity,weights[i],cov,m.volume[i],m.trades[i],protocol,buffer=.2,halted=halted)
            if action is not None:
                event=dict(decision_time=t,execution_time=t+HOUR,source_close=t,source_available=t,source_index=i,
                    source_prices=prices.tolist(),visible_equity=equity,position_before=ledger.qty.tolist(),
                    source_volume=m.volume[i].tolist(),source_trades=m.trades[i].tolist(),raw_target=weights[i].tolist(),**action)
                decisions.append(event);pending={**event,'source_volume':m.volume[i].copy(),'source_trades':m.trades[i].copy(),'sigma':cov}
        snapshots.append(dict(time=t,equity=equity,cash=ledger.cash,qty=ledger.qty.tolist(),unrealized=equity-ledger.cash,
            fees=ledger.fees,funding=ledger.funding,realized=ledger.realized,**risk_values(ledger.qty,equity,prices,cov)))
    daily=[dict(time=t,return_=daily_values[t]/daily_values[t-86400]-1) for t in sorted(daily_values) if t-86400 in daily_values]
    metrics={'roi':equity/ledger.initial-1,'equity':equity,'net_pnl':equity-ledger.initial,
       'daily_std_sample':float(np.std([d['return_'] for d in daily],ddof=1)) if len(daily)>1 else None,
       'daily_std_population':float(np.std([d['return_'] for d in daily])) if daily else None,
       'complete_days':len(daily),'max_drawdown_magnitude':-mdd,'fees':ledger.fees,'funding':ledger.funding,
       'realized':ledger.realized,'unrealized':equity-ledger.cash,'turnover':ledger.turnover,'fills':len(ledger.fills),
       'orders':len(orders),'completed_lifecycles':sum(f['position_after']==0 and f['position_before']!=0 for f in ledger.fills),
       'mean_gross':float(np.mean([s['gross'] for s in snapshots])),'maximum_gross':max(s['gross'] for s in snapshots),
       'incomplete_orders':sum(o['status']!='FILLED' for o in orders),'final_qty':ledger.qty.tolist(),'maximum_ledger_error':ledger.max_error,
       'rejected_asset_increases':sum(sum(d.get('entry_blocked',[])) for d in decisions)}
    result=dict(scenario=scenario_name,start=int(m.times[first]),end=int(m.times[last]),metrics=metrics,snapshots=snapshots,
       decisions=decisions,orders=orders,fills=ledger.fills,funding_events=ledger.funding_events,daily=daily,risk_breaches=violations,
       gates={'accounting':ledger.max_error<1e-6,'information_time':all(f['source_available']<=f['decision_time']<f['time'] for f in ledger.fills),
              'risk':max_streak<2,'not_halted':not halted},live_orders_sent=0)
    result['independent_accounting_audit']=audit(result,m,protocol)
    if not result['independent_accounting_audit']['passed']:raise AssertionError(result['independent_accounting_audit'])
    return result
