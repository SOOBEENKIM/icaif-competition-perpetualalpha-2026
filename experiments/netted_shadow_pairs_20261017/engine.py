"""A causal shadow-inventory signal, one physical ledger and net orders."""
import copy
import numpy as np
from replay_lab.common_comparison.models import Market
from replay_lab.common_comparison.engine import Ledger,project,rounded,risk_breaches,risk_values
from experiments.quote_anchored_pairs_20261017.pricing import execute_intent
from experiments.basis_carry_20261017.score import score


def map_shadow(times,run):
    st=np.asarray([s['time'] for s in run['snapshots']]);q=np.asarray([s['qty'] for s in run['snapshots']])
    if not len(st) or not np.all(np.diff(st)>0):raise ValueError('Shadow observations must be ordered and unique')
    ix=np.searchsorted(st,times,side='right')-1
    if np.any(ix<0) or np.any(times>st[-1]):raise ValueError('No shadow extrapolation beyond observed causal trace')
    assert np.all(st[ix]<=times)
    return q[ix],st[ix]


def aggregate(base,pair,share):
    if not 0<=share<=1:raise ValueError('Capital fractions must sum to one')
    return (1-share)*np.asarray(base)+share*np.asarray(pair)


class TargetState:
    def __init__(self):
        self.task=None;self.pair_key=None;self.suppressed=None;self.last_desired=None;self.events=[]

    def propose(self,now,qty,base,pair,share,prices,step,cfg):
        key=tuple(pair.tolist());fallback=(1-share)*base
        if key!=self.pair_key:self.suppressed=None;self.pair_key=key
        desired=fallback if self.suppressed==key else aggregate(base,pair,share)
        error=float(abs(desired-qty)@prices);gross=float(abs(desired)@prices)
        tolerance=max(10.,cfg['minimum_tracking_fraction']*gross)
        if self.task is not None and error<=tolerance:self.task=None
        if self.task is not None and now>=self.task['deadline']:
            if share and np.any(pair):
                self.suppressed=key;desired=fallback
                self.events.append(dict(time=now,event='target_timeout_pair_suppressed',tracking_error=error))
            self.task=None
        changed=self.last_desired is None or not np.array_equal(desired,self.last_desired)
        if changed:
            # A new source observation does not keep an incomplete trade alive forever.
            deadline=self.task['deadline'] if self.task is not None else now+cfg['completion_window_steps']*step
            self.task=dict(target=desired.copy(),deadline=deadline)
            self.last_desired=desired.copy()
        elif self.task is not None:self.task['target']=desired.copy()
        # Baseline-only fallback remains actionable, even if its first reduction was unfilled.
        if self.suppressed==key or share==0 or not np.any(pair):
            return desired.copy()
        return self.task['target'].copy() if self.task is not None else None


def replay(d,p,cfg,scenario,start,end,base,pair,share,source_times):
    p=copy.deepcopy(p);sc=cfg['scenarios'][scenario];p['execution']['fee_bps']=sc['fee_bps']
    t=d['times'];step=int(t[1]-t[0]);assert np.all(np.diff(t)==step)
    m=Market(t,d['par_close'],d['par_volume'],d['par_trades'],d['funding'],d['funding_observed'],tuple(p['symbols']))
    first,last=m.exact(start),m.exact(end);ledger=Ledger(p['initial'],m.symbols);state=TargetState()
    pending=None;snapshots=[];orders=[];decisions=[];peak=ledger.initial;mdd=0.;halted=False
    audit_eq=ledger.initial;price_move=impact=0.;max_error=0.;breaches=[]
    for i in range(first,last+1):
        now=int(t[i]);prices=m.close[i];sigma=d['sigma'][i];prior=ledger.qty.copy()
        if i>first:
            move=float(prior@(prices-m.close[i-1]));price_move+=move;audit_eq+=move
            if now%3600==0:
                rates=m.funding[i].copy();cash=-prior*d['reference_close'][i]*rates
                rates=np.where(cash>0,rates*sc['funding_receipt_multiplier'],rates)
                ledger.charge_funding(now,d['reference_close'][i],rates,m.observed[i],0.)
                audit_eq-=float(prior@(d['reference_close'][i]*rates))
        nf=len(ledger.fills)
        if pending is not None:
            assert pending['execution_time']==now;pending['sigma']=sigma
            orders.extend(execute_intent(ledger,pending,m,i,p,sc));pending=None
        for f in ledger.fills[nf:]:
            cost=f['signed_qty']*(f['price']-prices[f['asset']]);impact+=cost;audit_eq-=cost+f['fee']
        equity=ledger.reconcile(prices);max_error=max(max_error,abs(equity-audit_eq));assert max_error<1e-6
        peak=max(peak,equity);mdd=min(mdd,equity/peak-1);halted|=equity/peak-1<=-p['risk']['drawdown_halt']
        breach=risk_breaches(ledger.qty,equity,prices,sigma,p['risk'])
        if breach:breaches.append(dict(time=now,limits=breach))
        if i<last:
            assert source_times[i]<=now
            forced=bool(breach) or halted
            desired=np.zeros(len(prices)) if forced else state.propose(now,ledger.qty,base[i],pair[i],share,prices,step,cfg)
            if desired is not None:
                target=project(desired*prices/equity,equity,sigma,p['risk'])*equity/prices
                if forced:target=np.zeros_like(target)
                else:
                    delta=float(abs(target-ledger.qty)@prices/equity)
                    if delta>p['risk']['normal_turnover_fraction']:
                        target=ledger.qty+(target-ledger.qty)*p['risk']['normal_turnover_fraction']/delta
                target=np.asarray([rounded(q,dec) for q,dec in zip(target,p['execution']['decimals'])])
                if not np.array_equal(target,ledger.qty):
                    pending=dict(decision_time=now,execution_time=now+step,source_close=now,source_available=int(source_times[i]),
                        source_volume=m.volume[i].copy(),source_trades=m.trades[i].copy(),source_prices=prices.tolist(),
                        target_qty=target.tolist(),risk_forced=forced)
                    decisions.append(dict(time=now,source_time=int(source_times[i]),baseline_component=((1-share)*base[i]).tolist(),
                        pair_component=(share*pair[i]).tolist(),net_desired=desired.tolist(),actual_before=ledger.qty.tolist(),
                        target_qty=target.tolist(),forced=forced))
        snapshots.append(dict(time=now,equity=equity,cash=ledger.cash,qty=ledger.qty.tolist(),fees=ledger.fees,
            funding=ledger.funding,**risk_values(ledger.qty,equity,prices,sigma)))
    daily=[s for s in snapshots if s['time']%86400==43200];values=np.asarray([s['equity'] for s in daily]);returns=values[1:]/values[:-1]-1
    assert daily[0]['time']==start and daily[-1]['time']==end
    roi=equity/ledger.initial-1;finance=score(roi,returns,trading_days=len({f['time']//86400 for f in ledger.fills}),start_sample=True)
    residual=float(abs(ledger.qty)@prices);reserve=residual*(sc['fee_bps']+sc['impact_bps'])/10000
    metrics=dict(net_pnl=equity-ledger.initial,roi=roi,**finance,fees=ledger.fees,funding=ledger.funding,price_move_pnl=price_move,
        execution_cost=impact,max_drawdown=-mdd,mean_gross=float(np.mean([s['gross'] for s in snapshots])),maximum_gross=max(s['gross'] for s in snapshots),
        fills=len(ledger.fills),filled_decision_times=len({f['decision_time'] for f in ledger.fills}),orders=len(orders),
        residual_gross=residual,liquidation_cost_estimate=reserve,pnl_after_liquidation_reserve=equity-ledger.initial-reserve,
        maximum_accounting_error=max(max_error,ledger.max_error),risk_breach_bars=len(breaches),
        suppressed_pair_events=len(state.events))
    assert abs(np.prod(1+returns)-1-roi)<1e-10
    assert abs(metrics['net_pnl']-price_move+impact+ledger.fees-ledger.funding)<1e-6
    assert all(f['source_available']<=f['decision_time']<f['time'] for f in ledger.fills)
    return dict(start=start,end=end,metrics=metrics,snapshots=snapshots,orders=orders,decisions=decisions,
        fills=ledger.fills,funding_events=ledger.funding_events,events=state.events,risk_breaches=breaches,
        daily=[dict(time=x['time'],return_=float(r)) for x,r in zip(daily[1:],returns)],orders_sent=0)
