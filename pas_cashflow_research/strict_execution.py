"""Audited minute replay: only bars wholly after the decision are fill eligible.

There is no order client. Transaction participation is an explicit assumption,
not a claim of queue priority, available resting depth or exact live replication.
"""
import copy
import numpy as np
from replay_lab.common_comparison.engine import rounded,risk_breaches
from experiments.shared_execution_pairs_20261017.account_v2 import SharedAccount,difference
from replay_lab.ema_rebuild.engine import Controller
from competition_live.scoring import evaluate


def execute(account,pending,batch,prices,sigma,p,sc):
    ledger=account.physical;j=p['symbols'].index(batch['coin'][4:])
    old=ledger.qty[j];delta=difference(pending['qty'][j],old)
    if not delta:return None
    side=1 if delta>0 else -1;reducing=old*delta<0
    request=min(abs(delta),abs(old)) if reducing else abs(delta)
    price=batch['buy_price' if side>0 else 'sell_price']*(1+side*sc['impact_bps']/10000)
    request=abs(rounded(min(request,7500/price),p['execution']['decimals'][j]))
    full_close=reducing and request==abs(old)
    # Venue minimum constrains the submitted request, not the partial result.
    eligible=request*price>=10 or full_close
    limit=(pending.get('entry_limits') or {}).get(j,pending['source_prices'][j]*(1+side*.05))
    if pending.get('economic_execution') and reducing:limit=float('inf') if side>0 else 0.
    blocked=side*(price-limit)>1e-8
    available=batch['qty']*sc['participation']
    amount=abs(rounded(min(request,available),p['execution']['decimals'][j])) if eligible and not blocked else 0.
    if amount:amount=account.feasible_size(j,side,amount,price,prices,sigma,p,sc['fee_bps'])
    if amount:
        account.physical_fill(j,side*amount,price,sc['fee_bps'],batch['time']/1000,
            decision_time=pending['time']/1000,source_time=pending['source']/1000)
        account.owner_fill('base',j,side*amount,price,sc['fee_bps'],batch['time']/1000,'conditional_economic_fill')
    return dict(time=batch['time'],coin=batch['coin'],requested=request,filled=amount,
                limit_blocked=blocked,minimum_request_pass=eligible,available=available,price=price)


def simulate(cfg,d,signals,tape,funding,p,sc,*,controller=None,entry_lifetime=900,economic_limits=False,sticky_exits=False):
    p=copy.deepcopy(p);p['execution']['fee_bps']=sc['fee_bps']
    account=SharedAccount(p['initial'],p['symbols']);ledger=account.physical
    controller=Controller() if controller is None else controller
    start,end=cfg['start_ms'],cfg['cutoff_ms'];events=[]
    before=np.flatnonzero(d['times']<=start)
    prices=d['close'][before[-1]].copy()
    priced=np.isfinite(prices)&(prices>0);prices=np.where(priced,prices,1.)
    ages=d['age_ms'][before[-1]].copy();last_trade=start-ages
    for i,t in enumerate(d['times']):
        if start<=t<=end:events.append((int(t),3,'mark',i))
    for s in signals:
        if start<=s['time']<end:events.append((int(s['time']),2,'signal',s))
    for row in tape:
        if start<=row['time']<=end:events.append((int(row['time']),1,'trade',row))
    for row in funding:
        if start<row['time']<=end:events.append((int(row['time']),0,'funding',row))
    # At identical milliseconds funding is processed before trades; collisions
    # are explicitly counted. They cannot affect these runs without holdings.
    collisions=[];pending=None;sigma=np.eye(len(prices))*p['risk']['hourly_sigma_floor']**2
    peak=p['initial'];halted=False;snapshots=[];orders=[];decisions=[];breaches=[];fallback=0;rounding=0.
    last_source=None
    def snapshot(t):
        eq=account.reconcile(prices)
        return dict(time=t,equity=eq+50000,user_equity=eq,qty=ledger.qty.tolist(),prices=prices.tolist(),
            notional=float(abs(ledger.qty)@prices),fees=ledger.fees,funding=ledger.funding,
            valuation_age_seconds=float(np.max(np.where(ledger.qty!=0,(t-last_trade)/1000,0))))
    for at,priority,kind,value in sorted(events,key=lambda x:x[:2]):
        if kind=='funding':
            j=p['symbols'].index(value['coin'][4:]);unit=value.get('cash_per_unit')
            if unit is None:unit=-prices[j]*value['rate'];fallback+=int(ledger.qty[j]!=0)
            cash=ledger.qty[j]*unit
            if cash>0:unit*=sc['receipt_multiplier']
            if value.get('rounding_unit_bound'):rounding+=abs(ledger.qty[j])*value['rounding_unit_bound']
            rates=np.zeros(len(prices));rates[j]=-unit/prices[j]
            account.charge_funding(at/1000,prices,rates,np.ones(len(prices),bool))
        elif kind=='trade':
            j=p['symbols'].index(value['coin'][4:]);prices[j]=value['vwap'];priced[j]=True;last_trade[j]=at
            if pending is not None and pending['time']<at-60000 and (at<=pending['expires'] or (sticky_exits and ledger.qty[j]*(pending['qty'][j]-ledger.qty[j])<0)):
                row=execute(account,pending,value,prices,sigma,p,sc)
                if row is not None:orders.append(row)
        elif kind=='mark':
            # The trade tape is more temporally precise. Candle closes fill
            # gaps in transaction coverage but are never used before completion.
            i=value;known=np.isfinite(d['close'][i])&(d['close'][i]>0)
            prices[known]=d['close'][i,known];priced|=known
            last_trade=np.maximum(last_trade,np.where(d['active'][i],at-d['age_ms'][i],last_trade))
        else:
            sigma=np.asarray(value['sigma']);last_source=value['source_time']
            equity=account.reconcile(prices)
            action=controller.propose(value['source_time'],prices,ledger.qty.copy(),equity,np.asarray(value['weights']),
                sigma,np.where(priced,value['volume'],0.),np.where(priced,value['trades'],0.),p,buffer=.2,halted=halted)
            if action is not None:
                qty=np.asarray(action['target_qty']);qty[~priced]=0.
                pending=dict(time=at,source=value['source_time']*1000,qty=qty,
                    expires=value['source_time']*1000+entry_lifetime*1000,source_prices=prices.copy(),economic_execution=sticky_exits)
                if economic_limits:
                    k=int(np.searchsorted(controller.times,value['source_time']))
                    mu=controller.forecast[k];carry=controller.carry[k]
                    limits={}
                    for j in range(len(prices)):
                        delta=qty[j]-ledger.qty[j];side=np.sign(delta)
                        if not side or ledger.qty[j]*delta<0:continue
                        terminal=prices[j]*(1+mu[j]);fc=-side*carry[j]*prices[j]
                        if fc>0:fc*=controller.receipt
                        fee=sc['fee_bps']/10000;exit_cost=(sc['fee_bps']+sc['impact_bps'])/10000
                        limits[j]=((terminal*(1-exit_cost)+fc-prices[j]*.0005)/(1+fee) if side>0 else (terminal*(1+exit_cost)-fc+prices[j]*.0005)/(1-fee))
                    pending['entry_limits']=limits
                decisions.append(dict(time=at,source=value['source_time'],prices=prices.tolist(),
                    prior_qty=ledger.qty.tolist(),target_qty=qty.tolist(),equity=equity))
        equity=account.reconcile(prices);peak=max(peak,equity)
        halted|=equity/peak-1<=-p['risk']['drawdown_halt']
        bad=risk_breaches(ledger.qty,equity,prices,sigma,p['risk'])
        if bad:breaches.append(dict(time=at,limits=bad))
        if halted or bad:
            # Continuous risk monitoring; do not overwrite an earlier active
            # reduction each millisecond and make it forever ineligible to fill.
            if pending is None or not pending.get('forced'):
                pending=dict(time=at,source=at,qty=np.zeros(len(prices)),expires=end,
                             source_prices=prices.copy(),forced=True,economic_execution=True)
        if kind=='mark':snapshots.append(snapshot(at))
    final=snapshot(end);daily=[dict(time=start//1000,equity=100000.)]
    daily += [dict(time=s['time']//1000,equity=s['equity']) for s in snapshots if s['time']>start and s['time']%86400000==43200000]
    score=evaluate(daily,trading_days=len({int(f['time'])//86400 for f in ledger.fills}))
    curve=np.array([100000.]+[s['equity'] for s in snapshots]);reserve=final['notional']*(sc['fee_bps']+sc['impact_bps'])/10000
    metrics=dict(net_pnl=final['equity']-100000.,return_percent=(final['equity']/100000.-1)*100,
        **score,fees=ledger.fees,funding=ledger.funding,terminal_cost_reserve=reserve,
        pnl_after_liquidation_reserve=final['equity']-100000.-reserve,
        fills=len(ledger.fills),mean_notional=float(np.mean([s['notional'] for s in snapshots])),
        max_drawdown=float(-np.min(curve/np.maximum.accumulate(curve)-1)),
        first_fill=min((f['time'] for f in ledger.fills),default=None),
        traded_symbols=sorted({f['symbol'] for f in ledger.fills}),
        final_qty=ledger.qty.tolist(),risk_breach_events=len(breaches),
        maximum_accounting_error=max(account.maximum_error,ledger.max_error),funding_proxy_events=fallback,
        estimated_funding_rounding_bound=rounding,final_valuation_age_seconds=final['valuation_age_seconds'],
        timing_pass=all(f['decision_time']+60<f['time'] for f in ledger.fills),
        pair_gate_nonzero=0,official_Q=False,orders_sent=0)
    assert metrics['timing_pass'] and metrics['maximum_accounting_error']<1e-6
    return dict(metrics=metrics,snapshots=snapshots,decisions=decisions,fills=ledger.fills,orders=orders,
        funding=ledger.funding_events,daily=daily,breaches=breaches)
