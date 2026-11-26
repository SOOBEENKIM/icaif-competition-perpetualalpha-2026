"""Historical par event replay with separate funding/price ledgers and leg risk.

All arms share the same price, capacity, fee and delay model. Candles are actual
venue trades; fills and marks remain explicitly modeled, not exchange records.
"""
import copy
import numpy as np
from replay_lab.common_comparison.models import Market
from replay_lab.common_comparison.engine import Ledger,execute_intent,risk_breaches,risk_values,rounded
from replay_lab.ema_rebuild.engine import Controller
from experiments.cost_horizon_20261016.research import SecondController
from experiments.basis_carry_20261017.score import score


def pair_proposals(i,d,signals,cfg,p,sc,qty,equity,mode,baseline_raw):
    price=d['close'][i]; sigma=d['sigma'][i]; result=[]
    cost=2*(sc['fee_bps']+sc['impact_bps'])/10000
    for k,pair in enumerate(cfg['pairs']):
        ids=[p['symbols'].index(s) for s in pair]
        if np.any(qty[ids]) or np.any(abs(baseline_raw[ids])>1e-12): continue
        be=signals['beta'][i,k]; co=signals['correlation'][i,k]
        if not np.isfinite(be) or not .1<=be<=10 or co<cfg['minimum_correlation']: continue
        if np.any(d['volume'][i,ids]<=0) or np.any(d['trades'][i,ids]<=0): continue
        fund=signals['funding_forecast'][i,ids]
        if not np.isfinite(fund).all(): continue
        base=np.array([1.,-be])/(1+be)
        basis=price[ids]/d['oracle_proxy'][i,ids]-1
        reference=signals['ridge'][i,k] if 'ridge' in mode else 0.
        if not np.isfinite(reference): continue
        for direction in (-1,1):
            w=direction*base
            carry=-float(w@fund); carry=carry*cfg['positive_carry_haircut'] if carry>0 else carry
            convergence=-cfg['basis_convergence']*float(w@basis) if 'basis' in mode else 0.
            alpha=direction*reference+convergence+carry
            if alpha-cost<=cfg['entry_buffer_bps']/10000: continue
            gross=min(cfg['pair_max_gross'],float(np.min(sc['participation']*d['volume'][i,ids]*price[ids]/abs(w))))
            # Reserve both independent leg fill corners, not just hedged target.
            def allowed(g):
                for mask in ([1,0],[0,1],[1,1]):
                    q=qty.copy(); q[ids]+=g*w/price[ids]*mask
                    if risk_breaches(q,equity-g*cost,price,sigma,p['risk']): return False
                return True
            if not allowed(gross):
                lo,hi=0.,gross
                for _ in range(35):
                    mid=(lo+hi)/2
                    if allowed(mid): lo=mid
                    else: hi=mid
                gross=lo
            target=np.array([rounded(gross*z/price[j],p['execution']['decimals'][j]) for j,z in zip(ids,w)])
            if np.any(abs(target)*price[ids]<p['execution']['minimum_order_value']): continue
            actual_gross=float(abs(target)@price[ids])
            result.append(dict(pair=pair,ids=ids,target=target.tolist(),beta=float(be),correlation=float(co),
                direction=direction,expected_net_bps=(alpha-cost)*10000,gross=actual_gross,
                expected_net_cash=actual_gross*(alpha-cost),price_forecast=direction*reference+convergence,
                funding_forecast=carry,source_time=int(d['times'][i]),funding_known_through=int(d['times'][i]-3600)))
    return sorted(result,key=lambda r:(-r['expected_net_cash'],r['pair'],r['direction']))


def run(d,signals,cfg,protocol,arm,scenario,start,end):
    p=copy.deepcopy(protocol); sc=cfg['scenarios'][scenario];p['execution']['fee_bps']=sc['fee_bps']
    m=Market(d['times'],d['close'],d['volume'],d['trades'],d['funding'],np.isfinite(d['funding']),tuple(p['symbols']))
    first,last=m.exact(start),m.exact(end); ledger=Ledger(p['initial'],m.symbols)
    ctrl=Controller() if arm=='first_strategy' else SecondController(m.times,d['retention'])
    has_base=arm in ('first_strategy','second_strategy','second_plus_basis_carry')
    has_pair=arm.startswith('pairs_') or arm=='second_plus_basis_carry'
    raw=d['first'] if arm=='first_strategy' else d['second']
    pending=None; pair=None; events=[]; snapshots=[]; decisions=[];orders=[];breaches=[]
    peak=ledger.initial; mdd=0.;halted=False; price_move=execution_cost=0.;max_audit=0.;audit_eq=ledger.initial
    balance_hours=one_leg_hours=0;completed=aborted=0;denials=0
    for i in range(first,last+1):
        t=int(m.times[i]);prices=m.close[i];sigma=d['sigma'][i]
        prior_qty=ledger.qty.copy()
        if i>first:
            move=float(prior_qty@(prices-m.close[i-1]));price_move+=move;audit_eq+=move
            # Actual par settlement rate, reference price as explicitly declared
            # oracle proxy. Stress cuts receipts only and leaves payments intact.
            rates=m.funding[i].copy(); cash_by_leg=-prior_qty*d['oracle_proxy'][i]*rates
            rates=np.where(cash_by_leg>0,rates*sc['funding_receipt_multiplier'],rates)
            ledger.charge_funding(t,d['oracle_proxy'][i],rates,m.observed[i],0.)
            audit_eq-=float(prior_qty@(d['oracle_proxy'][i]*rates))
        old_count=len(ledger.fills)
        if pending is not None:
            pending['sigma']=sigma
            orders.extend(execute_intent(ledger,pending,m,i,p,sc))
            if pending.get('pair_enter'):
                ids=pair['ids'];got=ledger.qty[ids];want=np.array(pair['target'])
                ratios=np.divide(got,want,out=np.zeros(2),where=want!=0)
                if np.all(got==0):
                    events.append(dict(time=t,event='entry_unfilled',pair=pair['pair']));pair=None
                elif min(ratios)<=0 or abs(ratios[0]-ratios[1])/max(ratios)>cfg['pair_balance_tolerance']:
                    pair['closing']=True;pair['aborted']=True;aborted+=1
                    events.append(dict(time=t,event='entry_unbalanced_abort',pair=pair['pair'],ratios=ratios.tolist()))
                else:
                    pair['opened']=t;pair['closing']=False
                    events.append(dict(time=t,event='balanced_open',pair=pair['pair'],qty=got.tolist()))
            pending=None
        for f in ledger.fills[old_count:]:
            ec=f['signed_qty']*(f['price']-prices[f['asset']]);execution_cost+=ec;audit_eq-=ec+f['fee']
        equity=ledger.reconcile(prices);max_audit=max(max_audit,abs(equity-audit_eq))
        assert max_audit<1e-6
        if pair:
            ids=pair['ids'];q=ledger.qty[ids]
            if pair.get('closing') and np.all(q==0):
                if pair.get('opened') is not None: completed+=1
                events.append(dict(time=t,event='closed',pair=pair['pair'],aborted=pair.get('aborted',False)));pair=None
            elif pair:
                one_leg_hours+=int(np.count_nonzero(q)==1); balance_hours+=int(np.count_nonzero(q)==2)
        peak=max(peak,equity);mdd=min(mdd,equity/peak-1);halted|=equity/peak-1<=-p['risk']['drawdown_halt']
        breach=risk_breaches(ledger.qty,equity,prices,sigma,p['risk'])
        if breach: breaches.append(dict(time=t,constraints=breach))
        terminal=t>=end-cfg['terminal_reduce_hours']*3600
        if i<last:
            target=ledger.qty.copy();action=None;entry=None
            if terminal or halted or breach:
                target=np.zeros_like(target);action=dict(risk_forced=True)
                if pair:pair['closing']=True
            else:
                base_action=None
                if has_base:
                    base_qty=ledger.qty.copy()
                    if pair:base_qty[pair['ids']]=0.
                    base_action=ctrl.propose(t,prices,base_qty,equity,raw[i],sigma,m.volume[i],m.trades[i],p,buffer=.2)
                    if base_action:
                        target=np.array(base_action['target_qty']);action=base_action
                        if pair:
                            conflict=np.any(abs(target[pair['ids']])>1e-12)
                            if conflict:pair['closing']=True
                            target[pair['ids']]=ledger.qty[pair['ids']]
                if pair:
                    if pair.get('opened') is not None and t>=pair['opened']+cfg['holding_hours']*3600:pair['closing']=True
                    if pair.get('closing'):
                        target[pair['ids']]=0.;action=dict(risk_forced=True)
                elif has_pair and t%21600==0 and t+(cfg['holding_hours']+1)*3600<=end-cfg['terminal_reduce_hours']*3600:
                    reserved=raw[i] if has_base else np.zeros(10)
                    proposals=pair_proposals(i,d,signals,cfg,p,sc,ledger.qty,equity,arm,reserved)
                    if proposals:
                        for proposal in proposals:
                            q=target.copy();q[proposal['ids']]=proposal['target']
                            # Pending baseline increases share the same risk limit.
                            if risk_breaches(q,equity,prices,sigma,p['risk']):denials+=1;continue
                            pair=dict(proposal,opened=None,closing=False,aborted=False)
                            target=q;entry=proposal;action=dict(risk_forced=False);break
            if action is not None:
                # Never close-and-reverse in a single simulated intent.
                event=dict(decision_time=t,execution_time=t+3600,source_close=t,source_available=t,
                    target_qty=target.tolist(),risk_forced=bool(action.get('risk_forced',False)),
                    source_volume=m.volume[i].copy(),source_trades=m.trades[i].copy(),sigma=sigma)
                if entry:event['pair_enter']=True;events.append(dict(time=t,event='pair_intent',**entry))
                pending=event;decisions.append({k:v for k,v in event.items() if k!='sigma'})
        snapshots.append(dict(time=t,equity=equity,cash=ledger.cash,qty=ledger.qty.tolist(),fees=ledger.fees,
            funding=ledger.funding,realized=ledger.realized,**risk_values(ledger.qty,equity,prices,sigma)))
    days=[s for s in snapshots if s['time']%86400==43200];values=np.array([s['equity'] for s in days])
    returns=values[1:]/values[:-1]-1
    assert days[0]['time']==start and days[-1]['time']==end
    roi=equity/p['initial']-1;financial=score(roi,returns,trading_days=len({f['time']//86400 for f in ledger.fills}),start_sample=True)
    assert abs(np.prod(1+returns)-1-roi)<1e-10
    gross=float(abs(ledger.qty)@prices);reserve=gross*(sc['fee_bps']+sc['impact_bps'])/10000
    metrics=dict(net_pnl=equity-p['initial'],roi=roi,**financial,fees=ledger.fees,funding=ledger.funding,
        price_move_pnl=price_move,execution_cost=execution_cost,max_drawdown=-mdd,turnover=ledger.turnover,
        mean_gross=np.mean([s['gross'] for s in snapshots]),maximum_gross=max(s['gross'] for s in snapshots),
        fills=len(ledger.fills),orders=len(orders),completed_pairs=completed,aborted_pairs=aborted,
        one_leg_hours=one_leg_hours,both_leg_hours=balance_hours,residual_gross=gross,liquidation_cost_estimate=reserve,
        pnl_after_liquidation_reserve=equity-p['initial']-reserve,final_qty=ledger.qty.tolist(),
        risk_breach_hours=len(breaches),maximum_accounting_error=max(max_audit,ledger.max_error),risk_denials=denials)
    assert abs(metrics['net_pnl']-(price_move-execution_cost-ledger.fees+ledger.funding))<1e-6
    assert all(f['source_available']<=f['decision_time']<f['time'] for f in ledger.fills)
    return dict(arm=arm,scenario=scenario,start=start,end=end,metrics=metrics,snapshots=snapshots,
        daily=[dict(time=s['time'],return_=float(r)) for s,r in zip(days[1:],returns)],
        fills=ledger.fills,orders=orders,decisions=decisions,funding_events=ledger.funding_events,events=events,
        risk_breaches=breaches,orders_sent=0)
