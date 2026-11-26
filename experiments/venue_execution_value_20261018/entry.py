"""Same source-only order plan for learning, matched price control and new value."""
import numpy as np
from experiments.decision_gate_pairs_20261017.entry import proposals as original
from replay_lab.common_comparison.engine import rounded,risk_breaches
from replay_lab.pullback.broker import execution_price
from .model import features,predict,adjusted_net


def plan(i,k,direction,h,d,de,cfg,p,sc,risk_qty,equity):
    spec=cfg['venue_cash'];ids=list(map(int,de['pairs'][k]));beta=float(de['beta'][i,k])
    if not de['fresh'][i,k] or not np.isfinite(de['x'][i,k]).all() or not .1<=beta<=10 or de['correlation'][i,k]<spec['minimum_correlation']:return None
    prices=d['par_close'][i];sigma=d['sigma'][i];w=direction*np.asarray([1.,-beta])/(1+beta)
    gross=min(spec['pair_max_gross'],float(np.min(sc['participation']*d['par_volume'][i,ids]*prices[ids]/abs(w))))
    cost=2*(sc['fee_bps']+sc['impact_bps'])/10000
    def safe(g):
        for mask in ([1,0],[0,1],[1,1]):
            q=risk_qty.copy();q[ids]+=g*w/prices[ids]*mask
            if risk_breaches(q,equity-g*cost,prices,sigma,p['risk']):return False
        return True
    if not safe(gross):
        low,high=0.,gross
        for _ in range(35):
            middle=(low+high)/2
            if safe(middle):low=middle
            else:high=middle
        gross=low
    q=np.asarray([rounded(gross*weight/prices[j],p['execution']['decimals'][j]) for j,weight in zip(ids,w)])
    if np.any(abs(q)*prices[ids]<p['execution']['minimum_order_value']):return None
    actual_beta=abs(q[1])*prices[ids[1]]/(abs(q[0])*prices[ids[0]])
    if abs(actual_beta/beta-1)>cfg['pair_balance_tolerance']:return None
    limits={}
    for j,quantity in zip(ids,q):
        side=np.sign(quantity);preview=execution_price(prices[j],side,sc['impact_bps'],p['execution']['decimals'][j])
        limits[str(j)]=float(preview*(1+side*spec['entry_limit_buffer_bps']/10000))
    now=int(d['times'][i])
    return dict(dense=True,venue_cash=True,pair=spec['pairs'][k],pair_index=k,ids=ids,target=q.tolist(),horizon=h,
        beta=beta,direction=direction,entry_vol=float(de['vol'][i,k]),entry_limits=limits,
        intent_time=now,forecast_source=now,expected_net_cash=0.,expected_cash=0.,funding_known_through=now//3600*3600-3600)


def proposals(i,d,de,pred,h,cfg,p,sc,qty,equity,reserved,directions,arm,*,risk_qty=None):
    if 'venue_cash' not in cfg:return original(i,d,de,pred,h,cfg,p,sc,qty,equity,reserved,directions,arm,risk_qty=risk_qty)
    spec=cfg['venue_cash'];risk_qty=qty if risk_qty is None else risk_qty;now=int(d['times'][i]);sample=cfg.get('venue_label')
    if sample is not None and now!=sample['source']:return []
    results=[]
    for k,ids in enumerate(de['pairs']):
        if not directions[k] or np.any(qty[ids]) or np.any(abs(reserved[ids])>1e-12):continue
        if sample is not None and k!=sample['pair_index']:continue
        for direction in (-1,1):
            if sample is not None and direction!=sample['direction']:continue
            candidate=plan(i,k,direction,h,d,de,cfg,p,sc,risk_qty,equity)
            if candidate is None:continue
            if sample is not None:results.append(candidate);continue
            x,gross=features(candidate,d,de,i,p['initial'])
            if cfg.get('venue_model_kind')=='price':
                mu=cfg['venue_price_predictions'][f'ridge_{h}'][i,k]
                if not np.isfinite(mu):continue
                weights=np.array([1.,-candidate['beta']])/(1+candidate['beta'])*direction
                carry=-float(weights@d['known_hourly_funding'][i,ids])*h
                if carry>0:carry*=.5
                net=direction*mu+carry-2*(sc['fee_bps']+sc['impact_bps'])/10000
                fitted=None
            else:
                model=cfg['venue_models'].get((h,now))
                if model is None:continue
                net=adjusted_net(predict(model,x,now),cfg['scenarios']['base'],sc);fitted=model['fit_time']
            if net<=spec['entry_buffer_bps']/10000:continue
            results.append(dict(candidate,expected_net_cash=net*gross,expected_cash=net*gross,
                                expected_net_bps=net*10000,value_fit_time=fitted))
    return sorted(results,key=lambda r:(-r['expected_cash'],r['pair']))
