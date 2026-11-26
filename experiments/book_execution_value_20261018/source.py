"""Source-book plans and features, independent of later candles or fills."""
import numpy as np
from experiments.basis_carry_20261017.probe import walk
from replay_lab.common_comparison.engine import rounded,risk_breaches


def gross_capacity(ids,beta,quotes,marks,sigma,p,sc,spec):
    weights=np.array([1.,-beta])/(1+beta);caps=[]
    # Capacity is direction-specific; the caller supplies ordered entry levels.
    for j,w,quote in zip(ids,weights,quotes):
        levels=quote['entry_levels'];side=quote['side'];best=float(levels[0]['px'])
        limit=best*(1+side*spec['visible_depth_band_bps']/10000)
        caps.append(sum(float(r['sz'])*sc['depth_fraction'] for r in levels if side*(float(r['px'])-limit)<=1e-10))
    gross=min(spec['gross_ceiling'],float(np.min(np.array(caps)*marks[ids]/abs(weights))))
    actual_weights=weights*quotes[0]['side']
    def safe(value):
        for mask in ([1,0],[0,1],[1,1]):
            qty=np.zeros(len(marks));qty[ids]=value*actual_weights/marks[ids]*mask
            if risk_breaches(qty,p['initial']-value*.01,marks,sigma,p['risk']):return False
        return True
    if not safe(gross):
        lo,hi=0.,gross
        for _ in range(32):
            mid=(lo+hi)/2
            if safe(mid):lo=mid
            else:hi=mid
        gross=lo
    return gross


def make_plan(pair,ids,beta,quotes,now,marks,sigma,p,sc,spec,direction,fraction,buffer):
    if any(quotes.get(s) is None for s in pair):return None
    sides=direction*np.array([1.,-1.]);prepared=[]
    for s,side in zip(pair,sides):
        quote=quotes[s]
        prepared.append(dict(entry_levels=quote['book']['levels'][1 if side>0 else 0],side=side))
    capacity=gross_capacity(ids,beta,prepared,marks,sigma,p,sc,spec)
    weights=direction*np.array([1.,-beta])/(1+beta)
    target=np.array([rounded(capacity*fraction*w/marks[j],p['execution']['decimals'][j]) for j,w in zip(ids,weights)])
    if np.any(abs(target)*marks[ids]<p['execution']['minimum_order_value']):return None
    ratio=abs(target[1]*marks[ids[1]]/(target[0]*marks[ids[0]]))
    if abs(ratio/beta-1)>.05:return None
    previews=[];reverse=[];limits={}
    for s,j,q in zip(pair,ids,target):
        levels=quotes[s]['book']['levels'];side=np.sign(q)
        entry=walk(levels[1 if side>0 else 0],abs(q),sc['depth_fraction'])
        exit_=walk(levels[0 if side>0 else 1],abs(q),sc['depth_fraction'])
        if entry is None or exit_ is None:return None
        previews.append(entry);reverse.append(exit_);limits[str(j)]=entry*(1+side*buffer/10000)
    # Exact source limit risk, including either leg filling without its hedge.
    for mask in ([1,0],[0,1],[1,1]):
        change=target*mask;qty=np.zeros(len(marks));qty[ids]=change
        px=np.array([limits[str(j)] for j in ids])
        equity=p['initial']-change@(px-marks[ids])-abs(change)@px*sc['fee_bps']/10000
        if risk_breaches(qty,equity,marks,sigma,p['risk']):return None
    return dict(pair=pair,ids=ids,target=target.tolist(),beta=beta,direction=direction,forecast_source=now,horizon=24,
        entry_limits=limits,source_prices=marks[ids].tolist(),entry_preview=previews,reverse_preview=reverse,
        source_capacity=capacity,size_fraction=fraction,limit_buffer_bps=buffer,intended_gross=float(abs(target)@marks[ids]))


def features(plan,quotes,now,marks,sigma,p,sc,pair_index):
    ids=plan['ids'];target=np.asarray(plan['target']);gross=plan['intended_gross'];values=[];names=[]
    def add(name,value):names.append(name);values.append(float(value))
    add('log_gross',np.log(gross/p['initial']));add('beta',plan['beta']);add('direction',plan['direction'])
    add('fee_bps',sc['fee_bps']);add('latency_seconds',sc['latency_ms']/1000);add('depth_fraction',sc['depth_fraction'])
    for k,(symbol,j,q) in enumerate(zip(plan['pair'],ids,target)):
        quote=quotes[symbol];levels=quote['book']['levels'];side=np.sign(q);mid=quote['mid']
        entry=levels[1 if side>0 else 0];reverse=levels[0 if side>0 else 1]
        top=float(entry[0]['px']);limit=plan['entry_limits'][str(j)]
        entry_depth=sum(float(r['sz'])*sc['depth_fraction'] for r in entry if side*(float(r['px'])-limit)<=1e-10)
        reverse_depth=sum(float(r['sz'])*sc['depth_fraction'] for r in reverse)
        add(f'{k}_spread_bps',2*quote['halfspread']/mid*10000)
        add(f'{k}_entry_depth_log',np.log1p(entry_depth*mid));add(f'{k}_reverse_depth_log',np.log1p(reverse_depth*mid))
        add(f'{k}_quantity_entry_depth',abs(q)/max(entry_depth,1e-12));add(f'{k}_quantity_reverse_depth',abs(q)/max(reverse_depth,1e-12))
        add(f'{k}_limit_headroom_bps',side*(limit/top-1)*10000)
        add(f'{k}_oracle_basis_bps',side*(mid/quote['oracle']-1)*10000)
        add(f'{k}_funding_bps',side*float(quote['ctx']['funding'])*10000)
        add(f'{k}_log_lots',np.log1p(abs(q)*10**p['execution']['decimals'][j]))
        add(f'{k}_book_receipt_age',(now*1000-quote['received']['book'])/1000)
        add(f'{k}_context_receipt_age',(now*1000-quote['received']['ctx'])/1000)
        add(f'{k}_server_age',(now*1000-quote['book']['time'])/1000)
        add(f'{k}_hourly_sigma',np.sqrt(max(0,sigma[j,j])))
        add(f'{k}_entry_slippage_bps',side*(plan['entry_preview'][k]/mid-1)*10000)
        add(f'{k}_reverse_cost_bps',side*(1-plan['reverse_preview'][k]/mid)*10000)
    add('correlation',sigma[ids[0],ids[1]]/np.sqrt(max(1e-24,sigma[ids[0],ids[0]]*sigma[ids[1],ids[1]])))
    for k in range(6):add(f'pair_{k}',pair_index==k)
    x=np.array(values)
    if not np.isfinite(x).all():raise ValueError('Nonfinite source feature')
    return x,names
