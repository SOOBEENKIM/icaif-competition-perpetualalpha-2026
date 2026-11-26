"""Conditional PAS candle replay. Records limits of historical execution evidence."""
import json,gzip,math
import numpy as np
from .data import OUT,load
from .policy import intraday_targets, sticky_conflict_guard

DECIMALS=[5,4,2,3,3,3,2,4,1,2]

def simulate(cfg,d,targets,*,impact_bps,participation):
    times=d['times'];width=len(cfg['symbols']);qty=np.zeros(width);cash=50000.;fees=0.;funding=0.;turnover=0.;events=[];snapshots=[];daily=[{'time':cfg['start_ms'],'equity':100000.}]
    pending=None;target_idx=0;fund_idx=0;funds=d['funding_events'];lastprices=np.nan_to_num(d['close'][0],nan=0.)
    unfilled=0;short_entries=0;long_entries=0;reductions=0;active_days=set();holding_age=[]
    for i,t in enumerate(times):
        if t<cfg['start_ms']:lastprices=np.nan_to_num(d['close'][i],nan=0.);continue
        prices=np.where(np.isfinite(d['close'][i]),d['close'][i],lastprices)
        # Funding settlements inside this minute precede the modeled end-of-minute fills.
        while fund_idx<len(funds) and funds[fund_idx]['time']<=t:
            f=funds[fund_idx];pay=-qty[f['j']]*lastprices[f['j']]*f['rate'];cash+=pay;funding+=pay;fund_idx+=1
        if pending is not None and pending['time']<=t-60000:
            target=pending['qty'];dif=target-qty
            for j in sorted(range(width),key=lambda j:not(qty[j]*dif[j]<0)):
                delta=target[j]-qty[j]
                if abs(delta)<1e-10:continue
                reduce=qty[j]*delta<0
                if t>pending['expires'] and not reduce:continue
                if not d['active'][i,j]:continue
                request=min(abs(delta),abs(qty[j])) if reduce else abs(delta)
                amount=min(request,d['volume'][i,j]*participation)
                factor=10**DECIMALS[j];amount=math.floor(amount*factor+1e-8)/factor
                side=1 if delta>0 else -1
                px=d['high'][i,j]*(1+impact_bps/10000) if side>0 else d['low'][i,j]*(1-impact_bps/10000)
                if amount<=0 or amount*px<10 and not(reduce and abs(amount-abs(qty[j]))<1e-10):continue
                # Same strategy capital, 2000 gross sizing ceiling, original larger venue risk caps.
                gross_other=float(np.sum(abs(qty)*prices)-abs(qty[j])*prices[j])
                if not reduce:
                    amount=min(amount,max(0.,2000-gross_other-abs(qty[j])*px)/px)
                    amount=math.floor(amount*factor+1e-8)/factor
                    if amount*px<10:continue
                fee=amount*px*.0009;old=qty[j];qty[j]+=side*amount;cash-=side*amount*px+fee;fees+=fee;turnover+=amount*px
                if reduce:reductions+=1
                elif side>0:long_entries+=1
                else:short_entries+=1
                assert old*qty[j]>=-1e-9, 'Cannot reverse through zero in one fill'
                active_days.add(t//86400000)
                events.append({'time':int(t),'decision_time':int(pending['time']),'symbol':cfg['symbols'][j],'asset':j,'signed_qty':side*amount,'price':float(px),'fee':fee,'reduce_only':bool(reduce),'quantity_after':float(qty[j]),'observed_minute_volume':float(d['volume'][i,j])})
        equity=cash+float(qty@prices)+50000
        assert equity>50000 and np.isfinite(equity)
        if t%86400000==43200000 and t>cfg['start_ms']:daily.append({'time':int(t),'equity':equity})
        # New target availability is after model completion; never fill the decision candle.
        while target_idx<len(targets) and targets[target_idx]['time']<=t:
            event=targets[target_idx];weights=np.asarray(event['weights'],float)
            notionals=weights*(equity-50000) if event['units']=='weights' else weights*2000
            gross=abs(notionals).sum();notionals/=max(1.,gross/2000)
            target=np.divide(notionals,prices,out=np.zeros(width),where=prices>0)
            same=target*qty>0;relative=abs(target-qty)/np.maximum(np.maximum(abs(target),abs(qty)),1e-12)
            target=np.where(same&(relative<=.2),qty,target)
            target=np.array([math.trunc(q*10**v)/10**v for q,v in zip(target,DECIMALS)])
            if pending is not None:unfilled+=int(np.count_nonzero(abs(pending['qty']-qty)>1e-10))
            pending={'time':int(t),'expires':int(t)+900000,'qty':target};target_idx+=1
        gross=float(abs(qty)@prices);snapshots.append({'time':int(t),'equity':equity,'gross':gross,'net':float(qty@prices),'qty':qty.tolist(),'valuation_max_age_minutes':float(np.max(np.where(abs(qty)>1e-12,d['age_ms'][i]/60000,0)))})
        lastprices=prices
    terminal_reserve=float(abs(qty)@lastprices)*(.0009+impact_bps/10000)
    mark_pnl=snapshots[-1]['equity']-100000
    independent=50000+sum(-f['signed_qty']*f['price']-f['fee'] for f in events)+funding+float(qty@lastprices)+50000
    assert abs(independent-snapshots[-1]['equity'])<1e-7
    assert all(f['decision_time']<f['time'] for f in events)
    curve=np.array([s['equity'] for s in snapshots]);drawdown=float(np.min(curve/np.maximum.accumulate(curve)-1))
    metrics={'pnl':mark_pnl,'return_percent':mark_pnl/100000*100,'pnl_after_terminal_cost_reserve':mark_pnl-terminal_reserve,'fees':fees,'funding_proxy':funding,'turnover':turnover,'fills':len(events),'long_entry_fills':long_entries,'short_entry_fills':short_entries,'reduction_fills':reductions,'mean_gross':float(np.mean([s['gross'] for s in snapshots])),'max_gross':max(s['gross'] for s in snapshots),'max_drawdown_percent':100*drawdown,'valuation_age_minutes':snapshots[-1]['valuation_max_age_minutes'],'unfilled_replaced_asset_intents':unfilled,'final_qty':qty.tolist(),'daily_returns':len(daily)-1,'official_volatility':None,'official_Q':None,'accounting_error':abs(independent-snapshots[-1]['equity'])}
    return {'metrics':metrics,'fills':events,'snapshots':snapshots,'daily':daily}

def main():
    cfg,d=load();archive=np.load(OUT/'reference_targets.npz',allow_pickle=False);pub=json.loads((OUT/'publication_times.json').read_text())
    variants={}
    for name in ('incumbent_published','all_markets_long_only','all_markets_symmetric_daily'):
        events=[]
        for i,t in enumerate(archive['times']):
            at=int(t)*1000
            if cfg['start_ms']<=at<cfg['cutoff_ms'] and int(t)%21600==0:
                if str(at) not in pub:raise ValueError('Missing actual source publication time')
                events.append({'time':pub[str(at)],'weights':archive[name][i].tolist(),'units':'weights'})
        variants[name]=events
    w,reasons=intraday_targets(d)
    variants['pas_intraday_signed']=[{'time':int(t)+60000,'weights':w[i].tolist(),'units':'gross_fraction'} for i,t in enumerate(d['times']) if cfg['start_ms']<=t<cfg['cutoff_ms'] and t%900000==0]
    variants['symmetric_daily_pas_veto']=sticky_conflict_guard(
        variants['all_markets_symmetric_daily'],variants['pas_intraday_signed'])
    (OUT/'intraday_decisions.json').write_text(json.dumps(reasons,indent=2));rows=[]
    for name,events in variants.items():
        for scenario,impact,participation in [('base',10,.01),('stress',20,.005)]:
            result=simulate(cfg,d,events,impact_bps=impact,participation=participation)
            path=OUT/f'{name}_{scenario}.json.gz'
            with gzip.open(path,'wt') as f:json.dump(result,f)
            row={'candidate':name,'scenario':scenario,**result['metrics']};rows.append(row);print(json.dumps(row),flush=True)
    (OUT/'comparison.json').write_text(json.dumps(rows,indent=2))
if __name__=='__main__':main()
