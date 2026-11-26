"""Read-only fresh public quotes; model expectations are never recorded as fills."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
import json
import time
import urllib.request
from pathlib import Path
import numpy as np
from experiments.paired_venue_20261017.forward import published_pairs,current_snapshots
from experiments.paired_venue_20261017.engine import quote_pair
from experiments.cost_horizon_20261016.research import save
from .model import HOUR,panel,features,ridge_predict,ar1_forecast

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];OUT=HERE/'artifacts'


def request(payload):
    if payload['type'] not in ('fundingHistory','metaAndAssetCtxs','l2Book'):raise ValueError('Read-only endpoint allowlist')
    start=time.time()
    req=urllib.request.Request('https://api.hyperliquid-testnet.xyz/info',data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=25) as stream:response=json.load(stream)
    return dict(request=payload,sent_at=start,received_at=time.time(),response=response)


def funding_value(qty,oracle,forecast_bps,haircut):
    value=-float(np.array(qty)@(np.array(oracle)*np.array(forecast_bps)/10000))
    return value*haircut if value>0 else value


def current_forecasts(p,cfg,models,now):
    i=len(p['times'])-1;x=features(p);fit_at=now//(24*HOUR)*(24*HOUR)+1000
    forecasts={}
    for h in cfg['horizons_hours']:
        fitted=next((r for r in models if r.get('fit_time')==fit_at and r['horizon']==h and 'coef' in r),None)
        f={name:np.full(len(p['symbols']),np.nan) for name in cfg['models']}
        for j,symbol in enumerate(p['symbols']):
            if not np.isfinite(x[i,j]).all() or not 0<=now-p['seen'][i,j]<=HOUR+60000:continue
            f['zero'][j]=0.;f['persistence'][j]=p['rates'][i,j]*h;f['mean24'][j]=x[i,j,2]*h
            f['ar1'][j]=ar1_forecast(p,i,j,h,fit_at,cfg);f['ridge'][j]=ridge_predict(fitted,x[i,j],h)
        forecasts[h]=f
    return forecasts


def main():
    cfg=json.loads((HERE/'protocol.json').read_text());results=json.loads((OUT/'results.json').read_text())
    models=json.loads((OUT/'models.json').read_text())
    old={s:json.loads((ROOT/cfg['source']/f'funding_{s}.json').read_text())['rows'] for s in cfg['symbols']}
    end=int(time.time()*1000)
    payloads=[dict(type='fundingHistory',coin='par:'+s,startTime=max(r['time'] for r in old[s])+1,endTime=end) for s in cfg['symbols']]
    with ThreadPoolExecutor(max_workers=3) as pool:rate_receipts=list(pool.map(request,payloads))
    for symbol,receipt in zip(cfg['symbols'],rate_receipts):
        if not isinstance(receipt['response'],list) or len(receipt['response'])>=500:raise ValueError('Funding history incomplete')
        old[symbol]+=receipt['response']
    p=panel(old,cfg['symbols'])
    frame,obj=published_pairs()
    market_cfg=json.loads((ROOT/'experiments/paired_venue_20261017/protocol.json').read_text())
    coins=sorted({c for pair in obj['frame']['paired_metrics'] for c in pair['coins']})
    payloads=[dict(type='metaAndAssetCtxs',dex='par')]+[dict(type='l2Book',coin=c) for c in coins]
    with ThreadPoolExecutor(max_workers=7) as pool:receipts=list(pool.map(request,payloads))
    meta,ctx=receipts[0]['response'];metadata={r['name']:r for r in meta['universe']}
    contexts={m['name']:dict(received_ms=int(receipts[0]['received_at']*1000),ctx=c) for m,c in zip(meta['universe'],ctx)}
    books={r['response']['coin']:dict(received_ms=int(r['received_at']*1000),data=r['response']) for r in receipts[1:]}
    now=int(time.time()*1000);snapshots=current_snapshots(now,books,contexts,market_cfg)
    if not 0<=now/1000-frame.source_time<=market_cfg['maximum_signal_age_seconds']:raise ValueError('Published hedge input stale')
    forecasts=current_forecasts(p,cfg,models,now);rows=[]
    for pair in obj['frame']['paired_metrics']:
        if pair['correlation']<market_cfg['minimum_correlation'] or any(c not in snapshots for c in pair['coins']):continue
        symbols=[c.split(':')[1] for c in pair['coins']];ids=[cfg['symbols'].index(s) for s in symbols]
        oracle=[snapshots[c]['oracle'] for c in pair['coins']]
        for h in cfg['horizons_hours']:
            selected=results['selected'][str(h)]['model']
            allowance=next(r for r in results['error_allowances'] if r['horizon']==h and r['pair']==symbols)
            for scenario,sc in market_cfg['scenarios'].items():
                choices=[]
                for direction in (-1,1):
                    q=quote_pair(pair['coins'],snapshots,metadata,pair['beta'],direction,h,market_cfg,sc)
                    if q is None:continue
                    choices.append(dict(scenario=scenario,model='published_current_persistence',selected=False,proposal=q,
                        input='Then-published activeAssetCtx funding extrapolated for the horizon, positive carry halved',
                        error_allowance_bps=None,adjusted_net_bps=None,forecast_validation_passed=False))
                    for kind,pred in forecasts[h].items():
                        if not np.isfinite(pred[ids]).all():continue
                        carry=funding_value(q['target_qty'],oracle,pred[ids],market_cfg['positive_funding_haircut'])
                        revised=dict(q,predicted_funding=carry,predicted_net_pnl=q['predicted_price_pnl']+carry-q['predicted_fees'])
                        revised['predicted_net_bps']=revised['predicted_net_pnl']/q['gross_notional']*10000
                        error=allowance['diagnostic_error_allowance_bps'] if kind==selected else None
                        error_cash=(float(abs(np.array(q['target_qty']))@oracle)*error/10000) if error is not None else None
                        adjusted=(revised['predicted_net_pnl']-error_cash)/q['gross_notional']*10000 if error_cash is not None else None
                        check=next(r for r in results['checks'] if r['horizon']==h)
                        passed=kind==selected and all(check[k] for k in ('beats_persistence','beats_zero','adequate_nonoverlapping_blocks'))
                        choices.append(dict(scenario=scenario,model=kind,selected=kind==selected,proposal=revised,
                            input='Known settled funding history; this is different from the activeAssetCtx forecast',
                            error_allowance_bps=error,adjusted_net_bps=adjusted,forecast_validation_passed=passed))
                for kind in ('published_current_persistence',*cfg['models']):
                    candidates=[r for r in choices if r['model']==kind]
                    if candidates:rows.append(max(candidates,key=lambda r:r['proposal']['predicted_net_bps']))
    dest=OUT/('current_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'.json')
    save(dest,dict(at_ms=now,rate_receipts=rate_receipts,market_receipts=receipts,rows=rows,
        signal_time=frame.source_time,fresh_coins=list(snapshots),latest_settlements={s:int(p['seen'][-1,j]) for j,s in enumerate(cfg['symbols'])},
        no_orders_sent=True,actual_strategy_PnL=None,actual_strategy_Q=None,no_automatic_promotion=True,
        limits='Basis convergence and future exit depth remain assumptions. Error allowance is a diagnostic projection, not a statistical guarantee. No past quotes are fabricated.'))
    positives=[dict(pair=r['proposal']['coins'],horizon=r['proposal']['horizon_hours'],scenario=r['scenario'],model=r['model'],
         net_bps=r['proposal']['predicted_net_bps'],net_dollars=r['proposal']['predicted_net_pnl'],selected=r['selected'])
         for r in rows if r['proposal']['predicted_net_bps']>market_cfg['entry_buffer_bps']]
    print(json.dumps(dict(file=str(dest),quotes=len(rows),positive_expected_edges=positives),indent=2))


if __name__=='__main__':main()
