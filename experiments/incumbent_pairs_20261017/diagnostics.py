"""Post-selection diagnosis; no policy or frozen result is changed."""
from collections import Counter
import gzip
import json
from pathlib import Path
import numpy as np
from experiments.cost_horizon_20261016.research import data,ts,save
from experiments.pairs_forecast_20261017.model import labels
from .model import features

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];OUT=HERE/'artifacts'


def main():
    _,a,p,m=data();cfg=json.loads((HERE/'protocol.json').read_text());d=features(m,a,cfg)
    z=np.load(OUT/'models/predictions.npz');pred=z['predictions'];y,maturity=labels(m,d,24)
    realized=y*d['vol']*np.sqrt(24);counts=[];quality=[];attr=[]
    def number(x):return float(x) if np.isfinite(x) else None
    for phase in ('selection','comparison'):
        start,end=map(ts,cfg[phase]);scheduled=(m.times>=start)&(m.times<end)&(m.times%21600==0)
        for k,ids in enumerate(d['pairs']):
            wa,wb=a['candidate'][:,ids].T
            opposite=wa*wb<0;fresh=d['fresh'][:,k]
            correlated=d['x'][:,k,d['names'].index('correlation_168')]>=cfg['strategy']['minimum_correlation']
            expected=np.sign(wa)*pred[:,k];mask=scheduled&opposite&fresh&correlated
            record=dict(phase=phase,pair=cfg['pairs'][k],scheduled=int(scheduled.sum()),opposed=int((scheduled&opposite).sum()),
                        opposed_fresh_correlated=int(mask.sum()),valid_forecast=int((mask&np.isfinite(expected)).sum()),
                        clears_base_cost=int((mask&(expected>.004)).sum()),clears_stress_cost=int((mask&(expected>.006)).sum()),
                        largest_expected_bps=number(np.nanmax(expected[mask])*10000) if mask.any() else None)
            counts.append(record)
            valid=mask&(maturity<=end)&np.isfinite(realized[:,k])
            actual=np.sign(wa[valid])*realized[valid,k]
            quality.append(dict(phase=phase,pair=cfg['pairs'][k],opposed_future_labels=int(valid.sum()),
                mean_24h_net_funding_bps=number(np.mean(actual)*10000) if len(actual) else None,
                mean_after_base_cost_bps=number(np.mean(actual)*10000-38) if len(actual) else None,
                dependent_overlapping_observations=True,not_a_trade_backtest=True))
        for scenario in ('base','execution_stress'):
            selected=json.loads((OUT/'selection.json').read_text())['selected']
            for name in ('first','second',selected):
                with gzip.open(OUT/'runs'/f'{phase}-{name}-{scenario}.json.gz','rt') as f:r=json.load(f)
                ss=r['snapshots'];indexes=np.array([m.exact(s['time']) for s in ss]);qty=np.array([s['qty'] for s in ss])
                price=(qty[:-1]*np.diff(m.close[indexes],axis=0)).sum(0)
                impact=np.zeros(len(m.symbols));fees=impact.copy();fund=impact.copy()
                for fill in r['fills']:
                    j=fill['asset'];impact[j]+=fill['signed_qty']*(fill['price']-m.close[m.exact(fill['time']),j]);fees[j]+=fill['fee']
                for event in r['funding_events']:
                    fund-=np.array(event['qty'])*np.array(event['prices'])*np.array(event['rates'])
                net=price-impact-fees+fund
                assert abs(net.sum()-r['metrics']['net_pnl'])<1e-6
                attr.append(dict(phase=phase,scenario=scenario,policy=name,net_pnl=float(net.sum()),
                    gross_price_pnl=float(price.sum()),impact=float(impact.sum()),fees=float(fees.sum()),funding=float(fund.sum()),
                    assets=[dict(symbol=s,net_pnl=float(net[j]),gross_price_pnl=float(price[j]),impact=float(impact[j]),
                                 fees=float(fees[j]),funding=float(fund[j])) for j,s in enumerate(m.symbols)],
                    event_counts=dict(Counter(e['event'] for e in r.get('events',[])))))
    save(OUT/'diagnostics.json',dict(funnel=counts,label_diagnostics=quality,attribution=attr,
        generated_after_selection=True,thresholds_not_changed=True,no_additional_strategy_selected=True))
    print(json.dumps(dict(funnel=counts,label_diagnostics=quality),indent=2))


if __name__=='__main__':main()
