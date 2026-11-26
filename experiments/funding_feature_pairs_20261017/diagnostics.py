"""Post-selection cost attribution and like-for-like forecast diagnostics."""
import gzip,json
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
from experiments.adaptive_mean_pairs_20261017.reference_research_v2 import inputs
from experiments.risk_neutral_pairs_20261017.attribution import episode_attribution
from experiments.sector_pairs_20261017.diagnostics import funnel
from experiments.par_direct_forecast_20261017.data import ROOT,OUT as INPUT,save,sha,ts
from experiments.dense_history_pairs_20261017.research import load

HERE=Path(__file__).resolve().parent;ART=HERE/'artifacts'
PARENT=ROOT/'experiments/rolling_feature_pairs_20261017'
HOURLY=ROOT/'experiments/hourly_refresh_pairs_20261017'


def main():
    for rel,digest in json.loads((ART/'frozen.json').read_text())['sources'].items():
        assert sha(ROOT/rel)==digest,rel
    results=json.loads((ART/'results.json').read_text())
    cfg=json.loads((HOURLY/'reference_protocol.json').read_text())
    d,_,p=inputs(cfg);attribution=[]
    for row in results['rows']:
        if row['venue']!='reference' or '_funding_' not in row['arm']:continue
        key='-'.join(row[k] for k in ['venue','phase','arm','scenario'])
        with gzip.open(ART/'runs'/f'{key}.json.gz','rt') as f:run=json.load(f)
        audit=episode_attribution(run,d,cfg['scenarios'][row['scenario']])
        by_pair=defaultdict(Counter)
        for episode in audit['episodes']:
            bucket=by_pair['/'.join(episode['pair'])]
            for key in ['price_move','execution_cost','fees','funding','net_pnl','fills']:
                bucket[key]+=episode[key]
            bucket['episodes']+=1;bucket['closed']+=int(episode['closed'])
        attribution.append(dict(phase=row['phase'],arm=row['arm'],scenario=row['scenario'],
                                by_pair=dict(by_pair),**audit))
    save(ART/'attribution.json',dict(rows=attribution,not_used_for_selection=True,orders_sent=0))
    par=load(INPUT/'inputs.npz');pcfg=json.loads((HOURLY/'par_protocol.json').read_text());flow=[]
    for model,path in [('price_only',PARENT/'artifacts/par_direct_24_state.npz'),
                       ('funding_features',ART/'par_state.npz')]:
        flow.extend(funnel(par,load(path),pcfg,p,model))
    save(ART/'entry_funnel.json',dict(rows=flow,start=int(par['times'][0]),end=int(par['times'][-1]),
        scope='Pre-risk flat-account candle diagnostic, not realized PnL or order-book liquidity.',
        quantities_in_bps=True,not_account_returns=True,not_used_for_selection=True,orders_sent=0))
    arrays=load(ART/'reference_forecasts.npz')
    retained=load(PARENT/'artifacts/reference_issued_forecasts.npz')
    np.testing.assert_array_equal(arrays['times'],retained['times'])
    np.testing.assert_array_equal(arrays['target'],retained['target'])
    retained_prediction=retained['prediction_direct']
    pairs=json.loads((HERE/'protocol.json').read_text())['pairs'];forecasts=[]
    for phase in ['validation','comparison']:
        start=ts(cfg['validation_start'] if phase=='validation' else cfg['comparison_start'])
        end=ts(cfg['comparison_start'] if phase=='validation' else cfg['end'])
        time_mask=(arrays['times']>=start)&(arrays['maturity']<=end)&(arrays['times']%86400==0)
        for k,pair in enumerate(pairs):
            mask=time_mask&arrays['fresh'][:,k]&arrays['terminal_fresh'][:,k]
            mask &= np.isfinite(arrays['target'][:,k])&np.isfinite(arrays['predictions'][:,k])
            mask &= np.isfinite(retained_prediction[:,k])
            actual=arrays['target'][mask,k];unit=arrays['unit'][mask,k]
            row=dict(phase=phase,pair=pair,samples=int(mask.sum()),zero_rmse=float(np.sqrt(np.mean(actual**2))))
            for name,values in [('funding',arrays['predictions']),('price_only',retained_prediction)]:
                pred=values[mask,k]/unit
                row[name]=dict(normalized_rmse=float(np.sqrt(np.mean((actual-pred)**2))),
                    sign_accuracy=float(np.mean(np.sign(actual)==np.sign(pred))),
                    mean_abs_return_bps=float(np.mean(abs(values[mask,k]))*1e4))
            row['forecast_change_bps']=float(np.mean(abs(arrays['predictions'][mask,k]-retained_prediction[mask,k]))*1e4)
            forecasts.append(row)
    save(ART/'forecast_diagnostics.json',dict(rows=forecasts,common_samples=True,
        nonoverlapping_sources=True,not_used_for_selection=True))
    print('FUNNEL',[(r['model'],r['scenario'],r['counts']) for r in flow])
    for r in attribution:
        if r['arm'].startswith('first'):
            print('ATTRIBUTION',r['phase'],r['scenario'],dict(r['by_pair']))
    print('FORECASTS',forecasts)


if __name__=='__main__':main()
