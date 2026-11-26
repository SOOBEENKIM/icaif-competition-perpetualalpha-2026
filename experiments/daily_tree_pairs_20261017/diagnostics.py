"""Matched forecast errors, paid episode attribution and executable entry funnel."""
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
    for path,digest in json.loads((ART/'frozen.json').read_text())['sources'].items():assert sha(ROOT/path)==digest,path
    result=json.loads((ART/'results.json').read_text());cfg=json.loads((HOURLY/'reference_protocol.json').read_text())
    d,_,p=inputs(cfg);attribution=[]
    for row in result['rows']:
        if row['venue']!='reference' or not row['arm'].startswith('first_dense_hourlyhedge_'):continue
        kind=row['arm'].split('_')[-2]
        if kind not in ['direct','tree','mix']:continue
        directory=PARENT/'artifacts' if kind=='direct' else ART
        path=directory/'runs'/('-'.join(row[k] for k in ['venue','phase','arm','scenario'])+'.json.gz')
        with gzip.open(path,'rt') as f:run=json.load(f)
        audit=episode_attribution(run,d,cfg['scenarios'][row['scenario']]);by_pair=defaultdict(Counter)
        for e in audit['episodes']:
            bucket=by_pair['/'.join(e['pair'])]
            for k in ['price_move','execution_cost','fees','funding','net_pnl','fills']:bucket[k]+=e[k]
            bucket['episodes']+=1;bucket['closed']+=int(e['closed'])
        attribution.append(dict(phase=row['phase'],scenario=row['scenario'],kind=kind,
            by_pair=dict(by_pair),events=dict(Counter(e['event'] for e in run['events'])),
            exit_reasons=dict(Counter(e['reason'] for e in run['events'] if e['event']=='exit_requested')),**audit))
    save(ART/'attribution.json',dict(rows=attribution,not_used_for_selection=True,orders_sent=0))
    arrays=load(ART/'reference_forecasts.npz');pairs=json.loads((HERE/'protocol.json').read_text())['pairs'];forecasts=[]
    for phase in ['validation','comparison']:
        start=ts(cfg['validation_start'] if phase=='validation' else cfg['comparison_start'])
        end=ts(cfg['comparison_start'] if phase=='validation' else cfg['end'])
        eligible=(arrays['times']>=start)&(arrays['maturity']<=end)&(arrays['times']%86400==0)
        for k,pair in enumerate(pairs):
            mask=eligible&arrays['fresh'][:,k]&arrays['terminal_fresh'][:,k]&np.isfinite(arrays['target'][:,k])
            for kind in ['direct','tree','mix']:mask &= np.isfinite(arrays[f'prediction_{kind}'][:,k])
            actual=arrays['target'][mask,k];unit=arrays['unit'][mask,k]
            for kind in ['direct','tree','mix']:
                values=arrays[f'prediction_{kind}'][mask,k];pred=values/unit
                forecasts.append(dict(phase=phase,pair=pair,kind=kind,samples=int(mask.sum()),
                    normalized_rmse=float(np.sqrt(np.mean((actual-pred)**2))),
                    zero_rmse=float(np.sqrt(np.mean(actual**2))),sign_accuracy=float(np.mean(np.sign(actual)==np.sign(pred))),
                    mean_abs_forecast_bp=float(np.mean(abs(values))*1e4)))
    save(ART/'forecast_diagnostics.json',dict(rows=forecasts,common_samples=True,
        nonoverlapping_sources=True,not_used_for_selection=True))
    par=load(INPUT/'inputs.npz');pcfg=json.loads((HOURLY/'par_protocol.json').read_text());flow=[]
    for kind in ['direct','tree','mix']:
        flow.extend(funnel(par,load(ART/f'par_{kind}_24_state.npz'),pcfg,p,kind))
    save(ART/'entry_funnel.json',dict(rows=flow,start=int(par['times'][0]),end=int(par['times'][-1]),
        scope='Pre-risk flat-account candle diagnostic, not realized PnL or actual order-book capacity.',
        quantities_in_bps=True,not_account_returns=True,not_used_for_selection=True,orders_sent=0))
    model_hashes={}
    for venue in ['reference','par']:
        for r in json.loads((ART/f'{venue}_fit_audit.json').read_text()):
            path=ART/'models'/venue/r['model_file'];digest=sha(path)
            assert digest==r['tree_sha256'],str(path)
            model_hashes[str(path.relative_to(ROOT))]=digest
    save(ART/'trained_model_manifest.json',dict(sources=model_hashes,models=len(model_hashes),orders_sent=0))
    for r in forecasts:print('FORECAST',r)
    print('FUNNEL',[(r['model'],r['scenario'],r['counts']) for r in flow])
    for r in attribution:
        if r['kind']!='direct':print('ATTRIBUTION',r['kind'],r['phase'],r['scenario'],r['by_pair'])


if __name__=='__main__':main()
