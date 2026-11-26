"""One frozen economic candidate and one diagnostic component ablation."""
import gzip,json
from pathlib import Path
import numpy as np
import pandas as pd
from replay_lab.common_comparison.models import Market
from replay_lab.ema_rebuild.engine import Controller
from strategy.agent import TAIL
from experiments.risk_preserving_20261016.prepare import sha,save,json_default
from .collect import ROOT,OUT
from .model import features,fit,predict,CalibratedController
from .engine import simulate

def ts(s):return int(np.datetime64(s,'s').astype('int64'))

def forecasting(m,forecast,pred,start,end):
    # Daily, non-overlapping 24h labels. No inflated hourly sample count.
    actual=[];raw=[];cal=[]
    for i,t in enumerate(m.times):
        if t<start or t%86400 or i+25>=len(m.times) or m.times[i+25]>end:continue
        for j in range(3):
            if min(m.volume[i+1,j],m.volume[i+25,j])<=0:continue
            actual.append(float(np.log(m.close[i+25,j]/m.close[i+1,j])));raw.append(float(forecast[i,j]));cal.append(float(pred[i,j]))
    y=np.array(actual)
    return {'asset_days':len(y),'calendar_days':len(y)//3,'cross_asset_observations_independent':False,
       'raw_chronos':{'mse':float(np.mean((y-raw)**2)),'correlation':float(np.corrcoef(y,raw)[0,1])},
       'calibrated':{'mse':float(np.mean((y-cal)**2)),'correlation':float(np.corrcoef(y,cal)[0,1])},
       'zero_return_mse':float(np.mean(y*y))}

def main():
    OUT.mkdir(exist_ok=True);runs=OUT/'runs';runs.mkdir(exist_ok=True)
    if (OUT/'research_results.json').exists():raise FileExistsError('Completed economic comparison exists')
    source=ROOT/'experiments/risk_preserving_20261016/artifacts'
    prepared=json.loads((source/'prepared.json').read_text())
    if sha(source/'inputs.npz')!=prepared['input_sha256'] or sha(source/'targets.npz')!=prepared['targets_sha256']:raise ValueError('Comparison inputs changed')
    p=json.loads((TAIL/'baseline.json').read_text())['protocol']
    with np.load(source/'inputs.npz') as z:m=Market(*(z[k].copy() for k in ['market_times','close','volume','trades','funding','observed']),tuple(p['symbols']))
    with np.load(source/'targets.npz') as z:baseline=z['incumbent'].copy();forecast=z['forecast'].copy();sigma=z['sigma'].copy()
    with np.load(source/'components.npz') as z:trend=z['trend_after_original_allocation'].copy()
    scale=json.loads((TAIL/'selected.json').read_text())['total_scale'];trend*=scale
    x,vol=features(m,forecast);state=fit(m,x,vol,ts('2026-08-01'));pred=predict(state,x,vol)
    assert state['last_label_time']<ts('2026-08-01')
    save(OUT/'fitted_calibrator.json',state)
    # Verify causal features on a shortened input and identical fixed forecasts.
    b=m.exact(ts('2026-09-22'))+1
    prefix=Market(*(getattr(m,k)[:b] for k in ['times','close','volume','trades','funding','observed']),m.symbols)
    xx,vv=features(prefix,forecast[:b]);np.testing.assert_allclose(predict(state,xx,vv),pred[:b],rtol=0,atol=0)
    past_funding=pd.DataFrame(m.funding).shift(1).rolling(24).mean().fillna(0).to_numpy()
    cost=2*(p['execution']['fee_bps']+p['scenarios']['base']['impact_bps'])/10000
    save(OUT/'frozen_economic_manifest.json',{'source_hashes':{str(q.relative_to(ROOT)):sha(q) for q in Path(__file__).parent.glob('*.py')},
       'protocol_sha256':sha(Path(__file__).with_name('protocol.json')),'inputs':prepared,'fit':state,'round_trip_threshold':cost,
       'causal_feature_prefix_passed':True,'historical_independence':False,'no_after_result_tuning':True})
    periods=[('validation','2026-08-01','2026-09-01'),('historical_evaluation','2026-09-01','2026-10-13T12:00:00'),
             ('first_half','2026-09-01','2026-09-22'),('second_half','2026-09-22','2026-10-13T12:00:00')]
    rows=[];checks=[];stats=[]
    for phase,start,end in periods:
        first,last=m.exact(ts(start)),m.exact(ts(end))
        stats.append({'phase':phase,**forecasting(m,forecast,pred,ts(start),ts(end))})
        for name in ['incumbent','trend_only_same_scale','calibrated_entry_v1']:
            for scenario in ['base','stress']:
                controller=CalibratedController(m.times,pred,past_funding,cost) if name=='calibrated_entry_v1' else Controller()
                w=trend if name=='trend_only_same_scale' else baseline
                result=simulate(m,w,sigma,p,first,last,scenario,controller);result.update(policy=name,phase=phase)
                if phase=='historical_evaluation' and name=='incumbent':
                    with gzip.open(source/'runs'/f'period0-incumbent-{scenario}.json.gz','rt') as f:old=json.load(f)
                    assert result['fills']==old['fills']
                    np.testing.assert_allclose([s['equity'] for s in result['snapshots']],[s['equity'] for s in old['snapshots']],rtol=0,atol=0)
                with gzip.open(runs/f'{phase}-{name}-{scenario}.json.gz','wt') as f:json.dump(result,f,allow_nan=False,default=json_default)
                rows.append({k:result[k] for k in ['phase','policy','scenario','start','end','metrics','gates']})
                print(phase,name,scenario,'ROI%',round(result['metrics']['roi']*100,6),'STD%',round(result['metrics']['daily_std_sample']*100,6),'fills',result['metrics']['fills'],flush=True)
        for scenario in ['base','stress']:
            r={v['policy']:v for v in rows if v['phase']==phase and v['scenario']==scenario};base=r['incumbent']['metrics']
            for name in ['trend_only_same_scale','calibrated_entry_v1']:
                met=r[name]['metrics'];gate={'positive_return':met['roi']>0,'return_improved':met['roi']>base['roi'],
                   'daily_std_no_worse':met['daily_std_sample']<=base['daily_std_sample'],'mdd_no_worse':met['max_drawdown_magnitude']<=base['max_drawdown_magnitude'],
                   'engineering':all(r[name]['gates'].values())}
                checks.append({'phase':phase,'scenario':scenario,'policy':name,'checks':gate,'passes_all':all(gate.values())})
    save(OUT/'research_results.json',{'rows':rows,'checks':checks,'forecast_quality':stats,'baseline_execution_equivalent':True,
          'historical_independence':False,'live_orders_sent':0,'production_changed':False,
          'decision':'Historical research only; prospective par evidence required even for a positive result.'})

if __name__=='__main__':main()
