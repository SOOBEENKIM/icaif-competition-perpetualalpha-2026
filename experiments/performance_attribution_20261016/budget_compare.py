import gzip,json
from pathlib import Path
import numpy as np
from replay_lab.common_comparison.models import Market
from strategy.agent import TAIL
from experiments.risk_preserving_20261016.prepare import sha,save,json_default
from experiments.risk_preserving_20261016.replay import simulate
from .collect import ROOT,OUT
from .research import ts

def main():
    if (OUT/'budget_results.json').exists():raise FileExistsError('Completed comparison exists')
    source=ROOT/'experiments/risk_preserving_20261016/artifacts';prepared=json.loads((source/'prepared.json').read_text())
    multiplier=1/prepared['constant_scale_control'];p=json.loads((TAIL/'baseline.json').read_text())['protocol']
    with np.load(source/'inputs.npz') as z:m=Market(*(z[k].copy() for k in ['market_times','close','volume','trades','funding','observed']),tuple(p['symbols']))
    with np.load(source/'targets.npz') as z:arrays={k:z[k].copy() for k in z.files}
    save(OUT/'budget_frozen.json',{'multiplier':multiplier,'ratio_source_sha256':sha(source/'prepared.json'),
       'candidate_policy_sha256':sha(Path(__file__).with_name('budget_protocol.json')),'source_sha256':sha(__file__),
       'frozen_before_budget_results':True,'historical_independence':False,'live_orders_sent':0})
    rows=[];periods=[('august','2026-08-01','2026-09-01'),('historical_evaluation','2026-09-01','2026-10-13T12:00:00'),
        ('first_half','2026-09-01','2026-09-22'),('second_half','2026-09-22','2026-10-13T12:00:00')]
    for phase,start,end in periods:
        for policy in ['incumbent','original_risk_cap','risk_cap_budget_matched_v1']:
            for scenario in ['base','stress']:
                w=arrays['incumbent'] if policy=='incumbent' else arrays['candidate']*(multiplier if policy=='risk_cap_budget_matched_v1' else 1)
                result=simulate(m,w,arrays['retention'],arrays['sigma'],p,m.exact(ts(start)),m.exact(ts(end)),scenario,policy!='incumbent')
                result.update(phase=phase,policy=policy)
                with gzip.open(OUT/'runs'/f'budget-{phase}-{policy}-{scenario}.json.gz','wt') as f:json.dump(result,f,allow_nan=False,default=json_default)
                rows.append({k:result[k] for k in ['phase','policy','scenario','metrics','gates']})
                print(phase,policy,scenario,'ROI%',round(result['metrics']['roi']*100,6),'STD%',round(result['metrics']['daily_std_sample']*100,6),'MDD%',round(result['metrics']['max_drawdown_magnitude']*100,6),flush=True)
    checks=[]
    for phase,_,_ in periods:
        for scenario in ['base','stress']:
            rs={r['policy']:r for r in rows if r['phase']==phase and r['scenario']==scenario}
            b=rs['incumbent']['metrics'];c=rs['risk_cap_budget_matched_v1']['metrics']
            checks.append({'phase':phase,'scenario':scenario,'higher_return':c['roi']>b['roi'],
               'lower_std':c['daily_std_sample']<b['daily_std_sample'],'lower_mdd':c['max_drawdown_magnitude']<b['max_drawdown_magnitude'],
               'positive_return':c['roi']>0,'engineering':all(rs['risk_cap_budget_matched_v1']['gates'].values())})
    save(OUT/'budget_results.json',{'multiplier':multiplier,'rows':rows,'checks':checks,'historical_independence':False,'live_orders_sent':0})

if __name__=='__main__':main()
