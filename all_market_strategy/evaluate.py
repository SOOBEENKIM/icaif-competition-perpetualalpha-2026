"""Matched account replay: change the price-signal universe, nothing else."""
import copy
import gzip
import json
from pathlib import Path

import numpy as np

from competition_live.profile import ROOT, BUNDLE, load_profile
from competition_live.scoring import evaluate
from experiments.decision_gate_pairs_20261017.portfolio import replay

OUT=ROOT/'reports/all_markets_extension_20261020'


def main():
    p=load_profile()
    with np.load(OUT/'targets.npz',allow_pickle=False) as z:
        times=z['times'].copy(); weights=z['weights'].copy(); legacy=z['legacy'].copy()
    data=dict(np.load(BUNDLE/'data/reference.npz',allow_pickle=False))
    state=dict(np.load(BUNDLE/'models/reference_gate_state.npz',allow_pickle=False))
    ids=np.searchsorted(times,data['times'])
    np.testing.assert_array_equal(times[ids],data['times'])
    np.testing.assert_allclose(legacy[ids],data['first'],rtol=1e-11,atol=1e-13)
    source=json.loads((BUNDLE/'expected.json').read_text())
    rows=[]
    for row in source['runs']:
        if (row['venue']!='reference' or row['phase'] not in ('validation','comparison')
                or row['arm']!=p['candidate']): continue
        for variant in ('incumbent','all_markets'):
            d=dict(data)
            if variant=='all_markets': d['first']=weights[ids].copy()
            cfg=copy.deepcopy(p['configs']['reference']); cfg['conditional_completion']={'risk_aversion':0.}
            result=replay(d,state,None,cfg,p['trading'],row['arm'],row['scenario'],row['start'],row['end'])
            samples=[dict(time=int(s['time']),equity=s['equity']+p['reserved_capital'])
                     for s in result['snapshots'] if s['time']%86400==43200]
            team=evaluate(samples,trading_days=1); m=result['metrics']
            report=dict(variant=variant,phase=row['phase'],scenario=row['scenario'],
                start=row['start'],end=row['end'],net_pnl=m['net_pnl'],
                team_return=team['return_fraction'],Q=team['Q'],volatility=team['volatility'],
                completed_pairs=m['completed_pairs'],risk_breach_bars=m['risk_breach_bars'],
                accounting_error=m['maximum_accounting_error'],metrics=m)
            assert not m['risk_breach_bars'] and m['maximum_accounting_error']<1e-6
            name=f"{variant}_{row['phase']}_{row['scenario']}"
            with gzip.open(OUT/(name+'.json.gz'),'wt') as f:
                json.dump(result,f,default=lambda x:x.tolist() if hasattr(x,'tolist') else x.item())
            rows.append(report)
            (OUT/'comparison.json').write_text(json.dumps(rows,indent=2))
            print(json.dumps({k:v for k,v in report.items() if k!='metrics'}),flush=True)
    indexed={(r['variant'],r['phase'],r['scenario']):r for r in rows}
    checks=[]
    for phase in ('validation','comparison'):
        for scenario in ('base','stress'):
            a=indexed['incumbent',phase,scenario]; b=indexed['all_markets',phase,scenario]
            checks.append(dict(phase=phase,scenario=scenario,pnl_difference=b['net_pnl']-a['net_pnl'],
                               Q_difference=b['Q']-a['Q']))
    (OUT/'decision.json').write_text(json.dumps(dict(comparisons=checks,
        both_metrics_improved_all_windows=all(r['pnl_difference']>0 and r['Q_difference']>0 for r in checks),
        historical_independence=False,venue='reference_proxy_not_PAS_fills'),indent=2))


if __name__=='__main__': main()
