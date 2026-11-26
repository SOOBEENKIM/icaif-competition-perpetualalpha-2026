"""Apply previously frozen August-only scales to the continuous account path."""
import copy
import gzip
import json
import time
from pathlib import Path
import numpy as np
from .research import ROOT,HERE,OUT,inputs,save,sha,verify,protected,readrun
from .portfolio import replay


def main():
    result=json.loads((OUT/'results.json').read_text())
    source=ROOT/'experiments/decision_gate_robustness_20261017/artifacts/conclusion.json'
    calibration=json.loads(source.read_text())['calibration']
    assert [(r['arm'],r['scale']) for r in calibration]==[('first_strategy',1.625),('second_strategy',1.9375)]
    cfg=json.loads((ROOT/'experiments/hourly_refresh_pairs_20261017/reference_protocol.json').read_text())
    d,_,p=inputs(cfg);spec=json.loads((HERE/'protocol.json').read_text())
    files=[Path(__file__),source,OUT/'results.json',HERE/'portfolio.py',ROOT/'experiments/risk_preserving_20261016/artifacts/inputs.npz']
    sources=json.loads((OUT/'frozen.json').read_text())['sources'].copy()
    sources.update({str(path.relative_to(ROOT)):sha(path) for path in files});verify(sources)
    save(OUT/'continuous_risk_frozen.json',dict(at_unix=time.time(),sources=sources,calibration=calibration,
        method='Reuse pre-existing August-only volatility scales without refitting on the continuous interval.',protected=protected()))
    from experiments.dense_history_pairs_20261017.research import load
    state=load(ROOT/'experiments/lifecycle_value_pairs_20261018/artifacts/reference_state.npz')
    rows=[]
    for item in calibration:
        arm=item['arm'];modified={**d}
        key='first' if arm=='first_strategy' else 'second'
        modified[key]=d[key]*item['scale']
        for scenario in ['base','stress']:
            baseline=readrun(OUT/'runs'/f'reference-continuous-{arm}-{scenario}.json.gz')
            run=replay(modified,state,None,cfg,p,arm,scenario,baseline['start'],baseline['end'])
            label=arm+'_august_vol_matched';run['label']=label;run['fixed_scale']=item['scale']
            with gzip.open(OUT/'runs'/f'reference-continuous-{label}-{scenario}.json.gz','wt') as f:
                json.dump(run,f,allow_nan=False,default=lambda x:x.item() if isinstance(x,np.generic) else x.tolist())
            m=run['metrics'];candidate=next(r['metrics'] for r in result['rows'] if r['venue']=='reference' and r['phase']=='continuous' and r['arm']==spec['candidate'] and r['scenario']==scenario)
            rows.append(dict(arm=label,scenario=scenario,scale=item['scale'],metrics=m,
                             candidate_pnl_difference=candidate['net_pnl']-m['net_pnl'],
                             candidate_q_difference=candidate['financial_score']-m['financial_score']))
            print('RISK_MATCHED',arm,scenario,'PnL',m['net_pnl'],'Q',m['financial_score'],flush=True)
    verify(sources)
    save(OUT/'continuous_risk_results.json',dict(rows=rows,new_replays=4,protected=protected(),
        scope='Scales fitted on August before the present experiment. Approximate August volatility match only; realized continuous volatility may differ. Reused historical diagnostic, no candidate reselection.',
        orders_sent=0,historical_independence=False))
    print('DONE continuous fixed-scale controls',flush=True)


if __name__=='__main__':main()
