"""Replay the fixed three strategies with 100k capital and PAS lot precision."""
import copy
import csv
import gzip
import json
from pathlib import Path
import numpy as np
from experiments.decision_gate_pairs_20261017.portfolio import replay
from .profile import BUNDLE, load_profile, verify_bundle


def main(output=None):
    profile=load_profile();verify_bundle()
    out=Path(output).resolve() if output else Path(__file__).parent/'artifacts/backtest_100k'
    out.mkdir(exist_ok=False,parents=True);(out/'runs').mkdir()
    load=lambda name:dict(np.load(BUNDLE/name,allow_pickle=False))
    data=load('data/reference.npz');state=load('models/reference_gate_state.npz')
    old=json.loads((BUNDLE/'expected.json').read_text())
    rows=[]
    for r in old['runs']:
        if r['venue']!='reference':continue
        config=copy.deepcopy(profile['configs']['reference'])
        config['conditional_completion']={'risk_aversion':0.}
        result=replay(data,state,None,config,profile['trading'],r['arm'],r['scenario'],r['start'],r['end'])
        with gzip.open(out/'runs'/(r['id']+'.json.gz'),'wt') as f:
            json.dump(result,f,allow_nan=False,default=lambda v:v.tolist() if hasattr(v,'tolist') else v.item())
        row=dict(id=r['id'],phase=r['phase'],strategy=r['arm'],scenario=r['scenario'],
            start=r['start'],end=r['end'],**result['metrics'])
        rows.append(row)
        print(r['phase'],r['arm'],r['scenario'],'PnL',round(row['net_pnl'],4),'Q',row['financial_score'],flush=True)
    assert len(rows)==18
    advantages=[]
    for phase in ['validation','comparison','continuous']:
        for scenario in ['base','stress']:
            candidate=next(r for r in rows if (r['phase'],r['scenario'],r['strategy'])==(phase,scenario,profile['candidate']))
            for baseline in ['first_strategy','second_strategy']:
                control=next(r for r in rows if (r['phase'],r['scenario'],r['strategy'])==(phase,scenario,baseline))
                advantages.append(dict(phase=phase,scenario=scenario,baseline=baseline,
                    pnl_gain=candidate['net_pnl']-control['net_pnl'],Q_gain=candidate['financial_score']-control['financial_score'],
                    score_gain=candidate['score']-control['score']))
    summary=dict(capital=100000.,profile='PAS/USDP',historical_data='Reference historical data, not PAS historical execution.',
        model_changed=False,capital_and_lots_changed=True,orders_sent=0,rows=rows,advantages=advantages,
        all_pnl_Q_improved=all(r['pnl_gain']>0 and r['Q_gain']>0 for r in advantages))
    (out/'results.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n')
    fields=['phase','strategy','scenario','net_pnl','roi','volatility_horizon','financial_score','score','max_drawdown','completed_pairs']
    with (out/'comparison.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        writer.writerows({k:r[k] for k in fields} for r in rows)
    print('DONE',summary['all_pnl_Q_improved'],flush=True)


if __name__=='__main__':main()
