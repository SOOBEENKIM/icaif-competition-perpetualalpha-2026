"""Same historical evaluation, 50k user sleeve plus idle 50k team reserve."""
import copy
import gzip
import json
from pathlib import Path
import numpy as np
from experiments.decision_gate_pairs_20261017.portfolio import replay
from .profile import BUNDLE,ROOT,load_profile
from .scoring import evaluate


def main():
    p=load_profile();out=ROOT/'competition_launch/artifacts/replay_50k';out.mkdir(exist_ok=True)
    data=dict(np.load(BUNDLE/'data/reference.npz',allow_pickle=False))
    state=dict(np.load(BUNDLE/'models/reference_gate_state.npz',allow_pickle=False))
    old=json.loads((BUNDLE/'expected.json').read_text());rows=[]
    for row in old['runs']:
        if row['venue']!='reference' or row['phase']!='comparison':continue
        cfg=copy.deepcopy(p['configs']['reference']);cfg['conditional_completion']={'risk_aversion':0.}
        result=replay(data,state,None,cfg,p['trading'],row['arm'],row['scenario'],row['start'],row['end'])
        samples=[dict(time=int(s['time']),equity=s['equity']+p['reserved_capital']) for s in result['snapshots'] if s['time']%86400==43200]
        team=evaluate(samples,trading_days=1)
        m=result['metrics'];report=dict(strategy=row['arm'],scenario=row['scenario'],net_pnl=m['net_pnl'],
            user_roi=m['roi'],team_roi=team['return_fraction'],team_Q=team['Q'],team_volatility=team['volatility'],
            completed_pairs=m['completed_pairs'],risk_breach_bars=m['risk_breach_bars'],
            accounting_error=m['maximum_accounting_error'],initial_user=50000,initial_team=100000,reserved_cash=50000)
        with gzip.open(out/(row['id']+'.json.gz'),'wt') as f:
            json.dump(result,f,default=lambda x:x.tolist() if hasattr(x,'tolist') else x.item())
        assert not m['risk_breach_bars'] and m['maximum_accounting_error']<1e-6
        rows.append(report);print(json.dumps(report),flush=True)
    (out/'summary.json').write_text(json.dumps(rows,indent=2)+'\n')


if __name__=='__main__':main()
