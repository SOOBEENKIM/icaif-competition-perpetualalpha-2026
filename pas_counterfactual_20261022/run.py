import copy,gzip,json
import numpy as np
from .prepare import ROOT,OUT,START,END
from .engine import simulate
from pas_economic_allocation.lifecycle import LifecycleController
from pas_adaptive_development.study import save,sha,native


def main():
    from competition_live.profile import load_profile,verify_deployment
    from pas_strategy_audit.data import load
    p=json.loads((OUT/'protocol.json').read_text())
    for path,digest in json.loads((OUT/'source_freeze.json').read_text()).items():assert sha(ROOT/path)==digest,path
    pairs=json.loads((OUT/'pair_forecasts.json').read_text())
    actions=[f for row in pairs for f in row['forecasts'] if f['model']=='decision_gate']
    assert len(actions)==96 and all(f['economic_available'] and f['economic_action']==0 for f in actions)
    cfg,d=load(ROOT/'reports/pas_allocation_study_20261021/pas');cfg['cutoff_ms']=END*1000
    profile=load_profile();events=json.loads((OUT/'events.json').read_text());z=dict(np.load(OUT/'candidate_inputs.npz'))
    tape=json.loads((OUT/'trade_tape.json').read_text());funding=json.loads((OUT/'funding_tape.json').read_text())
    known={(r['coin'],r['time']//3600000) for r in funding}
    for r in d['funding_events']:
        coin='pas:'+cfg['symbols'][r['j']]
        if (coin,r['time']//3600000) not in known:
            funding.append(dict(coin=coin,time=r['time'],rate=r['rate'],cash_per_unit=None))
    scenarios=copy.deepcopy(p['scenarios']);scenarios['optimistic_100pct']={**scenarios['base_1pct'],'participation':1.}
    # This is an execution-assumption extension, sealed before any run results.
    freeze={str(f.relative_to(ROOT)):sha(f) for f in [* (ROOT/'pas_counterfactual_20261022').glob('*.py'),
        OUT/'trade_tape.json',OUT/'funding_tape.json',OUT/'tape_protocol.json']}
    path=OUT/'execution_freeze.json'
    if path.exists():assert json.loads(path.read_text())==freeze
    else:save(path,freeze)
    rows=[]
    for name,scenario in scenarios.items():
        for arm in p['arms']:
            signals=copy.deepcopy(events);controller=None
            if arm=='size_only':
                for e in signals:e['weights']=e['size_only']
            if arm=='budget_lifecycle':
                controller=LifecycleController(z['times'],z['forecast'],z['carry'],z['anchor'],z['protected'],5.)
            r=simulate(cfg,d,signals,tape,funding,profile['trading'],scenario,controller=controller)
            r['allocation_trace']=controller.trace if controller else []
            row=dict(arm=arm,scenario=name,**r['metrics']);rows.append(row)
            with gzip.open(OUT/f'{arm}_{name}.json.gz','wt') as f:json.dump(r,f,default=native,allow_nan=False)
            save(OUT/'results.json',rows);print(json.dumps(row),flush=True)
    assert verify_deployment()['files']==p['live_original']


if __name__=='__main__':main()
